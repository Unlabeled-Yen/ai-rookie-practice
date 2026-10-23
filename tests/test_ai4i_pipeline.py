"""#13 pipeline.py 測試：階段順序、依賴、跳過、失敗處理與命令列（TDD）。"""
import importlib
import inspect
from pathlib import Path

import pytest

from ai4i_lab import pipeline as PL

ROOT = Path(__file__).resolve().parent.parent
EXPECTED_ORDER = ["data", "rules", "pca", "framework", "models", "multilabel", "viz", "explain", "cost", "report"]


# ---------- 第 1 輪：階段定義 ----------
def test_stage_order_respects_issue_dependencies():
    names = [s.name for s in PL.STAGES]
    assert names == EXPECTED_ORDER


def test_every_dependency_comes_earlier_in_the_order():
    seen = set()
    for s in PL.STAGES:
        assert set(s.deps) <= seen, f"{s.name} 依賴尚未執行的階段 {set(s.deps) - seen}"
        seen.add(s.name)


def test_known_dependencies_match_the_dag_in_the_parent_issue():
    d = {s.name: set(s.deps) for s in PL.STAGES}
    assert "framework" in d["models"] and "framework" in d["multilabel"]      # #8、#9 需要 #7 的 splits
    assert {"models", "rules"} <= d["viz"]                                    # #10 需要 #5、#8
    assert {"models", "rules"} <= d["explain"]                                # #11 需要 #5、#8
    assert "models" in d["cost"]                                              # #12 需要 #8
    assert {"viz", "explain", "cost", "multilabel", "pca", "rules", "models"} <= d["report"]


def test_every_stage_resolves_to_a_real_run_function():
    for s in PL.STAGES:
        mod = importlib.import_module(s.module)
        assert callable(getattr(mod, "run")), s.module
        assert s.outputs, f"{s.name} 沒有宣告輸出檔"
        assert inspect.getdoc(PL.resolve_runner(s)) is not None or callable(PL.resolve_runner(s))


def test_declared_outputs_are_inside_outputs_or_docs():
    for s in PL.STAGES:
        for o in s.outputs:
            assert o.startswith(("outputs/ai4i/", "docs/")), o


# ---------- 第 2 輪：執行 ----------
class Recorder:
    def __init__(self, fail_on=None):
        self.calls, self.fail_on = [], fail_on

    def __call__(self, stage):
        self.calls.append(stage.name)
        if stage.name == self.fail_on:
            raise RuntimeError("boom")
        return {"ok": True}


def test_run_executes_all_stages_in_order():
    rec = Recorder()
    out = PL.run(runner=rec, log=lambda *_: None)
    assert rec.calls == EXPECTED_ORDER
    assert out["ok"] is True
    assert [r["status"] for r in out["stages"]] == ["ran"] * len(EXPECTED_ORDER)
    assert all(r["seconds"] >= 0 for r in out["stages"])


def test_run_stops_at_first_failure_and_reports_it():
    rec = Recorder(fail_on="models")
    with pytest.raises(PL.PipelineError) as ei:
        PL.run(runner=rec, log=lambda *_: None)
    assert ei.value.stage == "models"
    assert rec.calls == EXPECTED_ORDER[: EXPECTED_ORDER.index("models") + 1]
    assert isinstance(ei.value.__cause__, RuntimeError)


def test_only_selects_named_stages_but_keeps_pipeline_order():
    rec = Recorder()
    PL.run(only=["cost", "rules"], runner=rec, log=lambda *_: None)
    assert rec.calls == ["rules", "cost"]


def test_only_rejects_unknown_stage_names():
    with pytest.raises(ValueError, match="nope"):
        PL.run(only=["rules", "nope"], runner=Recorder(), log=lambda *_: None)


def test_report_only_runs_just_the_report_stage():
    rec = Recorder()
    PL.run(report_only=True, runner=rec, log=lambda *_: None)
    assert rec.calls == ["report"]


# ---------- 第 3 輪：skip-existing ----------
def _touch_all(root: Path):
    for s in PL.STAGES:
        for o in s.outputs:
            p = root / o
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("{}")


def test_skip_existing_skips_everything_when_outputs_present(tmp_path):
    _touch_all(tmp_path)
    rec = Recorder()
    out = PL.run(skip_existing=True, root=tmp_path, runner=rec, log=lambda *_: None)
    assert rec.calls == []
    assert {r["status"] for r in out["stages"]} == {"skipped"}


def test_skip_existing_reruns_stage_with_missing_output_and_all_its_downstream(tmp_path):
    _touch_all(tmp_path)
    models = next(s for s in PL.STAGES if s.name == "models")
    (tmp_path / models.outputs[0]).unlink()
    rec = Recorder()
    PL.run(skip_existing=True, root=tmp_path, runner=rec, log=lambda *_: None)
    downstream = {s.name for s in PL.STAGES if "models" in s.deps} | {"models"}
    assert downstream <= set(rec.calls), "上游重跑後，依賴它的階段不可沿用舊輸出"
    for untouched in ["data", "rules", "pca", "framework"]:
        assert untouched not in rec.calls


def test_without_skip_existing_everything_reruns_even_if_outputs_exist(tmp_path):
    _touch_all(tmp_path)
    rec = Recorder()
    PL.run(root=tmp_path, runner=rec, log=lambda *_: None)
    assert rec.calls == EXPECTED_ORDER


# ---------- 第 4 輪：資料檢查與命令列 ----------
def test_ensure_dataset_downloads_only_when_missing(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("ai4i_lab.data.download_dataset", lambda *a, **k: calls.append(1) or tmp_path / "x.csv")
    PL.ensure_dataset(path=tmp_path / "missing.csv")
    assert calls == [1]
    (tmp_path / "there.csv").write_text("a")
    PL.ensure_dataset(path=tmp_path / "there.csv")
    assert calls == [1]


def test_main_list_prints_every_stage_without_running(capsys, monkeypatch):
    monkeypatch.setattr(PL, "run", lambda **k: pytest.fail("--list 不應執行任何階段"))
    assert PL.main(["--list"]) == 0
    out = capsys.readouterr().out
    for name in EXPECTED_ORDER:
        assert name in out


def test_main_passes_flags_to_run(monkeypatch):
    seen = {}
    monkeypatch.setattr(PL, "run", lambda **k: seen.update(k) or {"ok": True, "stages": []})
    assert PL.main(["--skip-existing", "--only", "rules,cost"]) == 0
    assert seen["skip_existing"] is True and seen["only"] == ["rules", "cost"] and seen["report_only"] is False
    seen.clear()
    assert PL.main(["--report-only"]) == 0
    assert seen["report_only"] is True


def test_main_returns_nonzero_on_failure(monkeypatch, capsys):
    def boom(**k):
        raise PL.PipelineError("models", RuntimeError("x"))
    monkeypatch.setattr(PL, "run", boom)
    assert PL.main([]) == 1
    assert "models" in capsys.readouterr().err


def test_main_rejects_unknown_stage_with_exit_code_2(capsys):
    assert PL.main(["--only", "nope"]) == 2
