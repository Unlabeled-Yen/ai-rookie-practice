"""AI4I 2020 一鍵重現流程（#13）。

依相依順序呼叫各子單模組的 `run()`，最後由 `report` 產生 `docs/ai4i_paper.md` 與 `docs/ai4i_notes.md`。

    python -m ai4i_lab.pipeline                  # 完整重跑（耗時：MLP 搜尋、重複 CV、可解釋性最久）
    python -m ai4i_lab.pipeline --list           # 列出階段
    python -m ai4i_lab.pipeline --skip-existing  # 輸出已存在且上游沒有重跑的階段跳過
    python -m ai4i_lab.pipeline --only rules,cost
    python -m ai4i_lab.pipeline --report-only    # 只用現有結果重新產生論文與筆記
"""
import argparse
import importlib
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Stage:
    name: str
    module: str
    title: str
    deps: tuple
    outputs: tuple          # 相對於專案根目錄；全部存在才視為「已有輸出」
    est_minutes: float      # 2 顆 CPU 上的粗估（僅供 --list 顯示）


STAGES = (
    Stage("data", "ai4i_lab.data", "#4 資料品質檢查與 EDA", (), ("outputs/ai4i/eda/eda.json",), 1),
    Stage("rules", "ai4i_lab.rules", "#5 已知規則重建", ("data",), ("outputs/ai4i/rules/rule_agreement.json",), 1),
    Stage("pca", "ai4i_lab.pca", "#6 特徵工程與 PCA", ("data",), ("outputs/ai4i/pca/pca.json",), 2),
    Stage("framework", "ai4i_lab.evaluation", "#7 評估框架（切分、重複 CV、穩健性）", ("data",),
          ("outputs/ai4i/splits.json", "outputs/ai4i/framework/framework.json"), 5),
    Stage("models", "ai4i_lab.models", "#8 二元分類模型比較與 MLP 調參", ("pca", "framework"),
          ("outputs/ai4i/models/models.json", "outputs/ai4i/models/predictions_test.csv", "outputs/ai4i/models/cv_results.csv"), 30),
    Stage("multilabel", "ai4i_lab.multilabel", "#9 多標籤（故障模式）分類", ("pca", "framework"),
          ("outputs/ai4i/multilabel/multilabel.json",), 12),
    Stage("viz", "ai4i_lab.viz", "#10 評估視覺化與錯誤分析", ("rules", "models"),
          ("outputs/ai4i/evaluation/evaluation.json", "outputs/ai4i/evaluation/error_analysis.json"), 3),
    Stage("explain", "ai4i_lab.explain", "#11 可解釋性分析（A、B）", ("rules", "models"),
          ("outputs/ai4i/explain/explain.json",), 16),
    Stage("cost", "ai4i_lab.cost", "#12 成本敏感決策（C）", ("models",), ("outputs/ai4i/cost/cost.json",), 1),
    Stage("report", "ai4i_lab.report", "#13 小論文與學習筆記",
          ("data", "rules", "pca", "framework", "models", "multilabel", "viz", "explain", "cost"),
          ("docs/ai4i_paper.md", "docs/ai4i_notes.md"), 0.1),
)
STAGE_NAMES = [s.name for s in STAGES]


class PipelineError(RuntimeError):
    """某個階段失敗。`stage` 是階段名稱，原始例外在 `__cause__`。"""

    def __init__(self, stage: str, cause: BaseException):
        super().__init__(f"階段 {stage!r} 失敗：{type(cause).__name__}: {cause}")
        self.stage = stage
        self.__cause__ = cause


def ensure_dataset(path=None):
    """資料檔不存在時才從 UCI 下載。"""
    from ai4i_lab import data
    path = Path(path or data.DEFAULT_PATH)
    if not path.exists():
        data.download_dataset()
    return path


def resolve_runner(stage: Stage):
    return importlib.import_module(stage.module).run


def _default_runner(stage: Stage):
    if stage.name == "data":
        ensure_dataset()
    return resolve_runner(stage)()


def run(only=None, skip_existing: bool = False, report_only: bool = False, root=ROOT, runner=None, log=print) -> dict:
    """依序執行階段。

    skip_existing：階段輸出都已存在、且它依賴的階段在這次呼叫中都沒有重跑，才會跳過
    （上游重跑後，下游沿用舊輸出會與新結果不一致）。
    """
    root = Path(root)
    runner = runner or _default_runner
    if report_only:
        only = ["report"]
    if only is not None:
        unknown = [n for n in only if n not in STAGE_NAMES]
        if unknown:
            raise ValueError(f"未知的階段：{unknown}；可用 {STAGE_NAMES}")
    selected = [s for s in STAGES if only is None or s.name in only]
    ran, records = set(), []
    t_all = time.perf_counter()
    for i, s in enumerate(selected, 1):
        fresh = all((root / o).exists() for o in s.outputs)
        stale_upstream = bool(set(s.deps) & ran)
        if skip_existing and fresh and not stale_upstream:
            log(f"[{i}/{len(selected)}] {s.name:<10} 跳過（輸出已存在）")
            records.append({"name": s.name, "status": "skipped", "seconds": 0.0})
            continue
        log(f"[{i}/{len(selected)}] {s.name:<10} {s.title} ...")
        t0 = time.perf_counter()
        try:
            runner(s)
        except Exception as exc:  # noqa: BLE001  任何階段失敗都要停止並指出是哪一階段
            records.append({"name": s.name, "status": "failed", "seconds": time.perf_counter() - t0})
            raise PipelineError(s.name, exc) from exc
        dt = time.perf_counter() - t0
        ran.add(s.name)
        records.append({"name": s.name, "status": "ran", "seconds": dt})
        log(f"[{i}/{len(selected)}] {s.name:<10} 完成（{dt:.1f} 秒）")
    return {"ok": True, "stages": records, "seconds": time.perf_counter() - t_all}


def _print_list():
    print(f"{'階段':<11}{'依賴':<44}{'粗估(分)':<9}說明")
    for s in STAGES:
        print(f"{s.name:<11}{','.join(s.deps) or '-':<44}{s.est_minutes:<9g}{s.title}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ai4i_lab.pipeline", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="列出階段後結束")
    ap.add_argument("--only", help="只跑指定階段（逗號分隔，仍依流程順序）")
    ap.add_argument("--skip-existing", action="store_true", help="輸出已存在且上游未重跑的階段跳過")
    ap.add_argument("--report-only", action="store_true", help="只產生論文與筆記")
    args = ap.parse_args(argv)
    if args.list:
        _print_list()
        return 0
    only = [x.strip() for x in args.only.split(",") if x.strip()] if args.only else None
    bad = [n for n in (only or []) if n not in STAGE_NAMES]
    if bad:
        print(f"未知的階段：{bad}；可用 {STAGE_NAMES}", file=sys.stderr)
        return 2
    try:
        run(only=only, skip_existing=args.skip_existing, report_only=args.report_only)
    except PipelineError as e:
        print(str(e), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
