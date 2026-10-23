"""#13 report.py 測試：格式化工具、結果載入、事實彙整、論文與筆記產生（TDD 多輪）。"""
import copy
import json
import re
from pathlib import Path

import pytest

from ai4i_lab import report as R

OUT = Path(__file__).resolve().parent.parent / "outputs" / "ai4i"
HAVE_OUTPUTS = all((OUT / p).exists() for p in R.RESULT_FILES.values())
needs_outputs = pytest.mark.skipif(not HAVE_OUTPUTS, reason="需要先跑完 #4–#12 的實驗輸出")


# ---------- 第 1 輪：格式化工具 ----------
def test_fmt_num_rounds_and_handles_missing():
    assert R.fmt_num(0.123456) == "0.123"
    assert R.fmt_num(0.123456, 2) == "0.12"
    assert R.fmt_num(None) == "n/a"
    assert R.fmt_num(float("nan")) == "n/a"


def test_fmt_pct():
    assert R.fmt_pct(0.0339) == "3.4%"
    assert R.fmt_pct(0.0339, 2) == "3.39%"
    assert R.fmt_pct(None) == "n/a"


def test_fmt_ci():
    assert R.fmt_ci(0.9, 0.85, 0.95) == "0.900 [0.850, 0.950]"
    assert R.fmt_ci(0.9, None, None) == "0.900"
    assert R.fmt_ci(0.9, float("nan"), 0.95) == "0.900"


def test_md_table_layout_and_pipe_escape():
    t = R.md_table(["a", "b"], [[1, "x|y"], [2, "z"]])
    lines = t.splitlines()
    assert lines[0] == "| a | b |"
    assert lines[1] == "|---|---|"
    assert lines[2] == "| 1 | x\\|y |"
    assert len(lines) == 4


def test_md_table_rejects_ragged_rows():
    with pytest.raises(ValueError):
        R.md_table(["a", "b"], [[1]])


def test_img_uses_relative_prefix():
    assert R.img("eda/x.png", "說明", prefix="../outputs/ai4i") == "![說明](../outputs/ai4i/eda/x.png)"


# ---------- 第 2 輪：載入結果 ----------
def test_load_results_reports_every_missing_file(tmp_path):
    with pytest.raises(R.MissingResults) as ei:
        R.load_results(tmp_path)
    msg = str(ei.value)
    for rel in R.RESULT_FILES.values():
        assert rel in msg


def test_load_results_reads_json_and_csv(tmp_path):
    for key, rel in R.RESULT_FILES.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if rel.endswith(".csv"):
            p.write_text("a,b\n1,2\n", encoding="utf-8")
        else:
            p.write_text(json.dumps({"key": key}), encoding="utf-8")
    res = R.load_results(tmp_path)
    assert set(res) == set(R.RESULT_FILES)
    csv_keys = [k for k, v in R.RESULT_FILES.items() if v.endswith(".csv")]
    assert csv_keys, "至少要讀一份 CSV（CV 結果表）"
    assert res[csv_keys[0]] == [{"a": 1, "b": 2}]
    jk = next(k for k, v in R.RESULT_FILES.items() if v.endswith(".json"))
    assert res[jk] == {"key": jk}


# ---------- 第 3 輪：事實彙整 ----------
@pytest.fixture(scope="module")
def res():
    return R.load_results(OUT)


@needs_outputs
def test_facts_match_source_json(res):
    f = R.collect_facts(res)
    assert f["n_rows"] == 10000 and f["n_failures"] == 339
    assert f["failure_rate"] == pytest.approx(0.0339)
    best = res["models"]["conclusions"]["best_cv"]
    assert f["best_cv_model"] == best["model"] and f["best_cv_feature_set"] == best["feature_set"]
    assert f["best_cv_pr_auc"] == pytest.approx(best["pr_auc"])
    assert f["cost_break_even_ratio"] == pytest.approx(res["cost"]["break_even_ratio"])
    assert f["rnf_consistent_with_random"] is res["multilabel"]["rnf"]["consistent_with_random"]


@needs_outputs
def test_facts_never_contain_nan(res):
    import math
    f = R.collect_facts(res)
    bad = [k for k, v in f.items() if isinstance(v, float) and math.isnan(v)]
    assert bad == []


@needs_outputs
def test_test_table_rows_cover_all_models_and_feature_sets(res):
    rows = R.test_table_rows(res)
    assert len(rows) == 8
    names = {(r[0], r[1]) for r in rows}
    assert ("mlp", "raw") in names and ("random_forest", "engineered") in names
    assert all("[" in r[3] for r in rows), "測試集指標要附 95% 信賴區間"


# ---------- 第 4 輪：小論文 ----------
PAPER_SECTIONS = ["摘要", "1. 前言", "2. 文獻回顧", "3. 資料與探索", "4. 方法", "5. 結果", "6. 討論", "7. 結論", "參考文獻"]


@pytest.fixture(scope="module")
def paper(res):
    return R.render_paper(res)


@needs_outputs
def test_paper_has_required_sections_in_order(paper):
    heads = re.findall(r"^## (.+)$", paper, flags=re.M)
    idx = []
    for s in PAPER_SECTIONS:
        hit = [i for i, h in enumerate(heads) if h.startswith(s)]
        assert hit, f"缺少章節：{s}"
        idx.append(hit[0])
    assert idx == sorted(idx)


@needs_outputs
def test_paper_discussion_has_limitations_subsection(paper):
    assert re.search(r"^### 6\.\d+ 研究限制", paper, flags=re.M)


@needs_outputs
def test_paper_answers_three_research_questions_explicitly(paper):
    for q in "ABC":
        assert re.search(rf"^### .*研究問題 {q}", paper, flags=re.M), f"結果章缺少研究問題 {q} 的小節"
        assert f"**回答 {q}**" in paper, f"缺少研究問題 {q} 的明確回答"
    abstract = paper.split("## 1. 前言")[0]
    for q in "ABC":
        assert f"（{q}）" in abstract or f"{q}：" in abstract


@needs_outputs
def test_paper_states_the_leakage_and_not_forecasting_caveat(paper):
    assert "故障當下" in paper
    assert "資料洩漏" in paper


@needs_outputs
def test_paper_cites_matzka_and_core_references(paper):
    refs = paper.split("## 參考文獻")[1]
    assert "Matzka" in refs and "2020" in refs
    assert "Nadeau" in refs and "Chawla" in refs


@needs_outputs
def test_paper_has_no_unfilled_placeholders(paper):
    for bad in ["{", "}", "TODO", "XXX"]:
        assert bad not in paper, f"出現未填值：{bad}"
    assert not re.search(r"\b(nan|None|inf)\b", paper), "出現 nan／None／inf"


@needs_outputs
def test_paper_images_exist_on_disk(paper):
    docs = Path(__file__).resolve().parent.parent / "docs"
    paths = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", paper)
    assert len(paths) >= 12
    for p in paths:
        assert (docs / p).resolve().exists(), f"圖檔不存在：{p}"


@needs_outputs
def test_paper_numbers_come_from_results_not_hardcoded(res):
    """改動結果檔中的數字，論文文字必須跟著變（證明沒有手動抄寫）。"""
    r2 = copy.deepcopy(res)
    for row in r2["evaluation"]["table"]:
        if (row["model"], row["feature_set"]) == ("random_forest", "engineered"):
            row["pr_auc"], row["pr_auc_low"], row["pr_auc_high"] = 0.12345, 0.11111, 0.13579
    r2["models"]["conclusions"]["best_cv"]["pr_auc"] = 0.24680
    r2["cost"]["break_even_ratio"] = 0.7777
    r2["eda"]["exploration"]["n_failures"] = 4321
    text = R.render_paper(r2)
    for token in ["0.123 [0.111, 0.136]", "0.247", "0.778", "4,321"]:
        assert token in text, f"論文沒有反映改動後的數字：{token}"
    assert "0.901 [0.835, 0.954]" not in text


@needs_outputs
def test_paper_key_numbers_match_source(paper, res):
    f = R.collect_facts(res)
    assert f"{f['n_rows']:,}" in paper and f"{f['n_failures']:,}" in paper
    row = next(r for r in res["evaluation"]["table"] if (r["model"], r["feature_set"]) == ("random_forest", "engineered"))
    assert R.fmt_ci(row["pr_auc"], row["pr_auc_low"], row["pr_auc_high"]) in paper
    assert R.fmt_num(res["cost"]["break_even_ratio"]) in paper


@needs_outputs
def test_paper_reports_inconvenient_findings_honestly(paper, res):
    # TWF 無法重建、MLP 不如樹模型、時間切分表現下降都必須如實寫入
    assert "TWF" in paper and "無法" in paper
    tm = res["framework"]["conclusions"]["mean_pr_auc_holdout"]
    assert R.fmt_num(tm["temporal_test"]) in paper
    assert R.fmt_num(tm["stratified_val"]) in paper


# ---------- 第 5 輪：學習筆記 ----------
@pytest.fixture(scope="module")
def notes(res):
    return R.render_notes(res)


@needs_outputs
def test_notes_every_step_has_purpose_method_reading_and_pitfalls(notes):
    steps = re.split(r"^## ", notes, flags=re.M)[1:]
    step_blocks = [s for s in steps if re.match(r"步驟 \d+", s)]
    assert len(step_blocks) >= 10
    for blk in step_blocks:
        title = blk.splitlines()[0]
        for label in ["目的", "做法", "結果判讀", "踩到的坑"]:
            assert f"**{label}**" in blk, f"{title} 缺少「{label}」"


@needs_outputs
def test_notes_cover_all_analysis_stages(notes):
    for kw in ["資料取得", "探索", "規則重建", "特徵工程", "PCA", "評估框架", "二元分類", "多標籤", "錯誤分析", "可解釋性", "成本"]:
        assert re.search(rf"^## 步驟 \d+.*{kw}", notes, flags=re.M), f"缺少步驟：{kw}"


@needs_outputs
def test_notes_mention_key_pitfalls(notes):
    for kw in ["資料洩漏", "標記不一致", "SMOTE", "校準", "TWF", "RNF", "時間"]:
        assert kw in notes, f"筆記沒有提到：{kw}"


@needs_outputs
def test_notes_has_glossary_and_review_questions_and_reproduce_section(notes):
    for head in ["名詞對照", "自我檢查", "如何重現"]:
        assert re.search(rf"^## .*{head}", notes, flags=re.M), f"缺少章節：{head}"
    assert "python -m ai4i_lab.pipeline" in notes


@needs_outputs
def test_notes_images_exist_and_no_placeholders(notes):
    docs = Path(__file__).resolve().parent.parent / "docs"
    for p in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", notes):
        assert (docs / p).resolve().exists(), p
    assert not re.search(r"\b(nan|None|inf)\b", notes)
    assert "{" not in notes and "}" not in notes


@needs_outputs
def test_notes_numbers_are_generated_from_results(res):
    r2 = copy.deepcopy(res)
    r2["eda"]["exploration"]["n_failures"] = 4321
    r2["rules"]["tool_cycles"]["n_cycles"] = 777
    r2["cost"]["break_even_ratio"] = 0.7777
    t = R.render_notes(r2)
    for token in ["4,321", "777", "0.778"]:
        assert token in t, token


# ---------- 第 6 輪：寫檔與入口 ----------
@needs_outputs
def test_write_reports_creates_both_files_with_utf8(tmp_path):
    paths = R.write_reports(out_dir=OUT, docs_dir=tmp_path)
    assert (tmp_path / "ai4i_paper.md").read_text(encoding="utf-8").startswith("# ")
    assert (tmp_path / "ai4i_notes.md").read_text(encoding="utf-8").startswith("# ")
    assert set(paths) == {"paper", "notes"}


@needs_outputs
def test_write_reports_is_deterministic(tmp_path):
    R.write_reports(out_dir=OUT, docs_dir=tmp_path / "a")
    R.write_reports(out_dir=OUT, docs_dir=tmp_path / "b")
    for name in ["ai4i_paper.md", "ai4i_notes.md"]:
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()


def test_write_reports_fails_loudly_without_results(tmp_path):
    with pytest.raises(R.MissingResults):
        R.write_reports(out_dir=tmp_path / "empty", docs_dir=tmp_path / "docs")
    assert not (tmp_path / "docs" / "ai4i_paper.md").exists(), "失敗時不可留下半成品"


@needs_outputs
def test_committed_docs_are_in_sync_with_results():
    """已提交的 docs 必須就是目前結果重新產生的內容（防止手改或過期）。"""
    docs = Path(__file__).resolve().parent.parent / "docs"
    if not (docs / "ai4i_paper.md").exists():
        pytest.skip("尚未產生 docs")
    res = R.load_results(OUT)
    assert (docs / "ai4i_paper.md").read_text(encoding="utf-8") == R.render_paper(res)
    assert (docs / "ai4i_notes.md").read_text(encoding="utf-8") == R.render_notes(res)
