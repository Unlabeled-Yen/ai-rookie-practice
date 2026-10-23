"""#10 評估視覺化與錯誤分析：測試。"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ai4i_lab import rules

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "outputs" / "ai4i" / "models" / "predictions_test.csv"
MODELS_JSON = ROOT / "outputs" / "ai4i" / "models" / "models.json"
EVAL_DIR = ROOT / "outputs" / "ai4i" / "evaluation"
needs_upstream = pytest.mark.skipif(not (PRED.exists() and MODELS_JSON.exists()), reason="需要 #8 的輸出")


def _viz():
    from ai4i_lab import viz
    return viz


def _png(p):
    return Path(p).exists() and Path(p).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def mk(rows):
    """rows: (type, air, proc, rpm, torque, wear)"""
    return pd.DataFrame(rows, columns=["type", "air_temp_k", "process_temp_k", "rotational_speed_rpm",
                                       "torque_nm", "tool_wear_min"])


# ---------- 第 1 輪：混淆矩陣與校準 ----------

def test_confusion_matrix_hand_computed():
    y = [1, 1, 1, 0, 0, 0, 0]
    p = [0.9, 0.6, 0.2, 0.7, 0.1, 0.3, 0.5]
    cm = _viz().confusion_counts(y, p, 0.5)
    assert cm == {"tp": 2, "fn": 1, "fp": 2, "tn": 2}  # proba >= 0.5 為正，0.5 屬於正


def test_calibration_bins_hand_computed():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    b = _viz().calibration_bins(y, p, n_bins=2, strategy="quantile")
    assert [x["n"] for x in b["bins"]] == [2, 2]
    assert b["bins"][0]["mean_pred"] == pytest.approx(0.15)
    assert b["bins"][0]["frac_pos"] == 0.0
    assert b["bins"][1]["mean_pred"] == pytest.approx(0.85)
    assert b["bins"][1]["frac_pos"] == 1.0
    # ECE = 0.5*|0.15-0| + 0.5*|0.85-1|
    assert b["ece"] == pytest.approx(0.15)
    assert b["brier"] == pytest.approx(np.mean([0.01, 0.04, 0.04, 0.01]))


def test_calibration_perfect_has_zero_ece():
    y = np.array([0, 0, 0, 0, 1])
    p = np.array([0.2] * 5)
    b = _viz().calibration_bins(y, p, n_bins=1)
    assert b["ece"] == pytest.approx(0.0)


def test_calibration_bins_skip_empty_and_cover_all_rows():
    rng = np.random.default_rng(0)
    p = rng.random(200) ** 4
    y = (rng.random(200) < p).astype(int)
    for strat in ("uniform", "quantile"):
        b = _viz().calibration_bins(y, p, n_bins=10, strategy=strat)
        assert sum(x["n"] for x in b["bins"]) == 200
        assert all(x["n"] > 0 for x in b["bins"])
    with pytest.raises(ValueError):
        _viz().calibration_bins(y, p, strategy="x")


# ---------- 第 2 輪：bootstrap 指標表、模式偵測率、規則邊界距離 ----------

def test_bootstrap_metrics_point_matches_binary_metrics_and_reproducible():
    from ai4i_lab.evaluation import binary_metrics
    rng = np.random.default_rng(1)
    y = (rng.random(300) < 0.1).astype(int)
    p = np.clip(y * 0.4 + rng.random(300) * 0.6, 0, 1)
    a = _viz().bootstrap_metrics(y, p, 0.5, n_boot=60, seed=7)
    b = _viz().bootstrap_metrics(y, p, 0.5, n_boot=60, seed=7)
    assert a == b
    ref = binary_metrics(y, p, 0.5)
    for m in ("recall", "precision", "f1", "pr_auc", "roc_auc"):
        assert a[m]["point"] == pytest.approx(ref[m])
        assert a[m]["low"] <= a[m]["high"]
        assert 0 <= a[m]["low"] and a[m]["high"] <= 1


def test_bootstrap_metrics_perfect_separation_collapses_to_one():
    y = np.array([0] * 50 + [1] * 10)
    p = np.where(y == 1, 0.9, 0.1)
    r = _viz().bootstrap_metrics(y, p, 0.5, n_boot=40, seed=0)
    for m in ("recall", "precision", "f1", "pr_auc", "roc_auc"):
        assert (r[m]["low"], r[m]["high"]) == (1.0, 1.0)


def test_mode_detection_hand_computed():
    df = pd.DataFrame({
        "machine_failure": [1, 1, 1, 1, 0, 0],
        "twf": [1, 0, 0, 0, 0, 0],
        "hdf": [0, 1, 1, 0, 0, 0],
        "pwf": [0, 0, 0, 0, 0, 0],
        "osf": [0, 0, 1, 0, 0, 0],
        "rnf": [0, 0, 0, 0, 1, 0],
    })
    pred = np.array([True, True, False, False, False, True])
    r = _viz().mode_detection(df, pred)
    assert r["hdf"] == {"n_failures": 2, "detected": 1, "rate": 0.5, "n_rows_with_mode": 2}
    assert r["twf"]["rate"] == 1.0
    assert r["osf"]["rate"] == 0.0 and r["osf"]["n_failures"] == 1
    assert r["pwf"]["n_failures"] == 0 and r["pwf"]["rate"] is None
    # RNF 標了 1 但 machine_failure = 0：不算「應偵測的故障」，但要如實記錄
    assert r["rnf"]["n_failures"] == 0 and r["rnf"]["n_rows_with_mode"] == 1
    # 有故障但沒有任何模式（標記不一致）
    assert r["none"] == {"n_failures": 1, "detected": 0, "rate": 0.0, "n_rows_with_mode": 0}
    assert r["any"]["n_failures"] == 4 and r["any"]["detected"] == 2


def test_rule_margins_hand_computed():
    df = mk([
        ("M", 300.0, 308.0, 1300, 40.0, 10),    # HDF：溫差 8.0 < 8.6；rpm 1300 < 1380
        ("M", 300.0, 310.0, 1500, 60.0, 10),    # 溫差 10 -> 不觸發 HDF
        ("L", 300.0, 310.0, 1500, 60.0, 10),    # P = 60*1500*2pi/60 = 9424.78 > 9000
        ("L", 300.0, 310.0, 1500, 60.0, 200),   # OSF：200*60 = 12000 > 11000
        ("H", 300.0, 310.0, 1500, 40.0, 190),   # 磨耗 190：TWF 窗口下方 10 min
    ])
    m = _viz().rule_margins(df)
    # HDF 兩個條件：溫差 (8.6-8)/8.6 = 0.0698；rpm (1380-1300)/1380 = 0.0580 -> 取較小者（較先失守）
    assert m.loc[0, "hdf_margin"] == pytest.approx(80 / 1380)
    assert m.loc[0, "hdf_dist"] == pytest.approx(80.0) and m.loc[0, "hdf_unit"] == "rpm"
    assert m.loc[0, "hdf_fired"]
    # 第 2 列溫差 10 K：超出邊界 1.4 K（相對 -0.1628），rpm 條件為 (1380-1500)/1380 = -0.0870 -> 取較負者（溫差先失守）
    assert m.loc[1, "hdf_margin"] == pytest.approx(-1.4 / 8.6)
    assert m.loc[1, "hdf_dist"] == pytest.approx(-1.4) and m.loc[1, "hdf_unit"] == "K"
    assert not m.loc[1, "hdf_fired"]
    # PWF
    p = 60 * 1500 * 2 * np.pi / 60
    assert m.loc[2, "pwf_dist"] == pytest.approx(p - 9000)
    assert m.loc[2, "pwf_margin"] == pytest.approx((p - 9000) / 9000)
    assert m.loc[2, "pwf_fired"]
    # OSF（L 門檻 11000）
    assert m.loc[3, "osf_dist"] == pytest.approx(1000.0)
    assert m.loc[3, "osf_margin"] == pytest.approx(1000 / 11000)
    assert m.loc[3, "osf_fired"]
    # 第 1 列 OSF：10*40 = 400，離 12000 還差 11600
    assert m.loc[0, "osf_dist"] == pytest.approx(-11600.0)
    # TWF 窗口 [200, 240]：190 在外面 10 min；落在窗口內為正值（到最近邊緣的距離）
    assert m.loc[4, "twf_dist"] == pytest.approx(-10.0)
    assert not m.loc[4, "twf_fired"]
    inside = _viz().rule_margins(mk([("L", 300.0, 310.0, 1500, 40.0, 230)]))
    assert inside.loc[0, "twf_dist"] == pytest.approx(10.0) and inside.loc[0, "twf_fired"]


def test_rule_margins_fired_matches_rules_module_on_real_data():
    from ai4i_lab.data import DEFAULT_PATH, load_dataset
    if not Path(DEFAULT_PATH).exists():
        pytest.skip("無資料")
    df = load_dataset().iloc[:3000]
    m = _viz().rule_margins(df)
    assert (m["hdf_fired"].to_numpy() == rules.hdf_rule(df).to_numpy()).all()
    assert (m["pwf_fired"].to_numpy() == rules.pwf_rule(df).to_numpy()).all()
    assert (m["osf_fired"].to_numpy() == rules.osf_rule(df).to_numpy()).all()
    assert (m["twf_fired"].to_numpy() == rules.twf_window(df).to_numpy()).all()


# ---------- 第 3 輪：逐筆錯誤分析 ----------

def _err_df():
    base = dict(udi=0, product_id="x", machine_failure=0, twf=0, hdf=0, pwf=0, osf=0, rnf=0)
    rows = [
        # 0: 漏判，HDF 區域邊緣（距 rpm 邊界 10 rpm = 0.7%）
        dict(base, udi=1, type="M", air_temp_k=300.0, process_temp_k=307.0, rotational_speed_rpm=1370,
             torque_nm=40.0, tool_wear_min=10, machine_failure=1, hdf=1),
        # 1: 漏判，OSF 區域內（L: 250*50 = 12500 > 11000，相對 13.6%）
        dict(base, udi=2, type="L", air_temp_k=300.0, process_temp_k=310.0, rotational_speed_rpm=1500,
             torque_nm=50.0, tool_wear_min=250, machine_failure=1, osf=1),
        # 2: 漏判，只有 TWF（窗口內 220 min、沒有確定性規則）
        dict(base, udi=3, type="M", air_temp_k=300.0, process_temp_k=310.0, rotational_speed_rpm=1500,
             torque_nm=40.0, tool_wear_min=220, machine_failure=1, twf=1),
        # 3: 漏判，標記不一致（有故障、沒任何模式、遠離所有規則）
        dict(base, udi=4, type="M", air_temp_k=300.0, process_temp_k=310.0, rotational_speed_rpm=1500,
             torque_nm=40.0, tool_wear_min=30, machine_failure=1),
        # 4: 命中（TP）
        dict(base, udi=5, type="M", air_temp_k=300.0, process_temp_k=307.0, rotational_speed_rpm=1300,
             torque_nm=40.0, tool_wear_min=10, machine_failure=1, hdf=1),
        # 5: 誤報，HDF 邊界外側（溫差 8.7 > 8.6，rpm 1300）
        dict(base, udi=6, type="M", air_temp_k=300.0, process_temp_k=308.7, rotational_speed_rpm=1300,
             torque_nm=40.0, tool_wear_min=10),
        # 6: 真正常（TN）
        dict(base, udi=7, type="M", air_temp_k=300.0, process_temp_k=310.0, rotational_speed_rpm=1500,
             torque_nm=40.0, tool_wear_min=10),
    ]
    return pd.DataFrame(rows)


def test_error_analysis_hand_computed():
    df = _err_df()
    proba = np.array([0.1, 0.2, 0.05, 0.01, 0.9, 0.8, 0.02])
    r = _viz().error_analysis(df, proba, threshold=0.5)
    assert r["counts"] == {"tp": 1, "fn": 4, "fp": 1, "tn": 1}
    fn = {x["udi"]: x for x in r["false_negatives"]}
    assert set(fn) == {1, 2, 3, 4}
    # 每個漏判都標註最近的規則邊界與距離
    for x in fn.values():
        assert x["nearest_rule"] in ("hdf", "pwf", "osf", "twf")
        assert isinstance(x["nearest_dist"], float) and x["nearest_unit"]
    assert fn[1]["category"] == "rule_near_boundary"
    assert fn[1]["nearest_rule"] == "hdf" and fn[1]["nearest_dist"] == pytest.approx(10.0)
    assert fn[1]["true_modes"] == ["hdf"]
    assert fn[2]["category"] == "rule_deep_inside" and fn[2]["true_modes"] == ["osf"]
    # 有標記模式時，最近邊界取該模式對應的規則（OSF 超出 1500 min·Nm；不是無關的 TWF 窗口）
    assert fn[2]["nearest_rule"] == "osf" and fn[2]["nearest_dist"] == pytest.approx(1500.0)
    assert fn[3]["nearest_rule"] == "twf" and fn[3]["nearest_dist"] == pytest.approx(20.0)
    assert fn[3]["category"] == "twf_window"
    assert fn[4]["category"] == "no_rule_explains" and fn[4]["true_modes"] == []
    fp = r["false_positives"]
    assert [x["udi"] for x in fp] == [6]
    # 溫差 8.7 K 距 8.6 K 邊界 0.1 K（相對 1.2%，浮點誤差內）；邊界外側
    assert fp[0]["nearest_rule"] == "hdf" and fp[0]["nearest_dist"] == pytest.approx(-0.1, abs=1e-6)
    assert fp[0]["nearest_unit"] == "K"
    assert fp[0]["category"] == "near_boundary_outside"
    assert r["summary"]["fn_by_category"] == {"rule_near_boundary": 1, "rule_deep_inside": 1,
                                              "twf_window": 1, "no_rule_explains": 1}
    # 排序：最嚴重（機率最低）的漏判在前
    assert [x["udi"] for x in r["false_negatives"]] == [4, 3, 1, 2]


def test_error_analysis_without_errors_is_empty():
    df = _err_df().iloc[[4, 6]].reset_index(drop=True)
    r = _viz().error_analysis(df, np.array([0.9, 0.1]), 0.5)
    assert r["false_negatives"] == [] and r["false_positives"] == []
    assert r["counts"] == {"tp": 1, "fn": 0, "fp": 0, "tn": 1}


def test_error_analysis_length_mismatch():
    with pytest.raises(ValueError):
        _viz().error_analysis(_err_df(), np.array([0.1, 0.2]), 0.5)


# ---------- 第 4 輪：run、圖表、介面約定、真實輸出 ----------

def _synth(n=600, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "udi": np.arange(1, n + 1), "product_id": [f"M{i}" for i in range(n)],
        "type": rng.choice(["L", "M", "H"], n),
        "air_temp_k": 300 + rng.normal(0, 1, n), "process_temp_k": 310 + rng.normal(0, 1, n),
        "rotational_speed_rpm": rng.integers(1200, 2000, n), "torque_nm": rng.normal(40, 10, n).clip(5),
        "tool_wear_min": rng.integers(0, 250, n),
    })
    df["hdf"] = rules.hdf_rule(df).astype(int)
    df["pwf"] = rules.pwf_rule(df).astype(int)
    df["osf"] = rules.osf_rule(df).astype(int)
    df["twf"] = ((rng.random(n) < 0.02) & rules.twf_window(df)).astype(int)
    df["rnf"] = (rng.random(n) < 0.005).astype(int)
    df["machine_failure"] = ((df[["twf", "hdf", "pwf", "osf", "rnf"]].sum(axis=1)) > 0).astype(int)
    return df


def _synth_preds(df, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for model in ("logreg", "random_forest"):
        for fs in ("raw", "engineered"):
            noise = 0.4 if fs == "raw" else 0.3
            p = np.clip(df["machine_failure"].to_numpy() * (0.75 - noise) + rng.random(len(df)) * (0.3 + noise), 0, 1)
            out.append(pd.DataFrame({"row_id": df["udi"], "model": model, "feature_set": fs,
                                     "y_true": df["machine_failure"], "proba": p}))
    return pd.concat(out, ignore_index=True)


def test_run_on_synthetic_writes_all_deliverables(tmp_path):
    df = _synth()
    preds = _synth_preds(df)
    thr = {f"{m}|{fs}": 0.5 for m in ("logreg", "random_forest") for fs in ("raw", "engineered")}
    chosen = {"random_forest|engineered": 0.9, "logreg|raw": 0.2, "logreg|engineered": 0.3, "random_forest|raw": 0.8}
    res = _viz().run(out_dir=tmp_path, df=df, preds=preds, thresholds=thr, cv_pr_auc=chosen, n_boot=20)
    for f in _viz().FIGURES:
        assert _png(tmp_path / f), f
    for f in ("model_comparison.csv", "model_comparison.md", "error_analysis.json", "evaluation.json",
              "false_negatives.csv", "false_positives.csv"):
        assert (tmp_path / f).exists(), f
    tab = pd.read_csv(tmp_path / "model_comparison.csv")
    assert len(tab) == 4
    for m in ("f1", "recall", "precision", "pr_auc", "roc_auc"):
        assert {m, f"{m}_low", f"{m}_high"} <= set(tab.columns)
    assert (tab["pr_auc_low"] <= tab["pr_auc"]).all() and (tab["pr_auc"] <= tab["pr_auc_high"]).all()
    ea = json.loads((tmp_path / "error_analysis.json").read_text())
    assert ea["model"] == "random_forest|engineered"      # 依 CV（非測試集）選主模型
    assert res["primary_model"] == "random_forest|engineered"
    # 每個漏判樣本都標註最接近的規則邊界與距離
    assert ea["false_negatives"]
    assert all(x["nearest_rule"] and x["nearest_dist"] is not None and x["nearest_unit"] for x in ea["false_negatives"])
    fn_csv = pd.read_csv(tmp_path / "false_negatives.csv")
    assert len(fn_csv) == len(ea["false_negatives"])


def test_primary_model_is_chosen_by_cv_not_by_test():
    v = _viz()
    assert v.pick_primary({"a|raw": 0.5, "b|engineered": 0.7, "c|raw": 0.6}) == "b|engineered"
    with pytest.raises(ValueError):
        v.pick_primary({})


def test_viz_never_trains_or_loads_models():
    src = Path(_viz().__file__).read_text(encoding="utf-8")
    for bad in (".fit(", "train_final", "get_final_model", "joblib", "partial_fit"):
        assert bad not in src, bad


def test_run_rejects_rows_not_in_data(tmp_path):
    df = _synth(100)
    preds = _synth_preds(df)
    preds.loc[0, "row_id"] = 99999
    with pytest.raises(ValueError):
        _viz().run(out_dir=tmp_path, df=df, preds=preds, thresholds={"logreg|raw": 0.5}, cv_pr_auc={"logreg|raw": 0.1},
                   n_boot=5)


@needs_upstream
@pytest.mark.skipif(not (EVAL_DIR / "evaluation.json").exists(), reason="尚未對真實資料執行 run()")
def test_real_outputs_consistent_with_upstream():
    from ai4i_lab.data import load_dataset
    ev = json.loads((EVAL_DIR / "evaluation.json").read_text())
    up = json.loads(MODELS_JSON.read_text())
    pred = pd.read_csv(PRED)
    # 比較表的 PR-AUC 點估計必須與 #8 的測試指標完全一致（同一份機率）
    tab = pd.read_csv(EVAL_DIR / "model_comparison.csv")
    for _, r in tab.iterrows():
        k = f"{r['model']}|{r['feature_set']}"
        assert r["pr_auc"] == pytest.approx(up["test_metrics"][k]["pr_auc"])
        assert r["threshold"] == pytest.approx(up["thresholds"][k]["threshold"])
    # 錯誤分析：FN + TP = 測試集故障數；每個 FN 都有最近邊界
    ea = json.loads((EVAL_DIR / "error_analysis.json").read_text())
    c = ea["counts"]
    df = load_dataset()
    test_udi = set(pred.loc[pred["model"] == "logreg", "row_id"])
    assert c["tp"] + c["fn"] == int(df[df["udi"].isin(test_udi)]["machine_failure"].sum())
    assert len(ea["false_negatives"]) == c["fn"] and len(ea["false_positives"]) == c["fp"]
    assert all(x["nearest_rule"] in ("hdf", "pwf", "osf", "twf") for x in ea["false_negatives"])
    # 主模型的混淆矩陣與 #8 的 confusion_tuned 一致
    assert c == {k: up["test_metrics"][ea["model"]]["confusion_tuned"][k] for k in ("tp", "fp", "fn", "tn")}
    assert ev["primary_model"] == ea["model"]


def test_recall_by_depth_hand_computed():
    # 4 筆故障，相對深度 0.005, 0.02, 0.02, 0.5；命中 F, T, F, T；邊界 (0.01, 0.03, 0.1)
    depth = np.array([0.005, 0.02, 0.02, 0.5])
    hit = np.array([False, True, False, True])
    r = _viz().recall_by_depth(depth, hit, edges=(0.01, 0.03, 0.1))
    assert [b["n"] for b in r] == [1, 2, 0, 1]
    assert [b["recall"] for b in r] == [0.0, 0.5, None, 1.0]
    assert [b["label"] for b in r] == ["<=1%", "1-3%", "3-10%", ">10%"]
