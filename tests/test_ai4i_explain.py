"""#11 可解釋性分析的測試（permutation importance、ICE/PDP 介入、邊界吻合、乘積規則學習）。"""
import json
import re

import numpy as np
import pandas as pd
import pytest

from ai4i_lab import explain as E
from ai4i_lab import rules as R
from ai4i_lab.data import ROOT
from ai4i_lab.features import FEATURE_SETS, build_features

EXPLAIN_DIR = ROOT / "outputs" / "ai4i" / "explain"


def _df(n=60, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "udi": np.arange(1, n + 1),
        "type": rng.choice(["L", "M", "H"], n),
        "air_temp_k": rng.uniform(298, 302, n).round(1),
        "process_temp_k": rng.uniform(308, 312, n).round(1),
        "rotational_speed_rpm": rng.integers(1200, 2500, n),
        "torque_nm": rng.uniform(20, 60, n).round(1),
        "tool_wear_min": rng.integers(0, 200, n),
    })


class ThresholdModel:
    """只看 torque_nm 的玩具模型：torque > 40 -> 機率 0.9，否則 0.1。"""
    def predict_proba(self, X):
        p = np.where(X["torque_nm"].to_numpy() > 40, 0.9, 0.1)
        return np.column_stack([1 - p, p])


# ---------- 第 1 輪：permutation importance ----------

def test_permutation_importance_finds_the_only_used_feature():
    df = _df(300)
    X = build_features(df, "raw")
    y = (df["torque_nm"] > 40).astype(int).to_numpy()
    imp = E.permutation_importance(ThresholdModel(), X, y, n_repeats=3, seed=1)
    top = imp.sort_values("mean", ascending=False).iloc[0]
    assert top["feature"] == "torque_nm"
    assert top["mean"] > 0.3
    others = imp[imp["feature"] != "torque_nm"]
    assert (others["mean"].abs() < 1e-12).all()      # 模型根本不看這些欄位 -> 重要度恰為 0


def test_permutation_importance_groups_type_one_hot_columns():
    X = build_features(_df(80), "raw")
    y = (X["torque_nm"] > 40).astype(int).to_numpy()
    imp = E.permutation_importance(ThresholdModel(), X, y, n_repeats=2, seed=0)
    assert "type" in set(imp["feature"])
    assert not any(f.startswith("type_") for f in imp["feature"])


def test_permutation_importance_is_reproducible_and_does_not_mutate_input():
    X = build_features(_df(120), "engineered")
    y = (X["torque_nm"] > 40).astype(int).to_numpy()
    X0 = X.copy()
    a = E.permutation_importance(ThresholdModel(), X, y, n_repeats=3, seed=5)
    b = E.permutation_importance(ThresholdModel(), X, y, n_repeats=3, seed=5)
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(X, X0)


def test_permutation_importance_columns_and_repeats():
    X = build_features(_df(100), "raw")
    y = (X["torque_nm"] > 40).astype(int).to_numpy()
    imp = E.permutation_importance(ThresholdModel(), X, y, n_repeats=4, seed=0)
    assert {"feature", "mean", "std", "n_repeats", "metric"} <= set(imp.columns)
    assert (imp["n_repeats"] == 4).all()


# ---------- 第 2 輪：在規則座標上介入原始特徵 ----------

def test_power_intervention_hits_target_power_and_keeps_rpm():
    df = _df(20)
    out = E.set_power(df, 5000.0)
    assert np.allclose(R.power_w(out), 5000.0)
    assert (out["rotational_speed_rpm"] == df["rotational_speed_rpm"]).all()


def test_temp_diff_intervention_hits_target():
    df = _df(20)
    out = E.set_temp_diff(df, 8.6)
    assert np.allclose(R.temp_diff_k(out), 8.6)
    assert (out["air_temp_k"] == df["air_temp_k"]).all()


def test_strain_ratio_intervention_hits_ratio_per_type():
    df = _df(30)
    out = E.set_strain_ratio(df, 1.0)
    assert np.allclose(R.strain_minnm(out), R.osf_threshold(df))
    assert (out["torque_nm"] == df["torque_nm"]).all()


def test_wear_intervention_sets_wear():
    out = E.set_wear(_df(5), 220)
    assert (out["tool_wear_min"] == 220).all()


def test_ice_valid_mask_marks_out_of_range_interventions():
    df = _df(10)
    lo, hi = 3.8, 76.6
    # torque = P / omega；極大功率需要超出觀測範圍的扭矩 -> 無效
    out, valid = E.apply_intervention(df, "power", 1e6, bounds={"torque_nm": (lo, hi), "tool_wear_min": (0, 253)})
    assert not valid.any()
    out, valid = E.apply_intervention(df, "power", 5000.0, bounds={"torque_nm": (0, 1e9), "tool_wear_min": (0, 253)})
    assert valid.all()


def test_ice_matrix_shape_and_nan_for_invalid():
    df = _df(12)
    grid = np.array([3000.0, 6000.0, 1e6])
    bounds = {"torque_nm": (0, 80), "tool_wear_min": (0, 253)}
    ice = E.ice_curves(ThresholdModel(), df, "raw", "power", grid, bounds=bounds)
    assert ice.shape == (12, 3)
    assert np.isnan(ice[:, 2]).all()
    assert not np.isnan(ice[:, :2]).any()


def test_intervention_axes_come_from_rules_constants():
    ax = E.axis_spec("power")
    assert ax["thresholds"] == [R.PWF_POWER_MIN_W, R.PWF_POWER_MAX_W]
    assert E.axis_spec("temp_diff")["thresholds"] == [R.HDF_TEMP_DIFF_MAX_K]
    assert E.axis_spec("strain_ratio")["thresholds"] == [1.0]
    assert E.axis_spec("wear")["window"] == [R.TWF_WEAR_MIN, R.TWF_WEAR_MAX]


def test_explain_source_has_no_hardcoded_rule_thresholds():
    src = (ROOT / "ai4i_lab" / "explain.py").read_text(encoding="utf-8")
    code = "\n".join(l for l in src.splitlines() if not l.strip().startswith("#"))
    for banned in [r"\b3500\b", r"\b9000\b", r"\b8\.6\b", r"\b1380\b", r"\b11000\b", r"\b12000\b", r"\b13000\b"]:
        assert not re.search(banned, code), banned


# ---------- 第 3 輪：邊界交點與吻合度 ----------

def test_find_crossings_interpolates_linearly():
    grid = np.array([0.0, 1.0, 2.0, 3.0])
    p = np.array([0.0, 0.2, 0.8, 1.0])
    cr = E.find_crossings(grid, p, 0.5)
    assert len(cr) == 1 and cr[0] == pytest.approx(1.5)


def test_find_crossings_multiple_and_none_and_nan():
    grid = np.arange(6.0)
    p = np.array([0.9, 0.9, 0.1, 0.1, 0.9, 0.9])
    cr = E.find_crossings(grid, p, 0.5)
    assert len(cr) == 2 and cr[0] == pytest.approx(1.5) and cr[1] == pytest.approx(3.5)
    assert E.find_crossings(grid, np.full(6, 0.1), 0.5) == []
    p2 = p.copy(); p2[2:4] = np.nan
    assert E.find_crossings(grid, p2, 0.5) == []        # NaN 段落不產生交點


def test_boundary_offsets_nearest_crossing_within_window():
    grid = np.linspace(0, 10, 101)
    ice = np.vstack([(grid > 5.2).astype(float), (grid > 4.8).astype(float), np.zeros_like(grid)])
    res = E.boundary_offsets(grid, ice, thr=0.5, rule_value=5.0, window=1.0)
    assert res["n_curves"] == 3 and res["n_crossing"] == 2
    assert res["median_offset"] == pytest.approx(0.0, abs=0.06)
    assert res["median_abs_offset"] == pytest.approx(0.2, abs=0.06)


def test_boundary_offsets_ignores_crossings_outside_window():
    grid = np.linspace(0, 10, 101)
    ice = np.vstack([(grid > 9.0).astype(float)])
    res = E.boundary_offsets(grid, ice, thr=0.5, rule_value=5.0, window=1.0)
    assert res["n_crossing"] == 0 and res["median_offset"] is None


def test_decision_agreement_hand_computed():
    pred = np.array([1, 1, 0, 0, 1, 0])
    rule = np.array([1, 0, 0, 1, 1, 0])
    a = E.decision_agreement(pred, rule)
    assert (a["tp"], a["fp"], a["fn"], a["tn"]) == (2, 1, 1, 2)
    assert a["accuracy"] == pytest.approx(4 / 6)
    assert a["jaccard"] == pytest.approx(2 / 4)
    assert a["f1"] == pytest.approx(2 * 2 / (2 * 2 + 1 + 1))
    # kappa: po=4/6, pe = (3/6*3/6)+(3/6*3/6)=0.5 -> (0.6667-0.5)/0.5
    assert a["kappa"] == pytest.approx(1 / 3)


def test_decision_agreement_edge_all_negative():
    a = E.decision_agreement(np.zeros(4), np.zeros(4))
    assert a["accuracy"] == 1.0 and a["jaccard"] is None and a["kappa"] is None


# ---------- 第 4 輪：MLP 能否學到乘積型規則 ----------

def _rule_df(n=1500, seed=3):
    rng = np.random.default_rng(seed)
    df = _df(n, seed)
    df["torque_nm"] = rng.uniform(10, 70, n).round(1)
    df["tool_wear_min"] = rng.integers(0, 250, n)
    return df


def test_rule_targets_match_rules_module():
    df = _rule_df(200)
    t = E.rule_targets(df)
    assert set(t) == {"osf", "pwf", "hdf", "rpm_only"}
    assert (t["osf"] == R.osf_rule(df).astype(int)).all()
    assert (t["pwf"] == R.pwf_rule(df).astype(int)).all()
    assert (t["hdf"] == R.hdf_rule(df).astype(int)).all()
    assert (t["rpm_only"] == (df["rotational_speed_rpm"] < R.HDF_RPM_MAX).astype(int)).all()


FAST = {"hidden_layer_sizes": (16,), "max_iter": 60}


def test_mlp_rule_learning_returns_metrics_and_is_reproducible():
    df = _rule_df()
    tr, va, te = df.iloc[:900], df.iloc[900:1200], df.iloc[1200:]
    a = E.mlp_rule_learning(tr, va, te, "osf", "raw", params=FAST, seed=0)
    b = E.mlp_rule_learning(tr, va, te, "osf", "raw", params=FAST, seed=0)
    assert {"rule", "feature_set", "pr_auc", "f1", "recall", "precision", "threshold", "n_pos_train"} <= set(a)
    assert a == b
    assert 0 <= a["pr_auc"] <= 1


def test_mlp_rule_learning_threshold_comes_from_val_only():
    df = _rule_df()
    tr, va, te = df.iloc[:900], df.iloc[900:1200], df.iloc[1200:].copy()
    a = E.mlp_rule_learning(tr, va, te, "osf", "raw", params=FAST, seed=0)
    te2 = te.copy()
    te2["torque_nm"] = te2["torque_nm"].to_numpy()[::-1]      # 竄改測試集特徵
    b = E.mlp_rule_learning(tr, va, te2, "osf", "raw", params=FAST, seed=0)
    assert a["threshold"] == b["threshold"]


def test_mlp_rule_learning_never_uses_failure_mode_columns():
    df = _rule_df()
    df["osf"] = 1                                   # 若被當成特徵會讓結果異常完美
    df["machine_failure"] = 1
    tr, va, te = df.iloc[:900], df.iloc[900:1200], df.iloc[1200:]
    out = E.mlp_rule_learning(tr, va, te, "osf", "raw", params=FAST, seed=0)
    assert out["pr_auc"] < 0.999


def test_engineered_product_feature_makes_product_rule_easy():
    """合成資料：給 MLP 乘積特徵（engineered）應不輸給只有原始特徵。小模型、小資料，只檢查方向。"""
    df = _rule_df(2400, seed=1)
    tr, va, te = df.iloc[:1500], df.iloc[1500:1950], df.iloc[1950:]
    p = {"hidden_layer_sizes": (16,), "max_iter": 200}
    raw = E.mlp_rule_learning(tr, va, te, "osf", "raw", params=p, seed=0)
    eng = E.mlp_rule_learning(tr, va, te, "osf", "engineered", params=p, seed=0)
    assert eng["pr_auc"] >= raw["pr_auc"] - 0.02
    assert eng["pr_auc"] > 0.8


def test_per_mode_recall_hand_computed():
    df = pd.DataFrame({"twf": [0, 0, 0, 0], "hdf": [1, 1, 0, 0], "pwf": [0, 0, 0, 0],
                       "osf": [0, 0, 1, 1], "rnf": [0, 0, 0, 0]})
    pred = np.array([1, 0, 1, 1])
    r = E.per_mode_recall(df, pred)
    assert r["hdf"] == {"n": 2, "detected": 1, "recall": 0.5}
    assert r["osf"] == {"n": 2, "detected": 2, "recall": 1.0}
    assert r["twf"]["recall"] is None


# ---------- 第 5 輪：介入後的規則預測（反事實區域吻合度）與健康列選取 ----------

def _healthy_row():
    return pd.DataFrame({"udi": [1], "type": ["M"], "air_temp_k": [300.0], "process_temp_k": [311.0],
                         "rotational_speed_rpm": [1500], "torque_nm": [40.0], "tool_wear_min": [10]})


def test_ice_rule_matrix_power_axis_matches_pwf_thresholds():
    df = _healthy_row()
    grid = np.array([R.PWF_POWER_MIN_W - 100, 5000.0, R.PWF_POWER_MAX_W + 100])
    rule, valid = E.ice_rule_matrix(df, "power", grid, bounds={"torque_nm": (0, 1e9)})
    assert rule.tolist() == [[True, False, True]] and valid.all()


def test_ice_rule_matrix_temp_axis_requires_low_rpm():
    df = _healthy_row()
    df.loc[0, "rotational_speed_rpm"] = R.HDF_RPM_MAX - 1
    grid = np.array([R.HDF_TEMP_DIFF_MAX_K - 0.5, R.HDF_TEMP_DIFF_MAX_K + 0.5])
    rule, _ = E.ice_rule_matrix(df, "temp_diff", grid)
    assert rule.tolist() == [[True, False]]
    df.loc[0, "rotational_speed_rpm"] = R.HDF_RPM_MAX + 100
    rule, _ = E.ice_rule_matrix(df, "temp_diff", grid)
    assert rule.tolist() == [[False, False]]


def test_ice_rule_matrix_strain_axis_ratio_one_is_boundary():
    df = _healthy_row()
    rule, _ = E.ice_rule_matrix(df, "strain_ratio", np.array([0.9, 1.1]))
    assert rule.tolist() == [[False, True]]


def test_ice_rule_matrix_is_union_of_all_deterministic_rules():
    """介入功率時扭矩改變，OSF（磨耗×扭矩）可能也被觸發；規則預測必須是三條規則的聯集。"""
    df = _healthy_row()
    df.loc[0, "tool_wear_min"] = 200
    rule, _ = E.ice_rule_matrix(df, "power", np.array([R.PWF_POWER_MAX_W - 100]), bounds=None)
    t = R.PWF_POWER_MAX_W - 100
    torque = t / (1500 * 2 * np.pi / 60)
    assert rule[0, 0] == bool(200 * torque > R.OSF_STRAIN_MAX_MINNM["M"])


def test_healthy_pool_excludes_rule_active_and_failure_rows():
    df = _df(200)
    df["machine_failure"] = 0
    df.loc[df.index[:5], "machine_failure"] = 1
    pool = E.healthy_pool(df)
    assert not R.hdf_rule(pool).any() and not R.pwf_rule(pool).any() and not R.osf_rule(pool).any()
    assert (pool["machine_failure"] == 0).all()


def test_sample_rows_is_reproducible_and_capped():
    df = _df(200)
    a, b = E.sample_rows(df, 30, seed=1), E.sample_rows(df, 30, seed=1)
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == 30 and len(E.sample_rows(df, 500, seed=1)) == 200


# ---------- 第 6 輪：以重抽樣合成更多資料（資料量對乘積規則學習的影響） ----------

def test_synthetic_frame_shape_reproducible_and_valid():
    src = _rule_df(300)
    a = E.synthetic_frame(src, 1000, seed=2)
    b = E.synthetic_frame(src, 1000, seed=2)
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == 1000 and set(a["type"]) <= {"L", "M", "H"}
    assert {"udi", "type", "air_temp_k", "process_temp_k", "rotational_speed_rpm", "torque_nm",
            "tool_wear_min"} <= set(a.columns)
    assert (a["torque_nm"] > 0).all() and (a["rotational_speed_rpm"] > 0).all()


def test_synthetic_frame_uses_only_source_rows():
    """rpm／torque 來自來源列（加小幅抖動）：來源之外的列不可能被洩漏進來。"""
    src = _rule_df(100)
    a = E.synthetic_frame(src, 500, seed=0, rpm_jitter=0.0, torque_jitter=0.0)
    pairs = set(zip(src["rotational_speed_rpm"], src["torque_nm"]))
    assert set(zip(a["rotational_speed_rpm"], a["torque_nm"])) <= pairs
    assert set(a["tool_wear_min"]) <= set(src["tool_wear_min"])


def test_synthetic_frame_temperature_difference_comes_from_source():
    src = _rule_df(100)
    a = E.synthetic_frame(src, 400, seed=0, temp_jitter=0.0)
    diffs = set(np.round(R.temp_diff_k(src), 6))
    assert set(np.round(R.temp_diff_k(a), 6)) <= diffs | {round(d, 6) for d in diffs}


# ---------- 第 7 輪：單一模型的完整分析與輸出一致性 ----------

def _full_df(n=400, seed=7):
    df = _rule_df(n, seed)
    for m in ["twf", "hdf", "pwf", "osf", "rnf"]:
        df[m] = 0
    df["hdf"] = R.hdf_rule(df).astype(int)
    df["pwf"] = R.pwf_rule(df).astype(int)
    df["osf"] = R.osf_rule(df).astype(int)
    df["machine_failure"] = ((df[["twf", "hdf", "pwf", "osf", "rnf"]].sum(axis=1)) > 0).astype(int)
    return df


class RuleOracle:
    """完美模型：機率 = 規則聯集（以特徵還原），用來驗證分析管線能把『完全學到規則』量測出來。"""
    def predict_proba(self, X):
        power = X["torque_nm"] * X["rotational_speed_rpm"] * 2 * np.pi / 60
        pwf = (power < R.PWF_POWER_MIN_W) | (power > R.PWF_POWER_MAX_W)
        hdf = ((X["process_temp_k"] - X["air_temp_k"]) < R.HDF_TEMP_DIFF_MAX_K) & \
              (X["rotational_speed_rpm"] < R.HDF_RPM_MAX)
        thr = X["type_L"] * R.OSF_STRAIN_MAX_MINNM["L"] + X["type_M"] * R.OSF_STRAIN_MAX_MINNM["M"] + \
              X["type_H"] * R.OSF_STRAIN_MAX_MINNM["H"]
        osf = X["tool_wear_min"] * X["torque_nm"] > thr
        p = (pwf | hdf | osf).astype(float).to_numpy()
        return np.column_stack([1 - p, p])


def test_analyze_model_recovers_rule_boundaries_for_oracle():
    df = _full_df(600)
    bounds = E.observed_bounds(df)
    res = E.analyze_model(RuleOracle(), "oracle", "raw", 0.5, df, bounds, n_ice=40, n_repeats=2, seed=0, wear_ref=df)
    assert res["test_agreement"]["vs_rule_union"]["accuracy"] == 1.0
    for axis in ("power", "temp_diff", "strain_ratio"):
        ca = res["axes"][axis]["counterfactual_agreement"]
        assert ca["accuracy"] == 1.0
        for name, b in res["axes"][axis]["boundaries"].items():
            if b["n_crossing"]:
                assert b["median_abs_offset_rel"] < 0.02, (axis, name)
    assert {"below", "in_window", "above"} == set(res["axes"]["wear"]["bands"])
    assert set(res["per_mode_recall"]) == {"twf", "hdf", "pwf", "osf", "rnf"}


def test_analyze_model_uses_only_given_test_rows():
    df = _full_df(500)
    test = df.iloc[:200]
    res = E.analyze_model(RuleOracle(), "oracle", "raw", 0.5, test, E.observed_bounds(df), n_ice=500,
                          n_repeats=1, seed=0, wear_ref=test)
    assert res["axes"]["power"]["n_rows"] <= len(test)
    assert res["test_agreement"]["vs_label"]["n"] == len(test)


def test_learned_boundary_criterion():
    good = {"frac_crossing": 0.9, "median_abs_offset_rel": 0.03}
    assert E.learned_boundary(good)
    assert not E.learned_boundary({"frac_crossing": 0.5, "median_abs_offset_rel": 0.03})
    assert not E.learned_boundary({"frac_crossing": 0.9, "median_abs_offset_rel": 0.2})
    assert not E.learned_boundary({"frac_crossing": 0.0, "median_abs_offset_rel": None})


needs_outputs = pytest.mark.skipif(not (EXPLAIN_DIR / "boundary_agreement.json").exists(),
                                   reason="尚未執行 python -m ai4i_lab.explain")


@needs_outputs
def test_real_outputs_exist_and_json_is_consistent():
    for f in E.FIGURES + ["boundary_agreement.json", "explain.json", "permutation_importance.csv", "pdp_curves.csv"]:
        assert (EXPLAIN_DIR / f).exists(), f
    ba = json.loads((EXPLAIN_DIR / "boundary_agreement.json").read_text(encoding="utf-8"))
    assert len(ba["boundary_agreement"]) == 8
    ex = json.loads((EXPLAIN_DIR / "explain.json").read_text(encoding="utf-8"))
    assert ex["conclusions"]["A"] and ex["conclusions"]["B"]
    imp = pd.read_csv(EXPLAIN_DIR / "permutation_importance.csv")
    allowed = (set(FEATURE_SETS["engineered"]) - {"type_L", "type_M", "type_H"}) | {"type"}
    assert set(imp["feature"]) <= allowed


@needs_outputs
def test_real_importance_never_contains_forbidden_columns():
    imp = pd.read_csv(EXPLAIN_DIR / "permutation_importance.csv")
    assert not set(imp["feature"]) & {"twf", "hdf", "pwf", "osf", "rnf", "machine_failure", "udi", "product_id"}


@needs_outputs
def test_real_thresholds_in_output_match_models_json():
    ba = json.loads((EXPLAIN_DIR / "boundary_agreement.json").read_text(encoding="utf-8"))
    mj = json.loads((ROOT / "outputs" / "ai4i" / "models" / "models.json").read_text(encoding="utf-8"))
    for k, v in ba["model_thresholds"].items():
        assert v == pytest.approx(mj["thresholds"][k]["threshold"])
