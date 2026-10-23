"""#9 多標籤分類（預測 TWF/HDF/PWF/OSF，驗證 RNF 不可預測）的測試。"""
import json
import math

import numpy as np
import pandas as pd
import pytest

from ai4i_lab import multilabel as M
from ai4i_lab.data import FAILURE_MODES, ROOT


# ---------- 第 1 輪：目標、多標籤彙總指標、bootstrap CI ----------

def test_mode_targets_extracts_mode_columns_as_int():
    df = pd.DataFrame({"twf": [1, 0], "hdf": [0, 1], "pwf": [0, 0], "osf": [1, 1], "rnf": [0, 0],
                       "torque_nm": [40.0, 41.0]})
    Y = M.mode_targets(df)
    assert list(Y.columns) == FAILURE_MODES
    assert Y.to_numpy().tolist() == [[1, 0, 0, 1, 0], [0, 1, 0, 1, 0]]
    assert M.PREDICTABLE_MODES == ["twf", "hdf", "pwf", "osf"]
    assert M.mode_targets(df, ["hdf"]).columns.tolist() == ["hdf"]


def test_multilabel_summary_hand_computed():
    # 3 列 × 2 標籤
    # Y:  [1,0] [0,1] [1,1]
    # P:  [1,0] [1,1] [0,1]  → 標籤 a: TP1 FP1 FN1；標籤 b: TP2 FP0 FN0
    Y = np.array([[1, 0], [0, 1], [1, 1]])
    P = np.array([[.9, .1], [.8, .7], [.2, .6]])
    s = M.multilabel_summary(Y, P, threshold=0.5)
    # micro：TP3 FP1 FN1 → P=R=F1=0.75
    assert s["micro_f1"] == pytest.approx(0.75)
    assert s["micro_precision"] == pytest.approx(0.75)
    assert s["micro_recall"] == pytest.approx(0.75)
    # macro：a 的 F1 = 0.5、b 的 F1 = 1 → 0.75
    assert s["macro_f1"] == pytest.approx(0.75)
    # Hamming loss：6 個格子錯 2 個
    assert s["hamming_loss"] == pytest.approx(2 / 6)
    # subset accuracy：只有第 1 列全對
    assert s["subset_accuracy"] == pytest.approx(1 / 3)


def test_bootstrap_metrics_perfect_predictions_give_degenerate_ci():
    y = np.array([0] * 20 + [1] * 5)
    p = y.astype(float)
    b = M.bootstrap_metrics(y, p, n_boot=200, seed=0)
    for m in ("recall", "precision", "f1"):
        assert b[m]["point"] == 1.0 and b[m]["low"] == 1.0 and b[m]["high"] == 1.0
    assert b["n_boot"] == 200


def test_bootstrap_metrics_interval_contains_point_and_is_reproducible():
    rng = np.random.default_rng(1)
    y = (rng.random(400) < 0.2).astype(int)
    p = np.clip(y * 0.4 + rng.random(400) * 0.6, 0, 1)
    a = M.bootstrap_metrics(y, p, n_boot=300, seed=42)
    b = M.bootstrap_metrics(y, p, n_boot=300, seed=42)
    assert a == b
    for m in ("recall", "precision", "f1", "pr_auc", "roc_auc"):
        assert a[m]["low"] <= a[m]["point"] <= a[m]["high"]
        assert 0.0 <= a[m]["low"] and a[m]["high"] <= 1.0
    assert a["recall"]["high"] - a["recall"]["low"] > 0


def test_bootstrap_metrics_handles_resamples_without_positives():
    # 只有 1 個正例：多數重抽樣會沒有正例，AUC 為 NaN 應被忽略而不是讓整個區間變 NaN
    y = np.array([1] + [0] * 30)
    p = np.linspace(1, 0, 31)
    b = M.bootstrap_metrics(y, p, n_boot=100, seed=0)
    assert not math.isnan(b["roc_auc"]["low"])
    assert b["roc_auc"]["n_valid"] < 100


# ---------- 第 2 輪：與隨機基準比較（偵測率、OOF 分數、置換檢定） ----------

def _logreg():
    from sklearn.linear_model import LogisticRegression
    return lambda: LogisticRegression(max_iter=500)


def test_random_detection_hand_computed():
    # 10 列、2 個正例（索引 0、3）；分數由高到低即索引順序
    y = np.array([1, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    s = np.linspace(1, 0, 10)
    rows = M.random_detection(y, s, fractions=(0.2, 0.4))
    r20, r40 = rows
    assert (r20["n_flagged"], r20["detected"]) == (2, 1)
    assert r20["recall"] == pytest.approx(0.5)
    assert r20["random_recall"] == pytest.approx(0.2)
    assert r20["lift"] == pytest.approx(2.5)
    # 超幾何：N=10、K=2、抽 2 → P(X≥1) = 1 − C(8,2)/C(10,2) = 17/45
    assert r20["p_value"] == pytest.approx(17 / 45)
    assert (r40["n_flagged"], r40["detected"], r40["recall"]) == (4, 2, 1.0)


def test_oof_scores_each_row_predicted_once_by_model_not_trained_on_it():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=60), "b": rng.normal(size=60)})
    y = (X["a"] > 0.8).astype(int).to_numpy()
    seen = []
    s = M.oof_scores(_logreg(), X, y, n_splits=5, seed=0, _spy=lambda tr, va: seen.append((set(tr), set(va))))
    assert s.shape == (60,) and np.all((s >= 0) & (s <= 1))
    assert len(seen) == 5
    covered = set()
    for tr, va in seen:
        assert not tr & va
        covered |= va
    assert covered == set(range(60))
    assert sum(len(va) for _, va in seen) == 60
    # 可重現
    assert np.array_equal(s, M.oof_scores(_logreg(), X, y, n_splits=5, seed=0))


def test_permutation_test_detects_signal_and_accepts_noise():
    rng = np.random.default_rng(3)
    n = 400
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y_signal = (X["a"] + 0.3 * rng.normal(size=n) > 1.0).astype(int).to_numpy()
    y_noise = (rng.random(n) < 0.1).astype(int)
    sig = M.permutation_test(_logreg(), X, y_signal, n_perm=19, n_splits=4, seed=0)
    noi = M.permutation_test(_logreg(), X, y_noise, n_perm=19, n_splits=4, seed=0)
    assert sig["observed"]["roc_auc"] > 0.9
    assert sig["p_value"]["roc_auc"] == pytest.approx(1 / 20)    # 觀察值高於全部 19 次置換
    assert abs(noi["observed"]["roc_auc"] - 0.5) < 0.15
    assert noi["p_value"]["roc_auc"] > 0.05
    assert len(noi["null"]["pr_auc"]) == 19
    assert noi["prevalence"] == pytest.approx(y_noise.mean())
    # 可重現
    assert noi == M.permutation_test(_logreg(), X, y_noise, n_perm=19, n_splits=4, seed=0)


# ---------- 第 3 輪：各模式 CV、測試集評估、防洩漏、run() ----------

def synthetic_ai4i(n=1000, seed=0, rnf_rate=0.04):
    """依已知產生規則合成的小型 AI4I 資料（RNF 與特徵無關）。"""
    rng = np.random.default_rng(seed)
    air = 300 + rng.normal(0, 2, n)
    proc = air + 10 + rng.normal(0, 1, n)
    rpm = rng.normal(1500, 180, n).clip(1170, 2800)
    torque = rng.normal(40, 10, n).clip(3, 77)
    wear = rng.integers(0, 250, n).astype(float)
    typ = rng.choice(["L", "M", "H"], n, p=[0.6, 0.3, 0.1])
    df = pd.DataFrame({"udi": np.arange(1, n + 1), "product_id": [f"{t}{i}" for t, i in zip(typ, range(n))],
                       "type": typ, "air_temp_k": air, "process_temp_k": proc, "rotational_speed_rpm": rpm,
                       "torque_nm": torque, "tool_wear_min": wear})
    power = torque * rpm * 2 * np.pi / 60
    lim = pd.Series(typ).map({"L": 11000, "M": 12000, "H": 13000}).to_numpy()
    df["twf"] = ((wear >= 200) & (rng.random(n) < 0.3)).astype(int)
    df["hdf"] = ((proc - air < 8.6) & (rpm < 1380)).astype(int)
    df["pwf"] = ((power < 3500) | (power > 9000)).astype(int)
    df["osf"] = (wear * torque > lim).astype(int)
    df["rnf"] = (rng.random(n) < rnf_rate).astype(int)
    df["machine_failure"] = df[FAILURE_MODES].max(axis=1)
    return df


@pytest.fixture(scope="module")
def syn():
    from ai4i_lab import evaluation as E
    df = synthetic_ai4i()
    return df, E.make_splits(df)


def test_cv_per_mode_uses_only_feature_columns_and_reports_ci(syn, monkeypatch):
    from ai4i_lab import evaluation as E
    from ai4i_lab.features import FEATURE_SETS
    df, _ = syn
    seen = []
    real = E.repeated_cv

    def spy(make_model, X, y, **kw):
        seen.append(list(X.columns))
        return real(make_model, X, y, **kw)
    monkeypatch.setattr(E, "repeated_cv", spy)
    res = M.cv_per_mode(df, ["hdf", "pwf"], "engineered", "fast_forest", n_splits=3, n_repeats=1)
    assert seen == [FEATURE_SETS["engineered"]] * 2
    for mode in ("hdf", "pwf"):
        s = res[mode]["summary"]
        for m in ("precision", "recall", "f1"):
            assert s[m]["low"] <= s[m]["mean"] <= s[m]["high"]
        assert res[mode]["n_positive"] == int(df[mode].sum())
    # 規則產生的模式在 engineered 特徵下應很好預測
    assert res["pwf"]["summary"]["f1"]["mean"] > 0.8


def test_holdout_per_mode_trains_on_dev_and_scores_test_once(syn):
    df, splits = syn
    from ai4i_lab import evaluation as E
    masks = E.split_masks(df, splits)
    fits = []
    res, P = M.holdout_per_mode(df, masks, ["hdf", "rnf"], "engineered", "fast_forest", n_boot=50,
                                _spy=lambda mode, tr_udi, te_udi: fits.append((mode, tr_udi, te_udi)))
    dev_udi = set(df.loc[masks["train"] | masks["val"], "udi"])
    test_udi = set(df.loc[masks["test"], "udi"])
    assert [f[0] for f in fits] == ["hdf", "rnf"]
    for _, tr, te in fits:
        assert tr == dev_udi and te == test_udi and not tr & te
    assert P.shape == (int(masks["test"].sum()), 2)
    assert res["hdf"]["n_train"] == len(dev_udi) and res["hdf"]["n_test"] == len(test_udi)
    assert res["hdf"]["recall"]["low"] <= res["hdf"]["recall"]["point"] <= res["hdf"]["recall"]["high"]


FAST = dict(n_splits=3, n_repeats=1, n_perm=3, n_boot=50, model="fast_forest", perm_model="fast_forest",
            n_sims=200, block_size=50)


@pytest.fixture(scope="module")
def syn_run(syn, tmp_path_factory):
    """在合成資料上跑一次 run()，同時用 spy 記錄 CV／OOF 看到的列。"""
    from ai4i_lab import evaluation as E
    df, splits = syn
    seen_idx = []
    real_cv, real_oof = E.repeated_cv, M.oof_scores
    mp = pytest.MonkeyPatch()
    mp.setattr(E, "repeated_cv", lambda mk, X, y, **kw: (seen_idx.append(set(X.index)),
                                                          real_cv(mk, X, y, **kw))[1])
    mp.setattr(M, "oof_scores", lambda mk, X, y, *a, **kw: (seen_idx.append(set(X.index)),
                                                            real_oof(mk, X, y, *a, **kw))[1])
    out_dir = tmp_path_factory.mktemp("ml")
    try:
        out = M.run(out_dir=out_dir, df=df, splits=splits, **FAST)
    finally:
        mp.undo()
    return out, out_dir, seen_idx


def test_run_writes_outputs_and_rnf_is_compared_with_random(syn, syn_run):
    from ai4i_lab import evaluation as E
    df, splits = syn
    out, out_dir, seen_idx = syn_run
    test_idx = set(df.index[E.split_masks(df, splits)["test"]])
    # 測試集不參與 CV、OOF（選門檻）、置換檢定
    assert seen_idx and all(not s & test_idx for s in seen_idx)
    j = json.loads((out_dir / "multilabel.json").read_text(encoding="utf-8"))
    for f in M.FIGURES:
        assert (out_dir / f).exists()
    for fs in j["config"]["feature_sets"]:
        for mode in FAILURE_MODES:
            s = j["cv"][fs][mode]["summary"]
            assert {"mean", "low", "high"} <= set(s["f1"])
    for mode in M.PREDICTABLE_MODES:
        assert {"point", "low", "high"} <= set(j["test"]["engineered"][mode]["f1"])
    r = j["rnf"]
    assert r["random_baseline"]["roc_auc"] == 0.5
    assert r["random_baseline"]["pr_auc"] == pytest.approx(r["permutation"]["prevalence"])
    assert {"observed", "p_value", "null_mean"} <= set(r["permutation"])
    assert r["detection"] and {"recall", "random_recall", "p_value"} <= set(r["detection"][0])
    for v in ("threshold_0.5", "threshold_tuned"):
        assert set(j["test_multilabel"][v]) >= {"micro_f1", "macro_f1", "hamming_loss", "subset_accuracy"}
    assert {"permutation_blocked", "detection_blocked", "udi_cluster", "checks"} <= set(r)
    # 選門檻只用開發集 OOF：測試集結果的 tuned 門檻 = config 中記錄的開發集門檻
    for mode in FAILURE_MODES:
        assert j["test"]["engineered"][mode]["tuned"]["threshold"] == j["config"]["tuned_thresholds"][mode]
    assert "rnf_verdict" in out and "seconds" in out


def test_run_is_reproducible(syn, syn_run, tmp_path):
    df, splits = syn
    _, first_dir, _ = syn_run
    M.run(out_dir=tmp_path, df=df, splits=splits, **FAST)
    ja = json.loads((first_dir / "multilabel.json").read_text(encoding="utf-8"))
    jb = json.loads((tmp_path / "multilabel.json").read_text(encoding="utf-8"))
    for j in (ja, jb):
        j.pop("seconds"), j.pop("timing")
    assert ja == jb


# ---------- 第 4 輪：真實資料產出的檢查（需先執行 python -m ai4i_lab.multilabel） ----------

REAL_JSON = ROOT / "outputs" / "ai4i" / "multilabel" / "multilabel.json"
needs_real = pytest.mark.skipif(not REAL_JSON.exists(), reason="尚未產生 outputs/ai4i/multilabel/multilabel.json")


@pytest.fixture(scope="module")
def real():
    return json.loads(REAL_JSON.read_text(encoding="utf-8"))


@needs_real
def test_real_uses_shared_split_sizes(real):
    assert real["config"]["dev_rows"] == 8000 and real["config"]["test_rows"] == 2000
    assert real["config"]["n_splits"] == 5 and real["config"]["n_repeats"] == 3


@needs_real
def test_real_every_mode_has_cv_and_test_ci(real):
    for fs in real["config"]["feature_sets"]:
        for mode in M.PREDICTABLE_MODES + ["rnf"]:
            for m in ("precision", "recall", "f1"):
                c = real["cv"][fs][mode]["summary"][m]
                assert c["low"] <= c["mean"] <= c["high"]
                t = real["test"][fs][mode][m]
                if t["n_valid"] == 0:
                    # 所有重抽樣都沒有預測為正 → precision 無定義（只可能發生在 precision、且測試集 TP+FP=0）
                    assert m == "precision" and real["test"][fs][mode]["tp"] + real["test"][fs][mode]["fp"] == 0
                    continue
                assert t["low"] <= t["point"] <= t["high"]
    for f in M.FIGURES:
        assert (REAL_JSON.parent / f).exists()


@needs_real
def test_real_rnf_indistinguishable_from_random_but_deterministic_modes_are_learned(real):
    r = real["rnf"]
    assert r["consistent_with_random"] is True
    assert r["permutation"]["p_value"]["roc_auc"] > 0.05
    assert r["cv"]["roc_auc"]["low"] <= 0.5 <= r["cv"]["roc_auc"]["high"]
    fs = real["config"]["primary_feature_set"]
    # 決定性規則產生的模式在 engineered 特徵下應明顯可學
    for mode in ("hdf", "pwf", "osf"):
        assert real["cv"][fs][mode]["summary"]["pr_auc"]["mean"] > 0.8
        assert real["cv"][fs][mode]["summary"]["pr_auc"]["low"] > real["cv"][fs][mode]["prevalence"]


# ---------- 第 5 輪：開發集選門檻、時間區塊 CV、RNF 時間群聚 ----------

def test_best_f1_threshold_hand_computed():
    # 分數 .9(+) .8(-) .7(+) .2(-)：門檻 .7 → TP2 FP1 → F1 = 0.8；門檻 .9 → F1 = 2/3；門檻 .2 → F1 = 2/3
    y = np.array([1, 0, 1, 0])
    s = np.array([.9, .8, .7, .2])
    t = M.best_f1_threshold(y, s)
    assert t["threshold"] == pytest.approx(0.7)
    assert t["f1"] == pytest.approx(0.8)


def test_best_f1_threshold_without_positives_falls_back_to_half():
    t = M.best_f1_threshold(np.zeros(5, int), np.linspace(0, 1, 5))
    assert t["threshold"] == 0.5 and t["f1"] == 0.0


def test_oof_scores_with_groups_never_splits_a_block():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=100)})
    y = (rng.random(100) < 0.3).astype(int)
    groups = np.arange(100) // 10
    seen = []
    M.oof_scores(_logreg(), X, y, n_splits=5, seed=0, groups=groups,
                 _spy=lambda tr, va: seen.append((tr, va)))
    for tr, va in seen:
        assert not set(groups[tr]) & set(groups[va])
    assert sorted(np.concatenate([va for _, va in seen]).tolist()) == list(range(100))


def test_scan_cluster_known_cases():
    # 5 個事件全落在長度 10 的窗內 → 最大窗計數 5；在 1000 列中隨機出現這種群聚的機率極小
    c = M.scan_cluster([100, 102, 104, 106, 109], n_total=1000, window=10, n_sims=2000, seed=0)
    assert c["max_count"] == 5 and c["window_start"] == 100
    assert c["p_value"] < 0.01
    # 均勻分散的事件 → 最大窗計數 1，p = 1
    d = M.scan_cluster([0, 200, 400, 600, 800], n_total=1000, window=10, n_sims=500, seed=0)
    assert d["max_count"] == 1 and d["p_value"] == pytest.approx(1.0)


def test_holdout_per_mode_applies_dev_tuned_threshold(syn):
    from ai4i_lab import evaluation as E
    df, splits = syn
    masks = E.split_masks(df, splits)
    res, _ = M.holdout_per_mode(df, masks, ["hdf"], "engineered", "fast_forest", n_boot=30,
                                thresholds={"hdf": 0.3})
    assert res["hdf"]["threshold"] == 0.5
    assert res["hdf"]["tuned"]["threshold"] == 0.3
    assert {"point", "low", "high"} <= set(res["hdf"]["tuned"]["recall"])


# ---------- 第 6 輪：bootstrap 用的快速 AUC（與 scikit-learn 一致） ----------

def test_fast_auc_matches_sklearn_including_ties():
    from sklearn.metrics import average_precision_score, roc_auc_score
    rng = np.random.default_rng(5)
    for _ in range(20):
        n = int(rng.integers(5, 300))
        y = (rng.random(n) < rng.uniform(0.05, 0.6)).astype(int)
        if y.sum() in (0, n):
            continue
        s = np.round(rng.random(n), int(rng.integers(1, 4)))      # 四捨五入製造同分
        assert M.fast_roc_auc(y, s) == pytest.approx(roc_auc_score(y, s), abs=1e-12)
        assert M.fast_average_precision(y, s) == pytest.approx(average_precision_score(y, s), abs=1e-12)


def test_fast_auc_single_class_is_nan():
    assert math.isnan(M.fast_roc_auc(np.zeros(4, int), np.arange(4.0)))
    assert math.isnan(M.fast_average_precision(np.zeros(4, int), np.arange(4.0)))
