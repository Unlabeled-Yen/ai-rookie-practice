"""#7 評估框架（切分、重複交叉驗證、指標、信賴區間）的測試。"""
import json
import math

import numpy as np
import pandas as pd
import pytest

from ai4i_lab import evaluation as E


# ---------- 第 1 輪：二元指標與信賴區間 ----------

def test_binary_metrics_hand_computed():
    # y:      1 1 1 0 0 0 0 0
    # proba: .9 .6 .2 .7 .1 .1 .3 .4  → 門檻 .5：TP=2 FN=1 FP=1 TN=4
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    p = np.array([.9, .6, .2, .7, .1, .1, .3, .4])
    m = E.binary_metrics(y, p, threshold=0.5)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (2, 1, 1, 4)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)
    # ROC-AUC：15 個正負配對中，正樣本分數較高的有 .9>全部(5)、.6>4、.2>2(.1,.1) → 11/15
    assert m["roc_auc"] == pytest.approx(11 / 15)
    # PR-AUC（average precision）：依分數排序 .9(+) .7(-) .6(+) .4 .3 .2(+) ...
    # AP = (1/3)(1/1) + (1/3)(2/3) + (1/3)(3/6) = 0.7222…
    assert m["pr_auc"] == pytest.approx((1 + 2 / 3 + 3 / 6) / 3)
    assert m["threshold"] == 0.5


def test_binary_metrics_threshold_is_inclusive_and_changes_counts():
    y = np.array([1, 0, 1, 0])
    p = np.array([.5, .5, .49, .1])
    m = E.binary_metrics(y, p, threshold=0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 1)
    m2 = E.binary_metrics(y, p, threshold=0.0)
    assert m2["recall"] == 1.0 and m2["fp"] == 2


def test_binary_metrics_no_predicted_positive_gives_zero_precision_not_error():
    y = np.array([1, 0, 0, 0])
    m = E.binary_metrics(y, np.zeros(4), threshold=0.5)
    assert m["precision"] == 0.0 and m["recall"] == 0.0 and m["f1"] == 0.0


def test_binary_metrics_single_class_auc_is_nan():
    m = E.binary_metrics(np.zeros(5, int), np.linspace(0, 1, 5))
    assert math.isnan(m["roc_auc"]) and math.isnan(m["pr_auc"])


def test_binary_metrics_rejects_bad_input():
    with pytest.raises(ValueError):
        E.binary_metrics([0, 1], [0.1])
    with pytest.raises(ValueError):
        E.binary_metrics([0, 2], [0.1, 0.2])


def test_confidence_interval_t_known_answer():
    # [1..5]：平均 3、s = sqrt(2.5)、se = s/√5 = 0.70711；t(0.975, df=4) = 2.776445
    ci = E.confidence_interval([1, 2, 3, 4, 5])
    half = 2.776445105 * math.sqrt(2.5) / math.sqrt(5)
    assert ci["mean"] == pytest.approx(3.0)
    assert ci["low"] == pytest.approx(3 - half, abs=1e-6)
    assert ci["high"] == pytest.approx(3 + half, abs=1e-6)
    assert ci["n"] == 5


def test_confidence_interval_90_percent_known_answer():
    # t(0.95, df=4) = 2.131847
    ci = E.confidence_interval([1, 2, 3, 4, 5], confidence=0.90)
    assert ci["high"] - ci["mean"] == pytest.approx(2.131846786 * math.sqrt(0.5), abs=1e-6)


def test_confidence_interval_nadeau_bengio_correction_known_answer():
    # 校正變異數 = s² (1/n + n_test/n_train)；5 折 → n_test/n_train = 1/4
    # [1..5]：s² = 2.5 → var = 2.5 (0.2 + 0.25) = 1.125 → se = 1.06066
    ci = E.confidence_interval([1, 2, 3, 4, 5], test_train_ratio=0.25)
    assert ci["high"] - ci["mean"] == pytest.approx(2.776445105 * math.sqrt(1.125), abs=1e-6)
    naive = E.confidence_interval([1, 2, 3, 4, 5])
    assert ci["high"] - ci["low"] > naive["high"] - naive["low"]


def test_confidence_interval_edge_cases():
    same = E.confidence_interval([0.7, 0.7, 0.7])
    assert same["low"] == pytest.approx(0.7) and same["high"] == pytest.approx(0.7)
    one = E.confidence_interval([0.4])
    assert one["mean"] == 0.4 and math.isnan(one["low"]) and math.isnan(one["high"])
    with_nan = E.confidence_interval([1.0, float("nan"), 3.0])
    assert with_nan["n"] == 2 and with_nan["mean"] == pytest.approx(2.0)
    with pytest.raises(ValueError):
        E.confidence_interval([])
    with pytest.raises(ValueError):
        E.confidence_interval([1, 2], confidence=1.5)


# ---------- 第 2 輪：分層切分、時間切分與固定切分檔 ----------

def _toy_df(n=1000, pos_rate=0.05, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < pos_rate).astype(int)
    modes = {m: np.zeros(n, int) for m in ["twf", "hdf", "pwf", "osf", "rnf"]}
    pos = np.flatnonzero(y)
    for i, k in enumerate(pos):
        modes[["twf", "hdf", "pwf", "osf"][i % 4]][k] = 1
    return pd.DataFrame({"udi": np.arange(1, n + 1), "machine_failure": y, **modes})


def _assert_partition(parts, n):
    allidx = np.concatenate([parts["train"], parts["val"], parts["test"]])
    assert len(allidx) == n and len(np.unique(allidx)) == n
    assert set(parts["train"]).isdisjoint(parts["test"])
    assert set(parts["val"]).isdisjoint(parts["test"])
    assert set(parts["train"]).isdisjoint(parts["val"])


def test_split_stratified_partitions_and_sizes():
    y = _toy_df()["machine_failure"].to_numpy()
    parts = E.split_stratified(y, test_size=0.2, val_size=0.2, seed=E.SEED)
    _assert_partition(parts, len(y))
    assert (len(parts["train"]), len(parts["val"]), len(parts["test"])) == (600, 200, 200)


def test_split_stratified_preserves_positive_rate():
    y = _toy_df(n=2000, pos_rate=0.034)["machine_failure"].to_numpy()
    parts = E.split_stratified(y)
    overall = y.mean()
    for name in ("train", "val", "test"):
        assert abs(y[parts[name]].mean() - overall) < 0.004, name


def test_split_stratified_reproducible_and_seed_dependent():
    y = _toy_df()["machine_failure"].to_numpy()
    a, b = E.split_stratified(y, seed=42), E.split_stratified(y, seed=42)
    c = E.split_stratified(y, seed=7)
    assert all(np.array_equal(a[k], b[k]) for k in a)
    assert not np.array_equal(a["test"], c["test"])


def test_split_stratified_uses_custom_strata():
    # strata 讓少數群（此處 g=1，只有 10 筆）在三個部分都出現
    y = np.zeros(100, int)
    strata = np.array([1] * 10 + [0] * 90)
    parts = E.split_stratified(y, strata=strata, test_size=0.2, val_size=0.2)
    assert [int(strata[parts[k]].sum()) for k in ("train", "val", "test")] == [6, 2, 2]


def test_failure_strata_merges_rare_groups():
    df = _toy_df(n=2000, pos_rate=0.05)
    df.loc[0, ["machine_failure", "rnf"]] = [0, 1]       # 單筆罕見組合 → 併入 rare
    s = E.failure_strata(df, min_count=10)
    assert s[0] == "0|rare"
    assert set(s[df["machine_failure"] == 0][1:]) == {"0|none"}
    counts = pd.Series(s).value_counts()
    assert (counts[counts.index != "0|rare"] >= 10).all()


def test_split_temporal_keeps_order():
    order = np.array([5, 3, 1, 4, 2, 10, 9, 8, 7, 6])     # 例如 UDI
    parts = E.split_temporal(order, train_frac=0.6, val_frac=0.2)
    _assert_partition(parts, 10)
    assert sorted(order[parts["train"]]) == [1, 2, 3, 4, 5, 6]
    assert sorted(order[parts["val"]]) == [7, 8]
    assert sorted(order[parts["test"]]) == [9, 10]
    assert order[parts["train"]].max() < order[parts["val"]].min() < order[parts["test"]].min()


def test_split_temporal_rejects_bad_fractions():
    with pytest.raises(ValueError):
        E.split_temporal(np.arange(10), train_frac=0.9, val_frac=0.2)


def test_make_splits_uses_udi_and_roundtrips(tmp_path):
    df = _toy_df(n=500)
    sp = E.make_splits(df)
    assert set(sp["schemes"]) == {"stratified", "temporal"}
    for scheme in sp["schemes"].values():
        u = [set(scheme[k]["udi"]) for k in ("train", "val", "test")]
        assert u[0].isdisjoint(u[2]) and u[1].isdisjoint(u[2]) and u[0].isdisjoint(u[1])
        assert len(u[0] | u[1] | u[2]) == 500
    path = tmp_path / "splits.json"
    E.save_splits(sp, path)
    assert E.load_splits(path) == json.loads(path.read_text(encoding="utf-8"))


def test_split_masks_work_on_row_subset(tmp_path):
    # drop_inconsistent 會移除部分列：遮罩依 UDI 對齊，被移除的列不影響其他列的歸屬
    df = _toy_df(n=500)
    sp = E.make_splits(df)
    sub = df.iloc[10:].reset_index(drop=True)
    masks = E.split_masks(sub, sp, "stratified")
    assert sum(int(m.sum()) for m in masks.values()) == len(sub)
    test_udi = set(sp["schemes"]["stratified"]["test"]["udi"]) - set(df["udi"][:10])
    assert set(sub.loc[masks["test"], "udi"]) == test_udi
    with pytest.raises(ValueError):
        E.split_masks(sub.assign(udi=sub["udi"] + 10_000), sp, "stratified")


REAL_SPLITS = E.SPLITS_PATH


@pytest.mark.skipif(not REAL_SPLITS.exists(), reason="尚未產生 outputs/ai4i/splits.json")
def test_real_splits_fixed_and_disjoint():
    from ai4i_lab.data import load_dataset
    sp = E.load_splits()
    st = sp["schemes"]["stratified"]
    train, val, test = (set(st[k]["udi"]) for k in ("train", "val", "test"))
    assert len(test) == 2000 and not (test & train) and not (test & val)
    assert len(train | val | test) == 10_000
    # 固定：以相同種子重新產生，測試集逐筆一致
    again = E.make_splits(load_dataset())
    assert again["schemes"]["stratified"]["test"]["udi"] == st["test"]["udi"]
    assert again["schemes"]["temporal"]["test"]["udi"] == sp["schemes"]["temporal"]["test"]["udi"]


# ---------- 第 3 輪：重複分層交叉驗證（含過採樣防洩漏） ----------

class SpyModel:
    """X 第 0 欄是列編號、第 1 欄是分數；記錄 fit 與 predict_proba 看到的列編號。"""
    log = []

    def fit(self, X, y):
        self.fit_ids = np.asarray(X)[:, 0].astype(int)
        SpyModel.log.append(("fit", self.fit_ids))
        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        SpyModel.log.append(("predict", X[:, 0].astype(int)))
        s = X[:, 1]
        return np.column_stack([1 - s, s])


class SpySampler:
    """把少數類複製一次（像 RandomOverSampler），並記錄被要求重抽樣的列編號。"""
    seen = []

    def fit_resample(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        SpySampler.seen.append(X[:, 0].astype(int))
        pos = y == 1
        return np.vstack([X, X[pos]]), np.concatenate([y, y[pos]])


def _cv_data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.15).astype(int)
    score = np.clip(0.6 * y + rng.random(n) * 0.5, 0, 1)
    X = np.column_stack([np.arange(n), score])
    return X, y


def test_repeated_cv_summary_shape_and_fold_count():
    X, y = _cv_data()
    res = E.repeated_cv(SpyModel, X, y, n_splits=5, n_repeats=2)
    assert len(res["folds"]) == 10
    assert set(res["summary"]) == set(E.METRICS)
    for m in E.METRICS:
        s = res["summary"][m]
        assert s["low"] <= s["mean"] <= s["high"]
        assert s["mean"] == pytest.approx(np.mean([f[m] for f in res["folds"]]))
    assert res["config"]["n_splits"] == 5 and res["config"]["n_repeats"] == 2


def test_repeated_cv_uses_corrected_ci_and_also_reports_naive():
    X, y = _cv_data()
    res = E.repeated_cv(SpyModel, X, y, n_splits=5, n_repeats=2)
    vals = [f["pr_auc"] for f in res["folds"]]
    corrected = E.confidence_interval(vals, test_train_ratio=1 / 4)
    naive = E.confidence_interval(vals)
    s = res["summary"]["pr_auc"]
    assert s["low"] == pytest.approx(corrected["low"]) and s["high"] == pytest.approx(corrected["high"])
    assert s["naive_low"] == pytest.approx(naive["low"])


def test_repeated_cv_validation_rows_never_seen_in_fit_even_with_oversampling():
    X, y = _cv_data()
    SpyModel.log.clear()
    SpySampler.seen.clear()
    E.repeated_cv(SpyModel, X, y, n_splits=5, n_repeats=2, make_sampler=SpySampler)
    fits = [ids for kind, ids in SpyModel.log if kind == "fit"]
    preds = [ids for kind, ids in SpyModel.log if kind == "predict"]
    assert len(fits) == len(preds) == len(SpySampler.seen) == 10
    for fit_ids, pred_ids, sampled_ids in zip(fits, preds, SpySampler.seen):
        assert set(fit_ids).isdisjoint(pred_ids)            # 驗證折未進入訓練
        assert set(sampled_ids).isdisjoint(pred_ids)        # 過採樣器只看得到訓練折
        assert set(fit_ids) == set(sampled_ids)             # 訓練資料只來自訓練折（可重複）
        assert len(fit_ids) > len(sampled_ids)              # 確實做了過採樣
        assert len(pred_ids) == len(X) // 5                 # 驗證折未被重抽樣
    # 每次重複，驗證折剛好覆蓋全部資料一次
    for r in range(2):
        covered = np.concatenate(preds[r * 5:(r + 1) * 5])
        assert sorted(covered) == list(range(len(X)))


def test_repeated_cv_with_real_oversampler_inside_imblearn_pipeline():
    pytest.importorskip("imblearn")
    from imblearn.over_sampling import RandomOverSampler
    from imblearn.pipeline import make_pipeline
    from sklearn.linear_model import LogisticRegression
    X, y = _cv_data()
    res = E.repeated_cv(lambda: make_pipeline(RandomOverSampler(random_state=0), LogisticRegression()),
                        X[:, 1:], y, n_splits=5, n_repeats=1)
    assert res["summary"]["roc_auc"]["mean"] > 0.8


def test_repeated_cv_new_model_per_fold_and_reproducible():
    X, y = _cv_data()
    calls = []

    def factory():
        calls.append(1)
        return SpyModel()
    a = E.repeated_cv(factory, X, y, n_splits=4, n_repeats=3, seed=1)
    b = E.repeated_cv(SpyModel, X, y, n_splits=4, n_repeats=3, seed=1)
    assert len(calls) == 12
    assert [f["pr_auc"] for f in a["folds"]] == [f["pr_auc"] for f in b["folds"]]


def test_repeated_cv_folds_are_stratified():
    X, y = _cv_data(n=300)
    res = E.repeated_cv(SpyModel, X, y, n_splits=5, n_repeats=1)
    pos = [f["tp"] + f["fn"] for f in res["folds"]]
    assert max(pos) - min(pos) <= 1


def test_repeated_cv_perfect_model_and_dataframe_input():
    X, y = _cv_data()
    Xdf = pd.DataFrame({"id": X[:, 0], "score": y.astype(float)})
    res = E.repeated_cv(SpyModel, Xdf, pd.Series(y), n_splits=5, n_repeats=1)
    for m in E.METRICS:
        assert res["summary"][m]["mean"] == pytest.approx(1.0)


def test_repeated_cv_requires_predict_proba():
    class NoProba:
        def fit(self, X, y):
            return self
    X, y = _cv_data()
    with pytest.raises(TypeError):
        E.repeated_cv(NoProba, X, y, n_splits=2, n_repeats=1)


# ---------- 第 4 輪：過採樣器、時間漂移、評估網格與 run() ----------

def test_random_oversampler_balances_by_duplicating_minority_only():
    X = np.column_stack([np.arange(10), np.arange(10) * 2.0])
    y = np.array([1, 1] + [0] * 8)
    Xr, yr = E.RandomOversampler(seed=0).fit_resample(X, y)
    assert (yr == 1).sum() == (yr == 0).sum() == 8
    assert set(map(tuple, Xr)) == set(map(tuple, X))                 # 只有複製，沒有新樣本
    Xr2, _ = E.RandomOversampler(seed=0).fit_resample(X, y)
    assert np.array_equal(Xr, Xr2)
    Xdf, ys = E.RandomOversampler(seed=0).fit_resample(pd.DataFrame(X, columns=["a", "b"]), pd.Series(y))
    assert list(Xdf.columns) == ["a", "b"] and len(ys) == 16


def _sensor_df(n=600, seed=0):
    rng = np.random.default_rng(seed)
    df = _toy_df(n=n, seed=seed)
    df["type"] = rng.choice(["L", "M", "H"], n)
    df["air_temp_k"] = 298 + np.linspace(0, 4, n) + rng.normal(0, 0.1, n)     # 隨時間漂移
    df["process_temp_k"] = df["air_temp_k"] + 10 + rng.normal(0, 0.5, n)
    df["rotational_speed_rpm"] = rng.normal(1500, 100, n)
    df["torque_nm"] = rng.normal(40, 8, n)
    df["tool_wear_min"] = rng.integers(0, 250, n)
    return df


def test_temporal_drift_detects_shifted_feature():
    df = _sensor_df()
    sp = E.make_splits(df)
    d = E.temporal_drift(df, sp)
    assert set(d["parts"]) == {"stratified", "temporal"}
    assert d["parts"]["temporal"]["test"]["air_temp_k_mean"] > d["parts"]["temporal"]["train"]["air_temp_k_mean"] + 2
    ks_t = d["ks_train_vs_test"]["temporal"]["air_temp_k"]["statistic"]
    ks_s = d["ks_train_vs_test"]["stratified"]["air_temp_k"]["statistic"]
    assert ks_t > 0.9 and ks_s < 0.2


def test_evaluate_grid_covers_every_combination():
    df = _sensor_df(n=400)
    df["machine_failure"] = (df["torque_nm"] > 50).astype(int)
    grid = E.evaluate_grid(df, models=["logreg"], feature_sets=["raw", "engineered"],
                           imbalance=["none", "class_weight", "oversample"], n_splits=3, n_repeats=1)
    combos = {(r["model"], r["feature_set"], r["imbalance"]) for r in grid}
    assert combos == {("logreg", fs, im) for fs in ("raw", "engineered") for im in ("none", "class_weight", "oversample")}
    for r in grid:
        assert set(r["summary"]) == set(E.METRICS) and r["summary"]["roc_auc"]["mean"] > 0.9


def test_holdout_check_never_uses_stratified_test_rows():
    df = _sensor_df(n=400)
    sp = E.make_splits(df)
    used = []
    res = E.holdout_check(df, sp, models=["logreg"], feature_sets=["raw"], _spy=used.append)
    test_udi = set(sp["schemes"]["stratified"]["test"]["udi"])
    # 審查修正：所有方案（含 temporal）的訓練列與評估列都不可碰到共用的分層測試集
    assert {scheme for scheme, _ in used} == {"stratified", "temporal"}
    assert all(u and not (u & test_udi) for _, u in used)
    assert {r["scheme"] for r in res} == {"stratified", "temporal"}


def test_holdout_check_temporal_parts_exclude_shared_test_and_report_sizes():
    df = _sensor_df(n=400)
    sp = E.make_splits(df)
    res = E.holdout_check(df, sp, models=["logreg"], feature_sets=["raw"])
    test_udi = set(sp["schemes"]["stratified"]["test"]["udi"])
    for r in res:
        part = sp["schemes"][r["scheme"]]
        n_train = len(set(part["train"]["udi"]) - test_udi)
        n_eval = len(set(part[r["eval_part"]]["udi"]) - test_udi)
        assert (r["n_train"], r["n_eval"]) == (n_train, n_eval)
        assert r["tp"] + r["fp"] + r["fn"] + r["tn"] == n_eval
    t = next(r for r in res if r["scheme"] == "temporal")
    assert t["n_train"] < len(sp["schemes"]["temporal"]["train"]["udi"])   # 確實剔除了重疊列


def test_run_cv_only_sees_development_rows(tmp_path, monkeypatch):
    df = _sensor_df(n=500)
    seen = []
    real = E.evaluate_grid

    def spy(dev, **kw):
        seen.append(set(dev["udi"]))
        return real(dev, **kw)
    monkeypatch.setattr(E, "evaluate_grid", spy)
    E.run(out_dir=tmp_path, df=df, models=["logreg"], n_splits=3, n_repeats=1)
    test_udi = set(E.load_splits(tmp_path / "splits.json")["schemes"]["stratified"]["test"]["udi"])
    assert len(seen) == 1 and seen[0] and not (seen[0] & test_udi)
    assert seen[0] | test_udi == set(df["udi"])


def test_build_features_never_contains_failure_modes_or_ids():
    from ai4i_lab.data import FAILURE_MODES, ID_COLUMNS, TARGET
    from ai4i_lab.features import build_features
    df = _sensor_df(n=50)
    df["product_id"] = [f"L{i}" for i in range(50)]
    for fs in E.FEATURE_SET_NAMES:
        cols = set(build_features(df, fs).columns)
        assert not cols & (set(FAILURE_MODES) | set(ID_COLUMNS) | {TARGET}), fs


def test_fold_seeded_oversampler_draws_differ_between_folds():
    make = E.fold_seeded_sampler(seed=7)
    seeds = [make().seed for _ in range(4)]
    assert seeds == [7, 8, 9, 10]
    assert [E.fold_seeded_sampler(seed=7)().seed for _ in range(2)] == [7, 7]   # 每個工廠各自從 seed 起算
    X = np.arange(40).reshape(20, 2)
    y = np.array([1] * 3 + [0] * 17)
    f = E.fold_seeded_sampler(seed=0)
    a, _ = f().fit_resample(X, y)
    b, _ = f().fit_resample(X, y)
    assert not np.array_equal(a, b)


DATA = E.ROOT / "data" / "ai4i2020.csv"


@pytest.mark.skipif(not DATA.exists(), reason="缺少 data/ai4i2020.csv")
def test_run_on_small_sample_writes_outputs(tmp_path):
    from ai4i_lab.data import load_dataset
    df = load_dataset().sample(1500, random_state=0).sort_values("udi").reset_index(drop=True)
    res = E.run(out_dir=tmp_path, df=df, models=["logreg"], n_splits=3, n_repeats=1)
    assert (tmp_path / "splits.json").exists()
    fw = tmp_path / "framework"
    saved = json.loads((fw / "framework.json").read_text(encoding="utf-8"))
    assert saved["cv"] and saved["drift"] and saved["holdout"] and saved["conclusions"]
    for png in E.FIGURES:
        assert (fw / png).stat().st_size > 10_000
    assert res["splits_path"].endswith("splits.json")


REAL_FW = E.FRAMEWORK_DIR / "framework.json"


@pytest.mark.skipif(not REAL_FW.exists(), reason="尚未執行 python -m ai4i_lab.evaluation")
def test_real_framework_results_are_complete():
    fw = json.loads(REAL_FW.read_text(encoding="utf-8"))
    combos = {(r["model"], r["feature_set"], r["imbalance"]) for r in fw["cv"]}
    assert len(combos) == len(fw["config"]["models"]) * 2 * 3
    assert fw["config"]["n_splits"] * fw["config"]["n_repeats"] >= 10
    for r in fw["cv"]:
        assert r["summary"]["pr_auc"]["low"] <= r["summary"]["pr_auc"]["mean"] <= r["summary"]["pr_auc"]["high"]
    rates = fw["drift"]["parts"]["temporal"]
    assert rates["train"]["n"] == 6000 and rates["test"]["n"] == 2000


# ---------- 第 5 輪（重構）：有界指標的區間截斷 ----------

def test_confidence_interval_clips_to_bounds():
    vals = [1.0, 1.0, 0.9, 1.0, 0.95]
    raw = E.confidence_interval(vals, test_train_ratio=0.25)
    assert raw["high"] > 1.0
    clipped = E.confidence_interval(vals, test_train_ratio=0.25, bounds=(0.0, 1.0))
    assert clipped["high"] == 1.0 and clipped["low"] == pytest.approx(raw["low"])
    assert clipped["mean"] == pytest.approx(raw["mean"])


def test_repeated_cv_summary_stays_within_unit_interval():
    X, y = _cv_data()
    Xp = np.column_stack([X[:, 0], np.where(y == 1, 0.9, 0.1)])     # 幾乎完美 → 上界容易超過 1
    Xp[:3, 1] = 0.6
    res = E.repeated_cv(SpyModel, Xp, y, n_splits=5, n_repeats=2)
    for m in E.METRICS:
        s = res["summary"][m]
        assert 0.0 <= s["low"] <= s["high"] <= 1.0 and s["naive_high"] <= 1.0
