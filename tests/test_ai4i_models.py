"""#8 二元故障分類（模型工廠、不平衡處理、CV 網格、MLP 調參、門檻、最終模型）的測試。"""
import json

import numpy as np
import pandas as pd
import pytest

from ai4i_lab import evaluation as E
from ai4i_lab import models as M

# 小資料、少迭代的 MLP 必然出現收斂警告，與測試目的無關
pytestmark = pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning",
                                        "ignore::DeprecationWarning")


def _xy(n=400, pos=0.1, seed=0, d=4):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = (rng.random(n) < pos).astype(int)
    X[y == 1, 0] += 2.5                       # 第一欄可分辨正類
    return pd.DataFrame(X, columns=[f"x{i}" for i in range(d)]), y


# ---------- 第 1 輪：模型工廠與不平衡處理 ----------

@pytest.mark.parametrize("model", M.MODEL_NAMES)
@pytest.mark.parametrize("imbalance", M.IMBALANCE)
def test_every_factory_fits_and_predicts_proba(model, imbalance):
    X, y = _xy()
    est = M.make_estimator(model, imbalance, params={"max_iter": 50} if model == "mlp" else None)
    est.fit(X, y)
    p = est.predict_proba(X)
    assert p.shape == (len(X), 2)
    assert np.all((p >= 0) & (p <= 1))


def test_model_and_imbalance_names():
    assert M.MODEL_NAMES == ("logreg", "random_forest", "hist_gb", "mlp")
    assert M.IMBALANCE == ("none", "class_weight", "smote")
    with pytest.raises(ValueError):
        M.make_estimator("svm", "none")
    with pytest.raises(ValueError):
        M.make_estimator("logreg", "undersample")


@pytest.mark.parametrize("model", ["logreg", "random_forest", "hist_gb"])
def test_class_weight_is_balanced_only_when_requested(model):
    final = lambda e: e.steps[-1][1] if hasattr(e, "steps") else e  # noqa: E731
    assert final(M.make_estimator(model, "class_weight")).class_weight == "balanced"
    assert final(M.make_estimator(model, "none")).class_weight is None


def test_mlp_class_weight_uses_random_duplication_and_early_stopping():
    # MLPClassifier（sklearn 1.6）不支援 sample_weight：以隨機複製少數類達到等效的類別權重
    from imblearn.over_sampling import RandomOverSampler
    est = M.make_estimator("mlp", "class_weight")
    assert any(isinstance(s, RandomOverSampler) for _, s in est.steps)
    mlp = est.steps[-1][1]
    assert mlp.early_stopping is True and mlp.max_iter <= 500 and mlp.random_state == M.SEED


def test_smote_pipelines_resample_only_during_fit():
    from imblearn.over_sampling import SMOTE
    for model in M.MODEL_NAMES:
        est = M.make_estimator(model, "smote", params={"max_iter": 30} if model == "mlp" else None)
        assert any(isinstance(s, SMOTE) for _, s in est.steps), model
    X, y = _xy(n=300)
    est = M.make_estimator("logreg", "smote").fit(X, y)
    assert est.predict_proba(X.iloc[:37]).shape[0] == 37      # 預測時不重抽樣


def test_smote_inside_cv_only_sees_training_folds(monkeypatch):
    from imblearn.over_sampling import SMOTE
    seen = []
    orig = SMOTE.fit_resample

    def spy(self, X, y):
        seen.append(len(y))
        return orig(self, X, y)
    monkeypatch.setattr(SMOTE, "fit_resample", spy)
    X, y = _xy(n=500)
    E.repeated_cv(lambda: M.make_estimator("logreg", "smote"), X, y, n_splits=5, n_repeats=1)
    assert seen == [400] * 5                                   # 每折只對 4/5 的訓練列重抽樣


def test_mlp_params_override_defaults():
    est = M.make_estimator("mlp", "none", params={"hidden_layer_sizes": (8,), "alpha": 0.1,
                                                  "learning_rate_init": 0.01})
    mlp = est.steps[-1][1]
    assert mlp.hidden_layer_sizes == (8,) and mlp.alpha == 0.1 and mlp.learning_rate_init == 0.01


# ---------- 第 2 輪：門檻、CV 網格、原始 vs 衍生特徵差異 ----------

def test_tune_threshold_maximizes_f1_hand_computed():
    # 門檻 .9→F1 .5；.8→.8；.7→.667；.6→TP3 FP1 FN0 → P .75 R 1 → F1 6/7；.3→.75；.1→.667
    y = np.array([1, 1, 0, 1, 0, 0])
    p = np.array([.9, .8, .7, .6, .3, .1])
    t = M.tune_threshold(y, p)
    assert t["threshold"] == pytest.approx(0.6)
    assert t["f1"] == pytest.approx(6 / 7)
    assert (t["precision"], t["recall"]) == (pytest.approx(0.75), pytest.approx(1.0))
    curve = t["curve"]
    assert {"threshold", "precision", "recall", "f1"} <= set(curve)
    assert len(curve["threshold"]) == 6


def test_tune_threshold_tie_prefers_higher_threshold():
    # .8 與 .4 的 F1 都是 2/3（TP1 FP0 FN1 vs TP2 FP2 FN0），取較高門檻（較少誤報）
    y = np.array([1, 0, 0, 1])
    p = np.array([.8, .6, .5, .4])
    assert M.tune_threshold(y, p)["threshold"] == pytest.approx(0.8)


def test_tune_threshold_without_positives_returns_default():
    t = M.tune_threshold(np.zeros(5, int), np.linspace(0, 1, 5))
    assert t["threshold"] == 0.5 and t["f1"] == 0.0


def _dev_df(n=600, seed=1):
    """小型合成資料（欄位與真實資料相同）：故障由 HDF 式的規則產生。"""
    rng = np.random.default_rng(seed)
    air = rng.normal(300, 2, n)
    proc = air + rng.normal(10, 1, n)
    rpm = rng.normal(1500, 150, n)
    torque = rng.normal(40, 10, n)
    wear = rng.integers(0, 250, n)
    hdf = ((proc - air) < 9.2) & (rpm < 1450)
    noise = rng.random(n) < 0.01
    fail = (hdf | noise).astype(int)
    return pd.DataFrame({
        "udi": np.arange(1, n + 1), "product_id": [f"L{i}" for i in range(n)],
        "type": rng.choice(["L", "M", "H"], n, p=[.6, .3, .1]),
        "air_temp_k": air, "process_temp_k": proc, "rotational_speed_rpm": rpm, "torque_nm": torque,
        "tool_wear_min": wear, "machine_failure": fail,
        "twf": 0, "hdf": hdf.astype(int), "pwf": 0, "osf": 0, "rnf": (noise & ~hdf).astype(int)})


FAST_MLP = {"hidden_layer_sizes": (8,), "max_iter": 40}


def test_cv_grid_rows_and_table_have_ci():
    df = _dev_df(n=400)
    rows = M.cv_grid(df, models=("logreg", "hist_gb"), feature_sets=("raw", "engineered"),
                     imbalance=("none", "smote"), n_splits=3, n_repeats=1)
    assert len(rows) == 2 * 2 * 2
    assert len(rows[0]["folds"]) == 3
    table = M.cv_table(rows)
    for m in E.METRICS:
        for s in ("mean", "std", "low", "high"):
            assert f"{m}_{s}" in table.columns
    assert set(table["model"]) == {"logreg", "hist_gb"}
    assert (table["pr_auc_low"] <= table["pr_auc_mean"]).all()
    assert (table["pr_auc_mean"] <= table["pr_auc_high"]).all()


def test_cv_grid_is_reproducible():
    df = _dev_df()
    kw = dict(models=("mlp",), feature_sets=("raw",), imbalance=("class_weight",), n_splits=3, n_repeats=1,
              mlp_params=FAST_MLP)
    a, b = M.cv_grid(df, **kw), M.cv_grid(df, **kw)
    assert [f["pr_auc"] for f in a[0]["folds"]] == [f["pr_auc"] for f in b[0]["folds"]]


def _row(model, fs, im, vals, n_splits):
    folds = [{"repeat": i // n_splits, "fold": i % n_splits, "pr_auc": v} for i, v in enumerate(vals)]
    return {"model": model, "feature_set": fs, "imbalance": im, "folds": folds, "n_splits": n_splits}


def test_feature_set_differences_paired_by_fold():
    raw = _row("rf", "raw", "none", [0.5, 0.6, 0.7, 0.4], 2)
    eng = _row("rf", "engineered", "none", [0.6, 0.7, 0.8, 0.5], 2)
    d = M.feature_set_differences([raw, eng], metric="pr_auc")
    assert len(d) == 1
    assert d[0]["diff_mean"] == pytest.approx(0.1)
    assert d[0]["diff_low"] == pytest.approx(0.1) and d[0]["diff_high"] == pytest.approx(0.1)
    assert d[0]["raw_mean"] == pytest.approx(0.55) and d[0]["engineered_mean"] == pytest.approx(0.65)


def test_feature_set_difference_ci_is_nadeau_bengio_hand_computed():
    # 差異 [0.1, 0.3]，k=2 → 平均 .2、s² = .02、校正因子 1/2 + 1/(2−1) = 1.5
    # se = √.03 = .17321；t(.975, df=1) = 12.7062 → 半寬 2.2008（截在 [−1, 1]）
    raw = _row("m", "raw", "none", [0.5, 0.5], 2)
    eng = _row("m", "engineered", "none", [0.6, 0.8], 2)
    d = M.feature_set_differences([raw, eng])[0]
    assert d["diff_mean"] == pytest.approx(0.2)
    assert d["diff_low"] == -1.0 and d["diff_high"] == 1.0
    raw4 = _row("m", "raw", "none", [0.5] * 4, 2)
    eng4 = _row("m", "engineered", "none", [0.6, 0.8, 0.6, 0.8], 2)
    d4 = M.feature_set_differences([raw4, eng4])[0]
    # s² = .013333、因子 1/4 + 1 = 1.25 → se = .12910；t(.975, 3) = 3.18245 → 半寬 .41085
    assert d4["diff_low"] == pytest.approx(0.2 - 0.41085, abs=1e-4)
    assert d4["significant"] is False


# ---------- 第 3 輪：MLP 超參數搜尋與訓練曲線 ----------

def _train_val(seed=3):
    X, y = _xy(n=500, pos=0.15, seed=seed)
    return X.iloc[:350], y[:350], X.iloc[350:], y[350:]


TINY_GRID = {"hidden_layer_sizes": [(4,), (16,)], "learning_rate_init": [1e-3, 1e-2], "alpha": [1e-4]}


def test_mlp_grid_default_covers_structure_lr_and_regularization():
    g = M.MLP_GRID
    assert len(g["hidden_layer_sizes"]) >= 3 and any(len(h) == 2 for h in g["hidden_layer_sizes"])
    assert len(g["learning_rate_init"]) >= 2 and len(g["alpha"]) >= 2


def test_mlp_search_scores_every_config_on_val_and_sorts():
    Xtr, ytr, Xva, yva = _train_val()
    res = M.mlp_search(Xtr, ytr, Xva, yva, grid=TINY_GRID, base={"max_iter": 60})
    assert len(res) == 4
    scores = [r["val_pr_auc"] for r in res]
    assert scores == sorted(scores, reverse=True)
    best = res[0]
    # 用最佳參數重新訓練，在 val 上的 PR-AUC 應與搜尋結果相同（可重現、且確實在 val 上評分）
    est = M.make_estimator("mlp", "none", params={"max_iter": 60} | M.params_of(best)).fit(Xtr, ytr)
    assert E.binary_metrics(yva, est.predict_proba(Xva)[:, 1])["pr_auc"] == pytest.approx(best["val_pr_auc"])
    assert {"hidden_layer_sizes", "learning_rate_init", "alpha", "n_iter", "val_roc_auc"} <= set(best)


def test_mlp_search_signature_has_no_test_argument():
    import inspect
    params = list(inspect.signature(M.mlp_search).parameters)
    assert params[:4] == ["X_train", "y_train", "X_val", "y_val"]
    assert not any("test" in p for p in params)


def test_mlp_training_curve_tracks_train_and_val():
    Xtr, ytr, Xva, yva = _train_val()
    c = M.mlp_training_curve(Xtr, ytr, Xva, yva, params={"hidden_layer_sizes": (8,)}, epochs=15)
    assert c["epoch"] == list(range(1, 16))
    for k in ("train_loss", "val_loss", "val_pr_auc"):
        assert len(c[k]) == 15 and np.all(np.isfinite(c[k]))
    assert c["train_loss"][-1] < c["train_loss"][0]            # 訓練損失下降
    assert 1 <= c["best_epoch"] <= 15
    assert c["best_epoch"] == int(np.argmin(c["val_loss"])) + 1


# ---------- 第 4 輪：run()、預測檔介面、測試集隔離、最終模型快取 ----------

FAST_RUN = dict(models=("logreg", "mlp"), imbalance=("none", "smote"), n_splits=3, n_repeats=1,
                mlp_grid={"hidden_layer_sizes": [(4,), (8,)], "learning_rate_init": [1e-2], "alpha": [1e-4]},
                mlp_base={"max_iter": 30}, curve_epochs=4, n_boot=50)


@pytest.fixture(scope="module")
def synth():
    df = _dev_df(n=900, seed=5)
    return df, E.make_splits(df)


@pytest.fixture(scope="module")
def synth_run(synth, tmp_path_factory):
    df, splits = synth
    out = tmp_path_factory.mktemp("models")
    return out, M.run(out_dir=out, df=df, splits=splits, **FAST_RUN)


def test_run_writes_all_deliverables(synth_run):
    out, res = synth_run
    for f in ["cv_results.csv", "cv_folds.csv", "feature_set_diff.csv", "mlp_search.csv", "models.json",
              "predictions_val.csv", "predictions_test.csv", *M.FIGURES]:
        assert (out / f).exists(), f
    for model in FAST_RUN["models"]:
        for fs in M.FEATURE_SET_NAMES:
            assert (out / "final" / f"{model}__{fs}.joblib").exists()
    cv = pd.read_csv(out / "cv_results.csv")
    assert len(cv) == 2 * 2 * 2 and {"pr_auc_low", "pr_auc_high"} <= set(cv.columns)
    info = json.loads((out / "models.json").read_text(encoding="utf-8"))
    assert {"config", "chosen", "mlp_best", "thresholds", "val_metrics", "test_metrics", "feature_set_diff",
            "mlp_curves", "protocol"} <= set(info)
    assert res["models_dir"] == str(out)


def test_predictions_follow_interface_contract(synth, synth_run):
    df, splits = synth
    out, _ = synth_run
    masks = E.split_masks(df, splits)
    for part in ("val", "test"):
        p = pd.read_csv(out / f"predictions_{part}.csv")
        assert list(p.columns) == ["row_id", "model", "feature_set", "y_true", "proba"]
        assert len(p) == int(masks[part].sum()) * 2 * 2
        for (_, _), g in p.groupby(["model", "feature_set"]):
            assert set(g["row_id"]) == set(df.loc[masks[part], "udi"])        # row_id = UDI
            truth = df.set_index("udi").loc[g["row_id"], "machine_failure"].to_numpy()
            assert (g["y_true"].to_numpy() == truth).all()
        assert p["proba"].between(0, 1).all()


def test_tuning_and_thresholds_never_use_test_set(synth, synth_run, tmp_path):
    """把測試集的特徵與標籤全部竄改後重跑：CV、MLP 搜尋、選擇與門檻必須完全不變。"""
    df, splits = synth
    out_a, _ = synth_run
    test = E.split_masks(df, splits)["test"]
    bad = df.copy()
    rng = np.random.default_rng(0)
    bad.loc[test, "machine_failure"] = 1 - bad.loc[test, "machine_failure"]
    for c in ("air_temp_k", "torque_nm", "rotational_speed_rpm"):
        bad.loc[test, c] = rng.permutation(bad.loc[test, c].to_numpy()) * 1.1
    M.run(out_dir=tmp_path, df=bad, splits=splits, **FAST_RUN)
    for f in ("cv_results.csv", "mlp_search.csv", "predictions_val.csv"):
        a = pd.read_csv(out_a / f).drop(columns=["seconds"], errors="ignore")
        b = pd.read_csv(tmp_path / f).drop(columns=["seconds"], errors="ignore")
        pd.testing.assert_frame_equal(a, b)
    ja = json.loads((out_a / "models.json").read_text(encoding="utf-8"))
    jb = json.loads((tmp_path / "models.json").read_text(encoding="utf-8"))
    strip = lambda d: {fs: {k: v for k, v in r.items() if k != "seconds"} for fs, r in d.items()}  # noqa: E731
    assert strip(ja["mlp_best"]) == strip(jb["mlp_best"])
    for k in ("chosen", "thresholds", "val_metrics", "mlp_curves"):
        assert ja[k] == jb[k], k
    assert ja["test_metrics"] != jb["test_metrics"]            # 測試集確實被（只）用於最後評估


def test_get_final_model_loads_cache_and_matches_predictions(synth, synth_run, monkeypatch):
    df, splits = synth
    out, _ = synth_run
    from sklearn.neural_network import MLPClassifier

    def boom(*a, **k):
        raise AssertionError("有快取時不應重新訓練")
    monkeypatch.setattr(MLPClassifier, "fit", boom)
    est = M.get_final_model("mlp", "engineered", out_dir=out)
    from ai4i_lab.features import build_features
    p = pd.read_csv(out / "predictions_test.csv")
    p = p[(p.model == "mlp") & (p.feature_set == "engineered")].sort_values("row_id")
    sub = df.set_index("udi").loc[p["row_id"]].reset_index()
    assert np.allclose(est.predict_proba(build_features(sub, "engineered"))[:, 1], p["proba"].to_numpy())


def test_get_final_model_retrains_reproducibly_without_cache(synth, synth_run, tmp_path):
    df, splits = synth
    out, _ = synth_run
    import shutil
    shutil.copy(out / "models.json", tmp_path / "models.json")
    a = M.get_final_model("logreg", "raw", out_dir=tmp_path, df=df, splits=splits)
    assert (tmp_path / "final" / "logreg__raw.joblib").exists()
    b = M.get_final_model("logreg", "raw", out_dir=out)
    from ai4i_lab.features import build_features
    X = build_features(df, "raw")
    assert np.allclose(a.predict_proba(X), b.predict_proba(X))


def test_mlp_threshold_tuned_on_val_predictions(synth_run):
    out, _ = synth_run
    info = json.loads((out / "models.json").read_text(encoding="utf-8"))
    p = pd.read_csv(out / "predictions_val.csv")
    g = p[(p.model == "mlp") & (p.feature_set == "engineered")]
    t = M.tune_threshold(g["y_true"], g["proba"])
    assert info["thresholds"]["mlp|engineered"]["threshold"] == pytest.approx(t["threshold"])


def test_repo_config_ignores_joblib_and_pins_packages():
    from ai4i_lab.data import ROOT
    assert "outputs/ai4i/models/**/*.joblib" in (ROOT / ".gitignore").read_text().splitlines()
    req = (ROOT / "requirements.txt").read_text()
    assert "imbalanced-learn==" in req and "scipy==" in req and "joblib==" in req


REAL = M.OUT_DIR / "models.json"


@pytest.mark.skipif(not REAL.exists(), reason="尚未在真實資料上執行 models.run()")
def test_real_run_outputs_are_consistent():
    info = json.loads(REAL.read_text(encoding="utf-8"))
    cv = pd.read_csv(M.OUT_DIR / "cv_results.csv")
    assert len(cv) == len(M.MODEL_NAMES) * len(M.FEATURE_SET_NAMES) * len(M.IMBALANCE)
    assert info["config"]["dev_rows"] == 8000
    pt = pd.read_csv(M.OUT_DIR / "predictions_test.csv")
    assert len(pt) == 2000 * len(M.MODEL_NAMES) * len(M.FEATURE_SET_NAMES)


def test_bootstrap_ci_bounds_and_perfect_separation():
    y = np.array([0] * 40 + [1] * 10)
    perfect = M.bootstrap_ci(y, y.astype(float), "pr_auc", n_boot=200)
    assert perfect["point"] == 1.0 and perfect["low"] == 1.0 and perfect["high"] == 1.0
    rng = np.random.default_rng(0)
    noisy = M.bootstrap_ci(y, rng.random(50), "pr_auc", n_boot=200)
    assert 0 <= noisy["low"] <= noisy["point"] <= noisy["high"] <= 1
    p = rng.random(50)
    assert M.bootstrap_ci(y, p, n_boot=50, seed=1) == M.bootstrap_ci(y, p, n_boot=50, seed=1)
