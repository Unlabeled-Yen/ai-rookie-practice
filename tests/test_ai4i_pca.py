"""#6 PCA 測試：標準化 PCA、解釋變異、可重現性、故障結構量化與輸出。"""
import json

import numpy as np
import pandas as pd
import pytest

from ai4i_lab import data as D
from ai4i_lab import pca as P

# sklearn 的 LogisticRegression 在新版 SciPy 觸發的棄用警告，與本單無關
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


# ---------- 第 3 輪：PCA 核心 ----------

def test_fit_pca_dominant_axis():
    rng = np.random.default_rng(0)
    t = rng.normal(size=500)
    # 三欄高度相關 + 一欄獨立雜訊：標準化後 PC1 應約占 3/4 變異
    X = np.c_[t, 2 * t + 0.01 * rng.normal(size=500), -t, rng.normal(size=500)]
    res = P.fit_pca(X)
    evr = res["explained_variance_ratio"]
    assert evr.shape == (4,)
    assert evr.sum() == pytest.approx(1.0)
    assert evr[0] == pytest.approx(0.75, abs=0.02)
    assert res["scores"].shape == (500, 4)


def test_fit_pca_is_scale_invariant():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 3)) @ np.array([[1, 0.5, 0], [0, 1, 0.3], [0, 0, 1]])
    a = P.fit_pca(X)["explained_variance_ratio"]
    X2 = X * np.array([1000.0, 1.0, 0.001])
    b = P.fit_pca(X2)["explained_variance_ratio"]
    np.testing.assert_allclose(a, b, atol=1e-10)


def test_fit_pca_reproducible():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(100, 5))
    np.testing.assert_array_equal(P.fit_pca(X, n_components=3)["scores"],
                                  P.fit_pca(X, n_components=3)["scores"])


def test_n_components_for_threshold():
    evr = np.array([0.5, 0.3, 0.15, 0.05])
    assert P.n_components_for(evr, 0.5) == 1   # 剛好等於門檻
    assert P.n_components_for(evr, 0.8) == 2   # 0.5+0.3=0.8 剛好等於
    assert P.n_components_for(evr, 0.9) == 3
    assert P.n_components_for(evr, 0.99) == 4


def test_oriented_auc():
    y = np.array([0, 0, 1, 1])
    assert P.oriented_auc(y, np.array([1, 2, 3, 4])) == pytest.approx(1.0)
    assert P.oriented_auc(y, np.array([4, 3, 2, 1])) == pytest.approx(1.0)   # 方向無關
    assert P.oriented_auc(y, np.array([1, 2, 1, 2])) == pytest.approx(0.5)


def test_centroid_shift_hand_computed():
    # 正常組在 PC 上的平均 (0,0)、標準差 (1,2)；目標組平均 (3,4) → 位移 (3, 2)，距離 sqrt(13)
    normal = np.array([[-1.0, -2.0], [1.0, 2.0]])
    group = np.array([[3.0, 4.0], [3.0, 4.0]])
    shift, dist = P.centroid_shift(group, normal)
    np.testing.assert_allclose(shift, [3.0, 2.0])
    assert dist == pytest.approx(np.sqrt(13))


# ---------- 第 4 輪：特徵組分析與可分性 ----------

def synthetic_ai4i(n=600, seed=0):
    """仿 AI4I 的小型合成資料（snake_case 欄名），故障由簡化規則產生。"""
    rng = np.random.default_rng(seed)
    air = rng.normal(300, 2, n)
    df = pd.DataFrame({
        "udi": np.arange(1, n + 1),
        "product_id": [f"P{i}" for i in range(n)],
        "type": rng.choice(D.TYPES, n, p=[0.5, 0.3, 0.2]),
        "air_temp_k": air,
        "process_temp_k": air + 10 + rng.normal(0, 1, n),
        "rotational_speed_rpm": rng.normal(1540, 180, n),
        "torque_nm": rng.normal(40, 10, n),
        "tool_wear_min": rng.integers(0, 250, n).astype(float),
    })
    power = df.torque_nm * df.rotational_speed_rpm * 2 * np.pi / 60
    df["twf"] = (df.tool_wear_min > 235).astype(int)
    df["hdf"] = ((df.process_temp_k - air < 8.6) & (df.rotational_speed_rpm < 1380)).astype(int)
    df["pwf"] = ((power < 3500) | (power > 9000)).astype(int)
    df["osf"] = (df.tool_wear_min * df.torque_nm > 11000).astype(int)
    df["rnf"] = (rng.random(n) < 0.01).astype(int)
    df["machine_failure"] = df[D.FAILURE_MODES].max(axis=1)
    return df


@pytest.fixture(scope="module")
def syn():
    return synthetic_ai4i()


@pytest.mark.parametrize("fs,k", [("raw", 8), ("engineered", 11)])
def test_analyze_feature_set_structure(syn, fs, k):
    res = P.analyze_feature_set(syn, fs, cv_repeats=1)
    assert res["feature_set"] == fs
    assert res["n_features"] == k
    assert len(res["explained_variance_ratio"]) == k
    assert res["cumulative_variance"][-1] == pytest.approx(1.0)
    assert res["cumulative_variance_3pc"] == pytest.approx(sum(res["explained_variance_ratio"][:3]), abs=1e-3)
    assert set(res["loadings"]) == {"PC1", "PC2", "PC3"}
    feats = set(res["loadings"]["PC1"])
    assert not feats & (set(D.FAILURE_MODES) | set(D.ID_COLUMNS) | {D.TARGET})
    assert set(res["mode_centroid_shift_3pc"]) == set(D.FAILURE_MODES) | {"any_failure"}
    assert len(res["pc_auc_failure"]) == k
    for key in ("logreg_3pc", "logreg_all_features"):
        assert 0.5 <= res["cv_auc"][key]["mean"] <= 1.0
    assert res["scores"].shape == (len(syn), 3)


def test_analyze_reproducible(syn):
    a = P.analyze_feature_set(syn, "engineered", cv_repeats=1)
    b = P.analyze_feature_set(syn, "engineered", cv_repeats=1)
    assert a["cv_auc"] == b["cv_auc"]
    np.testing.assert_array_equal(a["scores"], b["scores"])


def test_separability_pipeline_fits_scaler_and_pca_inside_folds():
    pipe = P.separability_pipeline(n_components=3)
    assert [name for name, _ in pipe.steps] == ["scale", "pca", "clf"]
    pipe_all = P.separability_pipeline(n_components=None)
    assert [name for name, _ in pipe_all.steps] == ["scale", "clf"]


def test_mode_centroid_skips_empty_mode(syn):
    df = syn.copy()
    df["rnf"] = 0
    res = P.analyze_feature_set(df, "raw", cv_repeats=1)
    assert res["mode_centroid_shift_3pc"]["rnf"] is None


# ---------- 第 5 輪：run() 輸出 ----------

FIGS = ["pca_3d_raw.png", "pca_3d_engineered.png", "pca_3d_modes_raw.png",
        "pca_3d_modes_engineered.png", "explained_variance.png", "loadings.png"]


def test_run_writes_outputs(syn, tmp_path):
    res = P.run(df=syn, out_dir=tmp_path, cv_repeats=1)
    for f in FIGS:
        assert (tmp_path / f).stat().st_size > 10_000, f
    js = json.loads((tmp_path / "pca.json").read_text(encoding="utf-8"))
    assert set(js["feature_sets"]) == {"raw", "engineered"}
    assert "scores" not in js["feature_sets"]["raw"]          # 大陣列不寫入 json
    assert js["n_samples"] == len(syn) and js["seed"] == P.SEED
    cmp_ = js["comparison"]
    assert set(cmp_) >= {"cumulative_variance_3pc", "n_components_90", "cv_auc_logreg_3pc",
                         "cv_auc_logreg_all_features"}
    assert isinstance(js["discussion"], list) and len(js["discussion"]) >= 3
    assert res["figures"] == sorted(FIGS)


def test_run_label_policy_recorded(syn, tmp_path):
    js = P.run(df=syn, out_dir=tmp_path, cv_repeats=1, plots=False)
    assert js["label_policy"] == "as_is"


OUT = P.PCA_DIR / "pca.json"


@pytest.mark.skipif(not OUT.exists(), reason="尚未執行 python -m ai4i_lab.pca")
def test_real_outputs_sane():
    js = json.loads(OUT.read_text(encoding="utf-8"))
    assert js["n_samples"] == 10_000 and js["n_failures"] == 339
    for fs, k in (("raw", 8), ("engineered", 11)):
        r = js["feature_sets"][fs]
        assert r["n_features"] == k
        assert sum(r["explained_variance_ratio"]) == pytest.approx(1.0, abs=1e-3)
    for f in FIGS:
        assert (P.PCA_DIR / f).exists()


# ---------- 第 6 輪：討論文字依實際資料計算（不得寫死 3.39%） ----------

def test_discussion_uses_actual_failure_rate(syn, tmp_path):
    js = P.run(df=syn, out_dir=tmp_path, cv_repeats=1, plots=False)
    rate = syn["machine_failure"].mean() * 100
    text = "".join(js["discussion"])
    assert f"{rate:.2f}%" in text
    assert "3.39%" not in text or abs(rate - 3.39) < 0.005
    assert js["failure_rate"] == pytest.approx(syn["machine_failure"].mean(), abs=1e-4)


def test_main_entrypoint_prints_comparison(syn, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(P, "run", lambda **kw: {"comparison": {"n_components_90": {"raw": 1}}})
    P.main()
    out = capsys.readouterr().out
    assert "n_components_90" in out and "done in" in out


# ---------- 第 7 輪：討論需指出「最能分辨故障的主成分」與其變異占比 ----------

def test_discussion_reports_best_pc_and_its_variance_share(syn, tmp_path):
    js = P.run(df=syn, out_dir=tmp_path, cv_repeats=1, plots=False)
    text = "".join(js["discussion"])
    for fs in ("raw", "engineered"):
        r = js["feature_sets"][fs]
        best = max(r["pc_auc_failure"], key=r["pc_auc_failure"].get)
        k = int(best[2:]) - 1
        assert f"{best}（解釋 {r['explained_variance_ratio'][k] * 100:.1f}% 變異）AUC {r['pc_auc_failure'][best]:.3f}" in text
        assert js["comparison"]["best_single_pc"][fs] == best
