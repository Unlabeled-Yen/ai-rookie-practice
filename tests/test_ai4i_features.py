"""#6 特徵工程測試：欄位組成、資料洩漏防護、衍生特徵手算驗證。"""
import numpy as np
import pandas as pd
import pytest

from ai4i_lab import data as D
from ai4i_lab import features as F

LEAK = set(D.FAILURE_MODES) | set(D.ID_COLUMNS) | {D.TARGET}


def _df():
    """三筆手算用小資料（snake_case 欄名，與 load_dataset 輸出一致）。"""
    return pd.DataFrame({
        "udi": [1, 2, 3],
        "product_id": ["L1", "M2", "H3"],
        "type": ["L", "M", "H"],
        "air_temp_k": [300.0, 298.0, 302.5],
        "process_temp_k": [310.0, 308.5, 310.0],
        "rotational_speed_rpm": [1500, 1200, 3000],
        "torque_nm": [40.0, 60.0, 10.0],
        "tool_wear_min": [100, 200, 0],
        "machine_failure": [0, 1, 0],
        "twf": [0, 0, 0], "hdf": [0, 1, 0], "pwf": [0, 0, 0], "osf": [0, 0, 0], "rnf": [0, 0, 0],
    })


# ---------- 第 1 輪：欄位組成與洩漏 ----------

def test_feature_sets_are_raw_and_engineered():
    assert set(F.FEATURE_SETS) == {"raw", "engineered"}
    raw, eng = F.FEATURE_SETS["raw"], F.FEATURE_SETS["engineered"]
    assert raw == ["air_temp_k", "process_temp_k", "rotational_speed_rpm", "torque_nm",
                   "tool_wear_min", "type_L", "type_M", "type_H"]
    assert eng == raw + ["temp_diff_k", "power_w", "overstrain_minnm"]


@pytest.mark.parametrize("fs", ["raw", "engineered"])
def test_no_leakage_columns(fs):
    X = F.build_features(_df(), fs)
    assert not (set(X.columns) & LEAK)
    assert list(X.columns) == F.FEATURE_SETS[fs]
    # 即使洩漏欄位改變，特徵矩陣也不變
    df2 = _df()
    df2[D.FAILURE_MODES] = 1
    df2["machine_failure"] = 1
    df2["udi"] = [99, 98, 97]
    df2["product_id"] = ["x", "y", "z"]
    pd.testing.assert_frame_equal(X, F.build_features(df2, fs))


def test_unknown_feature_set_raises():
    with pytest.raises(ValueError):
        F.build_features(_df(), "everything")


def test_works_without_label_columns():
    df = _df().drop(columns=list(LEAK))
    X = F.build_features(df, "engineered")
    assert X.shape == (3, 11)


def test_index_preserved_and_float_dtype():
    df = _df()
    df.index = [10, 20, 30]
    X = F.build_features(df, "engineered")
    assert list(X.index) == [10, 20, 30]
    assert all(np.issubdtype(t, np.floating) for t in X.dtypes)


# ---------- 第 2 輪：衍生特徵手算驗證與 one-hot ----------

def test_temp_diff_keeps_raw_float_subtraction():
    # 與 #5 rules.temp_diff_k 一致：不四捨五入（HDF 標記在 8.6 K 邊界由浮點誤差決定）
    df = _df()
    df.loc[0, ["air_temp_k", "process_temp_k"]] = [300.0, 308.6]
    assert F.build_features(df, "engineered").loc[0, "temp_diff_k"] == 308.6 - 300.0


def test_temp_diff_hand_computed():
    X = F.build_features(_df(), "engineered")
    # 310.0-300.0=10.0；308.5-298.0=10.5；310.0-302.5=7.5
    np.testing.assert_allclose(X["temp_diff_k"], [10.0, 10.5, 7.5])


def test_power_hand_computed():
    X = F.build_features(_df(), "engineered")
    # P = T * rpm * 2π/60：40*1500 → 40*157.0796=6283.185；60*1200 → 60*125.6637=7539.822；10*3000 → 10*314.1593=3141.593
    np.testing.assert_allclose(X["power_w"], [6283.185307, 7539.822369, 3141.592654], rtol=1e-9)


def test_overstrain_hand_computed():
    X = F.build_features(_df(), "engineered")
    # 100*40=4000；200*60=12000；0*10=0
    np.testing.assert_allclose(X["overstrain_minnm"], [4000.0, 12000.0, 0.0])


def test_type_one_hot():
    X = F.build_features(_df(), "raw")
    np.testing.assert_array_equal(X[F.TYPE_COLUMNS].to_numpy(), np.eye(3))
    # 子集只有單一類別時仍有三欄
    X1 = F.build_features(_df().iloc[[1]], "raw")
    assert list(X1[F.TYPE_COLUMNS].iloc[0]) == [0.0, 1.0, 0.0]


def test_unknown_type_raises():
    df = _df()
    df.loc[0, "type"] = "X"
    with pytest.raises(ValueError):
        F.build_features(df, "raw")


def test_raw_columns_identical_inside_engineered():
    df = _df()
    pd.testing.assert_frame_equal(F.build_features(df, "raw"),
                                  F.build_features(df, "engineered")[F.FEATURE_SETS["raw"]])


real = pytest.mark.skipif(not D.DEFAULT_PATH.exists(), reason="需要 data/ai4i2020.csv")


@real
def test_real_data_shapes_and_consistency_with_known_rules():
    df = D.load_dataset()
    X = F.build_features(df, "engineered")
    assert X.shape == (10_000, 11)
    assert not X.isna().any().any()
    assert (X[F.TYPE_COLUMNS].sum(axis=1) == 1).all()
    # 已知規則：所有 HDF 樣本溫差 < 8.6 K 且轉速 < 1380；所有 PWF 樣本功率 <3500 或 >9000
    from ai4i_lab import rules as R
    hdf = df["hdf"] == 1
    assert (X.loc[hdf, "temp_diff_k"] < R.HDF_TEMP_DIFF_MAX_K).all()
    assert (X.loc[hdf, "rotational_speed_rpm"] < R.HDF_RPM_MAX).all()
    pwf = df["pwf"] == 1
    assert ((X.loc[pwf, "power_w"] < R.PWF_POWER_MIN_W) | (X.loc[pwf, "power_w"] > R.PWF_POWER_MAX_W)).all()
    # 與 #5 規則模組的計算一致
    np.testing.assert_array_equal(X["temp_diff_k"], R.temp_diff_k(df))   # 逐位元相同
    np.testing.assert_allclose(X["power_w"], R.power_w(df))
    np.testing.assert_allclose(X["overstrain_minnm"], R.strain_minnm(df))
