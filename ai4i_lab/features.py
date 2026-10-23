"""AI4I 特徵工程（#6）。

所有建模子單一律透過 `build_features(df, feature_set)` 取得特徵矩陣。
特徵中絕不含故障模式（TWF/HDF/PWF/OSF/RNF）、目標欄 `machine_failure` 與識別欄（UDI、Product ID）。

特徵組
- raw：5 個數值感測特徵 + Type one-hot（type_L/type_M/type_H，三欄全保留；
  線性模型請搭配正則化或自行丟一欄以避免共線）
- engineered：raw + 三個依物理意義推導的衍生特徵
  - temp_diff_k      = process_temp_k − air_temp_k                [K]   （對應 HDF）
  - power_w          = torque_nm × rotational_speed_rpm × 2π/60  [W]   （對應 PWF）
  - overstrain_minnm = tool_wear_min × torque_nm                 [min·Nm]（對應 OSF）
"""
import math

import pandas as pd

from ai4i_lab.data import FAILURE_MODES, ID_COLUMNS, NUMERIC_FEATURES, TARGET, TYPES

TYPE_COLUMNS = [f"type_{t}" for t in TYPES]
DERIVED = ["temp_diff_k", "power_w", "overstrain_minnm"]
RAW = list(NUMERIC_FEATURES) + TYPE_COLUMNS
FEATURE_SETS = {
    "raw": RAW,
    "engineered": RAW + DERIVED,
}
FORBIDDEN = frozenset(FAILURE_MODES) | frozenset(ID_COLUMNS) | {TARGET}
RPM_TO_RAD_S = 2 * math.pi / 60

assert not any(set(cols) & FORBIDDEN for cols in FEATURE_SETS.values())


def derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """三個衍生特徵（只用感測欄位計算）。"""
    torque = df["torque_nm"].astype(float)
    return pd.DataFrame({
        # 刻意保留浮點減法誤差、不四捨五入：#5 發現資料中 27 列名義溫差恰為 8.6 K，
        # 其 HDF 標記由 IEEE-754 誤差決定（8.5999…→1、8.6000…→0），四捨五入會讓這些列無法區分
        "temp_diff_k": df["process_temp_k"].astype(float) - df["air_temp_k"].astype(float),
        "power_w": torque * df["rotational_speed_rpm"].astype(float) * RPM_TO_RAD_S,
        "overstrain_minnm": df["tool_wear_min"].astype(float) * torque,
    }, index=df.index)


def type_one_hot(type_series: pd.Series) -> pd.DataFrame:
    unknown = set(type_series.unique()) - set(TYPES)
    if unknown:
        raise ValueError(f"未知的 Type 類別：{sorted(unknown)}；只接受 {TYPES}")
    return pd.DataFrame({col: (type_series == t).astype(float) for t, col in zip(TYPES, TYPE_COLUMNS)},
                        index=type_series.index)


def build_features(df: pd.DataFrame, feature_set: str = "raw") -> pd.DataFrame:
    """回傳指定特徵組的 float 特徵矩陣（保留 df 的 index，欄序固定為 FEATURE_SETS[feature_set]）。"""
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"未知的 feature_set：{feature_set!r}；可用 {list(FEATURE_SETS)}")
    parts = [df[list(NUMERIC_FEATURES)].astype(float), type_one_hot(df["type"])]
    if feature_set == "engineered":
        parts.append(derived_features(df))
    X = pd.concat(parts, axis=1)[FEATURE_SETS[feature_set]]
    assert not set(X.columns) & FORBIDDEN
    return X
