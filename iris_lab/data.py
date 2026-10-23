"""Iris 資料的取得、保存與基本探索。"""
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_iris

FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
TARGET = "species"
CLASSES = ["setosa", "versicolor", "virginica"]


def load_iris_df() -> pd.DataFrame:
    """從 scikit-learn 內建資料集取得 Iris，轉成整齊的 DataFrame。"""
    raw = load_iris()
    df = pd.DataFrame(raw.data, columns=FEATURES)
    df[TARGET] = [CLASSES[i] for i in raw.target]
    return df


def save_dataset(df: pd.DataFrame, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_dataset(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"資料缺少欄位: {sorted(missing)}")
    return df[FEATURES + [TARGET]]


def explore(df: pd.DataFrame) -> dict:
    """基本探索：筆數、缺失值、重複列、類別分布、統計摘要與各類別平均。"""
    return {
        "n_rows": len(df),
        "n_features": len(FEATURES),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "class_counts": {c: int(n) for c, n in df[TARGET].value_counts().items()},
        "describe": df[FEATURES].describe().round(3).to_dict(),
        "class_means": df.groupby(TARGET)[FEATURES].mean().round(3).to_dict(orient="index"),
        "correlation": df[FEATURES].corr().round(3).to_dict(),
    }
