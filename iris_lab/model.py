"""以多層感知器（MLP）類神經網路進行 Iris 分類。"""
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from iris_lab.data import CLASSES, FEATURES, TARGET

HIDDEN_LAYERS = (16, 8)
LEARNING_RATE = 0.01


def split_data(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """分層抽樣切分訓練/測試資料，標籤轉成 0..2 的整數。"""
    X = df[FEATURES].to_numpy()
    y = df[TARGET].map(CLASSES.index).to_numpy()
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


def _new_mlp(seed: int, **kwargs) -> MLPClassifier:
    return MLPClassifier(hidden_layer_sizes=HIDDEN_LAYERS, activation="relu", solver="adam",
                         learning_rate_init=LEARNING_RATE, random_state=seed, **kwargs)


def train_mlp(X_train, y_train, epochs: int = 200, val_size: float = 0.2, seed: int = 42):
    """逐 epoch 訓練 MLP，並從訓練資料再切出驗證集記錄學習曲線（測試集完全不參與訓練）。

    回傳 (已訓練的 Pipeline[StandardScaler → MLP], history)。
    """
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=val_size, stratify=y_train, random_state=seed)
    scaler = StandardScaler().fit(X_fit)
    X_fit_s, X_val_s = scaler.transform(X_fit), scaler.transform(X_val)

    mlp = _new_mlp(seed, batch_size=16, shuffle=True)
    classes = np.arange(len(CLASSES))
    history = {"loss": [], "train_acc": [], "val_acc": []}
    for _ in range(epochs):
        mlp.partial_fit(X_fit_s, y_fit, classes=classes)
        history["loss"].append(float(mlp.loss_))
        history["train_acc"].append(float(mlp.score(X_fit_s, y_fit)))
        history["val_acc"].append(float(mlp.score(X_val_s, y_val)))
    return Pipeline([("scaler", scaler), ("mlp", mlp)]), history


def evaluate(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    labels = list(range(len(CLASSES)))
    p, r, f, s = precision_recall_fscore_support(y_test, y_pred, labels=labels, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, labels=labels, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=labels).tolist(),
        "per_class": {
            cls: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
            for i, cls in enumerate(CLASSES)
        },
    }


def cross_validate_mlp(df: pd.DataFrame, folds: int = 5, seed: int = 42) -> list:
    """以分層 K-fold 交叉驗證檢查模型表現的穩定度。"""
    X = df[FEATURES].to_numpy()
    y = df[TARGET].map(CLASSES.index).to_numpy()
    model = make_pipeline(StandardScaler(), _new_mlp(seed, max_iter=1000))
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return cross_val_score(model, X, y, cv=cv).tolist()
