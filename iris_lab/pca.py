"""PCA 降維與 3D 特徵分布圖。"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from iris_lab.data import CLASSES, FEATURES, TARGET
from iris_lab.style import CLASS_COLORS, CLASS_MARKERS, SURFACE, plt


def fit_pca(X: np.ndarray, n_components: int = 3):
    """先標準化（各特徵尺度不同）再做 PCA，回傳投影座標與各主成分解釋變異比例。"""
    X_std = StandardScaler().fit_transform(X)
    pca = PCA(n_components=n_components).fit(X_std)
    return pca.transform(X_std), pca.explained_variance_ratio_.tolist()


def plot_pca_3d(df: pd.DataFrame, path) -> list:
    coords, ratio = fit_pca(df[FEATURES].to_numpy(), n_components=3)
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(projection="3d")
    for cls in CLASSES:
        pts = coords[(df[TARGET] == cls).to_numpy()]
        ax.scatter(*pts.T, label=cls, color=CLASS_COLORS[cls], marker=CLASS_MARKERS[cls],
                   s=36, edgecolors=SURFACE, linewidths=0.8, depthshade=False)
    for setter, i in ((ax.set_xlabel, 0), (ax.set_ylabel, 1), (ax.set_zlabel, 2)):
        setter(f"PC{i + 1} ({ratio[i]:.1%})", labelpad=10)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.set_pane_color(SURFACE)
    ax.view_init(elev=20, azim=-60)
    ax.set_box_aspect(None, zoom=0.85)
    ax.set_title(f"Iris PCA 3D projection — {sum(ratio):.1%} variance explained")
    ax.legend(title="species", loc="upper left")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return ratio
