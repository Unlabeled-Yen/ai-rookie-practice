"""資料探索與模型訓練/評估的視覺化。"""
from pathlib import Path

import numpy as np
import pandas as pd

from iris_lab.data import CLASSES, FEATURES, TARGET
from iris_lab.style import (BLUE_RAMP, CLASS_COLORS, CLASS_MARKERS, SERIES_1, SERIES_2, SURFACE,
                            TEXT_PRIMARY, TEXT_SECONDARY, plt)


def _save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_feature_distributions(df: pd.DataFrame, path) -> None:
    """四個特徵各一格的小倍數圖：每類別的箱形圖 + 抖動散點。"""
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, feat in zip(axes.flat, FEATURES):
        groups = [df.loc[df[TARGET] == c, feat].to_numpy() for c in CLASSES]
        ax.boxplot(groups, widths=0.5, showfliers=False,
                   medianprops={"color": TEXT_PRIMARY, "linewidth": 2},
                   boxprops={"color": TEXT_SECONDARY}, whiskerprops={"color": TEXT_SECONDARY},
                   capprops={"color": TEXT_SECONDARY})
        for i, (cls, vals) in enumerate(zip(CLASSES, groups), start=1):
            ax.scatter(i + rng.uniform(-0.15, 0.15, len(vals)), vals, s=18, alpha=0.75,
                       color=CLASS_COLORS[cls], marker=CLASS_MARKERS[cls],
                       edgecolors=SURFACE, linewidths=0.5, zorder=3)
        ax.set_xticks([1, 2, 3], CLASSES)
        ax.set_title(f"{feat} (cm)", fontsize=11)
        ax.grid(axis="x", visible=False)
    fig.suptitle("Iris feature distributions by species", fontweight="bold", color=TEXT_PRIMARY)
    fig.tight_layout()
    _save(fig, path)


def plot_training_history(history: dict, path) -> None:
    """損失與準確率分成兩張圖（不共用雙 y 軸）。"""
    epochs = np.arange(1, len(history["loss"]) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))
    ax_loss.plot(epochs, history["loss"], color=SERIES_1)
    ax_loss.set_title("Training loss (cross-entropy)")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylim(bottom=0)

    for key, label, color in (("train_acc", "train", SERIES_1), ("val_acc", "validation", SERIES_2)):
        ax_acc.plot(epochs, history[key], color=color, label=label)
        ax_acc.annotate(f"{label} {history[key][-1]:.1%}", (epochs[-1], history[key][-1]),
                        xytext=(6, -12 if key == "val_acc" else 6), textcoords="offset points",
                        color=TEXT_PRIMARY, fontsize=9)
    ax_acc.set_title("Accuracy")
    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylim(0, 1.05)
    ax_acc.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax_acc.legend(loc="lower right")
    ax_acc.set_xlim(right=epochs[-1] * 1.18)
    fig.tight_layout()
    _save(fig, path)


def plot_confusion_matrix(cm, path) -> None:
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(cm, cmap=BLUE_RAMP, vmin=0)
    threshold = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=13, fontweight="bold",
                    color="white" if cm[i, j] > threshold else TEXT_PRIMARY)
    ax.set_xticks(range(len(CLASSES)), CLASSES)
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.grid(False)
    ax.set_title("Confusion matrix (test set)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save(fig, path)
