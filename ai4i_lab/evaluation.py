"""AI4I 共用評估框架（#7）。

所有建模子單（#8–#12）一律透過本模組取得切分與指標，確保各模型在完全相同的條件下比較。
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score

from ai4i_lab.data import ROOT

SEED = 42
OUT_DIR = ROOT / "outputs" / "ai4i"
SPLITS_PATH = OUT_DIR / "splits.json"
FRAMEWORK_DIR = OUT_DIR / "framework"
METRICS = ["recall", "precision", "f1", "pr_auc", "roc_auc"]


# ---------- 指標 ----------

def binary_metrics(y, proba, threshold: float = 0.5) -> dict:
    """二元分類指標。proba ≥ threshold 判為故障；PR-AUC 以 average precision 計算。

    只有單一類別時 ROC-AUC／PR-AUC 無定義，回傳 NaN；沒有預測為正時 precision 記為 0。
    """
    y = np.asarray(y).ravel()
    proba = np.asarray(proba, dtype=float).ravel()
    if y.shape != proba.shape:
        raise ValueError(f"y 與 proba 長度不同：{y.shape} vs {proba.shape}")
    if not set(np.unique(y)) <= {0, 1}:
        raise ValueError("y 只能含 0/1")
    y = y.astype(int)
    pred = proba >= threshold
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    recall = tp / (tp + fn) if tp + fn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    both = len(np.unique(y)) == 2
    return {
        "recall": recall, "precision": precision, "f1": f1,
        "pr_auc": float(average_precision_score(y, proba)) if both else float("nan"),
        "roc_auc": float(roc_auc_score(y, proba)) if both else float("nan"),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn, "threshold": float(threshold),
    }


# ---------- 信賴區間 ----------

def confidence_interval(values, confidence: float = 0.95, test_train_ratio: float | None = None,
                        bounds: tuple | None = None) -> dict:
    """平均值的 t 分布信賴區間。

    重複交叉驗證的各折分數彼此相關（訓練集大量重疊），樸素 t 區間會過窄。
    給定 test_train_ratio（= n_test / n_train，k 折時為 1/(k−1)）時改用
    Nadeau & Bengio (2003) 的校正變異數：s² · (1/n + n_test/n_train)。
    NaN 會被忽略；有效值只有 1 個時區間為 NaN。
    bounds=(lo, hi)：指標本身有界（如 0–1）時，把區間端點截斷在範圍內（平均值不變）。
    """
    if not 0 < confidence < 1:
        raise ValueError(f"confidence 必須介於 0 與 1：{confidence}")
    v = np.asarray(values, dtype=float).ravel()
    if v.size == 0:
        raise ValueError("values 不可為空")
    v = v[~np.isnan(v)]
    n = int(v.size)
    if n == 0:
        return {"mean": float("nan"), "low": float("nan"), "high": float("nan"), "std": float("nan"),
                "n": 0, "confidence": confidence}
    mean = float(v.mean())
    if n == 1:
        return {"mean": mean, "low": float("nan"), "high": float("nan"), "std": float("nan"),
                "n": 1, "confidence": confidence}
    s2 = float(v.var(ddof=1))
    factor = 1 / n + (test_train_ratio if test_train_ratio is not None else 0.0)
    se = math.sqrt(s2 * factor)
    half = float(stats.t.ppf(0.5 + confidence / 2, df=n - 1)) * se
    low, high = mean - half, mean + half
    if bounds is not None:
        low, high = max(low, bounds[0]), min(high, bounds[1])
    return {"mean": mean, "low": low, "high": high, "std": math.sqrt(s2),
            "n": n, "confidence": confidence}


# ---------- 切分 ----------

PARTS = ("train", "val", "test")


def failure_strata(df, min_count: int = 10) -> np.ndarray:
    """分層用的複合標籤：`<machine_failure>|<故障模式組合>`。

    只依目標分層時，罕見模式（RNF 19 筆、多重模式 24 筆）可能整組落在同一部分；
    以模式組合分層可讓 #9 多標籤與各模式的評估在三個部分都有樣本。
    筆數少於 min_count 的組合併入 `<machine_failure>|rare`。
    """
    from ai4i_lab.data import FAILURE_MODES, TARGET
    modes = df[FAILURE_MODES].to_numpy().astype(bool)
    combo = ["+".join(m for m, on in zip(FAILURE_MODES, row) if on) or "none" for row in modes]
    keys = pd.Series([f"{int(t)}|{c}" for t, c in zip(df[TARGET], combo)], index=df.index)
    counts = keys.map(keys.value_counts())
    rare = df[TARGET].astype(int).astype(str) + "|rare"
    return np.where(counts < min_count, rare, keys).astype(object)


def split_stratified(y, strata=None, test_size: float = 0.2, val_size: float = 0.2, seed: int = SEED) -> dict:
    """分層切分為 train／val／test（回傳位置索引，各自排序）。

    test_size、val_size 皆為占全體的比例；預設 60/20/20。strata 預設為 y。
    """
    from sklearn.model_selection import train_test_split
    y = np.asarray(y).ravel()
    strata = y if strata is None else np.asarray(strata).ravel()
    idx = np.arange(len(y))
    rest, test = train_test_split(idx, test_size=test_size, stratify=strata, random_state=seed)
    train, val = train_test_split(rest, test_size=val_size / (1 - test_size), stratify=strata[rest],
                                  random_state=seed)
    return {"train": np.sort(train), "val": np.sort(val), "test": np.sort(test)}


def split_temporal(order, train_frac: float = 0.6, val_frac: float = 0.2) -> dict:
    """依時間順序（如 UDI）切分：最早 train_frac 為 train、接著 val_frac 為 val、最後為 test。"""
    if not (0 < train_frac < 1 and 0 <= val_frac < 1 and train_frac + val_frac < 1):
        raise ValueError(f"比例不合法：train_frac={train_frac}, val_frac={val_frac}")
    order = np.asarray(order).ravel()
    ranked = np.argsort(order, kind="stable")
    n = len(order)
    a, b = int(round(n * train_frac)), int(round(n * (train_frac + val_frac)))
    return {"train": np.sort(ranked[:a]), "val": np.sort(ranked[a:b]), "test": np.sort(ranked[b:])}


def _part_summary(df, idx) -> dict:
    from ai4i_lab.data import TARGET
    sub = df.iloc[idx]
    return {"udi": sorted(int(u) for u in sub["udi"]), "n": int(len(sub)),
            "n_failures": int(sub[TARGET].sum()), "failure_rate": float(sub[TARGET].mean())}


def make_splits(df, seed: int = SEED) -> dict:
    """產生兩種切分（以 UDI 記錄，與列順序、label_policy 無關）。"""
    strat = split_stratified(df["machine_failure"].to_numpy(), strata=failure_strata(df), seed=seed)
    temp = split_temporal(df["udi"].to_numpy())
    return {
        "seed": seed, "key": "udi", "n_rows": int(len(df)),
        "fractions": {"train": 0.6, "val": 0.2, "test": 0.2},
        "schemes": {
            "stratified": {k: _part_summary(df, strat[k]) for k in PARTS},
            "temporal": {k: _part_summary(df, temp[k]) for k in PARTS},
        },
    }


def save_splits(splits: dict, path=SPLITS_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(splits, ensure_ascii=False, indent=1), encoding="utf-8")


def load_splits(path=SPLITS_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def split_masks(df, splits: dict | None = None, scheme: str = "stratified") -> dict:
    """依 UDI 把 df 的每一列對應到 train／val／test，回傳布林遮罩（numpy）。

    df 可以是子集（如 label_policy="drop_inconsistent"），但不可含切分檔中沒有的 UDI。
    """
    splits = load_splits() if splits is None else splits
    s = splits["schemes"][scheme]
    udi = df["udi"].to_numpy()
    masks = {k: np.isin(udi, s[k]["udi"]) for k in PARTS}
    unknown = ~(masks["train"] | masks["val"] | masks["test"])
    if unknown.any():
        raise ValueError(f"有 {int(unknown.sum())} 列的 UDI 不在切分檔中")
    return masks


# ---------- 重複分層交叉驗證 ----------

def _take(X, idx):
    return X.iloc[idx] if hasattr(X, "iloc") else X[idx]


def summarize_folds(folds: list, n_splits: int) -> dict:
    """各指標的平均、標準差、Nadeau–Bengio 校正 95% CI（主要）與樸素 t 區間（參考）。"""
    out = {}
    for m in METRICS:
        vals = [f[m] for f in folds]
        c = confidence_interval(vals, test_train_ratio=1 / (n_splits - 1), bounds=(0.0, 1.0))
        nv = confidence_interval(vals, bounds=(0.0, 1.0))
        out[m] = {"mean": c["mean"], "std": c["std"], "low": c["low"], "high": c["high"],
                  "naive_low": nv["low"], "naive_high": nv["high"], "n": c["n"]}
    return out


def repeated_cv(make_model, X, y, n_splits: int = 5, n_repeats: int = 3, seed: int = SEED,
                threshold: float = 0.5, make_sampler=None) -> dict:
    """重複分層 k 折交叉驗證。

    make_model：無參數、每次回傳全新模型的工廠（需有 fit／predict_proba；可為 imblearn Pipeline）。
    make_sampler：選用的重抽樣器工廠（需有 fit_resample），只套用在每個訓練折上；
                  驗證折維持原始分布，絕不參與重抽樣或訓練。
    回傳 {"folds": [每折指標], "summary": {指標: 平均與 95% CI}, "config": {...}}。
    """
    from sklearn.model_selection import RepeatedStratifiedKFold
    y = np.asarray(y).ravel().astype(int)
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    folds = []
    for i, (tr, va) in enumerate(cv.split(np.zeros(len(y)), y)):
        X_tr, y_tr = _take(X, tr), y[tr]
        if make_sampler is not None:
            X_tr, y_tr = make_sampler().fit_resample(X_tr, y_tr)
        model = make_model()
        if not hasattr(model, "predict_proba"):
            raise TypeError(f"{type(model).__name__} 沒有 predict_proba")
        model.fit(X_tr, y_tr)
        proba = np.asarray(model.predict_proba(_take(X, va)))[:, 1]
        m = binary_metrics(y[va], proba, threshold)
        folds.append({"repeat": i // n_splits, "fold": i % n_splits, "n_train_fit": int(len(y_tr)), **m})
    return {"folds": folds, "summary": summarize_folds(folds, n_splits),
            "config": {"n_splits": n_splits, "n_repeats": n_repeats, "seed": seed, "threshold": threshold,
                       "ci": "Nadeau-Bengio corrected t, 95%", "oversampling": make_sampler is not None}}


# ---------- 不平衡處理 ----------

class RandomOversampler:
    """隨機複製少數類直到與多數類同數（不產生新樣本）。只應在訓練折內使用：
    透過 repeated_cv(make_sampler=...) 傳入時，框架保證驗證折不經過它。"""

    def __init__(self, seed: int = SEED):
        self.seed = seed

    def fit_resample(self, X, y):
        y_arr = np.asarray(y).ravel()
        classes, counts = np.unique(y_arr, return_counts=True)
        rng = np.random.default_rng(self.seed)
        extra = [rng.choice(np.flatnonzero(y_arr == c), counts.max() - n, replace=True)
                 for c, n in zip(classes, counts) if n < counts.max()]
        idx = np.concatenate([np.arange(len(y_arr)), *extra]).astype(int)
        Xr = X.iloc[idx].reset_index(drop=True) if hasattr(X, "iloc") else np.asarray(X)[idx]
        yr = y.iloc[idx].reset_index(drop=True) if hasattr(y, "iloc") else y_arr[idx]
        return Xr, yr


def fold_seeded_sampler(seed: int = SEED):
    """回傳 make_sampler 工廠：第 i 次呼叫（即第 i 折）產生 RandomOversampler(seed + i)，
    讓各折的過採樣抽樣彼此不同，同時保持可重現。"""
    import itertools
    counter = itertools.count()
    return lambda: RandomOversampler(seed + next(counter))


IMBALANCE = ("none", "class_weight", "oversample")
MODELS = ("logreg", "random_forest")
FEATURE_SET_NAMES = ("raw", "engineered")


def make_model_factory(name: str, imbalance: str = "none", seed: int = SEED):
    """框架示範用基準模型（固定超參數，不調參；正式模型比較由 #8 負責）。"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if imbalance not in IMBALANCE:
        raise ValueError(f"未知的 imbalance：{imbalance!r}")
    cw = "balanced" if imbalance == "class_weight" else None
    if name == "logreg":
        return lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight=cw))
    if name == "random_forest":
        return lambda: RandomForestClassifier(n_estimators=200, min_samples_leaf=2, class_weight=cw,
                                              n_jobs=2, random_state=seed)
    raise ValueError(f"未知的模型：{name!r}")


def evaluate_grid(df, models=MODELS, feature_sets=FEATURE_SET_NAMES, imbalance=IMBALANCE,
                  n_splits: int = 5, n_repeats: int = 3, seed: int = SEED) -> list:
    """模型 × 特徵組 × 不平衡處理，每一格都用相同的重複分層 CV 折（相同種子）評估。"""
    import time

    from ai4i_lab.data import TARGET
    from ai4i_lab.features import build_features
    y = df[TARGET].to_numpy()
    rows = []
    for fs in feature_sets:
        X = build_features(df, fs)
        for model in models:
            for im in imbalance:
                t0 = time.perf_counter()
                res = repeated_cv(make_model_factory(model, im, seed), X, y, n_splits=n_splits,
                                  n_repeats=n_repeats, seed=seed,
                                  make_sampler=fold_seeded_sampler(seed) if im == "oversample" else None)
                rows.append({"model": model, "feature_set": fs, "imbalance": im,
                             "summary": res["summary"], "folds": res["folds"],
                             "seconds": round(time.perf_counter() - t0, 2)})
    return rows


# ---------- 時間切分穩健性 ----------

def temporal_drift(df, splits: dict, bin_size: int = 500) -> dict:
    """兩種切分下各部分的故障率、模式筆數、特徵平均，以及 train vs test 的 KS 統計量。"""
    from ai4i_lab.data import FAILURE_MODES, NUMERIC_FEATURES, TARGET
    parts, ks = {}, {}
    for scheme in splits["schemes"]:
        masks = split_masks(df, splits, scheme)
        parts[scheme] = {}
        for k in PARTS:
            sub = df[masks[k]]
            parts[scheme][k] = {
                "n": int(len(sub)), "n_failures": int(sub[TARGET].sum()), "failure_rate": float(sub[TARGET].mean()),
                "udi_min": int(sub["udi"].min()), "udi_max": int(sub["udi"].max()),
                "mode_counts": {m: int(sub[m].sum()) for m in FAILURE_MODES},
                **{f"{c}_mean": float(sub[c].mean()) for c in NUMERIC_FEATURES},
            }
        tr, te = df[masks["train"]], df[masks["test"]]
        ks[scheme] = {}
        for c in NUMERIC_FEATURES:
            r = stats.ks_2samp(tr[c], te[c])
            ks[scheme][c] = {"statistic": float(r.statistic), "p_value": float(r.pvalue)}
    d = df.sort_values("udi")
    b = ((d["udi"] - 1) // bin_size).to_numpy()
    g = d.groupby(b)
    bins = {"bin_size": bin_size, "udi_start": [int(i * bin_size + 1) for i in g.groups],
            "failure_rate": g[TARGET].mean().round(5).tolist(),
            "air_temp_k_mean": g["air_temp_k"].mean().round(4).tolist(),
            "hdf": g["hdf"].sum().astype(int).tolist()}
    return {"parts": parts, "ks_train_vs_test": ks, "bins": bins}


def holdout_check(df, splits: dict, models=MODELS, feature_sets=FEATURE_SET_NAMES, seed: int = SEED,
                  _spy=None) -> list:
    """穩健性檢驗：同一模型在「分層 train→val」與「時間 train→val、train→test」下的表現。

    分層切分的 test（全子單共用的最終測試集）在任何方案中都不使用：時間切分的
    train／val／test 會先剔除分層 test 的 UDI（約 60%／20%／20% 的分層 test 落在時間
    train／val／test），所以穩健性基準的訓練與評估都只用開發集（分層 train+val）的列。
    每筆結果附 n_train／n_eval，記錄剔除後實際使用的列數。
    """
    from ai4i_lab.data import TARGET
    from ai4i_lab.features import build_features
    targets = {"stratified": ["val"], "temporal": ["val", "test"]}
    dev = ~split_masks(df, splits, "stratified")["test"]
    y = df[TARGET].to_numpy()
    out = []
    for scheme, evals in targets.items():
        masks = {k: v & dev for k, v in split_masks(df, splits, scheme).items()}
        for fs in feature_sets:
            X = build_features(df, fs)
            for model in models:
                tr = masks["train"]
                if _spy:
                    _spy((scheme, set(df.loc[tr, "udi"])))
                est = make_model_factory(model, "none", seed)().fit(X[tr], y[tr])
                for part in evals:
                    m = masks[part]
                    if _spy:
                        _spy((scheme, set(df.loc[m, "udi"])))
                    out.append({"scheme": scheme, "eval_part": part, "model": model, "feature_set": fs,
                                "n_train": int(tr.sum()), "n_eval": int(m.sum()),
                                **binary_metrics(y[m], est.predict_proba(X[m])[:, 1])})
    return out


# ---------- 圖表 ----------

FIGURES = ["temporal_drift.png", "cv_comparison.png", "holdout_robustness.png"]


def plot_temporal_drift(drift: dict, path) -> None:
    import matplotlib.pyplot as plt

    from ai4i_lab.style import FAILURE, NORMAL, SERIES, TEXT_SECONDARY, save
    b = drift["bins"]
    x = np.array(b["udi_start"]) + b["bin_size"] / 2
    w = b["bin_size"] * 0.8
    t = drift["parts"]["temporal"]
    cuts = [t["val"]["udi_min"] - 0.5, t["test"]["udi_min"] - 0.5]
    fig, axes = plt.subplots(3, 1, figsize=(9, 8.5), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1, 1]})
    axes[0].bar(x, np.array(b["failure_rate"]) * 100, width=w, color=FAILURE)
    axes[0].set_ylabel("Failure rate (%)")
    axes[0].set_title("Failure rate per 500 UDI")
    axes[1].bar(x, b["hdf"], width=w, color=SERIES[1])
    axes[1].set_ylabel("HDF count")
    axes[1].set_title("Heat-dissipation failures (HDF) per 500 UDI")
    axes[2].plot(x, b["air_temp_k_mean"], color=NORMAL, marker="o", markersize=5)
    axes[2].set_ylabel("Air temperature (K)")
    axes[2].set_title("Mean air temperature per 500 UDI")
    axes[2].set_xlabel("UDI (time order)")
    for ax in axes:
        for c in cuts:
            ax.axvline(c, color=TEXT_SECONDARY, linestyle="--", linewidth=1)
    ymax = axes[0].get_ylim()[1]
    for (lo, hi), name in zip([(0, cuts[0]), (cuts[0], cuts[1]), (cuts[1], x.max() + w)],
                              ["temporal train", "val", "test"]):
        rate = t[{"temporal train": "train"}.get(name, name)]["failure_rate"] * 100
        axes[0].text((lo + hi) / 2, ymax * 0.97, f"{name}: {rate:.2f}%", ha="center", va="top",
                     color=TEXT_SECONDARY, fontsize=9)
    fig.suptitle("Temporal split: train (UDI 1-6000) / val / test - distribution drift", fontsize=12,
                 fontweight="bold")
    fig.tight_layout()
    save(fig, path)


def _cfg_label(r):
    return f"{r['model']} | {r['feature_set']}"


def plot_cv_comparison(cv_rows: list, path, metrics=("pr_auc", "recall", "precision", "f1")) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from ai4i_lab.style import SERIES, save
    cfgs = list(dict.fromkeys(_cfg_label(r) for r in cv_rows))
    colors = {im: SERIES[i] for i, im in enumerate(IMBALANCE)}
    offs = {im: (i - 1) * 0.22 for i, im in enumerate(IMBALANCE)}
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.2 * len(metrics) + 1.5, 0.75 * len(cfgs) + 1.8),
                             sharey=True)
    for ax, m in zip(axes, metrics):
        for r in cv_rows:
            s = r["summary"][m]
            yv = cfgs.index(_cfg_label(r)) + offs[r["imbalance"]]
            ax.errorbar(s["mean"], yv, xerr=[[s["mean"] - s["low"]], [s["high"] - s["mean"]]], fmt="o",
                        color=colors[r["imbalance"]], markersize=6, elinewidth=2, capsize=0)
        ax.set_title(m.replace("_", "-").upper() if "auc" in m else m.capitalize())
        ax.set_xlim(0, 1.02)
        ax.set_yticks(range(len(cfgs)))
        ax.set_yticklabels(cfgs)
        ax.invert_yaxis() if ax is axes[0] else None
        ax.grid(axis="y", visible=False)
    handles = [Line2D([], [], color=colors[im], marker="o", linestyle="-", label=im) for im in IMBALANCE]
    fig.legend(handles=handles, title="Imbalance handling", loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Repeated stratified CV on the development set (mean, 95% corrected CI)", fontsize=12,
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, path)


def plot_holdout(holdout: list, path, metrics=("pr_auc", "recall")) -> None:
    import matplotlib.pyplot as plt

    from ai4i_lab.style import SERIES, TEXT_PRIMARY, save
    groups = [("stratified", "val", "stratified train -> val"), ("temporal", "val", "temporal train -> val"),
              ("temporal", "test", "temporal train -> test")]
    cfgs = list(dict.fromkeys(_cfg_label(r) for r in holdout))
    fig, axes = plt.subplots(1, len(metrics), figsize=(6.2 * len(metrics), 4.6), sharey=True)
    width = 0.26
    for ax, m in zip(axes, metrics):
        for gi, (scheme, part, label) in enumerate(groups):
            vals = [next(r[m] for r in holdout if r["scheme"] == scheme and r["eval_part"] == part
                         and _cfg_label(r) == c) for c in cfgs]
            xs = np.arange(len(cfgs)) + (gi - 1) * width
            ax.bar(xs, vals, width=width - 0.03, color=SERIES[gi], label=label)
            for xv, v in zip(xs, vals):
                ax.text(xv, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, color=TEXT_PRIMARY)
        ax.set_xticks(np.arange(len(cfgs)))
        ax.set_xticklabels([c.replace(" | ", "\n") for c in cfgs])
        ax.set_ylim(0, 1.1)
        ax.set_title(m.replace("_", "-").upper() if "auc" in m else m.capitalize())
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Score (threshold 0.5 for recall)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Robustness: stratified vs temporal hold-out (no class weighting; shared test rows excluded)",
                 fontsize=12,
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, path)


# ---------- 結論與 run() ----------

def _conclusions(drift: dict, cv_rows: list, holdout: list) -> dict:
    t, s = drift["parts"]["temporal"], drift["parts"]["stratified"]
    ks_t, ks_s = drift["ks_train_vs_test"]["temporal"], drift["ks_train_vs_test"]["stratified"]
    worst = max(ks_t, key=lambda c: ks_t[c]["statistic"])

    def pr(scheme, part):
        vals = [r["pr_auc"] for r in holdout if r["scheme"] == scheme and r["eval_part"] == part]
        return float(np.mean(vals)) if vals else float("nan")
    best = max(cv_rows, key=lambda r: r["summary"]["pr_auc"]["mean"])
    return {
        "failure_rate_by_part": {sch: {k: round(p[k]["failure_rate"], 4) for k in PARTS}
                                 for sch, p in (("stratified", s), ("temporal", t))},
        "max_ks_temporal": {"feature": worst, "statistic": round(ks_t[worst]["statistic"], 4)},
        "max_ks_stratified": round(max(v["statistic"] for v in ks_s.values()), 4),
        "mean_pr_auc_holdout": {"stratified_val": pr("stratified", "val"), "temporal_val": pr("temporal", "val"),
                                "temporal_test": pr("temporal", "test")},
        "holdout_rows": {f"{r['scheme']}_{r['eval_part']}": {"n_train": r["n_train"], "n_eval": r["n_eval"]}
                         for r in holdout},
        "holdout_note": "穩健性檢驗的所有方案都先剔除分層 test（共用最終測試集）的 UDI，訓練與評估只用開發集列。",
        "best_cv_config": {k: best[k] for k in ("model", "feature_set", "imbalance")}
                          | {"pr_auc": best["summary"]["pr_auc"]},
        "why_temporal_split": (
            "#4 發現依 UDI 排序有強烈時間結構（刀具磨耗 98.8% 相鄰非遞減、空氣溫度 lag-1 自相關 0.999），"
            "且故障集中於 UDI 4000–5000 的高溫區段。隨機分層切分會讓相鄰（幾乎相同條件）的列同時出現在訓練與測試，"
            "可能高估泛化能力；依時間切分則模擬『用過去預測未來』，並暴露空氣溫度等特徵的分布漂移。"),
    }


def run(out_dir=OUT_DIR, df=None, models=MODELS, n_splits: int = 5, n_repeats: int = 3, seed: int = SEED) -> dict:
    """產生 splits.json、在開發集（分層 train+val）上跑評估網格、做時間切分穩健性檢驗並繪圖。"""
    import time

    from ai4i_lab.data import load_dataset
    t0 = time.perf_counter()
    out_dir = Path(out_dir)
    fw_dir = out_dir / "framework"
    fw_dir.mkdir(parents=True, exist_ok=True)
    df = load_dataset() if df is None else df
    splits = make_splits(df, seed)
    splits_path = out_dir / "splits.json"
    save_splits(splits, splits_path)

    masks = split_masks(df, splits, "stratified")
    dev = df[masks["train"] | masks["val"]].reset_index(drop=True)      # 測試集不參與 CV
    cv_rows = evaluate_grid(dev, models=models, n_splits=n_splits, n_repeats=n_repeats, seed=seed)
    drift = temporal_drift(df, splits)
    holdout = holdout_check(df, splits, models=models, seed=seed)
    result = {
        "config": {"seed": seed, "n_splits": n_splits, "n_repeats": n_repeats, "models": list(models),
                   "feature_sets": list(FEATURE_SET_NAMES), "imbalance": list(IMBALANCE), "threshold": 0.5,
                   "dev_rows": int(len(dev)), "dev_failures": int(dev["machine_failure"].sum()),
                   "ci": "Nadeau-Bengio corrected resampled t, 95%",
                   "holdout_excludes_shared_test": True},
        "cv": [{k: v for k, v in r.items() if k != "folds"} for r in cv_rows],
        "drift": drift, "holdout": holdout,
        "conclusions": _conclusions(drift, cv_rows, holdout),
    }
    (fw_dir / "cv_folds.json").write_text(json.dumps(
        [{k: r[k] for k in ("model", "feature_set", "imbalance", "folds")} for r in cv_rows], indent=1),
        encoding="utf-8")
    plot_temporal_drift(drift, fw_dir / FIGURES[0])
    plot_cv_comparison(cv_rows, fw_dir / FIGURES[1])
    plot_holdout(holdout, fw_dir / FIGURES[2])
    result["seconds"] = round(time.perf_counter() - t0, 1)
    (fw_dir / "framework.json").write_text(json.dumps(result, ensure_ascii=False, indent=1, default=float),
                                           encoding="utf-8")
    return {"splits_path": str(splits_path), "framework_dir": str(fw_dir), **result["conclusions"],
            "seconds": result["seconds"]}


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=1, default=float))
