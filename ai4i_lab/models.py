"""AI4I 二元故障分類（#8）：模型工廠、不平衡處理、CV 比較、MLP 調參、門檻調整與最終模型。

協定（測試集只在最後評估一次）
- 模型比較：在開發集（分層 train+val）上做重複分層 CV（#7 `repeated_cv`，相同種子 → 相同折）。
- MLP 超參數搜尋：在 train 上訓練、只用 val 評分。
- 門檻：在 val 預測上最大化 F1。
- 最終模型：只在 train 上訓練；輸出 val 與 test 的預測機率供下游使用。
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ai4i_lab.data import ROOT, TARGET
from ai4i_lab.evaluation import SEED

OUT_DIR = ROOT / "outputs" / "ai4i" / "models"
MODEL_NAMES = ("logreg", "random_forest", "hist_gb", "mlp")
IMBALANCE = ("none", "class_weight", "smote")
FEATURE_SET_NAMES = ("raw", "engineered")
MLP_DEFAULTS = {"hidden_layer_sizes": (64, 32), "alpha": 1e-3, "learning_rate_init": 1e-3,
                "max_iter": 300, "early_stopping": True, "validation_fraction": 0.1,
                "n_iter_no_change": 10}


# ---------- 模型工廠與不平衡處理 ----------

def make_estimator(model: str, imbalance: str = "none", seed: int = SEED, params: dict | None = None):
    """回傳一個全新、未訓練的估計器（sklearn 估計器或 imblearn Pipeline）。

    imbalance
    - none：不處理
    - class_weight：類別權重 "balanced"；MLPClassifier 不支援樣本權重，改以
      RandomOverSampler（隨機複製少數類到 1:1，期望上等同整數權重）實作
    - smote：SMOTE 合成少數類（放在 imblearn Pipeline 內，只在 fit 時對訓練資料作用，
      交叉驗證時驗證折絕不經過它）
    線性模型與 MLP 先標準化再重抽樣（SMOTE 的 kNN 距離才有意義）。
    """
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    from imblearn.pipeline import Pipeline
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    if model not in MODEL_NAMES:
        raise ValueError(f"未知的模型：{model!r}；可用 {MODEL_NAMES}")
    if imbalance not in IMBALANCE:
        raise ValueError(f"未知的 imbalance：{imbalance!r}；可用 {IMBALANCE}")
    params = dict(params or {})
    cw = "balanced" if imbalance == "class_weight" else None
    steps = []
    if model in ("logreg", "mlp"):
        steps.append(("scale", StandardScaler()))
    if imbalance == "smote":
        steps.append(("smote", SMOTE(random_state=seed)))
    if model == "logreg":
        clf = LogisticRegression(max_iter=2000, class_weight=cw, **params)
    elif model == "random_forest":
        clf = RandomForestClassifier(**({"n_estimators": 200, "min_samples_leaf": 2, "n_jobs": 2} | params),
                                     class_weight=cw, random_state=seed)
    elif model == "hist_gb":
        clf = HistGradientBoostingClassifier(**params, class_weight=cw, random_state=seed)
    else:
        if imbalance == "class_weight":
            steps.append(("oversample", RandomOverSampler(random_state=seed)))
        clf = MLPClassifier(**(MLP_DEFAULTS | params), random_state=seed)
    if not steps:
        return clf
    return Pipeline(steps + [("clf", clf)])


# ---------- 門檻調整 ----------

def tune_threshold(y, proba, default: float = 0.5) -> dict:
    """在（驗證集）預測上找出最大化 F1 的門檻（proba ≥ 門檻判為故障）。

    候選門檻為所有不重複的機率值；F1 相同時取較高門檻（較少誤報）。
    沒有正類時無從調整，回傳 default。curve 為每個候選門檻的 precision／recall／F1。
    """
    y = np.asarray(y).ravel().astype(int)
    proba = np.asarray(proba, dtype=float).ravel()
    cand = np.unique(proba)[::-1]                              # 由高到低
    pos = int(y.sum())
    # 依機率由高到低排序後累加，O(n log n) 算出每個候選門檻的 TP／FP
    order = np.argsort(-proba, kind="stable")
    ps, ys = proba[order], y[order]
    tp_cum, fp_cum = np.cumsum(ys), np.cumsum(1 - ys)
    last = np.searchsorted(-ps, -cand, side="right") - 1       # 每個門檻最後一個 ≥ 門檻的位置
    tp, fp = tp_cum[last], fp_cum[last]
    precision = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0.0)
    recall = tp / pos if pos else np.zeros_like(tp, dtype=float)
    f1 = np.where(precision + recall > 0, 2 * precision * recall / np.maximum(precision + recall, 1e-300), 0.0)
    curve = {"threshold": cand.tolist(), "precision": precision.tolist(), "recall": recall.tolist(),
             "f1": f1.tolist()}
    if pos == 0:
        return {"threshold": float(default), "f1": 0.0, "precision": 0.0, "recall": 0.0, "curve": curve}
    best = int(np.argmax(f1))                                  # argmax 取第一個 → 較高門檻
    return {"threshold": float(cand[best]), "f1": float(f1[best]), "precision": float(precision[best]),
            "recall": float(recall[best]), "curve": curve}


# ---------- 交叉驗證比較 ----------

def cv_grid(dev_df, models=MODEL_NAMES, feature_sets=FEATURE_SET_NAMES, imbalance=IMBALANCE,
            n_splits: int = 5, n_repeats: int = 2, seed: int = SEED, mlp_params: dict | None = None,
            log=None) -> list:
    """模型 × 特徵組 × 不平衡處理；每格都用 #7 `repeated_cv` 的同一組折（相同種子、相同 y）。

    dev_df 必須只含開發集（分層 train+val）；重抽樣在 Pipeline 內，只作用於訓練折。
    """
    from ai4i_lab.evaluation import repeated_cv
    from ai4i_lab.features import build_features
    y = dev_df[TARGET].to_numpy().astype(int)
    rows = []
    for fs in feature_sets:
        X = build_features(dev_df, fs)
        for model in models:
            params = mlp_params if model == "mlp" else None
            for im in imbalance:
                t0 = time.perf_counter()
                res = repeated_cv(lambda m=model, i=im: make_estimator(m, i, seed, params), X, y,
                                  n_splits=n_splits, n_repeats=n_repeats, seed=seed)
                rows.append({"model": model, "feature_set": fs, "imbalance": im, "n_splits": n_splits,
                             "n_repeats": n_repeats, "summary": res["summary"], "folds": res["folds"],
                             "seconds": round(time.perf_counter() - t0, 2)})
                if log:
                    log(f"cv {model:14s} {fs:10s} {im:12s} PR-AUC {res['summary']['pr_auc']['mean']:.3f} "
                        f"({rows[-1]['seconds']}s)")
    return rows


def cv_table(rows: list) -> pd.DataFrame:
    """CV 結果攤平成表（每個指標的 mean／std／Nadeau–Bengio 95% CI low／high）。"""
    from ai4i_lab.evaluation import METRICS
    out = []
    for r in rows:
        rec = {k: r[k] for k in ("model", "feature_set", "imbalance")}
        for m in METRICS:
            s = r["summary"][m]
            rec |= {f"{m}_mean": s["mean"], f"{m}_std": s["std"], f"{m}_low": s["low"], f"{m}_high": s["high"]}
        rec |= {"n_folds": len(r["folds"]), "seconds": r.get("seconds")}
        out.append(rec)
    return pd.DataFrame(out)


def feature_set_differences(rows: list, metric: str = "pr_auc") -> list:
    """研究問題 B：同一模型 × 不平衡處理下，engineered − raw 的逐折配對差異與校正 95% CI。

    兩個特徵組用的是同一組 CV 折（同種子、同 y），所以可依 (repeat, fold) 配對；
    配對差異的 CI 同樣用 Nadeau–Bengio 校正（n_test/n_train = 1/(k−1)），端點截在 [−1, 1]。
    """
    from ai4i_lab.evaluation import confidence_interval
    by = {(r["model"], r["imbalance"], r["feature_set"]): r for r in rows}
    out = []
    for (model, im, fs), raw in by.items():
        if fs != "raw" or (model, im, "engineered") not in by:
            continue
        eng = by[(model, im, "engineered")]
        rv = {(f["repeat"], f["fold"]): f[metric] for f in raw["folds"]}
        ev = {(f["repeat"], f["fold"]): f[metric] for f in eng["folds"]}
        keys = sorted(rv.keys() & ev.keys())
        diffs = np.array([ev[k] - rv[k] for k in keys], dtype=float)
        k = raw.get("n_splits", 5)
        c = confidence_interval(diffs, test_train_ratio=1 / (k - 1), bounds=(-1.0, 1.0))
        out.append({"model": model, "imbalance": im, "metric": metric, "n_folds": len(keys),
                    "raw_mean": float(np.mean([rv[q] for q in keys])),
                    "engineered_mean": float(np.mean([ev[q] for q in keys])),
                    "diff_mean": c["mean"], "diff_low": c["low"], "diff_high": c["high"],
                    "significant": bool(c["low"] > 0 or c["high"] < 0)})
    return out


# ---------- MLP 超參數搜尋與訓練曲線（只用 train／val） ----------

MLP_GRID = {"hidden_layer_sizes": [(32,), (64, 32), (128, 64)],
            "learning_rate_init": [1e-3, 1e-2],
            "alpha": [1e-4, 1e-3, 1e-2]}
SEARCH_KEYS = ("hidden_layer_sizes", "learning_rate_init", "alpha")


def params_of(result: dict) -> dict:
    """搜尋結果列 → make_estimator 用的 MLP 參數。"""
    return {k: (tuple(result[k]) if k == "hidden_layer_sizes" else result[k]) for k in SEARCH_KEYS}


def mlp_search(X_train, y_train, X_val, y_val, grid: dict | None = None, imbalance: str = "none",
               seed: int = SEED, base: dict | None = None, log=None) -> list:
    """網格搜尋 MLP 結構、學習率、L2 正則化：在 train 上訓練、在 val 上以 PR-AUC 評分。

    刻意只接受 train／val：函式介面上就拿不到測試集。結果依 val PR-AUC 由高到低排序。
    """
    import itertools

    from ai4i_lab.evaluation import binary_metrics
    grid = MLP_GRID if grid is None else grid
    y_train = np.asarray(y_train).ravel().astype(int)
    y_val = np.asarray(y_val).ravel().astype(int)
    out = []
    for combo in itertools.product(*(grid[k] for k in SEARCH_KEYS)):
        p = dict(zip(SEARCH_KEYS, combo))
        t0 = time.perf_counter()
        est = make_estimator("mlp", imbalance, seed, (base or {}) | p).fit(X_train, y_train)
        m = binary_metrics(y_val, est.predict_proba(X_val)[:, 1])
        out.append({"hidden_layer_sizes": list(p["hidden_layer_sizes"]), "learning_rate_init": p["learning_rate_init"],
                    "alpha": p["alpha"], "imbalance": imbalance, "val_pr_auc": m["pr_auc"],
                    "val_roc_auc": m["roc_auc"], "val_f1_at_0.5": m["f1"],
                    "n_iter": int(est.steps[-1][1].n_iter_), "seconds": round(time.perf_counter() - t0, 2)})
        if log:
            log(f"mlp {p} → val PR-AUC {m['pr_auc']:.3f}")
    return sorted(out, key=lambda r: -r["val_pr_auc"])


def mlp_training_curve(X_train, y_train, X_val, y_val, params: dict | None = None, imbalance: str = "none",
                       epochs: int = 100, seed: int = SEED) -> dict:
    """逐 epoch（partial_fit）記錄 train／val 的 log-loss 與 val PR-AUC。

    與 make_estimator 相同的前處理：標準化（只在 train 上 fit）→ 重抽樣（只作用於 train）→ MLP。
    損失以原始（未重抽樣）分布計算，不含 L2 懲罰項，train 與 val 可直接比較。
    """
    from sklearn.metrics import average_precision_score, log_loss
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    y_train = np.asarray(y_train).ravel().astype(int)
    y_val = np.asarray(y_val).ravel().astype(int)
    pipe = make_estimator("mlp", imbalance, seed, params)
    scaler = StandardScaler().fit(X_train)
    Xtr, Xva = scaler.transform(X_train), scaler.transform(X_val)
    Xfit, yfit = Xtr, y_train
    for name, step in pipe.steps[1:-1]:                        # smote／oversample
        Xfit, yfit = step.fit_resample(Xfit, yfit)
    cfg = pipe.steps[-1][1].get_params() | {"early_stopping": False, "max_iter": 1}
    mlp = MLPClassifier(**cfg)
    curve = {"epoch": [], "train_loss": [], "val_loss": [], "val_pr_auc": []}
    for ep in range(1, epochs + 1):
        mlp.partial_fit(Xfit, yfit, classes=np.array([0, 1]))
        ptr, pva = mlp.predict_proba(Xtr)[:, 1], mlp.predict_proba(Xva)[:, 1]
        curve["epoch"].append(ep)
        curve["train_loss"].append(float(log_loss(y_train, ptr, labels=[0, 1])))
        curve["val_loss"].append(float(log_loss(y_val, pva, labels=[0, 1])))
        curve["val_pr_auc"].append(float(average_precision_score(y_val, pva)) if y_val.any() else float("nan"))
    curve["best_epoch"] = int(np.argmin(curve["val_loss"])) + 1
    curve["params"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in (params or {}).items()}
    curve["imbalance"] = imbalance
    return curve


# ---------- 最終模型 ----------

def _key(model, fs):
    return f"{model}|{fs}"


def final_path(model: str, feature_set: str, out_dir=OUT_DIR) -> Path:
    return Path(out_dir) / "final" / f"{model}__{feature_set}.joblib"


def final_config(model: str, feature_set: str, info: dict | None) -> dict:
    """最終模型設定：models.json 中依 CV 選出的不平衡處理（MLP 另加 val 搜尋的最佳參數）；
    沒有 models.json 時用預設（不處理、MLP_DEFAULTS）。"""
    k = _key(model, feature_set)
    if info and k in info.get("chosen", {}):
        c = info["chosen"][k]
        params = c.get("params")
        if params and "hidden_layer_sizes" in params:
            params = params | {"hidden_layer_sizes": tuple(params["hidden_layer_sizes"])}
        return {"imbalance": c["imbalance"], "params": params}
    return {"imbalance": "none", "params": None}


def train_final(model: str, feature_set: str, train_df, config: dict, seed: int = SEED):
    from ai4i_lab.features import build_features
    est = make_estimator(model, config["imbalance"], seed, config["params"])
    return est.fit(build_features(train_df, feature_set), train_df[TARGET].to_numpy().astype(int))


def get_final_model(model_name: str, feature_set: str, out_dir=OUT_DIR, df=None, splits=None, seed: int = SEED):
    """取得最終模型：有快取（final/<model>__<feature_set>.joblib）就載入；
    沒有就依 models.json 的設定、用固定種子在分層 train 上重新訓練並寫入快取。"""
    import joblib

    from ai4i_lab.data import load_dataset
    from ai4i_lab.evaluation import split_masks
    path = final_path(model_name, feature_set, out_dir)
    if path.exists():
        return joblib.load(path)
    info_path = Path(out_dir) / "models.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else None
    df = load_dataset() if df is None else df
    train = df[split_masks(df, splits, "stratified")["train"]]
    est = train_final(model_name, feature_set, train, final_config(model_name, feature_set, info), seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(est, path)
    return est


def bootstrap_ci(y, proba, metric: str = "pr_auc", threshold: float = 0.5, n_boot: int = 1000,
                 seed: int = SEED, confidence: float = 0.95) -> dict:
    """測試集指標的百分位 bootstrap 信賴區間（重抽樣列，不重新訓練）。"""
    from ai4i_lab.evaluation import binary_metrics
    y = np.asarray(y).ravel().astype(int)
    proba = np.asarray(proba, dtype=float).ravel()
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        vals.append(binary_metrics(y[i], proba[i], threshold)[metric])
    a = (1 - confidence) / 2
    return {"point": binary_metrics(y, proba, threshold)[metric],
            "low": float(np.quantile(vals, a)), "high": float(np.quantile(vals, 1 - a)), "n_boot": len(vals)}


# ---------- 圖表 ----------

FIGURES = ["cv_comparison.png", "feature_set_diff.png", "mlp_search.png", "mlp_training_curve.png",
           "threshold_tuning.png", "test_metrics.png"]
METRIC_LABEL = {"pr_auc": "PR-AUC", "roc_auc": "ROC-AUC", "f1": "F1", "recall": "Recall", "precision": "Precision"}


def plot_cv_comparison(table: pd.DataFrame, path, metrics=("pr_auc", "f1", "recall", "precision")) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from ai4i_lab.style import SERIES, save
    cfgs = list(dict.fromkeys(f"{m} | {f}" for m, f in zip(table["model"], table["feature_set"])))
    ims = list(dict.fromkeys(table["imbalance"]))
    colors = {im: SERIES[i] for i, im in enumerate(IMBALANCE)}
    offs = {im: (i - (len(ims) - 1) / 2) * 0.22 for i, im in enumerate(ims)}
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.1 * len(metrics) + 2, 0.55 * len(cfgs) + 2),
                             sharey=True)
    for ax, m in zip(axes, metrics):
        for _, r in table.iterrows():
            yv = cfgs.index(f"{r['model']} | {r['feature_set']}") + offs[r["imbalance"]]
            mean = r[f"{m}_mean"]
            ax.errorbar(mean, yv, xerr=[[mean - r[f"{m}_low"]], [r[f"{m}_high"] - mean]], fmt="o",
                        color=colors[r["imbalance"]], markersize=5, elinewidth=2, capsize=0)
        ax.set_title(METRIC_LABEL[m])
        ax.set_xlim(0, 1.02)
        ax.set_yticks(range(len(cfgs)))
        ax.set_yticklabels(cfgs)
        ax.grid(axis="y", visible=False)
    axes[0].invert_yaxis()
    handles = [Line2D([], [], color=colors[im], marker="o", label=im) for im in ims]
    fig.legend(handles=handles, title="Imbalance handling", loc="lower center", ncol=len(ims),
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Model x feature set x imbalance: repeated stratified CV on dev set (mean, 95% corrected CI; "
                 "threshold 0.5)", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, path)


def plot_feature_set_diff(diffs: list, path, metrics=("pr_auc", "f1")) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from ai4i_lab.style import SERIES, TEXT_SECONDARY, save
    colors = {im: SERIES[i] for i, im in enumerate(IMBALANCE)}
    models = list(dict.fromkeys(d["model"] for d in diffs))
    ims = list(dict.fromkeys(d["imbalance"] for d in diffs))
    offs = {im: (i - (len(ims) - 1) / 2) * 0.22 for i, im in enumerate(ims)}
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.2 * len(metrics), 0.75 * len(models) + 2.2), sharey=True)
    for ax, m in zip(np.atleast_1d(axes), metrics):
        for d in (d for d in diffs if d["metric"] == m):
            yv = models.index(d["model"]) + offs[d["imbalance"]]
            ax.errorbar(d["diff_mean"], yv, xerr=[[d["diff_mean"] - d["diff_low"]], [d["diff_high"] - d["diff_mean"]]],
                        fmt="o", color=colors[d["imbalance"]], markersize=6, elinewidth=2, capsize=0)
        ax.axvline(0, color=TEXT_SECONDARY, linestyle="--", linewidth=1)
        ax.set_title(f"{METRIC_LABEL[m]}: engineered - raw")
        ax.set_xlabel("Paired fold difference (95% corrected CI)")
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models)
        ax.grid(axis="y", visible=False)
    np.atleast_1d(axes)[0].invert_yaxis()
    handles = [Line2D([], [], color=colors[im], marker="o", label=im) for im in ims]
    fig.legend(handles=handles, title="Imbalance handling", loc="lower center", ncol=len(ims),
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Research question B: does adding physics-derived features help? (> 0 = engineered better)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, path)


def plot_mlp_search(search: pd.DataFrame, path) -> None:
    import matplotlib.pyplot as plt

    from ai4i_lab.style import SERIES, TEXT_PRIMARY, save
    fss = list(dict.fromkeys(search["feature_set"]))
    s = search.assign(h=search["hidden_layer_sizes"].astype(str))
    cats = list(dict.fromkeys((h, a) for h, a in sorted(zip(s["h"], s["alpha"]),
                                                        key=lambda t: (len(t[0]), t[0], t[1]))))
    lrs = sorted(s["learning_rate_init"].unique())
    fig, axes = plt.subplots(1, len(fss), figsize=(6.5 * len(fss), 4.8), sharey=True, squeeze=False)
    for ax, fs in zip(axes[0], fss):
        sub = s[s["feature_set"] == fs]
        for i, lr in enumerate(lrs):
            g = sub[sub["learning_rate_init"] == lr]
            xs = [cats.index((h, a)) for h, a in zip(g["h"], g["alpha"])]
            ax.plot(xs, g["val_pr_auc"], "o", color=SERIES[i], markersize=7, label=f"learning rate {lr:g}")
        best = sub.loc[sub["val_pr_auc"].idxmax()]
        bx = cats.index((best["h"], best["alpha"]))
        ax.annotate(f"best {best['val_pr_auc']:.3f}", (bx, best["val_pr_auc"]), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=8.5, color=TEXT_PRIMARY)
        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels([f"{h.replace(' ', '')}\nL2 {a:g}" for h, a in cats], fontsize=7.5)
        ax.set_title(f"Feature set: {fs}")
        ax.set_xlabel("Hidden layers / L2 penalty (alpha)")
        ax.grid(axis="x", visible=False)
    axes[0][0].set_ylabel("Validation PR-AUC")
    lo = max(0.0, s["val_pr_auc"].min() - 0.05)
    axes[0][0].set_ylim(lo, min(1.04, s["val_pr_auc"].max() + 0.06))
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(lrs), bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("MLP hyper-parameter search: fit on train, scored on validation only", fontsize=12,
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, path)


def plot_mlp_curves(curves: dict, path) -> None:
    import matplotlib.pyplot as plt

    from ai4i_lab.style import SERIES, TEXT_SECONDARY, save
    fss = list(curves)
    fig, axes = plt.subplots(2, len(fss), figsize=(6 * len(fss), 7), sharex=True, squeeze=False)
    for j, fs in enumerate(fss):
        c = curves[fs]
        ax = axes[0][j]
        ax.plot(c["epoch"], c["train_loss"], color=SERIES[0], label="train")
        ax.plot(c["epoch"], c["val_loss"], color=SERIES[1], label="validation")
        ax.set_yscale("log")
        ax.set_title(f"{fs}: log-loss per epoch")
        ax.set_ylabel("Log-loss (log scale)")
        ax.legend()
        ax2 = axes[1][j]
        ax2.plot(c["epoch"], c["val_pr_auc"], color=SERIES[0])
        ax2.set_title(f"{fs}: validation PR-AUC per epoch")
        ax2.set_ylabel("Validation PR-AUC")
        ax2.set_xlabel("Epoch")
        for a in (ax, ax2):
            a.axvline(c["best_epoch"], color=TEXT_SECONDARY, linestyle="--", linewidth=1)
        ax.annotate(f"min val loss @ epoch {c['best_epoch']}", (c["best_epoch"], 1), xycoords=("data", "axes fraction"),
                    xytext=(4, -12), textcoords="offset points", fontsize=8.5, color=TEXT_SECONDARY)
    fig.suptitle("MLP training curves (best searched hyper-parameters, chosen imbalance handling)", fontsize=12,
                 fontweight="bold")
    fig.tight_layout()
    save(fig, path)


def plot_threshold(thresholds: dict, path, model: str = "mlp") -> None:
    import matplotlib.pyplot as plt

    from ai4i_lab.style import SERIES, TEXT_SECONDARY, save
    keys = [k for k in thresholds if k.startswith(model + "|")]
    fig, axes = plt.subplots(1, len(keys), figsize=(6 * len(keys), 4.4), sharey=True, squeeze=False)
    for ax, k in zip(axes[0], keys):
        t = thresholds[k]
        c = t["curve"]
        for i, m in enumerate(("precision", "recall", "f1")):
            ax.plot(c["threshold"], c[m], color=SERIES[i], label=METRIC_LABEL[m])
        ax.axvline(t["threshold"], color=TEXT_SECONDARY, linestyle="--", linewidth=1)
        ax.axvline(0.5, color=TEXT_SECONDARY, linestyle=":", linewidth=1)
        ax.annotate(f"max F1 {t['f1']:.3f}\n@ {t['threshold']:.3f}", (t["threshold"], 0.04),
                    xytext=(5, 0), textcoords="offset points", fontsize=8.5, color=TEXT_SECONDARY)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.03)
        ax.set_title(k.replace("|", " | "))
        ax.set_xlabel("Decision threshold (dashed: tuned, dotted: 0.5)")
    axes[0][0].set_ylabel("Validation score")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Threshold tuning on the validation set (maximise F1)", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, path)


def plot_test_metrics(test_metrics: dict, path) -> None:
    import matplotlib.pyplot as plt

    from ai4i_lab.style import SERIES, TEXT_PRIMARY, save
    models = list(dict.fromkeys(k.split("|")[0] for k in test_metrics))
    fss = list(dict.fromkeys(k.split("|")[1] for k in test_metrics))
    panels = [("pr_auc", "PR-AUC (95% bootstrap CI)"), ("f1_tuned", "F1 at val-tuned threshold")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    w = 0.38
    for ax, (m, title) in zip(axes, panels):
        for j, fs in enumerate(fss):
            xs = np.arange(len(models)) + (j - (len(fss) - 1) / 2) * w
            vals = [test_metrics[_key(md, fs)][m] for md in models]
            if m == "pr_auc":
                ci = [test_metrics[_key(md, fs)]["pr_auc_ci"] for md in models]
                err = [[v - c["low"] for v, c in zip(vals, ci)], [c["high"] - v for v, c in zip(vals, ci)]]
                ax.bar(xs, vals, width=w - 0.04, color=SERIES[j], label=fs, yerr=err,
                       error_kw={"ecolor": TEXT_PRIMARY, "elinewidth": 1.2})
            else:
                ax.bar(xs, vals, width=w - 0.04, color=SERIES[j], label=fs)
            for xv, v in zip(xs, vals):
                ax.text(xv, 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color="white",
                        fontweight="bold")
        ax.set_xticks(np.arange(len(models)))
        ax.set_xticklabels(models)
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Test-set score (evaluated once)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Feature set", loc="lower center", ncol=len(fss), bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Final models on the held-out test set (trained on train only)", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, path)


# ---------- run() ----------

def run(out_dir=OUT_DIR, df=None, splits=None, models=MODEL_NAMES, feature_sets=FEATURE_SET_NAMES,
        imbalance=IMBALANCE, n_splits: int = 5, n_repeats: int = 2, mlp_grid: dict | None = None,
        mlp_base: dict | None = None, curve_epochs: int = 120, n_boot: int = 1000, seed: int = SEED, verbose: bool = False) -> dict:
    """完整流程：開發集 CV 比較 → 依 CV 選不平衡處理 → MLP 在 val 上搜尋 → 只在 train 訓練最終模型
    → val 上調門檻 → 測試集評估一次 → 輸出 CSV／JSON／圖表與 .joblib。"""
    import joblib

    from ai4i_lab.data import load_dataset
    from ai4i_lab.evaluation import binary_metrics, load_splits, split_masks
    from ai4i_lab.features import build_features
    t_start = time.perf_counter()
    log = print if verbose else None
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_dataset() if df is None else df
    splits = load_splits() if splits is None else splits
    masks = split_masks(df, splits, "stratified")
    train, val, test = (df[masks[k]].reset_index(drop=True) for k in ("train", "val", "test"))
    dev = df[masks["train"] | masks["val"]].reset_index(drop=True)          # 測試集不參與 CV
    base = dict(mlp_base or {})

    # 1. 模型 × 特徵組 × 不平衡處理（開發集重複 CV）
    rows = cv_grid(dev, models, feature_sets, imbalance, n_splits, n_repeats, seed, base or None, log)
    table = cv_table(rows)
    table.to_csv(out_dir / "cv_results.csv", index=False)
    pd.DataFrame([{k: r[k] for k in ("model", "feature_set", "imbalance")} | f for r in rows for f in r["folds"]]) \
        .to_csv(out_dir / "cv_folds.csv", index=False)
    diffs = [d for m in ("pr_auc", "f1", "roc_auc") for d in feature_set_differences(rows, m)]
    pd.DataFrame(diffs).to_csv(out_dir / "feature_set_diff.csv", index=False)

    # 2. 每個 模型 × 特徵組 依 CV PR-AUC 選不平衡處理
    chosen = {}
    for model in models:
        for fs in feature_sets:
            sub = table[(table["model"] == model) & (table["feature_set"] == fs)]
            best = sub.loc[sub["pr_auc_mean"].idxmax()]
            chosen[_key(model, fs)] = {"imbalance": best["imbalance"], "cv_pr_auc": float(best["pr_auc_mean"]),
                                       "params": None}

    # 3. MLP 超參數搜尋（train → val）
    y_tr, y_va, y_te = (d[TARGET].to_numpy().astype(int) for d in (train, val, test))
    search_rows, mlp_best, curves = [], {}, {}
    if "mlp" in models:
        for fs in feature_sets:
            Xtr, Xva = build_features(train, fs), build_features(val, fs)
            im = chosen[_key("mlp", fs)]["imbalance"]
            res = mlp_search(Xtr, y_tr, Xva, y_va, grid=mlp_grid, imbalance=im, seed=seed, base=base, log=log)
            search_rows += [{"feature_set": fs} | r for r in res]
            mlp_best[fs] = res[0]
            chosen[_key("mlp", fs)]["params"] = base | {k: (list(v) if isinstance(v, tuple) else v)
                                                        for k, v in params_of(res[0]).items()}
            curves[fs] = mlp_training_curve(Xtr, y_tr, Xva, y_va, params=base | params_of(res[0]), imbalance=im,
                                            epochs=curve_epochs, seed=seed)
        pd.DataFrame(search_rows).to_csv(out_dir / "mlp_search.csv", index=False)

    # 4. 最終模型（只在 train 訓練）、val 調門檻、test 評估一次
    preds = {"val": [], "test": []}
    thresholds, val_metrics, test_metrics = {}, {}, {}
    info_stub = {"chosen": chosen}
    for model in models:
        for fs in feature_sets:
            k = _key(model, fs)
            t0 = time.perf_counter()
            est = train_final(model, fs, train, final_config(model, fs, info_stub), seed)
            path = final_path(model, fs, out_dir)
            path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(est, path)
            p_va = est.predict_proba(build_features(val, fs))[:, 1]
            p_te = est.predict_proba(build_features(test, fs))[:, 1]
            for part, d, y, p in (("val", val, y_va, p_va), ("test", test, y_te, p_te)):
                preds[part].append(pd.DataFrame({"row_id": d["udi"].to_numpy(), "model": model, "feature_set": fs,
                                                 "y_true": y, "proba": p}))
            thr = tune_threshold(y_va, p_va)
            thresholds[k] = thr
            vm05, vmt = binary_metrics(y_va, p_va), binary_metrics(y_va, p_va, thr["threshold"])
            val_metrics[k] = {"pr_auc": vm05["pr_auc"], "roc_auc": vm05["roc_auc"], "f1_05": vm05["f1"],
                              "f1_tuned": vmt["f1"], "recall_tuned": vmt["recall"], "precision_tuned": vmt["precision"]}
            tm05, tmt = binary_metrics(y_te, p_te), binary_metrics(y_te, p_te, thr["threshold"])
            test_metrics[k] = {"pr_auc": tm05["pr_auc"], "roc_auc": tm05["roc_auc"],
                               "pr_auc_ci": bootstrap_ci(y_te, p_te, "pr_auc", n_boot=n_boot, seed=seed),
                               "f1_05": tm05["f1"], "recall_05": tm05["recall"], "precision_05": tm05["precision"],
                               "f1_tuned": tmt["f1"], "recall_tuned": tmt["recall"],
                               "precision_tuned": tmt["precision"],
                               "confusion_tuned": {c: tmt[c] for c in ("tp", "fp", "fn", "tn")},
                               "threshold": thr["threshold"]}
            if model == "mlp":
                test_metrics[k]["n_iter"] = int(est.steps[-1][1].n_iter_)
            if log:
                log(f"final {k}: val PR-AUC {vm05['pr_auc']:.3f} test PR-AUC {tm05['pr_auc']:.3f} "
                    f"({time.perf_counter() - t0:.1f}s)")
    for part in preds:
        pd.concat(preds[part], ignore_index=True).to_csv(out_dir / f"predictions_{part}.csv", index=False)

    # 5. 圖表
    plot_cv_comparison(table, out_dir / FIGURES[0])
    plot_feature_set_diff(diffs, out_dir / FIGURES[1])
    if search_rows:
        plot_mlp_search(pd.DataFrame(search_rows), out_dir / FIGURES[2])
        plot_mlp_curves(curves, out_dir / FIGURES[3])
        plot_threshold(thresholds, out_dir / FIGURES[4])
    plot_test_metrics(test_metrics, out_dir / FIGURES[5])

    best_cv = table.loc[table["pr_auc_mean"].idxmax()]
    best_test = max(test_metrics, key=lambda q: test_metrics[q]["pr_auc"])
    info = {
        "config": {"seed": seed, "n_splits": n_splits, "n_repeats": n_repeats, "models": list(models),
                   "feature_sets": list(feature_sets), "imbalance": list(imbalance),
                   "mlp_defaults": {k: (list(v) if isinstance(v, tuple) else v) for k, v in MLP_DEFAULTS.items()},
                   "mlp_base": base, "mlp_grid": {k: [list(x) if isinstance(x, tuple) else x for x in v]
                                                  for k, v in (mlp_grid or MLP_GRID).items()},
                   "n_train": int(len(train)), "n_val": int(len(val)), "n_test": int(len(test)),
                   "dev_rows": int(len(dev)), "n_boot": n_boot, "ci": "Nadeau-Bengio corrected resampled t, 95%"},
        "protocol": {
            "cv": "開發集（分層 train+val）重複分層 CV；SMOTE／過採樣在 Pipeline 內，只作用於訓練折",
            "imbalance_choice": "每個 模型×特徵組 取 CV PR-AUC 最高的不平衡處理",
            "mlp_search": "在 train 訓練、val 評分（PR-AUC），不使用測試集",
            "threshold": "在 val 預測上最大化 F1，不使用測試集",
            "final_models": "只在 train 訓練（val 用於門檻，必須保持未見）；測試集只評估一次",
            "mlp_class_weight": "MLPClassifier 不支援 sample_weight，以 RandomOverSampler 隨機複製少數類實作"},
        "chosen": chosen,
        "mlp_best": mlp_best,
        "mlp_curves": curves,
        "thresholds": thresholds,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "feature_set_diff": diffs,
        "conclusions": {
            "best_cv": {k: best_cv[k] for k in ("model", "feature_set", "imbalance")}
                       | {"pr_auc": float(best_cv["pr_auc_mean"]), "low": float(best_cv["pr_auc_low"]),
                          "high": float(best_cv["pr_auc_high"])},
            "best_test": {"key": best_test, "pr_auc": test_metrics[best_test]["pr_auc"]},
            "b_significant_pr_auc": [f"{d['model']}|{d['imbalance']}" for d in diffs
                                     if d["metric"] == "pr_auc" and d["significant"]],
        },
    }
    info["seconds"] = round(time.perf_counter() - t_start, 1)
    (out_dir / "models.json").write_text(json.dumps(info, ensure_ascii=False, indent=1, default=float),
                                         encoding="utf-8")
    return {"models_dir": str(out_dir), "seconds": info["seconds"], **info["conclusions"]}


if __name__ == "__main__":
    print(json.dumps(run(verbose=True), ensure_ascii=False, indent=1, default=float))
