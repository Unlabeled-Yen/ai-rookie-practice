"""AI4I 多標籤分類（#9）：預測「是哪一種故障」（TWF/HDF/PWF/OSF），並以實驗驗證 RNF 不可預測。

設計
- 二元關聯（binary relevance）：每種模式各訓練一個獨立分類器，輸入只用
  `features.build_features(df, feature_set)`（不含任何故障模式、目標欄或識別欄）。
  刻意不用 classifier chain：chain 會把其他模式的標籤當成特徵，等於讓多標籤目標回流到特徵。
- 多標籤目標（`mode_targets`）只在本模組內使用，不寫回任何共用特徵或二元分類資料。
- 切分與指標沿用 #7 `ai4i_lab.evaluation`：
  - 開發集（分層 train+val）上做重複分層 CV（5 折 × 3 次），各模式指標附 Nadeau–Bengio 校正 95% CI
  - 最終測試集（分層 test）只在最後評估一次，指標附 bootstrap 95% CI
- RNF：同一套模型與特徵；與隨機基準比較（PR-AUC 對盛行率、ROC-AUC 對 0.5、
  前 k% 標記的偵測率對 k%），並以標籤置換檢定（permutation test）取得 p 值。
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ai4i_lab import evaluation as E
from ai4i_lab.data import FAILURE_MODES, ROOT

SEED = 42
OUT_DIR = ROOT / "outputs" / "ai4i" / "multilabel"
PREDICTABLE_MODES = ["twf", "hdf", "pwf", "osf"]
POINT_METRICS = ["recall", "precision", "f1", "pr_auc", "roc_auc"]


# ---------- 目標與彙總指標 ----------

def mode_targets(df: pd.DataFrame, modes=FAILURE_MODES) -> pd.DataFrame:
    """多標籤目標矩陣（0/1 int）。只在本模組使用，不可回流到特徵。"""
    return df[list(modes)].astype(int)


def multilabel_summary(Y, P, threshold: float = 0.5) -> dict:
    """多標籤彙總指標：micro／macro Precision、Recall、F1，Hamming loss，subset accuracy。"""
    Y = np.asarray(Y).astype(int)
    pred = (np.asarray(P, dtype=float) >= threshold).astype(int)
    if Y.shape != pred.shape:
        raise ValueError(f"Y 與 P 形狀不同：{Y.shape} vs {pred.shape}")
    tp = ((pred == 1) & (Y == 1)).sum(axis=0)
    fp = ((pred == 1) & (Y == 0)).sum(axis=0)
    fn = ((pred == 0) & (Y == 1)).sum(axis=0)

    def prf(tp, fp, fn):
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        return p, r, (2 * p * r / (p + r) if p + r else 0.0)
    mp, mr, mf = prf(tp.sum(), fp.sum(), fn.sum())
    per = [prf(a, b, c) for a, b, c in zip(tp, fp, fn)]
    return {
        "micro_precision": float(mp), "micro_recall": float(mr), "micro_f1": float(mf),
        "macro_precision": float(np.mean([x[0] for x in per])),
        "macro_recall": float(np.mean([x[1] for x in per])),
        "macro_f1": float(np.mean([x[2] for x in per])),
        "hamming_loss": float((pred != Y).mean()),
        "subset_accuracy": float((pred == Y).all(axis=1).mean()),
    }


def fast_roc_auc(y, s) -> float:
    """ROC-AUC（Mann–Whitney，同分取平均秩），與 sklearn.roc_auc_score 一致；單一類別回傳 NaN。"""
    from scipy.stats import rankdata
    y = np.asarray(y).astype(bool)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def fast_average_precision(y, s) -> float:
    """Average precision（同分視為同一門檻），與 sklearn.average_precision_score 一致；無正例回傳 NaN。"""
    y = np.asarray(y).astype(int)
    n_pos = int(y.sum())
    if n_pos == 0 or n_pos == len(y):
        return float("nan") if n_pos == 0 else 1.0
    order = np.argsort(-np.asarray(s, dtype=float), kind="mergesort")
    ss, yy = np.asarray(s, dtype=float)[order], y[order]
    last = np.r_[np.flatnonzero(np.diff(ss)), len(ss) - 1]       # 每個不同分數的最後一個位置
    tps = np.cumsum(yy)[last]
    precision = tps / (last + 1)
    recall = tps / n_pos
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def _metrics_or_nan(y, proba, threshold):
    """bootstrap 用的快速指標；無定義的情況記為 NaN（無正例 → recall/F1/AUC；無預測正例 → precision）。"""
    pred = proba >= threshold
    pos = y == 1
    tp = int(np.sum(pred & pos))
    fp = int(np.sum(pred & ~pos))
    fn = int(np.sum(~pred & pos))
    nan = float("nan")
    recall = tp / (tp + fn) if tp + fn else nan
    precision = tp / (tp + fp) if tp + fp else nan
    if tp + fn == 0:
        f1 = nan
    else:
        f1 = 2 * tp / (2 * tp + fp + fn)
    return {"recall": recall, "precision": precision, "f1": f1,
            "roc_auc": fast_roc_auc(y, proba), "pr_auc": fast_average_precision(y, proba)}


def bootstrap_metrics(y, proba, threshold: float = 0.5, n_boot: int = 1000, seed: int = SEED,
                      confidence: float = 0.95) -> dict:
    """測試集指標的百分位 bootstrap CI（整列重抽樣）。

    重抽樣中指標無定義者（如沒有正例）記為 NaN 並忽略；每個指標回報 n_valid。
    """
    y = np.asarray(y).ravel().astype(int)
    proba = np.asarray(proba, dtype=float).ravel()
    point = E.binary_metrics(y, proba, threshold)
    rng = np.random.default_rng(seed)
    boots = {m: [] for m in POINT_METRICS}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        mb = _metrics_or_nan(y[idx], proba[idx], threshold)
        for m in POINT_METRICS:
            boots[m].append(mb[m])
    a = (1 - confidence) / 2
    out = {"n_boot": n_boot, "confidence": confidence, "threshold": float(threshold),
           **{k: point[k] for k in ("tp", "fp", "fn", "tn")}}
    for m in POINT_METRICS:
        v = np.asarray(boots[m], dtype=float)
        v = v[~np.isnan(v)]
        lo, hi = (float(np.quantile(v, a)), float(np.quantile(v, 1 - a))) if v.size else (np.nan, np.nan)
        out[m] = {"point": float(point[m]), "low": min(lo, float(point[m])), "high": max(hi, float(point[m])),
                  "n_valid": int(v.size)}
    return out


# ---------- 與隨機基準比較 ----------

def random_detection(y, scores, fractions=(0.01, 0.02, 0.05, 0.10, 0.20)) -> list:
    """依分數由高到低標記前 k% 的列，比較偵測率（recall）與隨機標記的期望值 k%。

    p_value：隨機標記同樣列數時偵測數 ≥ 觀察值的機率（超幾何分布，單尾）。
    同分時以原始順序（stable）決定先後。
    """
    from scipy.stats import hypergeom
    y = np.asarray(y).ravel().astype(int)
    order = np.argsort(-np.asarray(scores, dtype=float), kind="stable")
    n, k_pos = len(y), int(y.sum())
    rows = []
    for f in fractions:
        n_flag = max(1, int(round(f * n)))
        det = int(y[order[:n_flag]].sum())
        recall = det / k_pos if k_pos else float("nan")
        rand = n_flag / n
        rows.append({"fraction": float(f), "n_flagged": n_flag, "detected": det, "n_positive": k_pos,
                     "recall": recall, "random_recall": rand, "lift": recall / rand,
                     "p_value": float(hypergeom.sf(det - 1, n, k_pos, n_flag))})
    return rows


def oof_scores(make_model, X, y, n_splits: int = 5, seed: int = SEED, groups=None, _spy=None) -> np.ndarray:
    """分層 k 折的 out-of-fold 預測機率：每列只由沒看過它的模型預測一次。

    groups：給定時改用 StratifiedGroupKFold，同一組（如連續 UDI 區塊）不會同時出現在訓練與驗證折，
    用來排除「時間上相鄰的列互相洩漏」的效應。
    """
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
    y = np.asarray(y).ravel().astype(int)
    out = np.full(len(y), np.nan)
    if groups is None:
        it = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(np.zeros(len(y)), y)
    else:
        it = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(
            np.zeros(len(y)), y, np.asarray(groups))
    for tr, va in it:
        if _spy:
            _spy(tr, va)
        model = make_model().fit(E._take(X, tr), y[tr])
        out[va] = np.asarray(model.predict_proba(E._take(X, va)))[:, 1]
    return out


def best_f1_threshold(y, scores) -> dict:
    """使 F1 最大的門檻（候選為各個分數值）；沒有正例時回傳 0.5。只應在開發集的 OOF 分數上使用。"""
    from sklearn.metrics import precision_recall_curve
    y = np.asarray(y).ravel().astype(int)
    if y.sum() == 0:
        return {"threshold": 0.5, "f1": 0.0}
    p, r, t = precision_recall_curve(y, np.asarray(scores, dtype=float))
    p, r = p[:-1], r[:-1]
    f1 = np.where(p + r > 0, 2 * p * r / np.where(p + r > 0, p + r, 1), 0.0)
    i = int(np.argmax(f1))
    return {"threshold": float(t[i]), "f1": float(f1[i])}


def scan_cluster(positions, n_total: int, window: int, n_sims: int = 10000, seed: int = SEED) -> dict:
    """掃描統計量：長度 window 的連續區間內最多含幾個事件，以及在「事件位置均勻隨機」下
    出現同等或更強群聚的機率（蒙地卡羅，p = (1 + #模擬 ≥ 觀察) / (1 + n_sims)）。"""
    pos = np.sort(np.asarray(positions, dtype=int))
    k = len(pos)

    def max_window(x):
        x = np.sort(x)
        counts = np.searchsorted(x, x + window, side="left") - np.arange(len(x))
        j = int(np.argmax(counts))
        return int(counts[j]), int(x[j])
    obs, start = max_window(pos)
    rng = np.random.default_rng(seed)
    sims = np.array([max_window(rng.choice(n_total, k, replace=False))[0] for _ in range(n_sims)])
    return {"n_events": k, "n_total": int(n_total), "window": int(window), "max_count": obs,
            "window_start": start, "expected_per_window": k * window / n_total,
            "p_value": float((1 + np.sum(sims >= obs)) / (1 + n_sims)), "n_sims": n_sims}


def _auc_pair(y, s) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score
    return {"roc_auc": float(roc_auc_score(y, s)), "pr_auc": float(average_precision_score(y, s))}


def permutation_test(make_model, X, y, n_perm: int = 30, n_splits: int = 5, seed: int = SEED, groups=None) -> dict:
    """標籤置換檢定：比較真實標籤的 OOF ROC-AUC／PR-AUC 與「打亂標籤後重訓」的虛無分布。

    p 值 = (1 + #{虛無 ≥ 觀察}) / (1 + n_perm)。打亂標籤破壞特徵與標籤的任何關聯，
    所以虛無分布就是「純隨機標籤」下同一流程能得到的分數。
    """
    y = np.asarray(y).ravel().astype(int)
    obs = _auc_pair(y, oof_scores(make_model, X, y, n_splits, seed, groups))
    rng = np.random.default_rng(seed)
    null = {"roc_auc": [], "pr_auc": []}
    for _ in range(n_perm):
        yp = rng.permutation(y)
        a = _auc_pair(yp, oof_scores(make_model, X, yp, n_splits, seed, groups))
        for k in null:
            null[k].append(a[k])
    p = {k: (1 + sum(v >= obs[k] for v in null[k])) / (1 + n_perm) for k in null}
    return {"observed": obs, "null": null, "p_value": p, "n_perm": n_perm, "n_splits": n_splits,
            "blocked": groups is not None,
            "prevalence": float(y.mean()),
            "null_mean": {k: float(np.mean(v)) for k, v in null.items()},
            "null_q95": {k: float(np.quantile(v, 0.95)) for k, v in null.items()}}


# ---------- 各模式的模型評估 ----------

MODEL = "random_forest"          # 沿用 #7 框架的基準 RF（200 棵、min_samples_leaf=2、固定種子）
PERM_MODEL = "forest_100"        # 置換檢定用（觀察值與虛無分布用同一模型，檢定仍有效）
BLOCK_SIZE = 500                 # 時間區塊 CV：連續 500 個 UDI 為一組
FEATURE_SETS = ("engineered", "raw")
PRIMARY_FS = "engineered"


def _model_factory(seed: int = SEED, model: str = MODEL):
    """"random_forest"／"logreg"：#7 框架的基準模型；"forest_100"：同設定但 100 棵，供置換檢定（需重訓
    (n_perm+1)×k 次）控制計算量；"fast_forest"：10 棵、單執行緒的小型 RF，只供單元測試加速。"""
    from sklearn.ensemble import RandomForestClassifier
    if model == "fast_forest":
        return lambda: RandomForestClassifier(n_estimators=10, min_samples_leaf=2, n_jobs=1, random_state=seed)
    if model == "forest_100":
        return lambda: RandomForestClassifier(n_estimators=100, min_samples_leaf=2, n_jobs=2, random_state=seed)
    return E.make_model_factory(model, "none", seed)


def cv_per_mode(dev_df, modes=FAILURE_MODES, feature_set: str = PRIMARY_FS, model: str = MODEL,
                n_splits: int = 5, n_repeats: int = 3, seed: int = SEED) -> dict:
    """在開發集上對每個模式各跑一次 #7 的重複分層 CV（每個模式依自己的標籤分層）。"""
    from ai4i_lab.features import build_features
    X = build_features(dev_df, feature_set)
    Y = mode_targets(dev_df, modes)
    out = {}
    for mode in modes:
        t0 = time.perf_counter()
        r = E.repeated_cv(_model_factory(seed, model), X, Y[mode].to_numpy(), n_splits=n_splits,
                          n_repeats=n_repeats, seed=seed)
        out[mode] = {"summary": r["summary"], "n_positive": int(Y[mode].sum()), "n_rows": int(len(Y)),
                     "prevalence": float(Y[mode].mean()), "seconds": round(time.perf_counter() - t0, 2)}
    return out


def holdout_per_mode(df, masks, modes=FAILURE_MODES, feature_set: str = PRIMARY_FS, model: str = MODEL,
                     n_boot: int = 1000, seed: int = SEED, thresholds=None, _spy=None):
    """在開發集（train+val）上訓練最終模型，於最終測試集評估一次，指標附 bootstrap 95% CI。

    thresholds：{模式: 門檻}（由開發集 OOF 選出）；給定時另以該門檻評估一次，存於 "tuned"。
    回傳 (各模式結果, 測試集預測機率矩陣[n_test × len(modes)])。
    """
    from ai4i_lab.features import build_features
    X = build_features(df, feature_set)
    Y = mode_targets(df, modes)
    dev, test = masks["train"] | masks["val"], masks["test"]
    res, P = {}, np.zeros((int(test.sum()), len(modes)))
    for j, mode in enumerate(modes):
        if _spy:
            _spy(mode, set(df.loc[dev, "udi"]), set(df.loc[test, "udi"]))
        est = _model_factory(seed, model)().fit(X[dev], Y.loc[dev, mode].to_numpy())
        P[:, j] = est.predict_proba(X[test])[:, 1]
        res[mode] = {**bootstrap_metrics(Y.loc[test, mode].to_numpy(), P[:, j], n_boot=n_boot, seed=seed),
                     "n_train": int(dev.sum()), "n_test": int(test.sum()),
                     "n_positive_test": int(Y.loc[test, mode].sum())}
        if thresholds and mode in thresholds:
            res[mode]["tuned"] = bootstrap_metrics(Y.loc[test, mode].to_numpy(), P[:, j], thresholds[mode],
                                                   n_boot=n_boot, seed=seed)
    return res, P


def detection_by_mode(dev_df, modes=FAILURE_MODES, feature_set: str = PRIMARY_FS, n_splits: int = 5,
                      seed: int = SEED, model: str = MODEL, fractions=(0.01, 0.02, 0.05, 0.10, 0.20)) -> dict:
    """各模式在開發集上的 OOF 分數 →
    - 前 k% 偵測率 vs 隨機（RNF 的對照組是 TWF 等可預測模式）
    - 使 F1 最大的門檻（只用開發集 OOF 選，之後原封不動套用到測試集）
    """
    from ai4i_lab.features import build_features
    X = build_features(dev_df, feature_set)
    Y = mode_targets(dev_df, modes)
    out = {}
    for mode in modes:
        y = Y[mode].to_numpy()
        s = oof_scores(_model_factory(seed, model), X, y, n_splits, seed)
        out[mode] = {"auc": _auc_pair(y, s), "detection": random_detection(y, s, fractions),
                     "best_threshold": best_f1_threshold(y, s)}
    return out


def rnf_experiment(dev_df, cv_rnf: dict, detection: dict, feature_set: str = PRIMARY_FS, n_perm: int = 20,
                   n_splits: int = 5, seed: int = SEED, perm_model: str = PERM_MODEL,
                   block_size: int = BLOCK_SIZE, df_all=None, n_sims: int = 10000) -> dict:
    """RNF 不可預測性的實驗。

    1. threshold 0.5 下的 Precision／Recall／F1（重複 CV，附 CI）
    2. 與隨機基準比較：ROC-AUC 對 0.5、PR-AUC 對盛行率、前 k% 偵測率對 k%
    3. 標籤置換檢定（一般分層 CV 與時間區塊 CV 各一次）
    4. RNF 事件在 UDI 上的群聚（掃描統計量；只描述標籤位置，不訓練模型）
    時間區塊 CV 用來區分「特徵真的能預測 RNF」與「空氣溫度這類緩慢漂移的特徵充當時間代理、
    讓 RF 記住時間上相鄰的 RNF」兩種解釋。
    """
    from ai4i_lab.features import build_features
    X = build_features(dev_df, feature_set)
    y = dev_df["rnf"].astype(int).to_numpy()
    groups = ((dev_df["udi"].to_numpy() - 1) // block_size)
    mk = _model_factory(seed, perm_model)
    perm = permutation_test(mk, X, y, n_perm=n_perm, n_splits=n_splits, seed=seed)
    perm_b = permutation_test(mk, X, y, n_perm=n_perm, n_splits=n_splits, seed=seed, groups=groups)
    det_b = random_detection(y, oof_scores(mk, X, y, n_splits, seed, groups),
                             [r["fraction"] for r in detection["detection"]])
    src = dev_df if df_all is None else df_all
    cluster = scan_cluster(src.loc[src["rnf"] == 1, "udi"].to_numpy() - 1, n_total=len(src), window=200,
                           n_sims=n_sims, seed=seed)
    cv = cv_rnf["summary"]
    ci_ok = bool(cv["roc_auc"]["low"] <= 0.5 <= cv["roc_auc"]["high"]) if not np.isnan(cv["roc_auc"]["low"]) else None
    checks = {
        "recall_at_0_5_is_zero": bool(cv["recall"]["mean"] == 0.0),
        "cv_roc_auc_ci_contains_0_5": ci_ok,
        "permutation_stratified_not_significant": bool(perm["p_value"]["roc_auc"] > 0.05
                                                       and perm["p_value"]["pr_auc"] > 0.05),
        "permutation_blocked_not_significant": bool(perm_b["p_value"]["roc_auc"] > 0.05
                                                    and perm_b["p_value"]["pr_auc"] > 0.05),
        "detection_stratified_not_better_than_random": bool(all(r["p_value"] > 0.05 for r in detection["detection"])),
        "detection_blocked_not_better_than_random": bool(all(r["p_value"] > 0.05 for r in det_b)),
    }
    # 主判準：時間區塊 CV 下（排除時間鄰近洩漏）置換檢定不顯著，且 0.5 門檻下完全偵測不到
    consistent = bool(checks["permutation_blocked_not_significant"] and checks["recall_at_0_5_is_zero"]
                      and ci_ok is not False)
    return {
        "n_positive_dev": int(y.sum()), "n_dev": int(len(y)), "block_size": block_size,
        "random_baseline": {"roc_auc": 0.5, "pr_auc": float(y.mean()),
                            "recall_at_k": "k（隨機標記 k% 的列，期望偵測到 k% 的 RNF）"},
        "cv": cv, "permutation": perm, "permutation_blocked": perm_b,
        "oof_auc": detection["auc"], "detection": detection["detection"], "detection_blocked": det_b,
        "udi_cluster": cluster, "checks": checks, "consistent_with_random": consistent,
    }


# ---------- 圖表 ----------

FIGURES = ["mode_metrics_cv.png", "mode_metrics_test.png", "rnf_vs_random.png"]
MODE_LABELS = {m: m.upper() for m in FAILURE_MODES}


def _metric_title(m):
    return m.replace("_", "-").upper() if "auc" in m else m.capitalize()


def plot_cv_metrics(cv: dict, path, metrics=("precision", "recall", "f1", "pr_auc")) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from ai4i_lab.style import SERIES, TEXT_SECONDARY, save
    fss = list(cv)
    modes = list(next(iter(cv.values())))
    colors = {fs: SERIES[i] for i, fs in enumerate(fss)}
    offs = {fs: (i - (len(fss) - 1) / 2) * 0.28 for i, fs in enumerate(fss)}
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.1 * len(metrics) + 1, 4.4), sharey=True)
    for ax, m in zip(axes, metrics):
        for fs in fss:
            for i, mode in enumerate(modes):
                s = cv[fs][mode]["summary"][m]
                lo, hi = s["low"], s["high"]
                xerr = [[s["mean"] - lo], [hi - s["mean"]]] if not np.isnan(lo) else None
                ax.errorbar(s["mean"], i + offs[fs], xerr=xerr, fmt="o", color=colors[fs], markersize=6,
                            elinewidth=2, capsize=0)
        if m == "pr_auc":
            for i, mode in enumerate(modes):
                ax.plot(cv[fss[0]][mode]["prevalence"], i, marker="|", color=TEXT_SECONDARY, markersize=14,
                        markeredgewidth=2, linestyle="none")
        ax.set_title(_metric_title(m))
        ax.set_xlim(-0.02, 1.02)
        ax.set_yticks(range(len(modes)))
        ax.set_yticklabels([MODE_LABELS[x] for x in modes])
        ax.grid(axis="y", visible=False)
    axes[0].invert_yaxis()
    handles = [Line2D([], [], color=colors[fs], marker="o", label=f"{fs} features") for fs in fss]
    if "pr_auc" in metrics:
        handles.append(Line2D([], [], color=TEXT_SECONDARY, marker="|", markersize=12, markeredgewidth=2,
                              linestyle="none", label="random baseline (prevalence)"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Per-mode classifiers, repeated stratified CV on dev set (mean, 95% corrected CI, threshold 0.5)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, path)


def plot_test_metrics(test: dict, path, metrics=("precision", "recall", "f1", "pr_auc")) -> None:
    """測試集各模式指標（點估計 + bootstrap 95% CI）；有 tuned 時並列「門檻 0.5」與「開發集選的門檻」。"""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from ai4i_lab.style import SERIES, save
    modes = list(test)
    variants = [("fixed", "threshold 0.5")]
    if all("tuned" in test[m] for m in modes):
        variants.append(("tuned", "threshold tuned on dev OOF (max F1)"))
    off = {v: (i - (len(variants) - 1) / 2) * 0.3 for i, (v, _) in enumerate(variants)}
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.1 * len(metrics) + 1.6, 4.4), sharey=True)
    for ax, m in zip(axes, metrics):
        for vi, (v, _) in enumerate(variants):
            if m == "pr_auc" and v == "tuned":
                continue            # PR-AUC 與門檻無關
            for i, mode in enumerate(modes):
                s = (test[mode] if v == "fixed" else test[mode]["tuned"])[m]
                ax.errorbar(s["point"], i + (off[v] if m != "pr_auc" else 0),
                            xerr=[[s["point"] - s["low"]], [s["high"] - s["point"]]], fmt="o",
                            color=SERIES[vi], markersize=6, elinewidth=2, capsize=0)
        ax.set_title(_metric_title(m) + (" (threshold-free)" if m == "pr_auc" else ""))
        ax.set_xlim(-0.02, 1.02)
        ax.set_yticks(range(len(modes)))
        ax.set_yticklabels([f"{MODE_LABELS[x]} (n+={test[x]['n_positive_test']})" for x in modes])
        ax.grid(axis="y", visible=False)
    axes[0].invert_yaxis()
    handles = [Line2D([], [], color=SERIES[i], marker="o", label=lab) for i, (_, lab) in enumerate(variants)]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, -0.02))
    n_test = next(iter(test.values()))["n_test"]
    fig.suptitle(f"Final test set (n={n_test}), engineered features: point estimate and 95% bootstrap CI",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, path)


def plot_rnf(rnf: dict, detection: dict, path, df_all=None, control: str = "twf") -> None:
    import matplotlib.pyplot as plt

    from ai4i_lab.style import BLUE_RAMP, FAILURE, NORMAL, SERIES, TEXT_PRIMARY, TEXT_SECONDARY, save
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, key, name in ((axes[0, 0], "permutation", "stratified 5-fold CV"),
                          (axes[0, 1], "permutation_blocked", f"time-blocked CV ({rnf['block_size']}-UDI blocks)")):
        perm = rnf[key]
        ax.hist(perm["null"]["roc_auc"], bins=10, color=BLUE_RAMP(0.45), edgecolor="white", linewidth=2,
                label="shuffled labels (null)")
        ax.axvline(perm["observed"]["roc_auc"], color=FAILURE, linewidth=2,
                   label=f"real RNF labels = {perm['observed']['roc_auc']:.3f}")
        ax.axvline(0.5, color=TEXT_SECONDARY, linewidth=2, linestyle="--", label="random baseline = 0.5")
        ax.set_title(f"RNF out-of-fold ROC-AUC, {name}")
        ax.set_xlabel(f"ROC-AUC  (permutation p = {perm['p_value']['roc_auc']:.3f}, {perm['n_perm']} permutations;"
                      f" PR-AUC {perm['observed']['pr_auc']:.4f} vs null mean {perm['null_mean']['pr_auc']:.4f})",
                      fontsize=8.5)
        ax.set_ylabel("Count")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(axis="x", visible=False)
    ax = axes[1, 0]
    fr = [0] + [r["fraction"] * 100 for r in rnf["detection"]]
    ax.plot(fr, [0] + [r["random_recall"] * 100 for r in rnf["detection"]], color=TEXT_SECONDARY,
            linestyle="--", label="random flagging")
    ax.plot(fr, [0] + [r["recall"] * 100 for r in detection[control]["detection"]], color=NORMAL, marker="o",
            markersize=6, label=f"{control.upper()} (positive control, stratified CV)")
    ax.plot(fr, [0] + [r["recall"] * 100 for r in rnf["detection"]], color=FAILURE, marker="o", markersize=6,
            label="RNF, stratified CV")
    ax.plot(fr, [0] + [r["recall"] * 100 for r in rnf["detection_blocked"]], color=SERIES[2], marker="o",
            markersize=6, label="RNF, time-blocked CV")
    ax.set_xlabel("Rows flagged, highest score first (%)")
    ax.set_ylabel("Failures of that mode detected (%)")
    ax.set_title(f"Detection rate vs random (dev set, n+ RNF = {rnf['n_positive_dev']})")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper left", fontsize=8)
    ax = axes[1, 1]
    c = rnf["udi_cluster"]
    if df_all is not None:
        d = df_all.sort_values("udi")
        ax.plot(d["udi"], d["air_temp_k"], color=NORMAL, linewidth=1, label="air temperature (K)")
        r = d[d["rnf"] == 1]
        ax.plot(r["udi"], r["air_temp_k"], linestyle="none", marker="o", markersize=8, color=FAILURE,
                markeredgecolor="white", markeredgewidth=1.5, label="RNF event")
        ax.legend(loc="lower right", fontsize=8)
    ax.axvspan(c["window_start"] + 1, c["window_start"] + c["window"], color=GREY_SPAN, alpha=0.6, linewidth=0)
    ax.text(c["window_start"] + c["window"] + 150, ax.get_ylim()[1] - 0.3,
            f"{c['max_count']} of {c['n_events']} RNF within {c['window']} UDI\n"
            f"(expected {c['expected_per_window']:.2f}; scan p = {c['p_value']:.4f})",
            va="top", fontsize=8.5, color=TEXT_PRIMARY)
    ax.set_xlabel("UDI (time order)")
    ax.set_ylabel("Air temperature (K)")
    ax.set_title("RNF events cluster in time; air temperature drifts with time")
    fig.suptitle("RNF vs random: never detected at threshold 0.5; ranking signal weak and not significant (n+ = "
                 f"{rnf['n_positive_dev']})",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, path)


GREY_SPAN = "#e4e3df"


# ---------- run() ----------

def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, set):
        return sorted(o)
    raise TypeError(type(o))


def _verdict(rnf: dict) -> str:
    c = rnf["checks"]
    pb, ps = rnf["permutation_blocked"]["p_value"]["roc_auc"], rnf["permutation"]["p_value"]["roc_auc"]
    if rnf["consistent_with_random"] and c["permutation_stratified_not_significant"]:
        return f"RNF 與隨機無法區分：0.5 門檻下 Recall=0，兩種 CV 的置換檢定皆不顯著（p={ps:.3f}、{pb:.3f}）"
    if rnf["consistent_with_random"]:
        pp = rnf["permutation"]["p_value"]["pr_auc"]
        c = rnf["udi_cluster"]
        return (f"RNF 實質上不可預測：0.5 門檻下 Recall=0、時間區塊 CV 置換檢定不顯著（ROC p={pb:.3f}）；"
                f"一般分層 CV 的排序有微弱、邊緣的訊號（ROC p={ps:.3f}、PR p={pp:.3f}），"
                f"可能與 RNF 在時間上的群聚（{c['max_count']}/{c['n_events']} 落在 {c['window']} 個 UDI 內，"
                f"掃描 p={c['p_value']:.4f}）及溫度的時間代理效應有關")
    return f"RNF 結果與隨機有差異（區塊 CV p={pb:.3f}、分層 CV p={ps:.3f}），需檢查"


def run(out_dir=OUT_DIR, df=None, splits=None, feature_sets=FEATURE_SETS, n_splits: int = 5, n_repeats: int = 3,
        n_perm: int = 20, n_boot: int = 1000, seed: int = SEED, model: str = MODEL,
        perm_model: str = PERM_MODEL, n_sims: int = 10000, block_size: int = BLOCK_SIZE) -> dict:
    """各模式 CV（開發集）→ 開發集 OOF 選門檻 → 最終測試集評估一次 → RNF 與隨機基準比較 → 圖表與 multilabel.json。"""
    from ai4i_lab.data import load_dataset
    t0 = time.perf_counter()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_dataset() if df is None else df
    masks = E.split_masks(df, splits, "stratified")
    dev_df = df[masks["train"] | masks["val"]]           # 保留原 index；測試集不參與 CV／OOF／選門檻／置換檢定
    primary = feature_sets[0]
    timing = {}

    def tick(name, t):
        timing[name] = round(time.perf_counter() - t, 1)

    t = time.perf_counter()
    cv = {fs: cv_per_mode(dev_df, FAILURE_MODES, fs, model, n_splits=n_splits, n_repeats=n_repeats, seed=seed)
          for fs in feature_sets}
    tick("cv", t)
    t = time.perf_counter()
    detection = detection_by_mode(dev_df, FAILURE_MODES, primary, n_splits=n_splits, seed=seed, model=model)
    thresholds = {m: detection[m]["best_threshold"]["threshold"] for m in FAILURE_MODES}
    tick("dev_oof", t)
    t = time.perf_counter()
    test, P = {}, None
    for fs in feature_sets:
        test[fs], Pfs = holdout_per_mode(df, masks, FAILURE_MODES, fs, model, n_boot=n_boot, seed=seed,
                                         thresholds=thresholds if fs == primary else None)
        if fs == primary:
            P = Pfs
    k = [FAILURE_MODES.index(m) for m in PREDICTABLE_MODES]
    Yt = mode_targets(df[masks["test"]], PREDICTABLE_MODES).to_numpy()
    test_ml = {"threshold_0.5": multilabel_summary(Yt, P[:, k]),
               "threshold_tuned": multilabel_summary(Yt, P[:, k] >= np.array([thresholds[m] for m in PREDICTABLE_MODES]))}
    tick("test", t)
    t = time.perf_counter()
    rnf = rnf_experiment(dev_df, cv[primary]["rnf"], detection["rnf"], primary, n_perm=n_perm, n_splits=n_splits,
                         seed=seed, perm_model=perm_model, df_all=df, n_sims=n_sims,
                         block_size=block_size)
    rnf["test"] = {k2: test[primary]["rnf"][k2] for k2 in ("pr_auc", "roc_auc", "recall", "tp", "fp", "fn")}
    rnf["test"]["n_positive_test"] = test[primary]["rnf"]["n_positive_test"]
    tick("rnf", t)
    for fs in cv:
        for m in cv[fs]:
            timing[f"cv_{fs}_{m}"] = cv[fs][m].pop("seconds")

    verdict = _verdict(rnf)
    result = {
        "config": {"seed": seed, "model": model, "permutation_model": perm_model, "feature_sets": list(feature_sets),
                   "primary_feature_set": primary,
                   "strategy": "binary relevance（每模式一個獨立分類器；不用 classifier chain，避免模式標籤回流為特徵）",
                   "threshold": 0.5, "tuned_thresholds": thresholds, "n_splits": n_splits, "n_repeats": n_repeats,
                   "n_perm": n_perm, "n_boot": n_boot, "block_size": block_size, "split": "stratified (#7 splits.json)",
                   "dev_rows": int(len(dev_df)), "test_rows": int(masks["test"].sum()),
                   "cv_ci": "Nadeau-Bengio corrected t, 95%", "test_ci": "percentile bootstrap, 95%"},
        "cv": cv, "test": test, "test_multilabel": test_ml,
        "detection_by_mode": detection, "rnf": rnf, "rnf_verdict": verdict,
        "timing": timing,      # 計時與結果分開存放，讓結果部分在相同種子下逐位元可重現
    }
    plot_cv_metrics(cv, out_dir / FIGURES[0])
    plot_test_metrics({m: test[primary][m] for m in FAILURE_MODES}, out_dir / FIGURES[1])
    plot_rnf(rnf, detection, out_dir / FIGURES[2], df_all=df)
    result["seconds"] = round(time.perf_counter() - t0, 1)
    (out_dir / "multilabel.json").write_text(json.dumps(result, ensure_ascii=False, indent=1, default=_json_default),
                                             encoding="utf-8")
    return {"out_dir": str(out_dir), "rnf_verdict": verdict, "seconds": result["seconds"],
            "test_f1": {m: round(test[primary][m]["f1"]["point"], 4) for m in FAILURE_MODES},
            "test_f1_tuned": {m: round(test[primary][m]["tuned"]["f1"]["point"], 4) for m in FAILURE_MODES},
            "test_multilabel": test_ml}


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=1, default=_json_default))
