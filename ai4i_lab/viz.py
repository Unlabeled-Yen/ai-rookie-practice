"""AI4I 評估視覺化與錯誤分析（#10）。

只讀取 #8 輸出的測試集預測機率（predictions_test.csv）與 val 上調好的門檻（models.json），
並以 #5 的規則常數對照；不重新訓練任何模型。
執行：python -m ai4i_lab.viz
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from ai4i_lab import rules
from ai4i_lab.data import FAILURE_MODES, TARGET
from ai4i_lab.evaluation import METRICS, binary_metrics
from ai4i_lab.style import BLUE_RAMP, FAILURE, MODE_COLORS, NORMAL, SERIES, TEXT_PRIMARY, TEXT_SECONDARY, save

SEED = 42


# ---------- 混淆矩陣與校準 ----------

def confusion_counts(y, proba, threshold: float = 0.5) -> dict:
    """proba >= threshold 判為故障。"""
    y = np.asarray(y).astype(int).ravel()
    pred = np.asarray(proba, dtype=float).ravel() >= threshold
    return {"tp": int(np.sum(pred & (y == 1))), "fn": int(np.sum(~pred & (y == 1))),
            "fp": int(np.sum(pred & (y == 0))), "tn": int(np.sum(~pred & (y == 0)))}


def calibration_bins(y, proba, n_bins: int = 10, strategy: str = "quantile") -> dict:
    """機率校準：分箱後比較平均預測機率與實際故障比例。

    strategy="quantile"（等樣本數，適合極度不平衡）或 "uniform"（等寬）。空箱略過。
    ECE = Σ (n_b/N)·|frac_pos_b − mean_pred_b|；Brier = mean((p − y)²)。
    """
    if strategy not in ("quantile", "uniform"):
        raise ValueError(f"strategy 必須是 quantile 或 uniform：{strategy}")
    y = np.asarray(y).astype(int).ravel()
    p = np.asarray(proba, dtype=float).ravel()
    if strategy == "uniform":
        ids = np.minimum((p * n_bins).astype(int), n_bins - 1)
    else:
        order = np.argsort(p, kind="stable")
        ids = np.empty(len(p), dtype=int)
        ids[order] = np.minimum(np.arange(len(p)) * n_bins // len(p), n_bins - 1)
    bins, ece = [], 0.0
    for b in range(n_bins):
        m = ids == b
        n = int(m.sum())
        if n == 0:
            continue
        mp, fp_ = float(p[m].mean()), float(y[m].mean())
        bins.append({"n": n, "mean_pred": mp, "frac_pos": fp_, "n_pos": int(y[m].sum())})
        ece += n / len(p) * abs(fp_ - mp)
    return {"bins": bins, "ece": float(ece), "brier": float(np.mean((p - y) ** 2)), "strategy": strategy}


# ---------- bootstrap 信賴區間 ----------

def bootstrap_metrics(y, proba, threshold: float = 0.5, n_boot: int = 1000, seed: int = SEED,
                      confidence: float = 0.95) -> dict:
    """測試集五項指標的百分位 bootstrap 區間（重抽樣列；同一組抽樣同時算五項指標；不重新訓練）。

    point 為原始測試集上的值。抽到單一類別的重抽樣略過（ROC/PR-AUC 無定義）。
    """
    y = np.asarray(y).astype(int).ravel()
    p = np.asarray(proba, dtype=float).ravel()
    rng = np.random.default_rng(seed)
    vals = {m: [] for m in METRICS}
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        r = binary_metrics(y[i], p[i], threshold)
        for m in METRICS:
            vals[m].append(r[m])
    point = binary_metrics(y, p, threshold)
    a = (1 - confidence) / 2
    return {m: {"point": float(point[m]), "low": float(np.quantile(vals[m], a)),
                "high": float(np.quantile(vals[m], 1 - a)), "n_boot": len(vals[m])} for m in METRICS}


# ---------- 各故障模式偵測率 ----------

def mode_detection(df, pred) -> dict:
    """各故障模式的偵測率 = 該模式且 machine_failure = 1 的列中，被預測為故障的比例。

    RNF 標了 1 但 machine_failure = 0 的列不算「應偵測的故障」，只記在 n_rows_with_mode。
    "none"：machine_failure = 1 但五種模式皆為 0（標記不一致）；"any"：所有故障列。
    多重模式的列會同時計入各模式。
    """
    pred = np.asarray(pred, dtype=bool)
    fail = df[TARGET].to_numpy() == 1

    def cell(mask, n_rows=None):
        n = int((mask & fail).sum())
        d = int((mask & fail & pred).sum())
        return {"n_failures": n, "detected": d, "rate": (d / n if n else None),
                "n_rows_with_mode": int(mask.sum()) if n_rows is None else n_rows}

    out = {m: cell(df[m].to_numpy() == 1) for m in FAILURE_MODES}
    no_mode = ~(df[list(FAILURE_MODES)].to_numpy() == 1).any(axis=1)
    out["none"] = cell(no_mode & fail, n_rows=0)
    out["any"] = cell(fail, n_rows=int(fail.sum()))
    return out


# ---------- 規則邊界距離 ----------

RULE_NAMES = ("hdf", "pwf", "osf", "twf")


def rule_margins(df) -> pd.DataFrame:
    """每列相對於四條規則邊界的位置（正值 = 位於該規則的故障區域內，負值 = 區域外）。

    - `<rule>_dist`：到邊界的原始單位距離（HDF 取「最先失守」條件的單位：K 或 rpm；
      PWF 為 W；OSF 為 min·Nm；TWF 為到 [200, 240] 窗口最近邊緣的 min）。
    - `<rule>_margin`：dist 除以對應門檻的相對距離，跨規則可比較。
    - `<rule>_unit`：dist 的單位；`<rule>_fired`：直接取自 #5 的規則函式（嚴格不等式）。
    HDF 為兩條件的交集，距離取較小者（要同時滿足，最難滿足的那個決定距離）。
    PWF 為兩側的聯集，距離取較大者（離哪一側最近）。
    """
    d_t = rules.HDF_TEMP_DIFF_MAX_K - rules.temp_diff_k(df)
    d_r = rules.HDF_RPM_MAX - df["rotational_speed_rpm"]
    rel_t, rel_r = d_t / rules.HDF_TEMP_DIFF_MAX_K, d_r / rules.HDF_RPM_MAX
    use_t = rel_t <= rel_r
    P = rules.power_w(df)
    lo, hi = rules.PWF_POWER_MIN_W - P, P - rules.PWF_POWER_MAX_W
    use_lo = (lo / rules.PWF_POWER_MIN_W) >= (hi / rules.PWF_POWER_MAX_W)
    pwf_dist = lo.where(use_lo, hi)
    thr = rules.osf_threshold(df)
    osf_dist = rules.strain_minnm(df) - thr
    w = df["tool_wear_min"].astype(float)
    twf_dist = np.minimum(w - rules.TWF_WEAR_MIN, rules.TWF_WEAR_MAX - w)
    out = pd.DataFrame(index=df.index)
    out["hdf_dist"] = d_t.where(use_t, d_r)
    out["hdf_margin"] = np.minimum(rel_t, rel_r)
    out["hdf_unit"] = np.where(use_t, "K", "rpm")
    out["hdf_fired"] = rules.hdf_rule(df)
    out["pwf_dist"] = pwf_dist
    out["pwf_margin"] = np.maximum(lo / rules.PWF_POWER_MIN_W, hi / rules.PWF_POWER_MAX_W)
    out["pwf_unit"] = "W"
    out["pwf_fired"] = rules.pwf_rule(df)
    out["osf_dist"] = osf_dist
    out["osf_margin"] = osf_dist / thr
    out["osf_unit"] = "min*Nm"
    out["osf_fired"] = rules.osf_rule(df)
    out["twf_dist"] = twf_dist
    out["twf_margin"] = twf_dist / rules.TWF_WEAR_MIN
    out["twf_unit"] = "min"
    out["twf_fired"] = rules.twf_window(df)
    return out


# ---------- 逐筆錯誤分析 ----------

NEAR_REL = 0.05   # 相對距離 < 5% 視為「貼近邊界」
DET_RULES = ("hdf", "pwf", "osf")
_SHOW = ["type", "air_temp_k", "process_temp_k", "rotational_speed_rpm", "torque_nm", "tool_wear_min"]


def _record(row, m, proba, threshold, kind):
    modes = [k for k in FAILURE_MODES if row[k] == 1]
    cand = [k for k in modes if k in RULE_NAMES] or list(RULE_NAMES)
    near = min(cand, key=lambda k: abs(m[f"{k}_margin"]))
    fired_det = [k for k in DET_RULES if m[f"{k}_fired"]]
    if kind == "fn":
        if fired_det:
            small = min(m[f"{k}_margin"] for k in fired_det)
            cat = "rule_near_boundary" if small < NEAR_REL else "rule_deep_inside"
        elif row["twf"] == 1:
            cat = "twf_window"
        else:
            cat = "no_rule_explains"
    else:
        if fired_det:
            cat = "rule_fires_label_normal"
        elif min(abs(m[f"{k}_margin"]) for k in DET_RULES) < NEAR_REL:
            cat = "near_boundary_outside"
        elif m["twf_fired"]:
            cat = "twf_window"
        else:
            cat = "far_from_rules"
    rec = {"udi": int(row["udi"]), "product_id": str(row.get("product_id", "")), "proba": float(proba),
           "threshold": float(threshold), "true_modes": modes, "category": cat,
           "nearest_rule": near, "nearest_dist": float(m[f"{near}_dist"]), "nearest_unit": str(m[f"{near}_unit"]),
           "nearest_margin": float(m[f"{near}_margin"]),
           "fired_rules": [k for k in RULE_NAMES if m[f"{k}_fired"]],
           "margins": {k: float(m[f"{k}_margin"]) for k in RULE_NAMES}}
    for c in _SHOW:
        v = row[c]
        rec[c] = v if isinstance(v, str) else float(v)
    return rec


def error_analysis(df, proba, threshold: float) -> dict:
    """逐筆檢視漏判（FN）與誤報（FP），每筆標註最接近的規則邊界與距離。

    nearest_*：若列有標記的故障模式，取這些模式所對應規則（hdf/pwf/osf/twf）中離邊界最近者
    （RNF 無規則，回頭看全部）；否則取四條規則中最近者。dist 為正 = 位於故障區域內。
    FN 類別：rule_near_boundary（確定性規則已成立但相對距離 < 5%）、rule_deep_inside（深入區域內）、
    twf_window（僅 TWF，窗口內隨機失效）、no_rule_explains（RNF／標記不一致）。
    FP 類別：near_boundary_outside、twf_window、far_from_rules、rule_fires_label_normal。
    """
    p = np.asarray(proba, dtype=float).ravel()
    if len(p) != len(df):
        raise ValueError(f"proba 長度 {len(p)} 與資料列數 {len(df)} 不同")
    df = df.reset_index(drop=True)
    m = rule_margins(df)
    y = df[TARGET].to_numpy().astype(int)
    pred = p >= threshold
    fn_idx = np.where((y == 1) & ~pred)[0]
    fp_idx = np.where((y == 0) & pred)[0]
    fns = sorted((_record(df.iloc[i], m.iloc[i], p[i], threshold, "fn") for i in fn_idx), key=lambda r: r["proba"])
    fps = sorted((_record(df.iloc[i], m.iloc[i], p[i], threshold, "fp") for i in fp_idx),
                 key=lambda r: -r["proba"])

    def tally(rs, cats):
        return {c: sum(r["category"] == c for r in rs) for c in cats}

    return {
        "threshold": float(threshold), "counts": confusion_counts(y, p, threshold),
        "false_negatives": fns, "false_positives": fps,
        "summary": {
            "fn_by_category": tally(fns, ["rule_near_boundary", "rule_deep_inside", "twf_window", "no_rule_explains"]),
            "fp_by_category": tally(fps, ["near_boundary_outside", "twf_window", "far_from_rules",
                                          "rule_fires_label_normal"]),
        },
    }


# ---------- 圖表 ----------


ROOT = Path(__file__).resolve().parents[1]
UP_DIR = ROOT / "outputs" / "ai4i" / "models"
OUT_DIR = ROOT / "outputs" / "ai4i" / "evaluation"
FIGURES = ["confusion_matrices.png", "pr_curves.png", "roc_curves.png", "model_comparison.png",
           "calibration.png", "mode_detection.png", "error_margins.png"]
MODEL_ORDER = ["logreg", "random_forest", "hist_gb", "mlp"]
MODEL_COLOR = dict(zip(MODEL_ORDER, SERIES))
FEATURE_SETS = ["raw", "engineered"]
METRIC_LABEL = {"pr_auc": "PR-AUC", "roc_auc": "ROC-AUC", "f1": "F1", "recall": "Recall", "precision": "Precision"}


def _split_key(k):
    m, fs = k.split("|")
    return m, fs


def _groups(preds):
    """依 (model, feature_set) 切成 {key: (udi, y, proba)}，順序固定。"""
    out = {}
    for (m, fs), g in preds.groupby(["model", "feature_set"], sort=False):
        out[f"{m}|{fs}"] = (g["row_id"].to_numpy(), g["y_true"].to_numpy().astype(int), g["proba"].to_numpy())
    order = [f"{m}|{fs}" for m in MODEL_ORDER for fs in FEATURE_SETS if f"{m}|{fs}" in out]
    order += [k for k in out if k not in order]
    return {k: out[k] for k in order}


def _by_fs_axes(n=2, w=5.2, h=4.4):
    fig, axes = plt.subplots(1, n, figsize=(w * n, h), sharey=True)
    return fig, np.atleast_1d(axes)


def _model_legend(ax, models):
    ax.legend(handles=[plt.Line2D([], [], color=MODEL_COLOR.get(m, SERIES[-1]), lw=2, label=m) for m in models],
              loc="best", fontsize=8)


def plot_confusion_matrices(groups, thresholds, path):
    keys = list(groups)
    ncol = min(4, len(keys))
    nrow = int(np.ceil(len(keys) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.5 * nrow), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, k in zip(axes.ravel(), keys):
        ax.axis("on")
        _, y, p = groups[k]
        c = confusion_counts(y, p, thresholds[k])
        mat = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
        share = mat / mat.sum(axis=1, keepdims=True)       # 以列（實際類別）為底的比例，色深不被 TN 壓倒
        ax.imshow(share, cmap=BLUE_RAMP, vmin=0, vmax=1)
        ax.grid(False)
        for i in range(2):
            for j in range(2):
                dark = share[i, j] > 0.55
                ax.text(j, i, f"{mat[i, j]}\n({share[i, j]:.0%})", ha="center", va="center", fontsize=10,
                        color="white" if dark else TEXT_PRIMARY)
        ax.set_xticks([0, 1], ["Normal", "Failure"])
        ax.set_yticks([0, 1], ["Normal", "Failure"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{k.replace('|', ' / ')}\nthr={thresholds[k]:.2f}", fontsize=9)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle("Confusion matrices on test set (threshold tuned on validation; % of actual class)",
                 fontsize=11, color=TEXT_PRIMARY, y=1.02)
    fig.tight_layout()
    save(fig, path)


def _curves(groups, path, kind):
    fig, axes = _by_fs_axes()
    for ax, fs in zip(axes, FEATURE_SETS):
        models = []
        for k, (_, y, p) in groups.items():
            m, f = _split_key(k)
            if f != fs:
                continue
            models.append(m)
            if kind == "pr":
                pr, rc, _ = precision_recall_curve(y, p)
                ax.plot(rc, pr, color=MODEL_COLOR.get(m, SERIES[-1]))
            else:
                fpr, tpr, _ = roc_curve(y, p)
                ax.plot(fpr, tpr, color=MODEL_COLOR.get(m, SERIES[-1]))
        if kind == "pr":
            base = float(np.mean(next(iter(groups.values()))[1]))
            ax.axhline(base, color=TEXT_SECONDARY, lw=1, ls="--")
            ax.text(0.01, base + 0.02, f"base rate {base:.3f}", ha="left", color=TEXT_SECONDARY, fontsize=8)
            ax.set_xlabel("Recall")
            axes[0].set_ylabel("Precision")
        else:
            ax.plot([0, 1], [0, 1], color=TEXT_SECONDARY, lw=1, ls="--")
            ax.set_xlabel("False positive rate")
            axes[0].set_ylabel("True positive rate")
        ax.set_title(f"{fs} features")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        _model_legend(ax, models)
    fig.suptitle("Precision-Recall curves (test set)" if kind == "pr" else "ROC curves (test set)",
                 fontsize=12, color=TEXT_PRIMARY)
    fig.tight_layout()
    save(fig, path)


def plot_pr_curves(groups, path):
    _curves(groups, path, "pr")


def plot_roc_curves(groups, path):
    _curves(groups, path, "roc")


def plot_model_comparison(table, path, metrics=("pr_auc", "roc_auc", "f1", "recall", "precision")):
    """各指標一個子圖：每個模型 × 特徵組一個點，附 bootstrap 95% 區間（橫線）。"""
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.3 * len(metrics), 4.4), sharey=True)
    n = len(table)
    ypos = np.arange(n)[::-1]
    for ax, m in zip(axes, metrics):
        for y_, (_, r) in zip(ypos, table.iterrows()):
            c = MODEL_COLOR.get(r["model"], SERIES[-1])
            ax.plot([r[f"{m}_low"], r[f"{m}_high"]], [y_, y_], color=c, lw=2)
            ax.plot(r[m], y_, "o", color=c, ms=6, mfc=c if r["feature_set"] == "engineered" else "white", mew=2)
            ax.text(min(r[f"{m}_high"] + 0.02, 1.0), y_, f"{r[m]:.2f}", va="center", fontsize=7.5,
                    color=TEXT_SECONDARY)
        ax.set_xlim(0, 1.18)
        ax.set_title(METRIC_LABEL[m], fontsize=10)
        ax.set_xlabel("95% bootstrap CI")
    axes[0].set_yticks(ypos, [f"{r['model']} / {r['feature_set'][:3]}" for _, r in table.iterrows()], fontsize=8)
    fig.suptitle("Test-set metrics with 95% bootstrap CI (filled = engineered, hollow = raw; threshold tuned on val)",
                 fontsize=10.5, color=TEXT_PRIMARY)
    fig.tight_layout()
    save(fig, path)


def plot_calibration(cal, path):
    """cal: {key: calibration_bins(...)}；左右兩欄 = raw／engineered；等樣本數分箱（故障機率集中在接近 0）。"""
    fig, axes = _by_fs_axes(w=5.4, h=4.8)
    for ax, fs in zip(axes, FEATURE_SETS):
        models = []
        for k, c in cal.items():
            m, f = _split_key(k)
            if f != fs:
                continue
            models.append(m)
            xs = [b["mean_pred"] for b in c["bins"]]
            ys = [b["frac_pos"] for b in c["bins"]]
            ax.plot(xs, ys, "-o", color=MODEL_COLOR.get(m, SERIES[-1]), ms=4)
        ax.plot([0, 0.5], [0, 0.5], color=TEXT_SECONDARY, lw=1, ls="--")
        ax.set_xlim(-0.01, 0.5)
        ax.set_ylim(-0.01, 0.5)
        ax.set_xlabel("Mean predicted probability (10 quantile bins)")
        ax.set_title(f"{fs} features")
        _model_legend(ax, models)
        ax.text(0.03, 0.95, "dashed = perfect calibration\n(67 failures: low bins hold 0-1 events)",
                transform=ax.transAxes, va="top", fontsize=7.5, color=TEXT_SECONDARY)
    axes[0].set_ylabel("Observed failure rate")
    fig.suptitle("Calibration curves (test set)", fontsize=12, color=TEXT_PRIMARY)
    fig.tight_layout()
    save(fig, path)


def plot_mode_detection(det, path, primary):
    """左：主模型各模式偵測率（長條，標 k/n）；右：所有模型 × 模式的偵測率熱圖。"""
    cols = list(FAILURE_MODES) + ["none"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw={"width_ratios": [1, 1.35]})
    d = det[primary]
    for i, c in enumerate(cols):
        r = d[c]
        rate = r["rate"] if r["rate"] is not None else 0.0
        a1.bar(i, rate, color=MODE_COLORS.get(c, SERIES[-1]) if c != "none" else TEXT_SECONDARY, width=0.62)
        lab = "n/a" if r["rate"] is None else f"{r['detected']}/{r['n_failures']}"
        a1.text(i, rate + 0.02, lab, ha="center", fontsize=9, color=TEXT_PRIMARY)
    a1.set_xticks(range(len(cols)), [c.upper() if c != "none" else "no mode\nlabel" for c in cols])
    a1.set_ylim(0, 1.12)
    a1.set_ylabel("Detection rate (recall)")
    a1.set_title(f"Detection rate by failure mode\n{primary.replace('|', ' / ')}", fontsize=10.5)
    keys = list(det)
    mat = np.array([[np.nan if det[k][c]["rate"] is None else det[k][c]["rate"] for c in cols] for k in keys])
    a2.imshow(np.nan_to_num(mat, nan=0.0), cmap=BLUE_RAMP, vmin=0, vmax=1, aspect="auto")
    a2.grid(False)
    for i in range(len(keys)):
        for j in range(len(cols)):
            v = mat[i, j]
            a2.text(j, i, "n/a" if np.isnan(v) else f"{v:.2f}", ha="center", va="center", fontsize=8.5,
                    color="white" if (not np.isnan(v) and v > 0.55) else TEXT_PRIMARY)
    a2.set_xticks(range(len(cols)), [f"{c.upper() if c != 'none' else 'no mode'}\n(n={det[primary][c]['n_failures']})"
                                    for c in cols], fontsize=8.5)
    a2.set_yticks(range(len(keys)), [k.replace("|", " / ") for k in keys], fontsize=8.5)
    a2.set_title("Detection rate, all models (threshold tuned on val)", fontsize=10.5)
    for s in a2.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    save(fig, path)


DEPTH_EDGES = (0.01, 0.03, 0.1)


def recall_by_depth(depth, hit, edges=DEPTH_EDGES) -> list:
    """依「落入故障區域的相對深度」分箱（右閉）計算偵測率；空箱 recall = None。"""
    depth, hit = np.asarray(depth, dtype=float), np.asarray(hit, dtype=bool)
    lo = [-np.inf] + list(edges)
    hi = list(edges) + [np.inf]
    labels = [f"<={edges[0]:.0%}"] + [f"{a * 100:.0f}-{b:.0%}" for a, b in zip(edges[:-1], edges[1:])] + [f">{edges[-1]:.0%}"]
    out = []
    for l, h, lab in zip(lo, hi, labels):
        m = (depth > l) & (depth <= h)
        n = int(m.sum())
        out.append({"label": lab, "n": n, "n_detected": int(hit[m].sum()),
                    "recall": float(hit[m].mean()) if n else None})
    return out


def rule_depth(df) -> tuple:
    """確定性規則（HDF/PWF/OSF）故障列的深度 = 其標記模式對應規則的相對距離（多模式取最小）。"""
    m = rule_margins(df)
    sel = (df[list(DET_RULES)].to_numpy() == 1).any(axis=1)
    cols = np.stack([np.where(df[r].to_numpy() == 1, m[f"{r}_margin"].to_numpy(), np.inf) for r in DET_RULES])
    return sel, cols.min(axis=0)[sel]


def plot_recall_by_depth(depth_tab, path):
    """depth_tab: {key: [bin dict]}；每個特徵組一欄，每個模型一條線。"""
    fig, axes = _by_fs_axes(w=5.4, h=4.4)
    for ax, fs in zip(axes, FEATURE_SETS):
        models = []
        for k, bins in depth_tab.items():
            m, f = _split_key(k)
            if f != fs:
                continue
            models.append(m)
            xs = [i for i, b in enumerate(bins) if b["recall"] is not None]
            ax.plot(xs, [bins[i]["recall"] for i in xs], "-o", color=MODEL_COLOR.get(m, SERIES[-1]), ms=5)
        first = next(iter(depth_tab.values()))
        ax.set_xticks(range(len(first)), [f"{b['label']}\n(n={b['n']})" for b in first], fontsize=8.5)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Depth inside rule failure region (distance to boundary / threshold)")
        ax.set_title(f"{fs} features")
        _model_legend(ax, models)
    axes[0].set_ylabel("Detection rate (recall)")
    fig.suptitle("Detection rate vs. distance to the HDF/PWF/OSF rule boundary (test set)", fontsize=11.5,
                 color=TEXT_PRIMARY)
    fig.tight_layout()
    save(fig, path)


# ---------- 組裝 ----------

def pick_primary(cv_pr_auc: dict) -> str:
    """主模型 = 開發集 CV PR-AUC 最高者（不看測試集，避免用測試集選模型）。"""
    if not cv_pr_auc:
        raise ValueError("cv_pr_auc 為空")
    return max(cv_pr_auc, key=lambda k: cv_pr_auc[k])


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def _table_md(tab: pd.DataFrame) -> str:
    lines = ["| model | features | thr | " + " | ".join(METRIC_LABEL[m] for m in METRICS) + " | TP/FP/FN/TN | Brier | ECE |",
             "|---|---|---|" + "---|" * len(METRICS) + "---|---|---|"]
    for _, r in tab.iterrows():
        cells = " | ".join(f"{r[m]:.3f} [{r[m + '_low']:.3f}, {r[m + '_high']:.3f}]" for m in METRICS)
        lines.append(f"| {r['model']} | {r['feature_set']} | {r['threshold']:.3f} | {cells} | "
                     f"{r['tp']}/{r['fp']}/{r['fn']}/{r['tn']} | {r['brier']:.4f} | {r['ece']:.4f} |")
    return "\n".join(lines) + "\n"


def _flat(rec: dict) -> dict:
    r = {k: v for k, v in rec.items() if k not in ("margins", "true_modes", "fired_rules")}
    r["true_modes"] = "+".join(rec["true_modes"]) or "none"
    r["fired_rules"] = "+".join(rec["fired_rules"]) or "none"
    return r


def run(out_dir=OUT_DIR, df=None, preds=None, thresholds=None, cv_pr_auc=None, n_boot: int = 1000,
        up_dir=UP_DIR, seed: int = SEED) -> dict:
    """讀取 #8 的測試集預測與 val 門檻，產生所有評估圖表、比較表與 error_analysis.json。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if df is None:
        from ai4i_lab.data import load_dataset
        df = load_dataset()
    if preds is None:
        preds = pd.read_csv(Path(up_dir) / "predictions_test.csv")
    if thresholds is None or cv_pr_auc is None:
        up = json.loads((Path(up_dir) / "models.json").read_text(encoding="utf-8"))
        thresholds = thresholds or {k: v["threshold"] for k, v in up["thresholds"].items()}
        cv_pr_auc = cv_pr_auc or {k: v["cv_pr_auc"] for k, v in up["chosen"].items()}
    groups = _groups(preds)
    by_udi = df.set_index("udi")
    for k, (udi, y, _) in groups.items():
        missing = set(udi) - set(by_udi.index)
        if missing:
            raise ValueError(f"{k}：預測檔中有 {len(missing)} 個 row_id 不在資料裡")
        if not np.array_equal(by_udi.loc[udi, TARGET].to_numpy().astype(int), y):
            raise ValueError(f"{k}：y_true 與資料的 {TARGET} 不一致")
    primary = pick_primary({k: v for k, v in cv_pr_auc.items() if k in groups})

    rows, cal, det, depth_tab = [], {}, {}, {}
    for k, (udi, y, p) in groups.items():
        m, fs = _split_key(k)
        thr = float(thresholds[k])
        bs = bootstrap_metrics(y, p, thr, n_boot=n_boot, seed=seed)
        c = confusion_counts(y, p, thr)
        cal[k] = calibration_bins(y, p, n_bins=10, strategy="quantile")
        row = {"model": m, "feature_set": fs, "threshold": thr, "n_boot": bs["f1"]["n_boot"], **c,
               "brier": cal[k]["brier"], "ece": cal[k]["ece"]}
        for mt in METRICS:
            row[mt], row[f"{mt}_low"], row[f"{mt}_high"] = bs[mt]["point"], bs[mt]["low"], bs[mt]["high"]
        rows.append(row)
        sub = by_udi.loc[udi].reset_index()
        det[k] = mode_detection(sub, p >= thr)
        sel, dep = rule_depth(sub)
        depth_tab[k] = recall_by_depth(dep, (p >= thr)[sel])
    tab = pd.DataFrame(rows)
    tab.to_csv(out_dir / "model_comparison.csv", index=False)
    (out_dir / "model_comparison.md").write_text(_table_md(tab), encoding="utf-8")

    udi, y, p = groups[primary]
    sub = by_udi.loc[udi].reset_index()
    thr = float(thresholds[primary])
    ea = error_analysis(sub, p, thr)
    mm = rule_margins(sub)
    pred = p >= thr
    margin_stats = {}
    for r in DET_RULES:
        sel = sub[r].to_numpy() == 1
        margin_stats[r] = {
            "n": int(sel.sum()),
            "median_margin_detected": float(np.median(mm.loc[sel & pred, f"{r}_margin"])) if (sel & pred).any() else None,
            "median_margin_missed": float(np.median(mm.loc[sel & ~pred, f"{r}_margin"])) if (sel & ~pred).any() else None,
            "n_missed": int((sel & ~pred).sum()),
            "n_missed_within_5pct": int((sel & ~pred & (mm[f"{r}_margin"].to_numpy() < NEAR_REL)).sum()),
            "n_all_within_5pct": int((sel & (mm[f"{r}_margin"].to_numpy() < NEAR_REL)).sum()),
        }
    ea = {"model": primary, "selection": "highest dev-set CV PR-AUC (test set not used for selection)",
          "near_boundary_rel": NEAR_REL, **ea, "mode_detection": det[primary], "margin_stats": margin_stats}
    (out_dir / "error_analysis.json").write_text(json.dumps(ea, ensure_ascii=False, indent=1, default=_jsonable),
                                                 encoding="utf-8")
    pd.DataFrame([_flat(r) for r in ea["false_negatives"]]).to_csv(out_dir / "false_negatives.csv", index=False)
    pd.DataFrame([_flat(r) for r in ea["false_positives"]]).to_csv(out_dir / "false_positives.csv", index=False)

    plot_confusion_matrices(groups, thresholds, out_dir / "confusion_matrices.png")
    plot_pr_curves(groups, out_dir / "pr_curves.png")
    plot_roc_curves(groups, out_dir / "roc_curves.png")
    plot_model_comparison(tab, out_dir / "model_comparison.png")
    plot_calibration(cal, out_dir / "calibration.png")
    plot_mode_detection(det, out_dir / "mode_detection.png", primary)
    plot_recall_by_depth(depth_tab, out_dir / "error_margins.png")

    result = {"issue": 10, "primary_model": primary, "n_test": int(len(groups[primary][1])),
              "n_test_failures": int(groups[primary][1].sum()), "table": rows, "calibration": cal,
              "mode_detection": det, "primary_counts": ea["counts"], "primary_summary": ea["summary"],
              "margin_stats": margin_stats, "recall_by_depth": depth_tab, "figures": FIGURES, "n_boot": n_boot, "seed": seed}
    (out_dir / "evaluation.json").write_text(json.dumps(result, ensure_ascii=False, indent=1, default=_jsonable),
                                             encoding="utf-8")
    return result


if __name__ == "__main__":
    r = run()
    print(f"主模型 {r['primary_model']}：{r['primary_counts']}")
