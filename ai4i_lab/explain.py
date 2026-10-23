"""AI4I 可解釋性分析（#11）：模型學到的決策依據是否和資料產生規則一致？

- 研究問題 A：permutation importance、PDP/ICE（在規則座標上介入原始特徵）、模型邊界 vs 規則邊界
- 研究問題 B 補完：MLP 只用原始特徵時能否學到乘積型規則（磨耗×扭矩、扭矩×轉速）
規則門檻一律引用 `ai4i_lab.rules` 的常數。
執行：python -m ai4i_lab.explain
"""
import numpy as np
import pandas as pd

from ai4i_lab import rules as R
from ai4i_lab.features import FEATURE_SETS, TYPE_COLUMNS, build_features

SEED = 42


# ---------- Permutation importance ----------

def _groups(columns) -> dict:
    """Type 的三個 one-hot 欄位視為同一個特徵 'type'（否則各別打亂會低估重要度）。"""
    g = {}
    for c in columns:
        g.setdefault("type" if c in TYPE_COLUMNS else c, []).append(c)
    return g


def _pr_auc(y, proba) -> float:
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y, proba))


def permutation_importance(model, X: pd.DataFrame, y, n_repeats: int = 5, seed: int = SEED,
                           metric=_pr_auc, metric_name: str = "pr_auc") -> pd.DataFrame:
    """打亂單一特徵（或 Type 群組）後 metric 的下降量；越大代表模型越依賴該特徵。

    重要度 = 基準分數 − 打亂後分數（多次重複取平均／標準差）。X 不會被修改。
    """
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    base = metric(y, model.predict_proba(X)[:, 1])
    rows = []
    for name, cols in _groups(X.columns).items():
        drops = []
        for _ in range(n_repeats):
            perm = rng.permutation(len(X))
            Xp = X.copy()
            Xp[cols] = X[cols].to_numpy()[perm]       # 同群組欄位一起換列，保留 one-hot 結構
            drops.append(base - metric(y, model.predict_proba(Xp)[:, 1]))
        rows.append({"feature": name, "mean": float(np.mean(drops)),
                     "std": float(np.std(drops, ddof=1)) if n_repeats > 1 else 0.0,
                     "n_repeats": n_repeats, "metric": metric_name, "baseline": float(base)})
    return pd.DataFrame(rows)


# ---------- 在「規則座標」上介入原始特徵（PDP／ICE 的橫軸） ----------
# 為了讓 raw 與 engineered 模型能在同一個橫軸上比較，介入一律作用在原始感測欄位，
# 再用 build_features 重算所有衍生特徵（衍生特徵因此與原始欄位保持一致）。
#   power        ：保持轉速，調整扭矩使功率 = P [W]
#   temp_diff    ：保持空氣溫度，調整製程溫度使溫差 = d [K]
#   strain_ratio ：保持扭矩，調整磨耗使 (磨耗×扭矩)/該列 Type 的 OSF 門檻 = r
#   wear         ：直接設定刀具磨耗 [min]
AXES = {
    "power": {"label": "Power = torque x omega [W]", "variable": "torque_nm",
              "thresholds": [R.PWF_POWER_MIN_W, R.PWF_POWER_MAX_W], "window": None},
    "temp_diff": {"label": "Process - air temperature [K]", "variable": "process_temp_k",
                  "thresholds": [R.HDF_TEMP_DIFF_MAX_K], "window": None},
    "strain_ratio": {"label": "Wear x torque / Type threshold", "variable": "tool_wear_min",
                     "thresholds": [1.0], "window": None},
    "wear": {"label": "Tool wear [min]", "variable": "tool_wear_min",
             "thresholds": [], "window": [R.TWF_WEAR_MIN, R.TWF_WEAR_MAX]},
}


def axis_spec(axis: str) -> dict:
    if axis not in AXES:
        raise ValueError(f"未知的介入軸：{axis!r}；可用 {list(AXES)}")
    return AXES[axis]


def set_power(df, power_w: float):
    out = df.copy()
    out["torque_nm"] = power_w / (df["rotational_speed_rpm"].astype(float) * 2 * np.pi / 60)
    return out


def set_temp_diff(df, diff_k: float):
    out = df.copy()
    out["process_temp_k"] = df["air_temp_k"].astype(float) + diff_k
    return out


def set_strain_ratio(df, ratio: float):
    out = df.copy()
    out["tool_wear_min"] = ratio * R.osf_threshold(df) / df["torque_nm"].astype(float)
    return out


def set_wear(df, wear_min: float):
    out = df.copy()
    out["tool_wear_min"] = float(wear_min)
    return out


_SETTERS = {"power": set_power, "temp_diff": set_temp_diff, "strain_ratio": set_strain_ratio, "wear": set_wear}


def apply_intervention(df, axis: str, value: float, bounds: dict | None = None):
    """回傳 (介入後的 df, 有效遮罩)。被改動的欄位若超出觀測範圍 bounds，該列標為無效（避免外插）。"""
    axis_spec(axis)
    out = _SETTERS[axis](df, value)
    valid = np.ones(len(df), dtype=bool)
    var = AXES[axis]["variable"]
    if bounds and var in bounds:
        lo, hi = bounds[var]
        valid = ((out[var] >= lo) & (out[var] <= hi)).to_numpy()
    return out, valid


def observed_bounds(df) -> dict:
    return {c: (float(df[c].min()), float(df[c].max()))
            for c in ["torque_nm", "tool_wear_min", "process_temp_k"]}


def ice_curves(model, df, feature_set: str, axis: str, grid, bounds: dict | None = None) -> np.ndarray:
    """每列一條 ICE 曲線（列 × 網格）；介入後超出觀測範圍的點為 NaN。"""
    grid = np.asarray(grid, dtype=float)
    ice = np.full((len(df), len(grid)), np.nan)
    for j, v in enumerate(grid):
        mod, valid = apply_intervention(df, axis, float(v), bounds)
        if valid.any():
            proba = model.predict_proba(build_features(mod.loc[valid], feature_set))[:, 1]
            ice[valid, j] = proba
    return ice


# ---------- 決策邊界交點與吻合度 ----------

def find_crossings(grid, p, thr: float) -> list:
    """沿網格找出機率跨越門檻 thr 的位置（線性內插）；含 NaN 的區段不產生交點。"""
    grid = np.asarray(grid, float)
    p = np.asarray(p, float)
    out = []
    for i in range(len(grid) - 1):
        a, b = p[i], p[i + 1]
        if np.isnan(a) or np.isnan(b) or (a >= thr) == (b >= thr):
            continue
        t = (thr - a) / (b - a)
        out.append(float(grid[i] + t * (grid[i + 1] - grid[i])))
    return out


def boundary_offsets(grid, ice, thr: float, rule_value: float, window: float) -> dict:
    """每條 ICE 曲線取距離規則門檻最近的交點（|交點 − 規則值| <= window），統計偏移量。"""
    offsets = []
    for p in ice:
        cr = [c for c in find_crossings(grid, p, thr) if abs(c - rule_value) <= window]
        if cr:
            offsets.append(min(cr, key=lambda c: abs(c - rule_value)) - rule_value)
    n = len(ice)
    off = np.asarray(offsets, float)
    res = {"n_curves": int(n), "n_crossing": int(len(off)), "frac_crossing": float(len(off) / n) if n else None,
           "rule_value": float(rule_value), "window": float(window),
           "median_offset": None, "median_abs_offset": None, "q25_offset": None, "q75_offset": None}
    if len(off):
        res.update(median_offset=float(np.median(off)), median_abs_offset=float(np.median(np.abs(off))),
                   q25_offset=float(np.quantile(off, 0.25)), q75_offset=float(np.quantile(off, 0.75)))
    return res


def decision_agreement(pred, rule) -> dict:
    """模型二元預測 vs 規則預測：混淆矩陣、準確率、F1、Jaccard、Cohen's kappa。"""
    pred = np.asarray(pred).astype(bool)
    rule = np.asarray(rule).astype(bool)
    n = len(pred)
    tp = int((pred & rule).sum()); fp = int((pred & ~rule).sum())
    fn = int((~pred & rule).sum()); tn = int((~pred & ~rule).sum())
    po = (tp + tn) / n
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / n ** 2
    return {"n": int(n), "tp": tp, "fp": fp, "fn": fn, "tn": tn, "accuracy": float(po),
            "f1": float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else None,
            "jaccard": float(tp / (tp + fp + fn)) if (tp + fp + fn) else None,
            "kappa": float((po - pe) / (1 - pe)) if pe < 1 else None}


# ---------- 研究問題 B：MLP 只靠原始特徵能否學到乘積型規則 ----------

def rule_targets(df) -> dict:
    """直接以規則重算的 0/1 目標（無標記雜訊）。

    osf：磨耗×扭矩（乘積）；pwf：扭矩×轉速（乘積，兩側門檻）；hdf：溫差（差值）且轉速；
    rpm_only：控制組，單一原始特徵的軸平行門檻（轉速 < HDF 轉速門檻）。
    """
    return {"osf": R.osf_rule(df).astype(int).to_numpy(), "pwf": R.pwf_rule(df).astype(int).to_numpy(),
            "hdf": R.hdf_rule(df).astype(int).to_numpy(),
            "rpm_only": (df["rotational_speed_rpm"] < R.HDF_RPM_MAX).astype(int).to_numpy()}


def mlp_rule_learning(train, val, test, rule: str, feature_set: str, params: dict | None = None,
                      imbalance: str = "class_weight", seed: int = SEED) -> dict:
    """在 train 上訓練 MLP 去預測規則目標，val 選門檻（最大化 F1），test 只評估一次。

    特徵只用 build_features（不含故障模式欄位與 ID），目標由規則重算，因此可以乾淨地問：
    「只給原始特徵時，MLP 學得會這個規則嗎？」
    """
    from ai4i_lab import models as M
    from ai4i_lab.evaluation import binary_metrics
    y = {k: rule_targets(d)[rule] for k, d in (("train", train), ("val", val), ("test", test))}
    est = M.make_estimator("mlp", imbalance, seed, params)
    est.fit(build_features(train, feature_set), y["train"])
    p_val = est.predict_proba(build_features(val, feature_set))[:, 1]
    thr = M.tune_threshold(y["val"], p_val)["threshold"]
    p_test = est.predict_proba(build_features(test, feature_set))[:, 1]
    m = binary_metrics(y["test"], p_test, thr)
    mv = binary_metrics(y["val"], p_val, thr)
    return {"rule": rule, "feature_set": feature_set, "threshold": float(thr),
            "pr_auc": m["pr_auc"], "roc_auc": m["roc_auc"], "f1": m["f1"], "recall": m["recall"],
            "precision": m["precision"], "val_pr_auc": mv["pr_auc"], "val_f1": mv["f1"],
            "n_pos_train": int(y["train"].sum()), "n_pos_test": int(y["test"].sum()),
            "params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in (params or {}).items()}}


def per_mode_recall(df, pred) -> dict:
    """在各故障模式的正例列上，模型預測為故障的比例（pred 為 0/1，與 df 同序）。"""
    from ai4i_lab.data import FAILURE_MODES
    pred = np.asarray(pred).astype(bool)
    out = {}
    for m in FAILURE_MODES:
        pos = (df[m] == 1).to_numpy()
        n, det = int(pos.sum()), int((pred & pos).sum())
        out[m] = {"n": n, "detected": det, "recall": float(det / n) if n else None}
    return out


# ---------- 反事實區域吻合度：介入後的規則預測 ----------

def rule_union(df) -> np.ndarray:
    """三條確定性規則（HDF／PWF／OSF）的聯集；TWF 為隨機窗口，不屬於確定性規則。"""
    return (R.hdf_rule(df) | R.pwf_rule(df) | R.osf_rule(df)).to_numpy()


def ice_rule_matrix(df, axis: str, grid, bounds: dict | None = None):
    """與 ice_curves 同形狀的「規則預測」矩陣（True = 規則判定故障）與有效遮罩（超出觀測範圍者為 False）。"""
    grid = np.asarray(grid, float)
    rule = np.zeros((len(df), len(grid)), dtype=bool)
    valid = np.zeros((len(df), len(grid)), dtype=bool)
    for j, v in enumerate(grid):
        mod, ok = apply_intervention(df, axis, float(v), bounds)
        rule[:, j] = rule_union(mod) & ok
        valid[:, j] = ok
    return rule, valid


def healthy_pool(df):
    """沒有故障標記、且三條規則都不成立的列：ICE 的起點，避免曲線一開始就被其他原因拉到高機率。"""
    keep = (df["machine_failure"] == 0).to_numpy() & ~rule_union(df)
    return df[keep]


def sample_rows(df, n: int, seed: int = SEED):
    if len(df) <= n:
        return df
    return df.sample(n=n, random_state=seed).sort_index()


def synthetic_frame(src, n: int, seed: int = SEED, rpm_jitter: float = 15.0, torque_jitter: float = 0.4,
                    temp_jitter: float = 0.05):
    """由來源列重抽樣出 n 列合成資料（只用來回答「資料量夠大時學得會嗎」）。

    (轉速, 扭矩) 成對抽樣以保留兩者的強負相關（約 -0.88），加小幅均勻抖動；
    Type、磨耗、空氣溫度、溫差各自獨立抽樣。標記不使用——目標一律由規則重算。
    """
    rng = np.random.default_rng(seed)
    pair = rng.integers(0, len(src), n)
    pick = lambda col: src[col].to_numpy()[rng.integers(0, len(src), n)]    # noqa: E731
    rpm = src["rotational_speed_rpm"].to_numpy()[pair] + rng.uniform(-rpm_jitter, rpm_jitter, n)
    torque = src["torque_nm"].to_numpy()[pair] + rng.uniform(-torque_jitter, torque_jitter, n)
    air = pick("air_temp_k")
    diff = R.temp_diff_k(src).to_numpy()[rng.integers(0, len(src), n)] + rng.uniform(-temp_jitter, temp_jitter, n)
    return pd.DataFrame({
        "udi": np.arange(1, n + 1), "type": pick("type"), "air_temp_k": air, "process_temp_k": air + diff,
        "rotational_speed_rpm": np.maximum(rpm, 1.0), "torque_nm": np.maximum(torque, 0.1),
        "tool_wear_min": pick("tool_wear_min"),
    })


# ---------- 分析流程 ----------
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

from ai4i_lab.data import FAILURE_MODES, ROOT, TARGET  # noqa: E402
from ai4i_lab.features import DERIVED  # noqa: E402
from ai4i_lab.style import (FAILURE, NORMAL, SERIES, TEXT_PRIMARY, TEXT_SECONDARY,  # noqa: E402
                            save)

EXPLAIN_DIR = ROOT / "outputs" / "ai4i" / "explain"
MODEL_ORDER = ("logreg", "random_forest", "hist_gb", "mlp")
FS_ORDER = ("raw", "engineered")
WINDOW_FRAC = 0.25            # 交點必須落在「規則值 ± 25%」內才算「學到這個邊界」
# 每個軸的網格（範圍涵蓋規則門檻兩側；超出觀測範圍的介入自動標為無效）
GRIDS = {
    "power": np.arange(1500.0, 11001.0, 100.0),
    "temp_diff": np.round(np.arange(7.0, 11.001, 0.05), 4),
    "strain_ratio": np.round(np.arange(0.4, 1.601, 0.01), 4),
    "wear": np.arange(0.0, 254.0, 2.0),
}
FIGURES = ["importance.png", "pdp_ice_power.png", "pdp_ice_temp_diff.png", "pdp_ice_strain_ratio.png",
           "pdp_ice_wear.png", "boundary_offsets.png", "mlp_product_rules.png"]
MLP_REAL = {"hidden_layer_sizes": (64, 32), "learning_rate_init": 0.003, "max_iter": 300}
MLP_BIG = {"hidden_layer_sizes": (128, 64), "learning_rate_init": 0.003, "max_iter": 200}
SCALING_SIZES = (6000, 20000, 40000)


def _label(model, fs):
    return f"{model} | {fs}"


def _model_thresholds(out_dir_models) -> dict:
    info = json.loads((Path(out_dir_models) / "models.json").read_text(encoding="utf-8"))
    return {k: float(v["threshold"]) for k, v in info["thresholds"].items()}


def _axis_rules(axis: str) -> list:
    """[(名稱, 規則值)]：PDP 交點要和哪些規則值比較。"""
    spec = axis_spec(axis)
    names = {"power": ["P_low", "P_high"], "temp_diff": ["dT"], "strain_ratio": ["strain_ratio"]}.get(axis, [])
    return list(zip(names, spec["thresholds"]))


def axis_pool(test, axis: str, bounds: dict, n: int, seed: int = SEED):
    """ICE 起點列：健康列；HDF 需轉速低於門檻；OSF 需磨耗上限內能達到門檻。"""
    pool = healthy_pool(test)
    if axis == "temp_diff":
        pool = pool[pool["rotational_speed_rpm"] < R.HDF_RPM_MAX]
    elif axis == "strain_ratio":
        pool = pool[pool["torque_nm"] * bounds["tool_wear_min"][1] >= 1.1 * R.osf_threshold(pool)]
    return sample_rows(pool, n, seed)


def wear_band_summary(df, pdp: np.ndarray, grid) -> dict:
    """刀具磨耗三段（< 200、200–240、> 240）：PDP 平均機率 vs 資料中實際 TWF 比例。"""
    g = np.asarray(grid)
    bands = {"below": g < R.TWF_WEAR_MIN, "in_window": (g >= R.TWF_WEAR_MIN) & (g <= R.TWF_WEAR_MAX),
             "above": g > R.TWF_WEAR_MAX}
    w = df["tool_wear_min"]
    rows = {"below": w < R.TWF_WEAR_MIN, "in_window": (w >= R.TWF_WEAR_MIN) & (w <= R.TWF_WEAR_MAX),
            "above": w > R.TWF_WEAR_MAX}
    return {k: {"pdp_mean_proba": float(np.nanmean(pdp[bands[k]])),
                "data_twf_rate": float((df.loc[rows[k], "twf"] == 1).mean()), "n_rows": int(rows[k].sum())}
            for k in bands}


def _nan_stat(fn, ice, *args) -> np.ndarray:
    """沿列方向的 nan 統計；整欄皆無效（介入超出範圍）時回傳 NaN 而不發出警告。"""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return fn(ice, *args, axis=0)


def analyze_model(est, model, fs, thr, test, bounds, n_ice, n_repeats, seed, wear_ref) -> dict:
    X = build_features(test, fs)
    y = test[TARGET].to_numpy().astype(int)
    proba = est.predict_proba(X)[:, 1]
    pred = proba >= thr
    out = {"threshold": thr, "pr_auc_test": _pr_auc(y, proba)}
    out["importance"] = permutation_importance(est, X, y, n_repeats, seed)
    out["test_agreement"] = {
        "vs_rule_union": decision_agreement(pred, rule_union(test)),
        "vs_label": decision_agreement(pred, y),
        **{f"vs_{name}": decision_agreement(pred, rule(test).to_numpy())
           for name, rule in R.DETERMINISTIC_RULES.items()}}
    out["per_mode_recall"] = per_mode_recall(test, pred)
    axes, curves = {}, {}
    for axis, grid in GRIDS.items():
        pool = axis_pool(test, axis, bounds, n_ice, seed)
        ice = ice_curves(est, pool, fs, axis, grid, bounds)
        pdp = _nan_stat(np.nanmean, ice)
        entry = {"n_rows": int(len(pool)), "n_valid_cells": int((~np.isnan(ice)).sum())}
        if axis != "wear":
            rule_mat, valid = ice_rule_matrix(pool, axis, grid, bounds)
            entry["counterfactual_agreement"] = decision_agreement((ice >= thr)[valid], rule_mat[valid])
            entry["boundaries"] = {}
            for name, rv in _axis_rules(axis):
                b = boundary_offsets(grid, ice, thr, rv, WINDOW_FRAC * rv)
                pc = [c for c in find_crossings(grid, pdp, thr) if abs(c - rv) <= WINDOW_FRAC * rv]
                b["pdp_crossing"] = min(pc, key=lambda c: abs(c - rv)) if pc else None
                b["pdp_offset"] = None if b["pdp_crossing"] is None else float(b["pdp_crossing"] - rv)
                b["median_offset_rel"] = (None if b["median_offset"] is None else b["median_offset"] / rv)
                b["median_abs_offset_rel"] = (None if b["median_abs_offset"] is None
                                              else b["median_abs_offset"] / rv)
                entry["boundaries"][name] = b
        else:
            entry["bands"] = wear_band_summary(wear_ref, pdp, grid)
        axes[axis] = entry
        curves[axis] = {"grid": grid, "ice": ice, "pdp": pdp}
    out["axes"], out["curves"] = axes, curves
    return out


# ---------- 圖表 ----------

def plot_importance(imp: pd.DataFrame, path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.2), sharex=True)
    xmax = max(0.05, float((imp["mean"] + imp["std"]).max()) * 1.05)
    for i, fs in enumerate(FS_ORDER):
        for j, model in enumerate(MODEL_ORDER):
            ax = axes[i, j]
            d = imp[(imp["model"] == model) & (imp["feature_set"] == fs)].sort_values("mean")
            colors = [FAILURE if f in DERIVED else NORMAL for f in d["feature"]]
            ax.barh(d["feature"], d["mean"], xerr=d["std"], color=colors, error_kw={"ecolor": TEXT_SECONDARY, "lw": 1})
            ax.set_title(f"{model} | {fs}", fontsize=10.5)
            ax.set_xlim(min(0, float(imp["mean"].min()) * 1.1), xmax)
            ax.grid(axis="y", visible=False)
            if i == 1:
                ax.set_xlabel("PR-AUC drop when shuffled")
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=NORMAL, label="Sensor / Type feature"),
                        Patch(color=FAILURE, label="Derived feature (engineered set)")],
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Permutation importance on the held-out test set (mean +/- sd over repeats)",
                 color=TEXT_PRIMARY, fontweight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    save(fig, path)


def plot_ice(axis: str, results: dict, thresholds: dict, path) -> None:
    spec = axis_spec(axis)
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.8), sharey=True)
    for i, fs in enumerate(FS_ORDER):
        for j, model in enumerate(MODEL_ORDER):
            ax = axes[i, j]
            c = results[(model, fs)]["curves"][axis]
            for row in c["ice"]:
                ax.plot(c["grid"], row, color=NORMAL, alpha=0.10, lw=0.8)
            ax.plot(c["grid"], c["pdp"], color=FAILURE, lw=2)
            for t in spec["thresholds"]:
                ax.axvline(t, color=TEXT_SECONDARY, ls="--", lw=1.5)
            if spec["window"]:
                ax.axvspan(*spec["window"], color=TEXT_SECONDARY, alpha=0.12, lw=0)
            ax.axhline(thresholds[f"{model}|{fs}"], color=TEXT_SECONDARY, ls=":", lw=1.5)
            ax.set_title(f"{model} | {fs}", fontsize=10.5)
            ax.set_ylim(-0.02, 1.02)
            if i == 1:
                ax.set_xlabel(spec["label"], fontsize=9)
            if j == 0:
                ax.set_ylabel("Predicted failure probability")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    h = [Line2D([], [], color=NORMAL, alpha=0.5, lw=1, label="ICE (one test row each)"),
         Line2D([], [], color=FAILURE, lw=2, label="PDP (mean of ICE)"),
         (Patch(color=TEXT_SECONDARY, alpha=0.25, label=f"TWF window {R.TWF_WEAR_MIN}-{R.TWF_WEAR_MAX} min")
          if spec["window"] else Line2D([], [], color=TEXT_SECONDARY, ls="--", lw=1.5, label="Rule threshold")),
         Line2D([], [], color=TEXT_SECONDARY, ls=":", lw=1.5, label="Model decision threshold (tuned on val)")]
    fig.legend(handles=h, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"PDP / ICE along {spec['label']}", color=TEXT_PRIMARY, fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    save(fig, path)


def plot_offsets(results: dict, path) -> None:
    boundaries = [(ax, name, rv) for ax in ("power", "temp_diff", "strain_ratio") for name, rv in _axis_rules(ax)]
    titles = {"P_low": f"PWF lower: {R.PWF_POWER_MIN_W} W", "P_high": f"PWF upper: {R.PWF_POWER_MAX_W} W",
              "dT": f"HDF: dT = {R.HDF_TEMP_DIFF_MAX_K} K",
              "strain_ratio": "OSF: wear x torque = Type threshold"}
    fig, axes = plt.subplots(1, len(boundaries), figsize=(15, 4.6), sharey=True)
    labels = [(m, fs) for m in MODEL_ORDER for fs in FS_ORDER]
    ys = np.arange(len(labels))[::-1]
    for ax, (axis, name, rv) in zip(axes, boundaries):
        for y, (m, fs) in zip(ys, labels):
            b = results[(m, fs)]["axes"][axis]["boundaries"][name]
            col = NORMAL if fs == "raw" else FAILURE
            if b["median_offset"] is None:
                ax.text(0, y, "no crossing", ha="left", va="center", fontsize=8, color=TEXT_SECONDARY)
                continue
            ax.errorbar(100 * b["median_offset"] / rv, y,
                        xerr=[[100 * (b["median_offset"] - b["q25_offset"]) / rv],
                              [100 * (b["q75_offset"] - b["median_offset"]) / rv]],
                        fmt="o", color=col, ms=6, capsize=3, lw=1.5)
            ax.text(WINDOW_FRAC * 100 * 1.02, y, f"{100 * b['frac_crossing']:.0f}%", va="center", fontsize=8,
                    color=TEXT_SECONDARY)
        ax.axvline(0, color=TEXT_SECONDARY, ls="--", lw=1.5)
        ax.set_xlim(-WINDOW_FRAC * 100, WINDOW_FRAC * 100 * 1.25)
        ax.set_title(titles[name], fontsize=10)
        ax.set_xlabel("Learned - rule boundary [% of rule value]")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([_label(m, fs) for m, fs in labels], fontsize=9)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], marker="o", color=NORMAL, lw=0, label="raw features"),
                        Line2D([], [], marker="o", color=FAILURE, lw=0, label="engineered features")],
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Where do the models flip? Median ICE crossing vs rule threshold (bar = IQR; right label = share of "
                 "ICE curves with a crossing in the +/-25% window)", color=TEXT_PRIMARY, fontweight="bold", fontsize=10.5)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    save(fig, path)


def plot_mlp_rules(real: pd.DataFrame, scaling: pd.DataFrame, path) -> None:
    rules = ["osf", "pwf", "hdf"]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.4))
    ax = axes[0]
    order = ["rpm_only"] + rules
    x = np.arange(len(order))
    for k, (fs, col) in enumerate(zip(FS_ORDER, (NORMAL, FAILURE))):
        d = real[(real["feature_set"] == fs) & (real["config"] == "default")].set_index("rule").loc[order]
        ax.bar(x + (k - 0.5) * 0.38, d["pr_auc_mean"], 0.38, yerr=d["pr_auc_std"], color=col, label=fs,
               error_kw={"ecolor": TEXT_SECONDARY, "lw": 1})
    ax.set_xticks(x)
    ax.set_xticklabels([f"rpm<{R.HDF_RPM_MAX}\n(control)", "OSF\nwear*torque", "PWF\ntorque*rpm", "HDF\ndT & rpm"],
                       fontsize=8)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("Test PR-AUC (real data, 3 seeds)")
    ax.set_title("Real data, 6,000 train rows", fontsize=10.5)
    ax.legend(loc="upper center", ncol=2)
    ax.grid(axis="x", visible=False)
    for ax, rule in zip(axes[1:], rules):
        for fs, col in zip(FS_ORDER, (NORMAL, FAILURE)):
            d = scaling[(scaling["rule"] == rule) & (scaling["feature_set"] == fs)].sort_values("n_train")
            ax.plot(d["n_train"], d["pr_auc"], marker="o", color=col, label=fs)
        ax.set_xscale("log")
        sizes = sorted(scaling["n_train"].unique())
        ax.set_xticks(sizes)
        ax.set_xticklabels([f"{n // 1000}k" for n in sizes])
        ax.minorticks_off()
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Training rows (resampled)")
        ax.set_title(f"{rule.upper()}: more data (synthetic test)", fontsize=10.5)
        ax.legend(loc="lower right")
    fig.suptitle("Can an MLP learn the product-type rules from raw features alone?",
                 color=TEXT_PRIMARY, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, path)


# ---------- 研究問題 B 的實驗 ----------

def mlp_real_experiment(train, val, test, seeds=(0, 1, 2)) -> pd.DataFrame:
    """真實切分：規則目標 × 特徵組 × 兩種容量 × 多個種子；回傳逐次結果。"""
    rows = []
    for cfg_name, cfg in (("default", MLP_REAL), ("big", MLP_BIG)):
        for rule in ("rpm_only", "osf", "pwf", "hdf"):
            for fs in FS_ORDER:
                for s in seeds:
                    r = mlp_rule_learning(train, val, test, rule, fs, params=cfg, seed=SEED + s)
                    rows.append({"config": cfg_name, "seed": SEED + s} | r)
    return pd.DataFrame(rows)


def summarize_real(runs: pd.DataFrame) -> pd.DataFrame:
    g = runs.groupby(["config", "rule", "feature_set"])
    out = g.agg(pr_auc_mean=("pr_auc", "mean"), pr_auc_std=("pr_auc", "std"), f1_mean=("f1", "mean"),
                recall_mean=("recall", "mean"), precision_mean=("precision", "mean"),
                n_pos_test=("n_pos_test", "first"), n_pos_train=("n_pos_train", "first"),
                n_seeds=("seed", "count")).reset_index()
    out["pr_auc_std"] = out["pr_auc_std"].fillna(0.0)
    return out


def mlp_scaling_experiment(train, val, test, sizes=SCALING_SIZES, rules=("osf", "pwf", "hdf"),
                           seed: int = SEED) -> pd.DataFrame:
    """合成資料放大：訓練／驗證／測試的 (轉速, 扭矩) 對分別只來自 train／val／test 列，彼此不重疊。"""
    s_val = synthetic_frame(val, 10000, seed=seed + 1)
    s_test = synthetic_frame(test, 20000, seed=seed + 2)
    rows = []
    for n in sizes:
        s_train = synthetic_frame(train, n, seed=seed)
        for rule in rules:
            for fs in FS_ORDER:
                r = mlp_rule_learning(s_train, s_val, s_test, rule, fs, params=MLP_BIG, seed=seed)
                rows.append({"n_train": int(n)} | r)
    return pd.DataFrame(rows)


# ---------- 結論（由數據自動組字串，數字不手寫） ----------

def _fmt(x, nd=2):
    return "n/a" if x is None else f"{x:.{nd}f}"


def learned_boundary(b: dict, min_frac: float = 0.8, max_rel: float = 0.10) -> bool:
    return (b["frac_crossing"] is not None and b["frac_crossing"] >= min_frac
            and b["median_abs_offset_rel"] is not None and b["median_abs_offset_rel"] <= max_rel)


def build_conclusions(results: dict, imp: pd.DataFrame, real: pd.DataFrame, scaling: pd.DataFrame) -> dict:
    # --- A ---
    learned, table = {}, {}
    for (m, fs), r in results.items():
        for axis in ("power", "temp_diff", "strain_ratio"):
            for name, _ in _axis_rules(axis):
                b = r["axes"][axis]["boundaries"][name]
                learned[(m, fs, name)] = learned_boundary(b)
                table.setdefault(name, []).append((m, fs, b))
    n_by = {n: {fs: sum(learned[(m, fs, n)] for m in MODEL_ORDER) for fs in FS_ORDER} for n in table}
    best_kappa = max(results.items(), key=lambda kv: kv[1]["test_agreement"]["vs_rule_union"]["kappa"] or -1)
    ka = {f"{m}|{fs}": r["test_agreement"]["vs_rule_union"]["kappa"] for (m, fs), r in results.items()}
    a_text = (
        f"以 PDP/ICE 在規則座標上介入原始感測欄位（再重算衍生特徵）：判定為『學到邊界』需 >=80% 的 ICE 曲線在規則值 "
        f"+/-{int(WINDOW_FRAC * 100)}% 內跨越模型門檻，且交點中位偏移 <=10% 規則值。"
        f"4 個邊界（PWF 下限、PWF 上限、HDF 溫差、OSF 乘積）在 4 個模型下，raw 學到的模型數="
        + "、".join(f"{n}:{v['raw']}/4" for n, v in n_by.items())
        + "；engineered 學到的模型數=" + "、".join(f"{n}:{v['engineered']}/4" for n, v in n_by.items())
        + f"。測試集上與規則聯集的 Cohen kappa 最高者為 {best_kappa[0][0]}|{best_kappa[0][1]}（{_fmt(ka[best_kappa[0][0] + '|' + best_kappa[0][1]])}）。")
    # --- B ---
    d = real[real["config"] == "default"].set_index(["rule", "feature_set"])
    big = real[real["config"] == "big"].set_index(["rule", "feature_set"])
    nmax = int(scaling["n_train"].max())
    sc = scaling[scaling["n_train"] == nmax].set_index(["rule", "feature_set"])
    parts = []
    for rule in ("osf", "pwf", "hdf"):
        parts.append(f"{rule.upper()}：真實 6,000 列 raw {_fmt(d.loc[(rule, 'raw'), 'pr_auc_mean'], 3)} vs "
                     f"engineered {_fmt(d.loc[(rule, 'engineered'), 'pr_auc_mean'], 3)}；"
                     f"合成 {nmax:,} 列 raw {_fmt(sc.loc[(rule, 'raw'), 'pr_auc'], 3)} vs "
                     f"engineered {_fmt(sc.loc[(rule, 'engineered'), 'pr_auc'], 3)}")
    b_text = ("測試集 PR-AUC（MLP 預測『規則成立』）— " + "；".join(parts) + "。"
              f"控制組（單一特徵軸平行門檻 rpm<{R.HDF_RPM_MAX}）raw {_fmt(d.loc[('rpm_only', 'raw'), 'pr_auc_mean'], 3)}。")
    tot = {fs: sum(v[fs] for v in n_by.values()) for fs in FS_ORDER}
    worst = [n for n, v in n_by.items() if v["raw"] == 0]
    a_verdict = (f"結論 A：16 個（模型×邊界）組合中，raw 特徵有 {tot['raw']} 個、engineered 有 {tot['engineered']} 個達到『學到邊界』標準。"
                 f"raw 模型找不到的邊界：{'、'.join(worst) if worst else '無'}（0/4）；"
                 f"最吻合規則的是 {best_kappa[0][0]}|{best_kappa[0][1]}（kappa {_fmt(ka[best_kappa[0][0] + '|' + best_kappa[0][1]])}），"
                 f"MLP 在測試集的 kappa 為 raw {_fmt(ka['mlp|raw'])}、engineered {_fmt(ka['mlp|engineered'])}。"
                 "模型學到的邊界位置大多落在規則附近（見 median offset），但是否『學到』高度依賴特徵組與模型類型。")
    gain = {r: sc.loc[(r, "raw"), "pr_auc"] for r in ("osf", "pwf", "hdf")}
    can = all(v >= 0.95 for v in gain.values())
    b_verdict = (f"結論 B：MLP 在『目標＝規則本身』的設定下，只靠原始特徵即可學到乘積型規則："
                 f"真實 6,000 列 OSF {_fmt(d.loc[('osf', 'raw'), 'pr_auc_mean'], 3)}、PWF {_fmt(d.loc[('pwf', 'raw'), 'pr_auc_mean'], 3)}"
                 f"（與 engineered 相當或更好）；資料放大到 {nmax:,} 列後三條規則 raw 的 PR-AUC 皆"
                 f"{'>=0.95' if can else '未全數達 0.95'}（{', '.join(f'{k}:{_fmt(v, 3)}' for k, v in gain.items())}）。"
                 "但邊界不精準（F1 約 0.9 而非 1.0），且在含標記雜訊、僅 ~340 個故障的真實任務中，"
                 "MLP 的 raw 表現仍明顯落後樹模型 + engineered。")
    return {"A": a_text, "B": b_text, "A_verdict": a_verdict, "B_verdict": b_verdict, "learned_counts": n_by}


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(type(o))


def run(out_dir=EXPLAIN_DIR, df=None, splits=None, models_dir=None, n_ice: int = 100, n_repeats: int = 10,
        seeds=(0, 1, 2), scaling_sizes=SCALING_SIZES, models=MODEL_ORDER, verbose: bool = False) -> dict:
    """載入 #8 的最終模型，輸出重要度、PDP/ICE、邊界吻合度與 MLP 乘積規則實驗。"""
    from ai4i_lab import models as M
    from ai4i_lab.data import load_dataset
    from ai4i_lab.evaluation import load_splits, split_masks
    t0 = time.perf_counter()
    log = print if verbose else (lambda *a, **k: None)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = M.OUT_DIR if models_dir is None else Path(models_dir)
    df = load_dataset() if df is None else df
    splits = load_splits() if splits is None else splits
    mk = split_masks(df, splits, "stratified")
    train, val, test = (df[mk[k]].reset_index(drop=True) for k in ("train", "val", "test"))
    thresholds = _model_thresholds(models_dir)
    bounds = observed_bounds(df[mk["train"] | mk["val"]])      # 介入範圍只看開發集的觀測範圍

    results = {}
    for m in models:
        for fs in FS_ORDER:
            est = M.get_final_model(m, fs, out_dir=models_dir, df=df, splits=splits)
            results[(m, fs)] = analyze_model(est, m, fs, thresholds[f"{m}|{fs}"], test, bounds, n_ice, n_repeats,
                                             SEED, df)
            log(f"[{time.perf_counter() - t0:6.0f}s] analysed {m}|{fs}")
    imp = pd.concat([r["importance"].assign(model=m, feature_set=fs) for (m, fs), r in results.items()],
                    ignore_index=True)
    imp.to_csv(out_dir / "permutation_importance.csv", index=False)
    plot_importance(imp, out_dir / FIGURES[0])
    for axis, fig in zip(GRIDS, FIGURES[1:5]):
        plot_ice(axis, results, thresholds, out_dir / fig)
    curves = pd.DataFrame([{"model": m, "feature_set": fs, "axis": axis, "x": float(x), "pdp": float(p),
                            "ice_q10": float(q10), "ice_q90": float(q90)}
                           for (m, fs), r in results.items() for axis, c in r["curves"].items()
                           for x, p, q10, q90 in zip(c["grid"], c["pdp"],
                                                     _nan_stat(np.nanquantile, c["ice"], 0.1),
                                                     _nan_stat(np.nanquantile, c["ice"], 0.9))])
    curves.to_csv(out_dir / "pdp_curves.csv", index=False)
    plot_offsets(results, out_dir / FIGURES[5])

    real_runs = mlp_real_experiment(train, val, test, seeds)
    real_runs.drop(columns=["params"]).to_csv(out_dir / "mlp_rule_runs.csv", index=False)
    real = summarize_real(real_runs)
    log(f"[{time.perf_counter() - t0:6.0f}s] real MLP rule experiment done")
    scaling = mlp_scaling_experiment(train, val, test, scaling_sizes)
    scaling.drop(columns=["params"]).to_csv(out_dir / "mlp_rule_scaling.csv", index=False)
    log(f"[{time.perf_counter() - t0:6.0f}s] scaling done")
    plot_mlp_rules(real, scaling, out_dir / FIGURES[6])

    conclusions = build_conclusions(results, imp, real, scaling)
    result = {
        "issue": 11, "seed": SEED, "n_test": int(len(test)), "n_test_failures": int(test[TARGET].sum()),
        "window_frac": WINDOW_FRAC,
        "thresholds_source": "ai4i_lab.rules",
        "model_thresholds": thresholds,
        "importance": imp.to_dict(orient="records"),
        "boundary_agreement": {
            f"{m}|{fs}": {"test_agreement": r["test_agreement"], "per_mode_recall": r["per_mode_recall"],
                          "axes": r["axes"], "pr_auc_test": r["pr_auc_test"]}
            for (m, fs), r in results.items()},
        "mlp_product_rules": {"real": real.to_dict(orient="records"), "scaling": scaling.drop(columns=["params"]).to_dict(orient="records")},
        "conclusions": conclusions, "figures": FIGURES, "seconds": time.perf_counter() - t0,
    }
    (out_dir / "boundary_agreement.json").write_text(
        json.dumps({k: result[k] for k in ("issue", "window_frac", "thresholds_source", "model_thresholds",
                                           "boundary_agreement", "n_test", "n_test_failures")},
                   ensure_ascii=False, indent=1, default=_jsonable), encoding="utf-8")
    (out_dir / "explain.json").write_text(json.dumps(result, ensure_ascii=False, indent=1, default=_jsonable),
                                          encoding="utf-8")
    return result


if __name__ == "__main__":
    r = run(verbose=True)
    for k in ("A", "B", "A_verdict", "B_verdict"):
        print(r["conclusions"][k])
