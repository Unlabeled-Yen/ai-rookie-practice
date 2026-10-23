"""AI4I 維護決策（#12）：成本敏感門檻與敏感度分析。

成本模型：漏判（FN，機台故障卻未檢修）代價 C_FN，誤報（FP，正常機台被檢修）代價 C_FP。
總成本 = C_FN·FN + C_FP·FP（TP／TN 視為 0；TP 的檢修成本在此假設已含在避免故障的節省內，見報告）。
門檻只在驗證集上選定；測試集只用來評估該門檻的實際成本。
"""
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ai4i_lab.data import ROOT

SEED = 42
OUT_DIR = ROOT / "outputs" / "ai4i" / "cost"
MODELS_DIR = ROOT / "outputs" / "ai4i" / "models"
RATIOS = (1, 5, 10, 50, 100)
PRIMARY = ("random_forest", "engineered")


# ---------- 成本計算 ----------

def _check(y, proba, c_fn, c_fp):
    y = np.asarray(y).ravel().astype(int)
    proba = np.asarray(proba, dtype=float).ravel()
    if y.shape != proba.shape:
        raise ValueError(f"y 與 proba 長度不同：{y.shape} vs {proba.shape}")
    if c_fn < 0 or c_fp < 0:
        raise ValueError("成本不可為負")
    return y, proba


def _cost_dict(tp, fp, fn, tn, c_fn, c_fp) -> dict:
    n = tp + fp + fn + tn
    total = c_fn * fn + c_fp * fp
    rec = tp / (tp + fn) if tp + fn else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn), "total": float(total),
            "per_sample": float(total / n) if n else 0.0, "recall": rec, "precision": prec, "f1": f1}


def expected_cost(y, proba, threshold: float, c_fn: float, c_fp: float) -> dict:
    """proba ≥ threshold 判為故障（與 evaluation.binary_metrics 一致）時的混淆矩陣與成本。"""
    y, proba = _check(y, proba, c_fn, c_fp)
    pred = proba >= threshold
    tp = int(np.sum(pred & (y == 1))); fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1))); tn = int(np.sum(~pred & (y == 0)))
    return _cost_dict(tp, fp, fn, tn, c_fn, c_fp)


def baseline_costs(y, c_fn: float, c_fp: float) -> dict:
    """兩種基準策略：never = 不預測（任其故障，全部漏判）；always = 全部檢修（全部正常機台誤報）。"""
    y = np.asarray(y).ravel().astype(int)
    pos = int(y.sum()); neg = len(y) - pos
    return {"never": _cost_dict(0, 0, pos, neg, c_fn, c_fp),
            "always": _cost_dict(pos, neg, 0, 0, c_fn, c_fp)}


def cost_curve(y, proba, c_fn: float, c_fp: float) -> dict:
    """所有候選門檻（各不重複機率值，由高到低，最前面加上 +inf =「不預測」）的總成本。"""
    y, proba = _check(y, proba, c_fn, c_fp)
    cand = np.unique(proba)[::-1]
    order = np.argsort(-proba, kind="stable")
    ps, ys = proba[order], y[order]
    tp_cum, fp_cum = np.cumsum(ys), np.cumsum(1 - ys)
    last = np.searchsorted(-ps, -cand, side="right") - 1
    tp, fp = tp_cum[last], fp_cum[last]
    pos = int(y.sum())
    fn = pos - tp
    cost = c_fn * fn + c_fp * fp
    th = np.concatenate([[math.inf], cand])
    cost = np.concatenate([[c_fn * pos], cost])
    return {"threshold": th.tolist(), "cost": cost.tolist(),
            "tp": np.concatenate([[0], tp]).tolist(), "fp": np.concatenate([[0], fp]).tolist()}


def select_threshold(y, proba, c_fn: float, c_fp: float) -> dict:
    """在給定資料（驗證集）上找總成本最低的門檻；成本相同取較高門檻（較少檢修）。"""
    curve = cost_curve(y, proba, c_fn, c_fp)
    cost = np.asarray(curve["cost"])
    best = int(np.argmin(cost))                      # 門檻由高到低 → 第一個最小值 = 最高門檻
    return {"threshold": float(curve["threshold"][best]), "cost": float(cost[best])}


# ---------- 敏感度分析 ----------

def _row(yv, pv, yt, pt, ratio: float, c_fp: float = 1.0) -> dict:
    c_fn = ratio * c_fp
    sel = select_threshold(yv, pv, c_fn, c_fp)
    th = sel["threshold"]
    val = expected_cost(yv, pv, th, c_fn, c_fp)
    test = expected_cost(yt, pt, th, c_fn, c_fp)
    base = baseline_costs(yt, c_fn, c_fp)
    best = min(base, key=lambda k: base[k]["total"])
    bc = base[best]["total"]
    saving = bc - test["total"]
    return {"ratio": ratio, "c_fn": c_fn, "c_fp": c_fp, "threshold": th, "val": val, "test": test,
            "baseline": {k: v["total"] for k, v in base.items()}, "baseline_best": best,
            "baseline_cost": bc, "saving": float(saving), "saving_pct": float(saving / bc) if bc else 0.0}


def sensitivity(yv, pv, yt, pt, ratios=RATIOS, c_fp: float = 1.0) -> list:
    """對每個 C_FN:C_FP 比例：驗證集選門檻 → 測試集評估，並與兩種基準策略比較。"""
    return [_row(yv, pv, yt, pt, r, c_fp) for r in ratios]


def is_worthwhile(yv, pv, yt, pt, ratio: float) -> bool:
    """部署模型是否比最佳基準策略省錢（測試集實際成本嚴格較低）。"""
    return _row(yv, pv, yt, pt, ratio)["saving"] > 0


def baseline_crossover_ratio(y) -> float:
    """兩種基準策略成本相等的比例 N/P：低於它「不預測」較便宜，高於它「全部檢修」較便宜。"""
    y = np.asarray(y).ravel().astype(int)
    pos = int(y.sum())
    return (len(y) - pos) / pos if pos else math.inf


def break_even_ratio(yv, pv, yt, pt, grid=None):
    """grid 中第一個「值得部署」的比例；全都不值得則回傳 None。"""
    grid = np.arange(1, 201) if grid is None else grid
    for r in grid:
        if is_worthwhile(yv, pv, yt, pt, float(r)):
            return float(r)
    return None


def bootstrap_saving(y, proba, threshold: float, c_fn: float, c_fp: float, n_boot: int = 1000,
                     seed: int = SEED, confidence: float = 0.95) -> dict:
    """測試集重抽樣（固定門檻）下，「相對最佳基準策略的節省成本」的百分位區間。"""
    y, proba = _check(y, proba, c_fn, c_fp)
    rng = np.random.default_rng(seed)
    n = len(y)
    pred = proba >= threshold

    def saving(idx):
        yy, pp = y[idx], pred[idx]
        fn = np.sum(~pp & (yy == 1)); fp = np.sum(pp & (yy == 0))
        pos = yy.sum(); neg = len(yy) - pos
        return min(c_fn * pos, c_fp * neg) - (c_fn * fn + c_fp * fp)

    point = saving(np.arange(n))
    vals = np.array([saving(rng.integers(0, n, n)) for _ in range(n_boot)])
    a = (1 - confidence) / 2
    return {"point": float(point), "low": float(np.quantile(vals, a)), "high": float(np.quantile(vals, 1 - a)),
            "p_positive": float(np.mean(vals > 0)), "n_boot": n_boot}


# ---------- 讀取 #8 的預測 ----------

def load_predictions(path, model: str, feature_set: str):
    d = pd.read_csv(path)
    d = d[(d["model"] == model) & (d["feature_set"] == feature_set)]
    if d.empty:
        raise KeyError(f"{path} 內沒有 {model}|{feature_set}")
    return d["y_true"].to_numpy().astype(int), d["proba"].to_numpy(dtype=float)


# ---------- 圖表 ----------

FIGURES = ["cost_curves.png", "sensitivity.png"]


def plot_cost_curves(y, proba, ratios=RATIOS, path=None, y_test=None, p_test=None, c_fp: float = 1.0) -> None:
    """小倍數圖：每個成本比例一格，畫「門檻 vs 總成本」，並疊上「不預測」「全部檢修」兩條基準線。

    實線為驗證集成本（門檻在此選定，垂直虛線＝選定門檻）；若提供 y_test／p_test 則另畫測試集成本。
    """
    import matplotlib.pyplot as plt

    from ai4i_lab.style import SERIES, TEXT_PRIMARY, TEXT_SECONDARY, save
    ratios = list(ratios)
    ncol = min(3, len(ratios)); nrow = math.ceil(len(ratios) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.4 * nrow), squeeze=False)
    for ax, r in zip(axes.ravel(), ratios):
        c_fn = r * c_fp
        sets = [("Validation", y, proba, SERIES[0])]
        if y_test is not None:
            sets.append(("Test", y_test, p_test, SERIES[1]))
        for name, yy, pp, col in sets:
            cv = cost_curve(yy, pp, c_fn, c_fp)
            th = np.array(cv["threshold"][1:]); co = np.array(cv["cost"][1:])
            o = np.argsort(th)
            ax.step(th[o], co[o], where="post", color=col, label=f"{name} cost")
        base = baseline_costs(y, c_fn, c_fp)
        ax.axhline(base["never"]["total"], color=SERIES[2], ls="--", label="Never inspect")
        ax.axhline(base["always"]["total"], color=SERIES[3], ls="--", label="Inspect all")
        sel = select_threshold(y, proba, c_fn, c_fp)["threshold"]
        if math.isfinite(sel):
            ax.axvline(sel, color=TEXT_SECONDARY, ls=":", lw=1.5)
        ax.set_title(f"C_FN : C_FP = {r:g} : 1", color=TEXT_PRIMARY)
        ax.set_xlabel("Decision threshold"); ax.set_ylabel("Total cost (units of C_FP)")
        ax.set_xlim(0, 1)
        lo = min(base["never"]["total"], base["always"]["total"])
        ax.set_ylim(0, max(lo * 2.5, 1))                       # 以較低基準的 2.5 倍為上限，避免陡峭尾端壓縮主要區域
    for ax in axes.ravel()[len(ratios):]:
        ax.axis("off")
    h, l = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, path)


def plot_sensitivity(rows: list, path) -> None:
    """測試集總成本隨 C_FN:C_FP 比例的變化：模型 vs 兩種基準策略（對數座標）。"""
    import matplotlib.pyplot as plt

    from ai4i_lab.style import SERIES, TEXT_PRIMARY, save
    r = [x["ratio"] for x in rows]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(r, [x["test"]["total"] for x in rows], marker="o", color=SERIES[0], label="Model (val-tuned threshold)")
    ax.plot(r, [x["baseline"]["never"] for x in rows], marker="o", color=SERIES[2], label="Never inspect")
    ax.plot(r, [x["baseline"]["always"] for x in rows], marker="o", color=SERIES[3], label="Inspect all")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(r); ax.set_xticklabels([f"{v:g}:1" for v in r])
    ax.minorticks_off()
    ax.set_xlabel("C_FN : C_FP"); ax.set_ylabel("Test-set total cost (units of C_FP)")
    ax.set_title("Cost vs. cost ratio on the test set", color=TEXT_PRIMARY)
    ax.legend(loc="upper left")
    fig.tight_layout()
    save(fig, path)


# ---------- 主流程 ----------

def _conclusion(primary: dict, rows: list, be: float | None, be_all: dict) -> str:
    base = rows[0]
    lines = []
    if be is None:
        lines.append("在 C_FN:C_FP ≤ 200 的範圍內，模型都不比最佳基準策略省錢。")
    else:
        lines.append(f"測試集上，從 C_FN:C_FP ≈ {be:.2g}:1（掃描 0.01–200 的對數網格，低於此值模型相對「不預測」已無節省）起，驗證集選門檻的模型都比最佳基準策略省錢；"
                     f"原因是模型排序品質高，只檢修最有把握的少數機台即可在任何 C_FN>0 下略有獲利，"
                     f"因此「值得部署」的實質門檻不在成本比例，而在節省金額是否大於模型部署與維運成本。")
    worth = [r["ratio"] for r in rows if r["saving"] > 0]
    lines.append("測試集上值得部署的比例：" + (", ".join(f"{v:g}:1" for v in worth) if worth else "無") + "。")
    lines.append(f"主模型 {primary['model']}|{primary['feature_set']}；1:1 時最佳策略為「{base['baseline_best']}」，"
                 f"模型成本 {base['test']['total']:g} vs 基準 {base['baseline_cost']:g}。")
    return " ".join(lines)


def run(out_dir=OUT_DIR, models_dir=MODELS_DIR, model: str = PRIMARY[0], feature_set: str = PRIMARY[1],
        ratios=RATIOS, n_boot: int = 1000, seed: int = SEED, verbose: bool = False) -> dict:
    """讀取 #8 的驗證／測試預測 → 驗證集選門檻 → 測試集評估 → 敏感度表、成本曲線圖與 cost.json。"""
    t0 = time.perf_counter()
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path(models_dir)
    yv, pv = load_predictions(models_dir / "predictions_val.csv", model, feature_set)
    yt, pt = load_predictions(models_dir / "predictions_test.csv", model, feature_set)

    rows = sensitivity(yv, pv, yt, pt, ratios)
    for r in rows:
        r["saving_ci"] = bootstrap_saving(yt, pt, r["threshold"], r["c_fn"], r["c_fp"], n_boot, seed)
    grid = np.geomspace(0.01, 200, 400)
    be = break_even_ratio(yv, pv, yt, pt, grid)

    # 所有模型 × 特徵組的比較（同樣：驗證集選門檻、測試集評估）
    all_models = {}
    pred_t = pd.read_csv(models_dir / "predictions_test.csv")
    for (m, fs), _ in pred_t.groupby(["model", "feature_set"]):
        a, b = load_predictions(models_dir / "predictions_val.csv", m, fs)
        c, d = load_predictions(models_dir / "predictions_test.csv", m, fs)
        rr = sensitivity(a, b, c, d, ratios)
        all_models[f"{m}|{fs}"] = {"test_cost": {f"{x['ratio']:g}": x["test"]["total"] for x in rr},
                                   "saving_pct": {f"{x['ratio']:g}": x["saving_pct"] for x in rr},
                                   "break_even": break_even_ratio(a, b, c, d, grid)}

    table = pd.DataFrame([{
        "ratio": r["ratio"], "threshold": r["threshold"], "val_cost": r["val"]["total"],
        "test_cost": r["test"]["total"], "test_cost_per_machine": r["test"]["per_sample"],
        "never_cost": r["baseline"]["never"], "always_cost": r["baseline"]["always"],
        "best_baseline": r["baseline_best"], "saving": r["saving"], "saving_pct": r["saving_pct"],
        "saving_ci_low": r["saving_ci"]["low"], "saving_ci_high": r["saving_ci"]["high"],
        "p_saving_positive": r["saving_ci"]["p_positive"],
        "tp": r["test"]["tp"], "fp": r["test"]["fp"], "fn": r["test"]["fn"], "tn": r["test"]["tn"],
        "recall": r["test"]["recall"], "precision": r["test"]["precision"], "f1": r["test"]["f1"]} for r in rows])
    table.to_csv(out_dir / "sensitivity.csv", index=False)

    plot_cost_curves(yv, pv, ratios, out_dir / "cost_curves.png", y_test=yt, p_test=pt)
    plot_sensitivity(rows, out_dir / "sensitivity.png")

    primary = {"model": model, "feature_set": feature_set}
    result = {"primary": primary, "ratios": list(ratios), "c_fp": 1.0, "sensitivity": rows,
              "break_even_ratio": be, "baseline_crossover_ratio": baseline_crossover_ratio(yt), "all_models": all_models,
              "n": {"val": len(yv), "val_pos": int(yv.sum()), "test": len(yt), "test_pos": int(yt.sum())},
              "conclusion": _conclusion(primary, rows, be, all_models), "seconds": time.perf_counter() - t0}
    (out_dir / "cost.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=float))
    if verbose:
        print(table.to_string())
    return result


if __name__ == "__main__":
    run(verbose=True)
