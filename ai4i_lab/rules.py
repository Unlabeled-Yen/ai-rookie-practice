"""AI4I 2020 已知故障規則重建（#5）：研究問題 A 的「標準答案」。

依 Matzka (2020) / UCI 資料說明，以原始特徵重算 TWF/HDF/PWF/OSF。
門檻以常數公開，供下游 PDP 對照與錯誤分析使用。
執行：python -m ai4i_lab.rules
"""
import numpy as np
import pandas as pd

# ---------- 規則門檻常數 ----------
HDF_TEMP_DIFF_MAX_K = 8.6        # 溫差（process − air）< 8.6 K
HDF_RPM_MAX = 1380               # 且轉速 < 1380 rpm
PWF_POWER_MIN_W = 3500          # 功率 < 3500 W
PWF_POWER_MAX_W = 9000          # 或 > 9000 W
OSF_STRAIN_MAX_MINNM = {"L": 11000, "M": 12000, "H": 13000}  # 磨耗 × 扭矩 > 門檻（依 Type）
TWF_WEAR_MIN = 200              # 刀具在 200–240 min 間的隨機時點失效（換刀或故障）
TWF_WEAR_MAX = 240


def temp_diff_k(df) -> pd.Series:
    """製程溫度 − 空氣溫度 [K]，以原始浮點減法計算（刻意不四捨五入）。

    資料中有 27 列的溫差「名義上」恰為 8.6 K 且轉速 < 1380；其 HDF 標記完全由 IEEE-754
    減法誤差決定（8.5999…97 → 1，8.6000…02 → 0）。產生器顯然直接以浮點值比較，
    因此本函式保留浮點誤差才能 100% 重現標記。
    """
    return df["process_temp_k"] - df["air_temp_k"]


def hdf_rule(df) -> pd.Series:
    """散熱不良：溫差 < 8.6 K 且轉速 < 1380 rpm（兩者皆為嚴格不等式）。"""
    return (temp_diff_k(df) < HDF_TEMP_DIFF_MAX_K) & (df["rotational_speed_rpm"] < HDF_RPM_MAX)


def power_w(df) -> pd.Series:
    """機械功率 [W] = 扭矩 [Nm] × 角速度 ω [rad/s]，ω = rpm × 2π/60。"""
    return df["torque_nm"] * df["rotational_speed_rpm"] * 2 * np.pi / 60


def pwf_rule(df) -> pd.Series:
    """功率故障：功率 < 3500 W 或 > 9000 W（嚴格不等式）。"""
    p = power_w(df)
    return (p < PWF_POWER_MIN_W) | (p > PWF_POWER_MAX_W)


def strain_minnm(df) -> pd.Series:
    """過度負荷指標 [min·Nm] = 刀具磨耗 [min] × 扭矩 [Nm]。"""
    return df["tool_wear_min"] * df["torque_nm"]


def osf_threshold(df) -> pd.Series:
    """每列依產品 Type 對應的 OSF 門檻（L 11000 / M 12000 / H 13000 minNm）。"""
    return df["type"].map(OSF_STRAIN_MAX_MINNM).astype(float)


def osf_rule(df) -> pd.Series:
    """過度負荷：磨耗 × 扭矩 > 依 Type 的門檻（嚴格不等式）。"""
    return strain_minnm(df) > osf_threshold(df)


def twf_window(df) -> pd.Series:
    """刀具磨耗位於 TWF 失效窗口 [200, 240] min（含兩端）。

    注意：這不是確定性規則。依資料產生方式，刀具會在窗口內某個「隨機」時點失效，
    該時點刀具被更換（不記故障）或造成故障（TWF=1），因此只能給出必要條件的候選區間。
    """
    w = df["tool_wear_min"]
    return (w >= TWF_WEAR_MIN) & (w <= TWF_WEAR_MAX)


DETERMINISTIC_RULES = {"hdf": hdf_rule, "pwf": pwf_rule, "osf": osf_rule}


def _ratio(num, den):
    return None if den == 0 else float(num / den)


def agreement(pred, label, udi) -> dict:
    """規則預測與資料標記的混淆矩陣、吻合率與不吻合樣本（以 UDI 表示）。"""
    pred = np.asarray(pred, dtype=bool)
    label = np.asarray(label).astype(bool)
    udi = np.asarray(udi)
    tp = int((pred & label).sum())
    fp = int((pred & ~label).sum())
    fn = int((~pred & label).sum())
    tn = int((~pred & ~label).sum())
    return {
        "n": int(len(pred)), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy": float((tp + tn) / len(pred)),
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "false_positive_udi": [int(u) for u in udi[pred & ~label]],
        "false_negative_udi": [int(u) for u in udi[~pred & label]],
    }


def rule_agreement(df) -> dict:
    """各模式規則與標記的吻合率；另比對確定性規則聯集與 machine_failure。"""
    out = {m: agreement(f(df), df[m], df["udi"]) for m, f in DETERMINISTIC_RULES.items()}
    out["twf"] = agreement(twf_window(df), df["twf"], df["udi"])
    out["twf"]["note"] = ("TWF 規則僅為窗口 [200,240] min 的候選條件，非確定性規則；"
                          "precision 低是預期結果，見 tool_cycles 分析。")
    union = hdf_rule(df) | pwf_rule(df) | osf_rule(df)
    u = agreement(union, df["machine_failure"], df["udi"])
    unexplained = df[(~union) & (df["machine_failure"] == 1)]
    u["unexplained_failures"] = {
        "total": int(len(unexplained)),
        "with_twf_label": int((unexplained["twf"] == 1).sum()),
        "with_rnf_label_only": int(((unexplained["rnf"] == 1) & (unexplained["twf"] == 0)).sum()),
        "no_mode_label": int((unexplained[["twf", "hdf", "pwf", "osf", "rnf"]].sum(axis=1) == 0).sum()),
    }
    out["machine_failure_union"] = u
    return out


def tool_cycles(df) -> pd.DataFrame:
    """依 UDI 順序，以刀具磨耗下降（換刀）切分刀具週期，回傳每個週期的最大磨耗與是否出現 TWF。"""
    d = df.sort_values("udi")
    cycle = (d["tool_wear_min"].diff() < 0).cumsum()
    g = d.groupby(cycle.values)
    return pd.DataFrame({
        "cycle": np.arange(g.ngroups),
        "start_udi": g["udi"].first().to_numpy(),
        "n_rows": g.size().to_numpy(),
        "max_wear_min": g["tool_wear_min"].max().to_numpy(),
        "has_twf": g["twf"].max().astype(bool).to_numpy(),
    })


# ---------- 輸出：吻合率 JSON 與規則邊界圖 ----------
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from ai4i_lab.data import ROOT, TYPES, load_dataset  # noqa: E402
from ai4i_lab.style import FAILURE, NORMAL, TEXT_PRIMARY, TEXT_SECONDARY, save  # noqa: E402

RULES_DIR = ROOT / "outputs" / "ai4i" / "rules"
SEED = 42
FIGURES = ["hdf_boundary.png", "pwf_boundary.png", "osf_boundary.png", "twf_window.png"]
_TIE_EPS = 1e-9

TWF_LIMIT_TEXT = (
    "TWF 無法 100% 重建：依產生規則，每把刀在磨耗 200–240 min 間的『隨機』時點失效，"
    "失效時點不是由任何觀測特徵決定，而且當下可能被換刀（不記故障）或造成故障（TWF=1）。"
    "因此特徵只能告訴我們『是否位於窗口內』這個候選條件：窗口內的列遠多於 TWF 事件（precision 低），"
    "且實際資料仍有少數 TWF 落在窗口外（可能為換刀時點紀錄的離散化/產生器實作差異），recall 也不到 100%。"
)


def _ties(values, thresholds) -> int:
    return int((np.abs(np.asarray(values, float) - np.asarray(thresholds, float)) < _TIE_EPS).sum())


def boundary_ties(df) -> dict:
    """數值恰落在規則門檻上的列數（浮點誤差範圍內），用來說明邊界判定的敏感度。"""
    p = power_w(df)
    return {
        "hdf_temp_diff_8_6_and_rpm_lt_1380": _ties(temp_diff_k(df)[df["rotational_speed_rpm"] < HDF_RPM_MAX],
                                                   HDF_TEMP_DIFF_MAX_K),
        "hdf_rpm_1380_and_temp_lt_8_6": _ties(df["rotational_speed_rpm"][temp_diff_k(df) < HDF_TEMP_DIFF_MAX_K],
                                              HDF_RPM_MAX),
        "pwf_power_3500_or_9000": _ties(p, PWF_POWER_MIN_W) + _ties(p, PWF_POWER_MAX_W),
        "osf_strain_eq_threshold": _ties(strain_minnm(df), osf_threshold(df)),
    }


def _describe(s) -> dict:
    """pandas describe() 轉成 JSON 友善 dict；NaN（如樣本數 1 的 std）轉為 None。"""
    return {k: (None if pd.isna(v) else float(v)) for k, v in s.describe().items()}


def twf_cycle_summary(df) -> dict:
    cyc = tool_cycles(df)
    complete = cyc.iloc[:-1]                      # 最後一個週期被資料結尾截斷（未換刀）
    reached = cyc[cyc["max_wear_min"] >= TWF_WEAR_MIN]
    wear_at_twf = df.loc[df["twf"] == 1, "tool_wear_min"]
    return {
        "n_cycles": int(len(cyc)),
        "n_cycles_reaching_window": int(len(reached)),
        "n_cycles_with_twf": int(cyc["has_twf"].sum()),
        "twf_rate_given_reaching_window": _ratio(int(reached["has_twf"].sum()), len(reached)),
        "max_wear_without_twf_complete_cycles": _describe(complete.loc[~complete["has_twf"], "max_wear_min"]),
        "wear_at_twf": _describe(wear_at_twf),
        "twf_outside_window_udi": [int(u) for u in
                                   df.loc[(df["twf"] == 1) & ~twf_window(df), "udi"]],
        "rows_in_window": int(twf_window(df).sum()),
        "rows_above_window": int((df["tool_wear_min"] > TWF_WEAR_MAX).sum()),
        "last_cycle_truncated_max_wear": float(cyc["max_wear_min"].iloc[-1]),
    }


def _scatter_two(ax, x, y, is_pos, pos_label, neg_label):
    ax.scatter(x[~is_pos], y[~is_pos], s=4, color=NORMAL, alpha=0.25, linewidths=0, label=neg_label,
               rasterized=True)
    ax.scatter(x[is_pos], y[is_pos], s=14, color=FAILURE, alpha=0.9, linewidths=0, label=pos_label)


def _legend(ax, **kw):
    leg = ax.legend(markerscale=2, **kw)
    for h in leg.legend_handles:
        h.set_alpha(1)
    return leg


def plot_hdf(df, path):
    fig, ax = plt.subplots(figsize=(7, 4.8))
    x, y, pos = temp_diff_k(df), df["rotational_speed_rpm"], df["hdf"] == 1
    _scatter_two(ax, x, y, pos.to_numpy(), f"HDF label = 1 (n={int(pos.sum())})", "HDF label = 0")
    ax.axvline(HDF_TEMP_DIFF_MAX_K, color=TEXT_SECONDARY, ls="--", lw=2,
               label="Rule boundary: dT = 8.6 K, rpm = 1380")
    ax.axhline(HDF_RPM_MAX, color=TEXT_SECONDARY, ls="--", lw=2)
    ax.fill_between([x.min() - 0.2, HDF_TEMP_DIFF_MAX_K], y.min() - 50, HDF_RPM_MAX,
                    color=FAILURE, alpha=0.08, lw=0, label="Rule region (HDF predicted)")
    ax.set_xlim(x.min() - 0.2, x.max() + 0.2)
    ax.set_ylim(y.min() - 50, y.max() + 50)
    ax.set_xlabel("Process - air temperature [K]")
    ax.set_ylabel("Rotational speed [rpm]")
    ax.set_title("HDF rule boundary: temperature difference x speed")
    _legend(ax, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    save(fig, path)


def plot_pwf(df, path):
    fig, ax = plt.subplots(figsize=(7, 4.8))
    x, y, pos = df["rotational_speed_rpm"], df["torque_nm"], df["pwf"] == 1
    _scatter_two(ax, x, y, pos.to_numpy(), f"PWF label = 1 (n={int(pos.sum())})", "PWF label = 0")
    rpm = np.linspace(x.min() - 30, x.max() + 30, 400)
    w = rpm * 2 * np.pi / 60
    for i, p_w in enumerate([PWF_POWER_MIN_W, PWF_POWER_MAX_W]):
        ax.plot(rpm, p_w / w, color=TEXT_SECONDARY, ls="--", lw=2,
                label="Rule boundary: P = 3500 W / 9000 W" if i == 0 else None)
    xr = rpm.max() - 10
    ax.text(xr, PWF_POWER_MAX_W / (xr * 2 * np.pi / 60) + 3, "P = 9000 W", ha="right", va="bottom",
            color=TEXT_PRIMARY, fontsize=9)
    xl = rpm.min() + 10
    ax.text(xl, PWF_POWER_MIN_W / (xl * 2 * np.pi / 60) - 3, "P = 3500 W", ha="left", va="top",
            color=TEXT_PRIMARY, fontsize=9)
    ax.set_xlim(rpm.min(), rpm.max())
    ax.set_ylim(0, y.max() + 5)
    ax.set_xlabel("Rotational speed [rpm]")
    ax.set_ylabel("Torque [Nm]")
    ax.set_title("PWF rule boundary: power = torque x omega outside [3500, 9000] W")
    _legend(ax, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    save(fig, path)


def plot_osf(df, path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
    wear = np.linspace(1, df["tool_wear_min"].max() + 5, 400)
    for ax, t in zip(axes, TYPES):
        s = df[df["type"] == t]
        pos = (s["osf"] == 1).to_numpy()
        _scatter_two(ax, s["tool_wear_min"].to_numpy(), s["torque_nm"].to_numpy(), pos,
                     "OSF label = 1", "OSF label = 0")
        thr = OSF_STRAIN_MAX_MINNM[t]
        ax.plot(wear, thr / wear, color=TEXT_SECONDARY, ls="--", lw=2,
                label="Rule boundary: wear x torque = threshold")
        ax.set_title(f"Type {t}: threshold {thr} minNm\n(OSF n={int(pos.sum())} of {len(s)} rows)",
                     fontsize=11)
        ax.set_xlabel("Tool wear [min]")
        ax.set_xlim(0, wear.max())
    axes[0].set_ylim(0, df["torque_nm"].max() + 5)
    axes[0].set_ylabel("Torque [Nm]")
    _legend(axes[0], loc="lower left", fontsize=8.5)
    fig.suptitle("OSF rule boundary: tool wear x torque > type-specific threshold",
                 color=TEXT_PRIMARY, fontweight="bold")
    fig.tight_layout()
    save(fig, path)


def plot_twf(df, path):
    cyc = tool_cycles(df)
    wear_at_twf = df.loc[df["twf"] == 1, "tool_wear_min"].to_numpy()
    complete = cyc.iloc[:-1]                      # 最後一個週期被資料結尾截斷（未換刀）
    no_twf_max = complete.loc[~complete["has_twf"], "max_wear_min"].to_numpy()
    fig, ax = plt.subplots(figsize=(7, 4.4))
    bins = np.arange(0, max(260, int(df["tool_wear_min"].max()) + 10), 5)
    ax.axvspan(TWF_WEAR_MIN, TWF_WEAR_MAX, color=TEXT_SECONDARY, alpha=0.10, lw=0,
               label="Literature window 200-240 min")
    ax.hist(no_twf_max, bins=bins, histtype="step", lw=2, color=NORMAL,
            label=f"Max wear of complete tool cycles without TWF (n={len(no_twf_max)})")
    ax.hist(wear_at_twf, bins=bins, histtype="step", lw=2, color=FAILURE,
            label=f"Wear at TWF events (n={len(wear_at_twf)})")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    lo = min([TWF_WEAR_MIN, *no_twf_max, *wear_at_twf]) - 15   # 任一組可能為空（小樣本）
    ax.set_xlim(lo, bins[-1])
    ax.set_xlabel("Tool wear [min]")
    ax.set_ylabel("Count")
    ax.set_title("TWF is a random window, not a deterministic boundary")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=1)
    save(fig, path)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    raise TypeError(type(o))


def run(out_dir=RULES_DIR, df=None) -> dict:
    """重算規則、比對標記、輸出 rule_agreement.json 與四張規則邊界圖。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    from_disk = df is None
    if from_disk:
        df = load_dataset(label_policy="as_is")
    agree = rule_agreement(df)
    result = {
        "issue": 5,
        "source": "Matzka (2020), UCI AI4I 2020 dataset description",
        "label_policy": "as_is",
        "n_rows": int(len(df)),
        "thresholds": {
            "hdf_temp_diff_max_k": HDF_TEMP_DIFF_MAX_K, "hdf_rpm_max": HDF_RPM_MAX,
            "pwf_power_w": [PWF_POWER_MIN_W, PWF_POWER_MAX_W],
            "osf_strain_max_minnm": OSF_STRAIN_MAX_MINNM,
            "twf_wear_window_min": [TWF_WEAR_MIN, TWF_WEAR_MAX],
        },
        "rules": {
            "hdf": "process_temp_k - air_temp_k < 8.6 and rotational_speed_rpm < 1380",
            "pwf": "torque_nm * rpm * 2*pi/60 < 3500 or > 9000",
            "osf": "tool_wear_min * torque_nm > {L:11000, M:12000, H:13000}[type]",
            "twf": "200 <= tool_wear_min <= 240 (candidate window only; random failure point)",
        },
        "agreement": agree,
        "boundary_ties": boundary_ties(df),
        "tool_cycles": twf_cycle_summary(df),
        "twf_reconstruction_limit": TWF_LIMIT_TEXT,
        "figures": FIGURES,
    }
    if from_disk:  # 敏感度分析：移除 27 筆標記不一致樣本後重算
        dropped = load_dataset(label_policy="drop_inconsistent")
        result["sensitivity_drop_inconsistent"] = {
            m: {"accuracy": v["accuracy"], "precision": v["precision"], "recall": v["recall"]}
            for m, v in rule_agreement(dropped).items()}
    plot_hdf(df, out_dir / FIGURES[0])
    plot_pwf(df, out_dir / FIGURES[1])
    plot_osf(df, out_dir / FIGURES[2])
    plot_twf(df, out_dir / FIGURES[3])
    (out_dir / "rule_agreement.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    return result


if __name__ == "__main__":
    r = run()
    for m in ["hdf", "pwf", "osf", "twf", "machine_failure_union"]:
        a = r["agreement"][m]
        print(f"{m:>22}: acc={a['accuracy']:.4f} P={a['precision']} R={a['recall']} "
              f"FP={a['fp']} FN={a['fn']}")
