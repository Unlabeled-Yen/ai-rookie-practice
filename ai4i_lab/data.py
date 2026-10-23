"""AI4I 2020 資料的取得、載入、品質檢查與探索性分析（#4）。

所有後續子單一律透過 `load_dataset()` 讀取資料，並使用本模組的欄位常數。
執行：python -m ai4i_lab.data            # 讀取既有資料並產生 EDA 輸出
     python -m ai4i_lab.data --download # 先從 UCI 下載
"""
import datetime
import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE_URL = "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = ROOT / "data" / "ai4i2020.csv"
EDA_DIR = ROOT / "outputs" / "ai4i" / "eda"

ORIGINAL_TO_SNAKE = {
    "UDI": "udi",
    "Product ID": "product_id",
    "Type": "type",
    "Air temperature [K]": "air_temp_k",
    "Process temperature [K]": "process_temp_k",
    "Rotational speed [rpm]": "rotational_speed_rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
    "Machine failure": "machine_failure",
    "TWF": "twf", "HDF": "hdf", "PWF": "pwf", "OSF": "osf", "RNF": "rnf",
}
COLUMNS = list(ORIGINAL_TO_SNAKE.values())
ID_COLUMNS = ["udi", "product_id"]                      # 識別用，不可作為特徵
FEATURES = ["type", "air_temp_k", "process_temp_k", "rotational_speed_rpm", "torque_nm", "tool_wear_min"]
NUMERIC_FEATURES = FEATURES[1:]
TARGET = "machine_failure"
FAILURE_MODES = ["twf", "hdf", "pwf", "osf", "rnf"]    # 直接決定 TARGET，預測 TARGET 時不可作為特徵
TYPES = ["L", "M", "H"]
LABEL_POLICIES = ("as_is", "drop_inconsistent")

FEATURE_INFO = {
    "type": ("產品品質等級", "L/M/H（約 50% / 30% / 20%）"),
    "air_temp_k": ("空氣溫度", "K"),
    "process_temp_k": ("製程溫度", "K"),
    "rotational_speed_rpm": ("主軸轉速", "rpm"),
    "torque_nm": ("扭矩", "Nm"),
    "tool_wear_min": ("刀具磨耗（累積使用時間）", "min"),
}
MODE_INFO = {
    "twf": "刀具磨耗故障（Tool Wear Failure）",
    "hdf": "散熱不良故障（Heat Dissipation Failure）",
    "pwf": "功率故障（Power Failure）",
    "osf": "過度負荷故障（Overstrain Failure）",
    "rnf": "隨機故障（Random Failure）",
}


# ---------- 取得資料 ----------

def sha256_of(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _http_get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as resp:
        return resp.read()


def download_dataset(dest=DEFAULT_PATH, fetch=_http_get, today=None) -> Path:
    """從 UCI 下載 zip、取出 CSV（去除 BOM）並寫入來源、日期與 SHA-256 的 meta 檔。"""
    dest = Path(dest)
    raw = fetch(SOURCE_URL)
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        text = z.read("ai4i2020.csv").decode("utf-8-sig")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text, encoding="utf-8")
    meta = {
        "source_url": SOURCE_URL,
        "uci_id": 601,
        "downloaded": today or datetime.date.today().isoformat(),
        "source_zip_sha256": hashlib.sha256(raw).hexdigest(),
        "sha256": sha256_of(dest),
        "sha256_note": "sha256 為去除 BOM 後保存之 CSV 的雜湊；source_zip_sha256 為 UCI 原始 zip 的雜湊",
        "n_rows": len(pd.read_csv(dest)),
    }
    dest.with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return dest


def load_dataset(path=DEFAULT_PATH, label_policy: str = "as_is") -> pd.DataFrame:
    """讀取資料並改為 snake_case 欄名。

    label_policy：
      - "as_is"（預設）：保留原始標記
      - "drop_inconsistent"：移除故障旗標與 machine_failure 不一致的列（供敏感度分析）
    """
    if label_policy not in LABEL_POLICIES:
        raise ValueError(f"未知的 label_policy: {label_policy}，可用 {LABEL_POLICIES}")
    path = Path(path)
    meta_path = path.with_suffix(".meta.json")
    if meta_path.exists():
        expected = json.loads(meta_path.read_text(encoding="utf-8"))["sha256"]
        if sha256_of(path) != expected:
            raise ValueError(f"SHA-256 不符：{path} 可能已被修改")
    df = pd.read_csv(path, encoding="utf-8-sig")
    missing = set(ORIGINAL_TO_SNAKE) - set(df.columns)
    if missing:
        raise ValueError(f"資料缺少欄位: {sorted(missing)}")
    df = df.rename(columns=ORIGINAL_TO_SNAKE)[COLUMNS]
    if label_policy == "drop_inconsistent":
        inc = find_label_inconsistencies(df)
        bad = set(inc["failure_without_mode"]) | set(inc["mode_without_failure"])
        df = df[~df["udi"].isin(bad)].reset_index(drop=True)
    return df


# ---------- 品質檢查與探索 ----------

def find_label_inconsistencies(df: pd.DataFrame) -> dict:
    """回傳兩類不一致列的 UDI：有故障但無任何模式、有模式但未計為故障。"""
    any_mode = df[FAILURE_MODES].sum(axis=1) > 0
    failed = df[TARGET] == 1
    return {
        "failure_without_mode": df.loc[failed & ~any_mode, "udi"].astype(int).tolist(),
        "mode_without_failure": df.loc[~failed & any_mode, "udi"].astype(int).tolist(),
    }


def explore(df: pd.DataFrame) -> dict:
    modes = df[FAILURE_MODES]
    cooc = modes.T.dot(modes)
    return {
        "n_rows": len(df),
        "n_columns": df.shape[1],
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_product_ids": int(df["product_id"].duplicated().sum()),
        "n_failures": int(df[TARGET].sum()),
        "failure_rate": float(df[TARGET].mean()),
        "mode_counts": {m: int(modes[m].sum()) for m in FAILURE_MODES},
        "multi_mode_rows": int((modes.sum(axis=1) > 1).sum()),
        "mode_cooccurrence": {a: {b: int(cooc.loc[a, b]) for b in FAILURE_MODES} for a in FAILURE_MODES},
        "type_counts": {t: int(n) for t, n in df["type"].value_counts().items()},
        "failure_rate_by_type": {t: float(r) for t, r in df.groupby("type")[TARGET].mean().items()},
        "ranges": {c: {"min": float(df[c].min()), "max": float(df[c].max())} for c in NUMERIC_FEATURES},
        "describe": df[NUMERIC_FEATURES].describe().round(3).to_dict(),
        "by_failure": {str(k): g[NUMERIC_FEATURES].mean().round(3).to_dict() for k, g in df.groupby(TARGET)},
        "correlation": df[NUMERIC_FEATURES + [TARGET]].corr().round(3).to_dict(),
    }


def temporal_structure(df: pd.DataFrame) -> dict:
    """檢查依 UDI 排序後是否有時間結構：刀具磨耗是否持續累積（換刀時歸零）、溫度是否緩慢漂移。"""
    d = df.sort_values("udi")
    wear_diff = np.diff(d["tool_wear_min"].to_numpy())
    air = d["air_temp_k"].to_numpy()
    nondecreasing = float((wear_diff >= 0).mean())
    autocorr = float(np.corrcoef(air[:-1], air[1:])[0, 1])
    return {
        "tool_wear_resets": int((wear_diff < 0).sum()),
        "tool_wear_nondecreasing_fraction": nondecreasing,
        "air_temp_lag1_autocorr": autocorr,
        # 若資料為隨機順序，相鄰磨耗非遞減的比例約 0.5、溫度自相關約 0
        "has_temporal_structure": bool(nondecreasing > 0.9 or autocorr > 0.9),
    }


# ---------- 圖表 ----------

def _plots(df: pd.DataFrame, rep: dict, temporal: dict, out_dir: Path) -> None:
    from ai4i_lab.style import (BLUE_RAMP, DIVERGING, FAILURE, MODE_COLORS, NORMAL, SURFACE,
                                TEXT_PRIMARY, plt, save)

    # 1) 故障 vs 正常：各數值特徵的密度分布（階梯線，避免填色互相遮蔽）
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, col in zip(axes.flat, NUMERIC_FEATURES):
        bins = np.histogram_bin_edges(df[col], bins=40)
        for val, color, label in ((0, NORMAL, "normal"), (1, FAILURE, "failure")):
            ax.hist(df.loc[df[TARGET] == val, col], bins=bins, density=True, histtype="step",
                    color=color, linewidth=2, label=label)
        name, unit = FEATURE_INFO[col]
        ax.set_title(f"{col} [{unit}]", fontsize=10)
        ax.set_yticks([])
    axes.flat[0].legend(loc="upper left")
    axes.flat[-1].remove()  # 5 個數值特徵，第 6 格留空
    fig.suptitle("Feature distributions: failure vs normal (density)", fontweight="bold", color=TEXT_PRIMARY)
    fig.tight_layout()
    save(fig, out_dir / "feature_by_failure.png")

    # 2) 相關係數熱圖（發散色階，0 為中性灰）
    cols = NUMERIC_FEATURES + [TARGET]
    corr = df[cols].corr().to_numpy()
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(corr, cmap=DIVERGING, vmin=-1, vmax=1)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if abs(corr[i, j]) > 0.6 else TEXT_PRIMARY)
    ax.set_xticks(range(len(cols)), cols, rotation=40, ha="right")
    ax.set_yticks(range(len(cols)), cols)
    ax.grid(False)
    ax.set_title("Correlation matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save(fig, out_dir / "correlation.png")

    # 3) 依產品等級的故障率
    fig, ax = plt.subplots(figsize=(6, 4))
    rates = [rep["failure_rate_by_type"][t] for t in TYPES]
    bars = ax.bar(TYPES, rates, color=NORMAL, width=0.55, edgecolor=SURFACE, linewidth=2)
    for b, t, r in zip(bars, TYPES, rates):
        ax.annotate(f"{r:.2%}\n(n={rep['type_counts'][t]})", (b.get_x() + b.get_width() / 2, r),
                    xytext=(0, 4), textcoords="offset points", ha="center", color=TEXT_PRIMARY, fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_ylim(0, max(rates) * 1.35)
    ax.grid(axis="x", visible=False)
    ax.set_title("Failure rate by product type")
    save(fig, out_dir / "failure_rate_by_type.png")

    # 4) 故障模式筆數（直接標註數值）與共同發生矩陣
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [1.2, 1]})
    counts = [rep["mode_counts"][m] for m in FAILURE_MODES]
    bars = ax1.bar([m.upper() for m in FAILURE_MODES], counts, width=0.6,
                   color=[MODE_COLORS[m] for m in FAILURE_MODES], edgecolor=SURFACE, linewidth=2)
    for b, n in zip(bars, counts):
        ax1.annotate(str(n), (b.get_x() + b.get_width() / 2, n), xytext=(0, 3),
                     textcoords="offset points", ha="center", color=TEXT_PRIMARY)
    ax1.grid(axis="x", visible=False)
    ax1.set_title(f"Failure mode counts ({rep['multi_mode_rows']} rows have >1 mode)")
    co = np.array([[rep["mode_cooccurrence"][a][b] for b in FAILURE_MODES] for a in FAILURE_MODES])
    ax2.imshow(co, cmap=BLUE_RAMP)
    for i in range(5):
        for j in range(5):
            ax2.text(j, i, co[i, j], ha="center", va="center", fontsize=9,
                     color="white" if co[i, j] > co.max() / 2 else TEXT_PRIMARY)
    labels = [m.upper() for m in FAILURE_MODES]
    ax2.set_xticks(range(5), labels)
    ax2.set_yticks(range(5), labels)
    ax2.grid(False)
    ax2.set_title("Mode co-occurrence (diagonal = count)")
    fig.tight_layout()
    save(fig, out_dir / "failure_modes.png")

    # 5) 時間結構：依 UDI 排序的刀具磨耗與空氣溫度（上下兩圖，不共用 y 軸）
    d = df.sort_values("udi")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax1.plot(d["udi"], d["tool_wear_min"], color=NORMAL, linewidth=0.8)
    fail = d[d[TARGET] == 1]
    ax1.scatter(fail["udi"], fail["tool_wear_min"], s=10, color=FAILURE, zorder=3, label="failure")
    ax1.set_ylabel("tool_wear_min")
    ax1.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=1, markerscale=1.5)
    ax1.set_title(f"Temporal structure by UDI — {temporal['tool_wear_resets']} tool-wear resets")
    ax2.plot(d["udi"], d["air_temp_k"], color=NORMAL, linewidth=0.8)
    ax2.set_ylabel("air_temp_k")
    ax2.set_xlabel("UDI (row order)")
    fig.tight_layout()
    save(fig, out_dir / "temporal_structure.png")


def run(path=DEFAULT_PATH, out_dir=EDA_DIR) -> dict:
    """產生 EDA 圖表與 eda.json。"""
    out_dir = Path(out_dir)
    df = load_dataset(path)
    inc = find_label_inconsistencies(df)
    rep = {
        "exploration": explore(df),
        "label_inconsistencies": {
            "rows": inc,
            "counts": {k: len(v) for k, v in inc.items()},
            "policy": {
                "default": "as_is",
                "reason": ("主分析保留原始標記（machine_failure 即為實際紀錄的結果）；"
                           "9 筆無模式的故障無法被規則解釋，18 筆 RNF 未計為故障屬資料產生過程的特性。"
                           "另以 label_policy='drop_inconsistent' 移除這 27 筆做敏感度分析。"),
            },
        },
        "temporal_structure": temporal_structure(df),
        "feature_info": {k: {"name": n, "unit": u} for k, (n, u) in FEATURE_INFO.items()},
        "mode_info": MODE_INFO,
    }
    _plots(df, rep["exploration"], rep["temporal_structure"], out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eda.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    return rep


if __name__ == "__main__":
    import sys
    if "--download" in sys.argv:
        print("downloaded:", download_dataset())
    r = run()
    print(json.dumps({"label_inconsistencies": r["label_inconsistencies"]["counts"],
                      "temporal_structure": r["temporal_structure"]}, ensure_ascii=False, indent=2))
