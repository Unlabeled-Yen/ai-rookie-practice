"""AI4I 的 PCA 降維視覺化與故障結構分析（#6）。

執行：python -m ai4i_lab.pca
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

SEED = 42


def fit_pca(X, n_components=None, seed=SEED) -> dict:
    """先標準化（各特徵量綱差異大，如功率為數千 W）再做 PCA。"""
    X = np.asarray(X, dtype=float)
    scaler = StandardScaler().fit(X)
    Z = scaler.transform(X)
    pca = PCA(n_components=n_components, random_state=seed).fit(Z)
    return {"scaler": scaler, "pca": pca, "scores": pca.transform(Z),
            "explained_variance_ratio": pca.explained_variance_ratio_}


def n_components_for(evr, threshold: float) -> int:
    """累積解釋變異達到 threshold（含等於）所需的最少主成分數。"""
    cum = np.cumsum(evr)
    return int(np.argmax(cum >= threshold - 1e-12) + 1)


def oriented_auc(y, score) -> float:
    """單一軸的 ROC-AUC，取 max(AUC, 1−AUC)，與主成分符號無關。"""
    auc = roc_auc_score(y, score)
    return float(max(auc, 1 - auc))


def centroid_shift(group, normal):
    """目標組重心相對正常組重心的位移，以正常組各軸標準差為單位；回傳 (位移向量, 歐氏距離)。"""
    group, normal = np.asarray(group, float), np.asarray(normal, float)
    shift = (group.mean(axis=0) - normal.mean(axis=0)) / normal.std(axis=0)
    return shift, float(np.linalg.norm(shift))


# ---------- 特徵組分析 ----------

def separability_pipeline(n_components=3, seed=SEED):
    """故障可分性探針：標準化、PCA 與分類器都在訓練折內擬合（避免洩漏）。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    steps = [("scale", StandardScaler())]
    if n_components is not None:
        steps.append(("pca", PCA(n_components=n_components, random_state=seed)))
    steps.append(("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)))
    return Pipeline(steps)


def _cv_auc(pipe, X, y, cv_repeats, seed=SEED) -> dict:
    from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=cv_repeats, random_state=seed)
    s = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=1)
    return {"mean": round(float(s.mean()), 4), "std": round(float(s.std()), 4), "n_folds": int(len(s))}


def analyze_feature_set(df, feature_set: str, cv_repeats: int = 2, seed=SEED) -> dict:
    """對單一特徵組做標準化 PCA，並量化故障樣本在主成分空間中的結構。"""
    from ai4i_lab.data import FAILURE_MODES, TARGET
    from ai4i_lab.features import build_features

    X = build_features(df, feature_set)
    y = df[TARGET].to_numpy()
    full = fit_pca(X, seed=seed)
    evr = full["explained_variance_ratio"]
    scores = full["scores"]
    comps = full["pca"].components_
    normal3 = scores[y == 0, :3]

    shifts = {}
    for mode in FAILURE_MODES + ["any_failure"]:
        mask = (y == 1) if mode == "any_failure" else (df[mode].to_numpy() == 1)
        if mask.sum() == 0:
            shifts[mode] = None
            continue
        shift, dist = centroid_shift(scores[mask, :3], normal3)
        shifts[mode] = {"n": int(mask.sum()), "shift_sd": [round(float(v), 3) for v in shift],
                        "distance_sd": round(dist, 3)}

    cum = np.cumsum(evr)
    return {
        "feature_set": feature_set,
        "features": list(X.columns),
        "n_features": int(X.shape[1]),
        "explained_variance_ratio": [round(float(v), 4) for v in evr],
        "cumulative_variance": [round(float(v), 4) for v in cum],
        "cumulative_variance_3pc": round(float(evr[:3].sum()), 4),
        "n_components_90": n_components_for(evr, 0.90),
        "n_components_95": n_components_for(evr, 0.95),
        "loadings": {f"PC{i + 1}": {f: round(float(w), 3) for f, w in zip(X.columns, comps[i])}
                     for i in range(3)},
        "pc_auc_failure": {f"PC{i + 1}": round(oriented_auc(y, scores[:, i]), 4) for i in range(X.shape[1])},
        "mode_centroid_shift_3pc": shifts,
        "cv_auc": {
            "logreg_3pc": _cv_auc(separability_pipeline(3, seed), X, y, cv_repeats, seed),
            "logreg_all_features": _cv_auc(separability_pipeline(None, seed), X, y, cv_repeats, seed),
        },
        "scores": scores[:, :3],
    }


# ---------- 圖表 ----------

def _axes_labels(ax, evr):
    for i, setter in enumerate((ax.set_xlabel, ax.set_ylabel, ax.set_zlabel)):
        setter(f"PC{i + 1} ({evr[i] * 100:.1f}%)", labelpad=2)
    ax.tick_params(labelsize=7, pad=0)


def _scatter3d(ax, scores, mask, hi_color, hi_label, lo_label, elev=22, azim=-60):
    from ai4i_lab.style import NORMAL, SURFACE
    ax.scatter(*scores[~mask].T, s=3, c=NORMAL, alpha=0.12, linewidths=0, label=lo_label, depthshade=False)
    ax.scatter(*scores[mask].T, s=14, c=hi_color, alpha=0.95, edgecolors=SURFACE, linewidths=0.4,
               label=hi_label, depthshade=False)
    ax.view_init(elev=elev, azim=azim)


def plot_3d(res, y, path, title):
    import matplotlib.pyplot as plt
    from ai4i_lab.style import FAILURE, TEXT_SECONDARY, save
    evr = res["explained_variance_ratio"]
    fig = plt.figure(figsize=(12, 5.6))
    for j, (elev, azim) in enumerate(((22, -60), (22, 30))):
        ax = fig.add_subplot(1, 2, j + 1, projection="3d")
        mask = y == 1
        _scatter3d(ax, res["scores"], mask, FAILURE, f"Failure (n={mask.sum()})",
                   f"Normal (n={(~mask).sum()})", elev, azim)
        _axes_labels(ax, evr)
        ax.set_title(f"View {j + 1} (elev {elev}, azim {azim})", fontsize=10)
        if j == 0:
            ax.legend(loc="upper left", markerscale=2)
    fig.suptitle(title, fontweight="bold", y=0.98)
    fig.text(0.5, 0.02, f"Standardized PCA, first 3 PCs explain {res['cumulative_variance_3pc'] * 100:.1f}% "
             f"of variance", ha="center", color=TEXT_SECONDARY, fontsize=9)
    save(fig, path)


def plot_3d_modes(res, df, path, title):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from ai4i_lab.data import FAILURE_MODES, TARGET
    from ai4i_lab.style import FAILURE, NORMAL, save
    evr = res["explained_variance_ratio"]
    panels = [("Any failure", df[TARGET].to_numpy() == 1)] + \
             [(m.upper(), df[m].to_numpy() == 1) for m in FAILURE_MODES]
    fig = plt.figure(figsize=(13, 8.6))
    for k, (name, mask) in enumerate(panels):
        ax = fig.add_subplot(2, 3, k + 1, projection="3d")
        _scatter3d(ax, res["scores"], mask, FAILURE, name, "others")
        _axes_labels(ax, evr)
        ax.set_title(f"{name} (n={mask.sum()})", fontsize=10)
    handles = [Line2D([], [], ls="", marker="o", color=FAILURE, label="Highlighted failure mode"),
               Line2D([], [], ls="", marker="o", color=NORMAL, alpha=0.4, label="All other samples")]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.955))
    fig.suptitle(title, fontweight="bold", y=0.99)
    save(fig, path)


def plot_explained_variance(results, path):
    import matplotlib.pyplot as plt
    from ai4i_lab.style import SERIES, TEXT_SECONDARY, save
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4))
    width = 0.4
    for j, (fs, r) in enumerate(results.items()):
        evr = np.array(r["explained_variance_ratio"])
        idx = np.arange(1, len(evr) + 1)
        a1.bar(idx + (j - 0.5) * width, evr * 100, width=width - 0.04, color=SERIES[j],
               label=f"{fs} ({len(evr)} features)")
        a2.plot(idx, np.cumsum(evr) * 100, marker="o", ms=5, color=SERIES[j], label=fs)
        n90 = r["n_components_90"]
        a2.text(0.97, 0.36 - 0.07 * j, f"{fs}: {n90} PCs reach 90%", transform=a2.transAxes,
                ha="right", fontsize=9, color=TEXT_SECONDARY)
    a1.set(title="Explained variance per component", xlabel="Principal component", ylabel="Variance (%)")
    a1.set_xticks(range(1, 12))
    a1.legend()
    a2.axhline(90, color=TEXT_SECONDARY, lw=1, ls="--")
    a2.set(title="Cumulative explained variance", xlabel="Number of components",
           ylabel="Cumulative variance (%)", ylim=(0, 102))
    a2.set_xticks(range(1, 12))
    a2.legend(loc="lower right", bbox_to_anchor=(1.0, 0.0))
    save(fig, path)


def plot_loadings(results, path):
    import matplotlib.pyplot as plt
    from ai4i_lab.style import DIVERGING, TEXT_PRIMARY, save
    fig, axes = plt.subplots(1, len(results), figsize=(12, 6), layout="constrained")
    fig.get_layout_engine().set(wspace=0.08)
    for ax, (fs, r) in zip(axes, results.items()):
        feats = r["features"]
        M = np.array([[r["loadings"][pc][f] for f in feats] for pc in ("PC1", "PC2", "PC3")])
        im = ax.imshow(M.T, cmap=DIVERGING, vmin=-1, vmax=1, aspect="auto")
        for (i, j), v in np.ndenumerate(M.T):
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=8, color=TEXT_PRIMARY)
        ax.set_xticks(range(3), ["PC1", "PC2", "PC3"])
        ax.set_yticks(range(len(feats)), feats)
        ax.grid(False)
        ax.set_title(f"{fs} feature set")
    fig.colorbar(im, ax=axes, shrink=0.7, label="Loading")
    fig.suptitle("PCA loadings (standardized features)", fontweight="bold")
    save(fig, path)


# ---------- 討論與 run() ----------

def best_single_pc(res) -> str:
    """單軸 ROC-AUC 最高的主成分名稱。"""
    return max(res["pc_auc_failure"], key=res["pc_auc_failure"].get)


def _discussion(results, n_fail, n_total) -> list:
    raw, eng = results["raw"], results["engineered"]
    rate = n_fail / n_total * 100
    delta = (eng["cumulative_variance_3pc"] - raw["cumulative_variance_3pc"]) * 100
    if delta > 0.05:
        trend = (f"衍生組前 3 個主成分的占比比原始組高 {delta:.1f} 個百分點：衍生特徵是原始欄位的乘積／差，"
                 "與原欄位高度相關，PCA 會把它們併入既有方向，")
    else:
        trend = (f"衍生組前 3 個主成分的占比反而比原始組低 {-delta:.1f} 個百分點：總變異隨特徵數增加（11 vs 8 個"
                 "標準化單位），而衍生特徵（乘積）引入了與原欄位不完全共線的新方向，變異被分散到更多主成分；")
    sh = eng["mode_centroid_shift_3pc"]
    def best(r):
        pc = best_single_pc(r)
        return f"{pc}（解釋 {r['explained_variance_ratio'][int(pc[2:]) - 1] * 100:.1f}% 變異）AUC {r['pc_auc_failure'][pc]:.3f}"
    dist = "、".join(f"{m.upper()} {sh[m]['distance_sd']:.2f}" for m in ("twf", "hdf", "pwf", "osf", "rnf")
                    if sh[m] is not None)
    return [
        f"解釋變異：原始組前 3 個主成分解釋 {raw['cumulative_variance_3pc'] * 100:.1f}% 變異（達 90% 需 "
        f"{raw['n_components_90']} 個）；原始＋衍生組為 {eng['cumulative_variance_3pc'] * 100:.1f}%（需 "
        f"{eng['n_components_90']} 個）。" + trend + "無論占比升降，解釋變異只反映特徵之間的共變結構，"
        "不代表多了或少了故障資訊。",
        f"3D 圖中 {n_fail} 筆故障多半落在正常樣本雲的邊緣，沒有形成獨立的群。各模式重心相對正常重心的位移"
        f"（以正常組標準差為單位，前 3 個主成分，衍生組）：{dist}。",
        f"線性可分性：只用前 3 個主成分的邏輯斯迴歸（PCA 在訓練折內擬合）交叉驗證 ROC-AUC，原始組 "
        f"{raw['cv_auc']['logreg_3pc']['mean']:.3f}、衍生組 {eng['cv_auc']['logreg_3pc']['mean']:.3f}；"
        f"用全部特徵的邏輯斯迴歸為 {raw['cv_auc']['logreg_all_features']['mean']:.3f} / "
        f"{eng['cv_auc']['logreg_all_features']['mean']:.3f}。單一主成分中最能分辨故障的是：原始組 {best(raw)}、"
        f"衍生組 {best(eng)}；若最佳主成分不是變異最大的前幾個，表示故障訊號藏在低變異方向，"
        "依變異排序截取前 3 個主成分會把它丟掉。",
        "PCA 呈現不出故障結構的原因：(1) PCA 是非監督的，只找變異最大的方向，不看標籤；只有 "
        f"{rate:.2f}% 的故障樣本幾乎不影響共變異矩陣。(2) 標準化後 Type 的三個 one-hot 欄各占 1 單位變異，會吃掉部分主成分。"
        "(3) 故障由非線性、局部的門檻規則產生：HDF 是兩個條件同時成立（溫差 < 8.6 K 且轉速 < 1380 rpm），"
        "PWF 落在功率的兩端（< 3500 W 或 > 9000 W，非單調），OSF 的門檻依 Type 而變，TWF 只在磨耗 200–240 min "
        "的窄帶，RNF 與特徵無關。這些區域是特徵空間的角落或兩端，投影到變異最大的線性方向後會和正常樣本重疊。",
        "結論：PCA 適合檢視特徵之間的相關結構（轉速與扭矩強負相關、衍生特徵與原欄位冗餘），不適合當作故障的"
        "分群或偵測工具；要捕捉這些門檻型結構，需要樹模型等非線性方法，或直接使用與規則對應的衍生特徵（#8）。",
    ]


PCA_DIR = None  # 於下方設定


def run(df=None, out_dir=None, label_policy="as_is", cv_repeats=2, plots=True, seed=SEED) -> dict:
    import json
    from pathlib import Path
    from ai4i_lab.data import TARGET, load_dataset
    from ai4i_lab.features import FEATURE_SETS

    out_dir = Path(out_dir or PCA_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    if df is None:
        df = load_dataset(label_policy=label_policy)
    y = df[TARGET].to_numpy()
    results = {fs: analyze_feature_set(df, fs, cv_repeats=cv_repeats, seed=seed) for fs in FEATURE_SETS}

    figures = []
    if plots:
        names = {"raw": "Raw features", "engineered": "Raw + engineered features"}
        for fs, r in results.items():
            plot_3d(r, y, out_dir / f"pca_3d_{fs}.png", f"PCA 3D: {names[fs]} - normal vs failure")
            plot_3d_modes(r, df, out_dir / f"pca_3d_modes_{fs}.png", f"PCA 3D by failure mode: {names[fs]}")
            figures += [f"pca_3d_{fs}.png", f"pca_3d_modes_{fs}.png"]
        plot_explained_variance(results, out_dir / "explained_variance.png")
        plot_loadings(results, out_dir / "loadings.png")
        figures += ["explained_variance.png", "loadings.png"]

    serial = {fs: {k: v for k, v in r.items() if k != "scores"} for fs, r in results.items()}
    report = {
        "issue": 6,
        "seed": seed,
        "label_policy": label_policy,
        "n_samples": int(len(df)),
        "n_failures": int(y.sum()),
        "failure_rate": round(float(y.mean()), 4),
        "method": "StandardScaler → PCA（全部主成分）；可分性探針為 5 折分層 CV × "
                  f"{cv_repeats} 次重複、class_weight=balanced 的邏輯斯迴歸，標準化與 PCA 皆在訓練折內擬合",
        "feature_sets": serial,
        "comparison": {
            key: {fs: serial[fs][src] if sub is None else serial[fs][src][sub]["mean"] for fs in serial}
            for key, src, sub in (
                ("cumulative_variance_3pc", "cumulative_variance_3pc", None),
                ("n_components_90", "n_components_90", None),
                ("n_components_95", "n_components_95", None),
                ("cv_auc_logreg_3pc", "cv_auc", "logreg_3pc"),
                ("cv_auc_logreg_all_features", "cv_auc", "logreg_all_features"),
            )
        } | {"best_single_pc": {fs: best_single_pc(serial[fs]) for fs in serial}},
        "discussion": _discussion(results, int(y.sum()), int(len(df))),
        "figures": sorted(figures),
    }
    (out_dir / "pca.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _set_dirs():
    global PCA_DIR
    from ai4i_lab.data import ROOT
    PCA_DIR = ROOT / "outputs" / "ai4i" / "pca"


_set_dirs()

def main():
    import json
    import time
    t0 = time.time()
    rep = run()
    print(json.dumps(rep["comparison"], ensure_ascii=False, indent=2))
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
