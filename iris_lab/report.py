"""依實驗結果自動產生「小論文」（Markdown）。"""
import statistics

from iris_lab.data import CLASSES, FEATURES

FEATURE_ZH = {"sepal_length": "花萼長度", "sepal_width": "花萼寬度",
              "petal_length": "花瓣長度", "petal_width": "花瓣寬度"}


def _fig(results, name, caption, n):
    return f"![{caption}](../{results['figures'][name]})\n\n*圖 {n}：{caption}*"


def _errors(cm):
    return [(CLASSES[i], CLASSES[j], cm[i][j])
            for i in range(len(cm)) for j in range(len(cm)) if i != j and cm[i][j]]


def render_paper(results: dict) -> str:
    ex, test, m = results["exploration"], results["test"], results["model"]
    ratio = results["pca"]["explained_variance_ratio"]
    cv = results["cv_scores"]
    cv_mean, cv_std = statistics.mean(cv), statistics.stdev(cv)
    hist = results["history"]
    n_val = round(m["n_train"] * 0.2)
    errors = _errors(test["confusion_matrix"])
    n_wrong = sum(e[2] for e in errors)

    feat_rows = "\n".join(
        f"| `{f}` | {FEATURE_ZH[f]} | {ex['describe'][f]['mean']:.3f} | {ex['describe'][f]['std']:.3f} "
        f"| {ex['describe'][f]['min']:.1f} | {ex['describe'][f]['max']:.1f} |" for f in FEATURES)
    mean_rows = "\n".join(
        f"| {c} | " + " | ".join(f"{ex['class_means'][c][f]:.3f}" for f in FEATURES) + " |"
        for c in CLASSES)
    class_rows = "\n".join(
        f"| {c} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} | {v['support']} |"
        for c, v in test["per_class"].items())
    corr = ex["correlation"]["petal_length"]["petal_width"]

    if not errors:
        discussion_err = f"測試集 {m['n_test']} 筆全部分類正確。"
    else:
        err_text = "、".join(f"{a} 被誤判為 {b} 共 {n} 筆" for a, b, n in errors)
        discussion_err = f"測試集 {m['n_test']} 筆中共有 {n_wrong} 筆誤判（{err_text}）。"
        if all({a, b} == {"versicolor", "virginica"} for a, b, _ in errors):
            discussion_err += "誤判皆發生在 versicolor 與 virginica 之間。"

    gap = hist["train_acc"][-1] - hist["val_acc"][-1]
    if gap <= 0.05:
        fit_text = f"兩者差距 {gap:.1%}，未見明顯過擬合。"
    else:
        fit_text = f"兩者差距 {gap:.1%}，模型對訓練資料可能有些過擬合，可考慮加入正則化或提早停止。"

    return f"""# 以多層感知器類神經網路進行鳶尾花（Iris）分類之研究

## 摘要

本研究使用經典的 Iris 資料集（{ex['n_rows']} 筆、{ex['n_features']} 個特徵、3 個類別），先進行資料探索與主成分分析（PCA），
再以多層感知器（Multilayer Perceptron, MLP）類神經網路建立分類模型。PCA 結果顯示前三個主成分可解釋
{sum(ratio):.2%} 的總變異，其中 setosa 與其他兩類可明顯分離。模型以分層抽樣切分 {m['n_train']} 筆訓練、
{m['n_test']} 筆測試資料，於測試集上達到 **{test['accuracy']:.2%}** 的準確率與 {test['macro_f1']:.3f} 的 macro F1；
5 折交叉驗證平均準確率為 {cv_mean:.2%}（標準差 {cv_std:.2%}，各折介於 {min(cv):.0%}–{max(cv):.0%}），與測試集結果一致。

## 1. 前言

Iris 資料集由 Fisher（1936）提出，是機器學習與統計分類最常用的入門資料集之一。
其特徵維度低、類別平衡，適合用來完整練習「資料取得 → 探索 → 降維視覺化 → 建模 → 評估」的流程。
本研究的目標為：（1）了解資料欄位與特徵分布；（2）以 PCA 觀察資料在低維空間的結構；
（3）以類神經網路建立分類器並評估其效果。

## 2. 資料與探索

### 2.1 資料來源與欄位

資料取自 scikit-learn 內建之 Iris 資料集（源自 UCI Machine Learning Repository），保存於 `data/iris.csv`。
共 {ex['n_rows']} 筆樣本，包含 4 個數值特徵（單位：公分）與 1 個類別欄位 `species`。

| 欄位 | 說明 | 平均 | 標準差 | 最小 | 最大 |
|---|---|---|---|---|---|
{feat_rows}

分類目標 `species` 共三類：**setosa**（山鳶尾）、**versicolor**（變色鳶尾）、**virginica**（維吉尼亞鳶尾），
各 {ex['class_counts']['setosa']} 筆，類別完全平衡。

### 2.2 資料檢查

- 缺失值：{ex['missing_values']} 個
- 重複列：{ex['duplicate_rows']} 筆（屬於原始資料中的真實觀測，予以保留）
- 類別分布：{', '.join(f'{c} {n} 筆' for c, n in ex['class_counts'].items())}

各類別特徵平均值如下：

| 類別 | sepal_length | sepal_width | petal_length | petal_width |
|---|---|---|---|---|
{mean_rows}

可以看出花瓣（petal）特徵在三類之間差異最大，且 petal_length 與 petal_width 的相關係數高達 {corr:.3f}。
setosa 的花瓣明顯較小，而 versicolor 與 virginica 的數值範圍有部分重疊。

{_fig(results, 'feature_distributions', '各特徵依類別之分布（箱形圖＋散點）', 1)}

### 2.3 主成分分析（PCA）

由於四個特徵尺度不同，先以 z-score 標準化後再進行 PCA。三個主成分的解釋變異比例分別為
PC1 {ratio[0]:.2%}、PC2 {ratio[1]:.2%}、PC3 {ratio[2]:.2%}，累積 {sum(ratio):.2%}。
圖 2 顯示 setosa 在 PC1 方向上與其他兩類完全分離，versicolor 與 virginica 則在邊界處略有交疊，
預期分類錯誤主要會出現在這兩類之間。

{_fig(results, 'pca_3d', 'Iris 資料前三個主成分之 3D 分布', 2)}

## 3. 研究方法

### 3.1 資料切分

以分層抽樣（stratified sampling）將資料依 8:2 切分為訓練集 {m['n_train']} 筆與測試集 {m['n_test']} 筆，
確保每類比例一致；並再從訓練集中切出 20%（約 {n_val} 筆）作為驗證集，用於觀察學習曲線。
測試集完全不參與訓練與調整，僅用於最終評估。隨機種子固定為 {results['seed']} 以確保可重現。

### 3.2 模型架構

| 項目 | 設定 |
|---|---|
| 前處理 | StandardScaler（僅以訓練資料擬合） |
| 網路結構 | 4 → {' → '.join(map(str, m['hidden_layers']))} → 3 |
| 隱藏層激活函數 | {m['activation'].upper()} |
| 輸出層 | Softmax |
| 損失函數 | Cross-entropy |
| 最佳化器 | {m['optimizer'].capitalize()}（學習率 {m['learning_rate']}） |
| Batch size | {m['batch_size']} |
| 訓練回合數 | {m['epochs']} epochs |

### 3.3 評估指標

以測試集上的準確率（accuracy）、各類別 precision / recall / F1 與混淆矩陣評估模型；
另以 5 折分層交叉驗證評估模型表現的穩定度。

## 4. 實驗結果

### 4.1 訓練過程

訓練損失由 {hist['loss'][0]:.3f} 下降至 {hist['loss'][-1]:.3f}；最終訓練準確率 {hist['train_acc'][-1]:.2%}、
驗證準確率 {hist['val_acc'][-1]:.2%}，{fit_text}

{_fig(results, 'training_history', '訓練損失與訓練／驗證準確率曲線', 3)}

### 4.2 測試集評估

測試集準確率 **{test['accuracy']:.2%}**，macro F1 = **{test['macro_f1']:.3f}**。

| 類別 | Precision | Recall | F1 | 樣本數 |
|---|---|---|---|---|
{class_rows}

{_fig(results, 'confusion_matrix', '測試集混淆矩陣', 4)}

### 4.3 交叉驗證

5 折交叉驗證準確率：{', '.join(f'{s:.2%}' for s in cv)}；平均 {cv_mean:.2%}，標準差 {cv_std:.2%}。

## 5. 討論

{discussion_err}整體結果與 PCA 觀察到的資料結構一致：setosa 在特徵空間中與其他類別線性可分，
而 versicolor 與 virginica 的花瓣尺寸有部分重疊，是本問題中唯一真正困難的部分。

本研究的限制包括：（1）資料量僅 150 筆，測試集只有 {m['n_test']} 筆，單一筆誤判即影響約
{1 / m['n_test']:.1%} 的準確率，因此另以交叉驗證輔助判斷；（2）Iris 為簡單且乾淨的資料集，
結果不一定能推廣到雜訊較多的真實資料；（3）未進行系統性的超參數搜尋。
未來可嘗試比較其他模型（如邏輯斯迴歸、SVM）或進行超參數調整。

## 6. 結論

本研究完成了 Iris 資料的取得、探索、PCA 3D 視覺化，並以兩層隱藏層的 MLP 類神經網路建立分類器。
模型在測試集上達到 {test['accuracy']:.2%} 準確率，交叉驗證平均 {cv_mean:.2%}，
顯示即使是小型類神經網路，在經過適當標準化與資料切分後，也能有效完成 Iris 分類任務。

## 參考文獻

1. Fisher, R. A. (1936). The use of multiple measurements in taxonomic problems. *Annals of Eugenics*, 7(2), 179–188.
2. Dua, D., & Graff, C. (2019). *UCI Machine Learning Repository*. University of California, Irvine.
3. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
4. Pearson, K. (1901). On lines and planes of closest fit to systems of points in space. *Philosophical Magazine*, 2(11), 559–572.
"""
