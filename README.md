# ai-rookie-practice

上課練習用。

## Iris 類神經網路分類（#2）

```bash
pip install -r requirements.txt
python -m pytest                 # 執行測試
python -m iris_lab.pipeline      # 跑完整實驗，產生資料、圖表、結果與小論文
```

| 路徑 | 內容 |
|---|---|
| `iris_lab/` | 程式碼：資料（data）、PCA（pca）、模型（model）、圖表（viz）、小論文（report）、流程（pipeline） |
| `tests/` | pytest 測試（以 TDD 方式開發） |
| `data/iris.csv` | Iris 資料集 |
| `outputs/figures/` | 特徵分布、PCA 3D、訓練曲線、混淆矩陣 |
| `outputs/results.json` | 完整實驗數據 |
| `docs/paper.md` | 小論文 |
