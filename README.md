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

## AI4I 2020 預測性維護：類神經網路能否學到故障規則（#3）

研究三個問題：**A** 模型的決策依據是否與資料產生規則一致；**B** 物理衍生特徵的價值，以及 MLP 能否自己學到乘積型規則；**C** 不同漏判／誤報成本下的最佳判斷門檻與預期成本。

```bash
python -m pytest                              # 執行全部測試
python -m ai4i_lab.pipeline                   # 一鍵重現：依序重跑所有分析，最後重新產生小論文與學習筆記
python -m ai4i_lab.pipeline --list            # 列出階段與依賴
python -m ai4i_lab.pipeline --skip-existing   # 輸出已存在且上游未重跑的階段跳過
python -m ai4i_lab.pipeline --report-only     # 只用現有結果重新產生 docs
```

完整重跑在 2 顆 CPU 上約需一小時（MLP 搜尋、重複 CV、可解釋性最久）；只改文件時用 `--report-only`。

| 路徑 | 內容 |
|---|---|
| `ai4i_lab/` | `data`（取得與 EDA）、`rules`（規則重建）、`features`／`pca`、`evaluation`（切分與評估框架）、`models`（二元分類）、`multilabel`、`viz`（評估圖與錯誤分析）、`explain`（可解釋性）、`cost`（成本決策）、`report`（論文與筆記產生）、`pipeline`（一鍵流程） |
| `tests/test_ai4i_*.py` | pytest 測試（TDD 開發） |
| `data/ai4i2020.csv` | AI4I 2020 資料集（SHA-256 見 `data/ai4i2020.meta.json`） |
| `outputs/ai4i/` | 各階段的圖表、表格與 JSON 結果 |
| `docs/ai4i_paper.md` | 小論文（所有數字、表格、圖表由結果自動產生） |
| `docs/ai4i_notes.md` | 學習筆記（每步驟：目的、做法、結果判讀、踩到的坑） |
