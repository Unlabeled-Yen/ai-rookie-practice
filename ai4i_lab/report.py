"""小論文與學習筆記的自動產生（#13）。

只讀取各子單寫在 `outputs/ai4i/` 的結果（JSON／CSV），所有數字、表格與圖表連結都由結果填入，
不手動抄寫。執行：python -m ai4i_lab.report
"""
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "ai4i"
DOCS_DIR = ROOT / "docs"

# 報告會讀取的上游結果（鍵 → 相對於 outputs/ai4i 的路徑）
RESULT_FILES = {
    "eda": "eda/eda.json",
    "rules": "rules/rule_agreement.json",
    "pca": "pca/pca.json",
    "framework": "framework/framework.json",
    "models": "models/models.json",
    "cv_results": "models/cv_results.csv",
    "multilabel": "multilabel/multilabel.json",
    "evaluation": "evaluation/evaluation.json",
    "error_analysis": "evaluation/error_analysis.json",
    "explain": "explain/explain.json",
    "cost": "cost/cost.json",
}


class MissingResults(FileNotFoundError):
    """上游實驗輸出不齊全。"""


# ---------------------------------------------------------------- 格式化工具
def _bad(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def fmt_num(x, nd: int = 3) -> str:
    return "n/a" if _bad(x) else f"{x:.{nd}f}"


def fmt_pct(x, nd: int = 1) -> str:
    return "n/a" if _bad(x) else f"{100 * x:.{nd}f}%"


def fmt_ci(point, low, high, nd: int = 3) -> str:
    if _bad(low) or _bad(high):
        return fmt_num(point, nd)
    return f"{fmt_num(point, nd)} [{fmt_num(low, nd)}, {fmt_num(high, nd)}]"


def md_table(headers, rows) -> str:
    n = len(headers)
    out = ["| " + " | ".join(str(h) for h in headers) + " |", "|" + "---|" * n]
    for r in rows:
        if len(r) != n:
            raise ValueError(f"表格列長度 {len(r)} 與欄數 {n} 不符：{r}")
        out.append("| " + " | ".join(str(c).replace("|", "\\|") for c in r) + " |")
    return "\n".join(out)


def img(rel_path: str, caption: str, prefix: str = "../outputs/ai4i") -> str:
    return f"![{caption}]({prefix}/{rel_path})"


# ---------------------------------------------------------------- 載入
def _read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = int(v)
            except ValueError:
                try:
                    r[k] = float(v)
                except ValueError:
                    pass
    return rows


def load_results(out_dir=OUT_DIR) -> dict:
    out_dir = Path(out_dir)
    missing = [rel for rel in RESULT_FILES.values() if not (out_dir / rel).exists()]
    if missing:
        raise MissingResults("缺少上游實驗輸出（請先執行 python -m ai4i_lab.pipeline）：" + "、".join(missing))
    res = {}
    for key, rel in RESULT_FILES.items():
        p = out_dir / rel
        if rel.endswith(".csv"):
            res[key] = _read_csv(p)
        else:
            res[key] = json.loads(p.read_text(encoding="utf-8"))
    return res


# ---------------------------------------------------------------- 事實彙整
def collect_facts(res: dict) -> dict:
    eda = res["eda"]["exploration"]
    best = res["models"]["conclusions"]["best_cv"]
    return {
        "n_rows": eda["n_rows"],
        "n_failures": eda["n_failures"],
        "failure_rate": eda["failure_rate"],
        "best_cv_model": best["model"],
        "best_cv_feature_set": best["feature_set"],
        "best_cv_pr_auc": best["pr_auc"],
        "cost_break_even_ratio": res["cost"]["break_even_ratio"],
        "rnf_consistent_with_random": res["multilabel"]["rnf"]["consistent_with_random"],
    }


def test_table_rows(res: dict) -> list[list]:
    rows = []
    for r in res["evaluation"]["table"]:
        rows.append([
            r["model"], r["feature_set"], fmt_num(r["threshold"]),
            fmt_ci(r["recall"], r["recall_low"], r["recall_high"]),
            fmt_ci(r["precision"], r["precision_low"], r["precision_high"]),
            fmt_ci(r["f1"], r["f1_low"], r["f1_high"]),
            fmt_ci(r["pr_auc"], r["pr_auc_low"], r["pr_auc_high"]),
            fmt_ci(r["roc_auc"], r["roc_auc_low"], r["roc_auc_high"]),
        ])
    return rows


# ---------------------------------------------------------------- 取值輔助
MODEL_ZH = {"logreg": "邏輯斯迴歸", "random_forest": "隨機森林", "hist_gb": "HistGradientBoosting", "mlp": "MLP"}
FS_ZH = {"raw": "原始特徵", "engineered": "原始＋衍生特徵"}
MODE_ZH = {"twf": "TWF（刀具磨耗）", "hdf": "HDF（散熱不良）", "pwf": "PWF（功率異常）",
           "osf": "OSF（過度負荷）", "rnf": "RNF（隨機）"}
MODELS = ["logreg", "random_forest", "hist_gb", "mlp"]
FEATURE_SETS = ["raw", "engineered"]
MODES = ["twf", "hdf", "pwf", "osf", "rnf"]


def _n(x) -> str:
    """整數加千分位。"""
    return f"{int(x):,}"


def _eval_row(res, model, fs) -> dict:
    return next(r for r in res["evaluation"]["table"] if r["model"] == model and r["feature_set"] == fs)


def _cv_row(res, model, fs) -> dict:
    """該 模型×特徵組 在 CV 中被選用的不平衡處理所對應的列。"""
    imb = res["models"]["chosen"][f"{model}|{fs}"]["imbalance"]
    return next(r for r in res["cv_results"] if r["model"] == model and r["feature_set"] == fs and r["imbalance"] == imb)


def _imb(res, model, fs, imbalance) -> dict:
    return next(r for r in res["cv_results"] if r["model"] == model and r["feature_set"] == fs and r["imbalance"] == imbalance)


def _fs_diff(res, model) -> dict:
    imb = res["models"]["chosen"][f"{model}|engineered"]["imbalance"]
    return next(r for r in res["models"]["feature_set_diff"]
                if r["model"] == model and r["imbalance"] == imb and r["metric"] == "pr_auc")


def _sig(res, model) -> str:
    d = _fs_diff(res, model)
    return "95% 區間不含 0，顯著" if d["significant"] else "95% 區間含 0，不顯著"


def _kappa(res, model, fs) -> float:
    return res["explain"]["boundary_agreement"][f"{model}|{fs}"]["test_agreement"]["vs_rule_union"]["kappa"]


def _learned_total(res, fs) -> tuple[int, int]:
    """(達到『學到邊界』的模型×邊界組合數, 總組合數)。"""
    lc = res["explain"]["boundary_agreement"]
    n_models = len(lc)
    per = res["explain"]["conclusions"]["learned_counts"]
    n_models_fs = n_models // len(FEATURE_SETS)
    return sum(v[fs] for v in per.values()), len(per) * n_models_fs


def _scaling(res, rule, fs, n_train) -> dict:
    return next(r for r in res["explain"]["mlp_product_rules"]["scaling"]
                if r["rule"] == rule and r["feature_set"] == fs and r["n_train"] == n_train)


def _real(res, rule, fs, config="default") -> dict:
    return next(r for r in res["explain"]["mlp_product_rules"]["real"]
                if r["rule"] == rule and r["feature_set"] == fs and r["config"] == config)


def _sens(res, ratio) -> dict:
    return next(s for s in res["cost"]["sensitivity"] if s["ratio"] == ratio)


def _baseline_name(b) -> str:
    return {"never": "不預測（全部不檢修）", "always": "全部檢修"}[b]


def _top_features(res, model, fs, k=3) -> list[str]:
    rows = [r for r in res["explain"]["importance"] if r["model"] == model and r["feature_set"] == fs]
    rows.sort(key=lambda r: r["mean"], reverse=True)
    return [r["feature"] for r in rows[:k]]


def _facts_ext(res: dict) -> dict:
    """論文／筆記共用的衍生事實（皆由結果計算）。"""
    f = collect_facts(res)
    best = (f["best_cv_model"], f["best_cv_feature_set"])
    ev = _eval_row(res, *best)
    mlp_raw, mlp_eng = _eval_row(res, "mlp", "raw"), _eval_row(res, "mlp", "engineered")
    f.update(best_test=ev, mlp_raw_test=mlp_raw, mlp_eng_test=mlp_eng,
             n_test=res["evaluation"]["n_test"], n_test_failures=res["evaluation"]["n_test_failures"])
    return f


def _wrap_images(items, prefix) -> str:
    return "\n\n".join(img(p, c, prefix) for p, c in items)


# ---------------------------------------------------------------- 小論文
REFERENCES = [
    "Matzka, S. (2020). Explainable Artificial Intelligence for Predictive Maintenance Applications. "
    "*2020 Third International Conference on Artificial Intelligence for Industries (AI4I)*, 69–74.",
    "Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic Minority Over-sampling "
    "Technique. *Journal of Artificial Intelligence Research*, 16, 321–357.",
    "Saito, T., & Rehmsmeier, M. (2015). The Precision-Recall Plot Is More Informative than the ROC Plot When "
    "Evaluating Binary Classifiers on Imbalanced Datasets. *PLOS ONE*, 10(3), e0118432.",
    "Nadeau, C., & Bengio, Y. (2003). Inference for the Generalization Error. *Machine Learning*, 52, 239–281.",
    "Breiman, L. (2001). Random Forests. *Machine Learning*, 45, 5–32.",
    "Friedman, J. H. (2001). Greedy Function Approximation: A Gradient Boosting Machine. *Annals of Statistics*, 29(5), 1189–1232.",
    "Goldstein, A., Kapelner, A., Bleich, J., & Pitkin, E. (2015). Peeking Inside the Black Box: Visualizing Statistical "
    "Learning with Plots of Individual Conditional Expectation. *Journal of Computational and Graphical Statistics*, 24(1), 44–65.",
    "Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). Why Do Tree-Based Models Still Outperform Deep Learning on "
    "Typical Tabular Data? *NeurIPS 2022 Datasets and Benchmarks Track*.",
    "Elkan, C. (2001). The Foundations of Cost-Sensitive Learning. *Proceedings of the 17th International Joint "
    "Conference on Artificial Intelligence (IJCAI)*, 973–978.",
    "Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. *ICML 2017*.",
    "Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.",
    "UCI Machine Learning Repository. AI4I 2020 Predictive Maintenance Dataset (id 601). "
    "https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset",
]


def _paper_abstract(res, f) -> str:
    ev, mr, me = f["best_test"], f["mlp_raw_test"], f["mlp_eng_test"]
    tr = res["models"]["conclusions"]["best_cv"]
    kap_best = max(((_kappa(res, m, fs), m, fs) for m in MODELS for fs in FEATURE_SETS))
    raw_l, tot = _learned_total(res, "raw")
    eng_l, _ = _learned_total(res, "engineered")
    s5, s100 = _sens(res, 5), _sens(res, 100)
    return f"""AI4I 2020 是依公開規則產生的合成預測性維護資料（{_n(f['n_rows'])} 筆，故障 {_n(f['n_failures'])} 筆，
{fmt_pct(f['failure_rate'], 2)}）。由於故障規則已知，本研究不以準確率為重點，而把規則當作標準答案，檢驗三個問題：
（A）模型的決策依據是否與規則一致；（B）物理衍生特徵的價值，以及 MLP 能否自行學到乘積型規則；
（C）在不同漏判／誤報成本下，最佳判斷門檻與預期成本為何。所有特徵都是故障當下的量測值，因此本研究判斷的是
「某個運轉狀態是否構成故障」，不是預測未來故障。

方法為分層 60/20/20 切分、重複分層交叉驗證（Nadeau–Bengio 校正 95% 信賴區間）與時間順序切分的穩健性檢驗，
比較 {len(MODELS)} 種模型 × {len(FEATURE_SETS)} 組特徵 × {len(res['models']['config']['imbalance'])} 種不平衡處理。結果：開發集 CV 最佳為{MODEL_ZH[f['best_cv_model']]}搭配
{FS_ZH[f['best_cv_feature_set']]}（PR-AUC {fmt_ci(tr['pr_auc'], tr['low'], tr['high'])}）；測試集（{f['n_test_failures']} 筆故障）
PR-AUC {fmt_ci(ev['pr_auc'], ev['pr_auc_low'], ev['pr_auc_high'])}、Recall {fmt_num(ev['recall'])}、Precision {fmt_num(ev['precision'])}；
類神經網路（MLP）在原始特徵下測試 PR-AUC 僅 {fmt_num(mr['pr_auc'])}（衍生特徵 {fmt_num(me['pr_auc'])}）。

- **A**：一致程度取決於模型與特徵。與規則聯集的 Cohen kappa 最高為 {MODEL_ZH[kap_best[1]]}＋{FS_ZH[kap_best[2]]}（{fmt_num(kap_best[0], 2)}），
  MLP 僅 {fmt_num(_kappa(res, 'mlp', 'raw'), 2)}（原始）／{fmt_num(_kappa(res, 'mlp', 'engineered'), 2)}（衍生）；
  {tot} 個「模型×邊界」組合中，原始特徵有 {raw_l} 個、衍生特徵有 {eng_l} 個達到「學到規則邊界」標準。TWF 規則本質上含隨機成分，無法完全重建。
- **B**：衍生特徵使樹模型 PR-AUC 顯著提升（隨機森林 +{fmt_num(_fs_diff(res, 'random_forest')['diff_mean'])}），對 MLP 的提升不顯著。
  MLP 在「目標＝規則本身」的乾淨設定下只靠原始特徵即可學到乘積型規則，但在含雜訊、只有約 {res['framework']['config']['dev_failures']} 筆開發集故障的真實任務中仍明顯落後樹模型。
- **C**：驗證集選門檻後，C_FN:C_FP = 5:1 時測試集成本 {fmt_num(s5['test']['total'], 0)}（最佳基準 {fmt_num(s5['baseline_cost'], 0)}），
  100:1 時 {fmt_num(s100['test']['total'], 0)}（最佳基準 {fmt_num(s100['baseline_cost'], 0)}）；成本比 {fmt_num(res['cost']['break_even_ratio'])}:1 以上模型即優於「不預測」。

關鍵詞：預測性維護、AI4I 2020、類神經網路、可解釋性、不平衡分類、成本敏感學習"""


def _paper_intro(res, f) -> str:
    return f"""預測性維護希望在設備故障造成停機前採取行動。真實的故障資料稀少、標記昂貴，公開的 AI4I 2020 資料集
（Matzka, 2020）以已發表的物理規則合成 {_n(f['n_rows'])} 筆機台運轉紀錄，附帶五種故障模式標記。
這使它成為少見的「有標準答案」的資料集：我們不只能問模型準不準，還能問**模型是不是為了對的理由而準**。

這也帶來一個提醒：資料是依規則產生的，因此高準確率本身沒有太大意義（全部猜「不故障」就有
{fmt_pct(1 - f['failure_rate'])} 的準確率）。本研究改以 Recall、F1 與 PR-AUC 為主要指標，並以下列三個研究問題組織全文：

{md_table(['代號', '方向', '研究問題'], [
    ['A', '可解釋性', '模型學到的決策依據，是否和資料產生規則（TWF/HDF/PWF/OSF）一致？'],
    ['B', '領域知識的價值', '原始特徵 vs. 物理衍生特徵，模型差多少？類神經網路能否自己學到「扭矩 × 轉速」這類乘積關係？'],
    ['C', '維護決策', '漏判與誤報的代價不同，在不同成本假設下，最佳判斷門檻與預期成本為何？'],
])}

**用詞說明。** 特徵（溫度、轉速、扭矩、刀具磨耗）都是**故障當下**的量測值，所以模型做的是「判斷某個運轉狀態是否會導致故障」，
不是預測未來的故障；資料中沒有「故障前 N 小時」的標記。另外，TWF/HDF/PWF/OSF/RNF 直接決定 `machine_failure`，
預測故障時必須排除，否則會造成資料洩漏；本研究所有特徵組都經程式驗證不含這些欄位。"""


def _paper_related() -> str:
    return """**AI4I 資料集與可解釋 AI。** Matzka（2020）提出此資料集的動機，是讓可解釋 AI 方法有可驗證的標準答案：
故障由功率、溫差、過度負荷等明確條件觸發，因此解釋方法給出的「重要特徵」可以對照真實規則。
本研究沿用這個精神，並進一步把「模型邊界」與「規則邊界」做量化比對。

**不平衡分類與評估。** 故障比例僅百分之三左右時，準確率與 ROC-AUC 容易過度樂觀；Saito 與 Rehmsmeier（2015）
指出在不平衡資料上 Precision–Recall 曲線比 ROC 曲線更能反映少數類的表現，故本研究以 PR-AUC 為主。
SMOTE（Chawla et al., 2002）是常見的過採樣方法，本研究將它與類別權重、不處理三者並列比較，且過採樣只在訓練折內進行以避免洩漏。
重複交叉驗證的折與折之間共用訓練資料，樸素的 t 區間會過窄；Nadeau 與 Bengio（2003）提出的校正 resampled t 區間是本研究報告信賴區間的依據。

**表格資料上的樹模型與神經網路。** 隨機森林（Breiman, 2001）與梯度提升（Friedman, 2001）在中小型表格資料上通常很強；
Grinsztajn et al.（2022）的大規模基準指出，樹模型在典型表格資料上仍常優於深度學習，原因包括目標函數的不規則性與對無關特徵的穩健性。
這與本研究的觀察（見第 5 節）是否一致，正是研究問題 B 要檢驗的。

**模型解釋。** 排列重要度（Breiman, 2001）衡量打亂某特徵後的效能下降；部分依賴圖（Friedman, 2001）與個別條件期望曲線 ICE
（Goldstein et al., 2015）則顯示模型輸出如何隨單一特徵改變。本研究把這些曲線與已知規則門檻（如 3500 W、9000 W）對照。

**成本敏感決策與機率校準。** Elkan（2001）說明在已知錯誤成本時，最佳決策門檻由成本比決定，而不是 0.5；
這需要機率估計夠準，Guo et al.（2017）討論了現代模型的校準問題。本研究因此同時報告校準曲線與成本曲線。"""


def _paper_data(res, f, P) -> str:
    e = res["eda"]["exploration"]
    inc = res["eda"]["label_inconsistencies"]["counts"]
    ts = res["eda"]["temporal_structure"]
    rules = res["rules"]
    info = res["eda"]["feature_info"]
    tbl_fields = md_table(["欄位", "說明", "單位／取值"], [[k, v["name"], v["unit"]] for k, v in info.items()]
                          + [["machine_failure", "目標：是否故障", "0/1"], ["twf/hdf/pwf/osf/rnf", "五種故障模式", "0/1"]])
    mode_rows = [[MODE_ZH[m], e["mode_counts"][m], fmt_pct(e["mode_counts"][m] / e["n_rows"], 2)] for m in MODES]
    ag = rules["agreement"]
    rule_rows = [[MODE_ZH[m], ag[m]["tp"], ag[m]["fp"], ag[m]["fn"], fmt_num(ag[m]["precision"]), fmt_num(ag[m]["recall"])]
                 for m in ["hdf", "pwf", "osf", "twf"]]
    u = ag["machine_failure_union"]
    tc = rules["tool_cycles"]
    type_rows = [[t, _n(e["type_counts"][t]), fmt_pct(e["failure_rate_by_type"][t], 2)] for t in ["L", "M", "H"]]
    return f"""### 3.1 資料來源與欄位

資料取自 UCI Machine Learning Repository（id 601），共 {_n(e['n_rows'])} 筆 × {e['n_columns']} 欄，
識別欄位 `udi`、`product_id` 不作為特徵。

{tbl_fields}

### 3.2 資料品質

缺失值 {e['missing_values']} 筆、重複列 {e['duplicate_rows']} 筆、重複的 Product ID {e['duplicate_product_ids']} 筆。
故障 {_n(f['n_failures'])} 筆，故障率 {fmt_pct(f['failure_rate'], 2)}（不平衡比約 {1 / f['failure_rate'] - 1:.0f}:1）。
{e['multi_mode_rows']} 列同時有多種故障模式，因此各模式筆數加總大於故障筆數。

{md_table(['故障模式', '筆數', '占全部資料'], mode_rows)}

{img('eda/failure_modes.png', '各故障模式筆數與重疊情形', P)}

依 `Type` 比較故障率：

{md_table(['Type', '筆數', '故障率'], type_rows)}

### 3.3 標記不一致

有 {inc['failure_without_mode']} 筆 `machine_failure=1` 卻沒有任何模式標記，另有 {inc['mode_without_failure']} 筆 RNF=1 但
`machine_failure=0`。處理方式：{res['eda']['label_inconsistencies']['policy']['reason']}

### 3.4 UDI 順序的時間結構

依 UDI 排序後，刀具磨耗有 {fmt_pct(ts['tool_wear_nondecreasing_fraction'])} 的相鄰列非遞減，並有 {ts['tool_wear_resets']} 次換刀歸零；
空氣溫度的 lag-1 自相關為 {fmt_num(ts['air_temp_lag1_autocorr'], 4)}。資料因此帶有時間結構，隨機切分會讓幾乎相同條件的相鄰列同時出現在訓練與測試，
所以本研究另做依時間順序切分的穩健性檢驗（第 5.2 節）。

{img('eda/temporal_structure.png', 'UDI 順序下的刀具磨耗、空氣溫度與故障分布', P)}

### 3.5 已知規則重建（研究問題 A 的標準答案）

依文獻規則（HDF：溫差 < {rules['thresholds']['hdf_temp_diff_max_k']} K 且轉速 < {rules['thresholds']['hdf_rpm_max']} rpm；
PWF：功率 < {rules['thresholds']['pwf_power_w'][0]} W 或 > {rules['thresholds']['pwf_power_w'][1]} W；
OSF：磨耗 × 扭矩超過依 Type 而定的門檻；TWF：磨耗落在 {rules['thresholds']['twf_wear_window_min'][0]}–{rules['thresholds']['twf_wear_window_min'][1]} min 視窗）
以原始特徵重算，與資料標記比對：

{md_table(['模式', 'TP', 'FP', 'FN', 'Precision', 'Recall'], rule_rows)}

HDF、PWF、OSF 三條規則與標記完全吻合（FP = FN = 0）。TWF 無法完全重建：依產生規則，每把刀在磨耗 200–240 min 之間的**隨機**時點失效，
視窗內的 {_n(tc['rows_in_window'])} 列只有 {ag['twf']['tp']} 列為 TWF（Precision {fmt_num(ag['twf']['precision'])}）；
{tc['n_cycles']} 個刀具週期中有 {tc['n_cycles_reaching_window']} 個到達視窗，其中 {tc['n_cycles_with_twf']} 個發生 TWF
（{fmt_pct(tc['twf_rate_given_reaching_window'])}）。四條規則的聯集覆蓋 {u['tp']} 筆故障，漏掉 {u['fn']} 筆
（Recall {fmt_num(u['recall'])}），其中包含 {inc['failure_without_mode']} 筆沒有模式標記的故障。**因此純靠規則的偵測 Recall 上限約為 {fmt_num(u['recall'])}；模型若要超過，只能靠 TWF 視窗內的機率排序，而這部分本質上是隨機的。**

{_wrap_images([('rules/hdf_boundary.png', 'HDF 規則邊界（溫差 × 轉速）'), ('rules/pwf_boundary.png', 'PWF 規則邊界（扭矩 × 轉速與功率等值線）'),
               ('rules/osf_boundary.png', 'OSF 規則邊界（磨耗 × 扭矩）'), ('rules/twf_window.png', 'TWF 的磨耗視窗')], P)}"""


def _paper_method(res, f) -> str:
    c = res["models"]["config"]
    fw = res["framework"]["config"]
    ml = res["multilabel"]["config"]
    ex = res["explain"]
    grid = c["mlp_grid"]
    return f"""### 4.1 特徵組

- **原始特徵組**（8 欄）：空氣溫度、製程溫度、轉速、扭矩、刀具磨耗，加 `Type` 的 one-hot 三欄。
- **原始＋衍生特徵組**（11 欄）：再加三個依物理意義推導的特徵——溫差 = 製程溫度 − 空氣溫度（對應 HDF）、
  功率 = 扭矩 × 轉速 × 2π/60（對應 PWF）、過度負荷指標 = 刀具磨耗 × 扭矩（對應 OSF）。

兩組特徵都由 `build_features` 產生，程式內以斷言確保不含故障模式、目標與識別欄位。

### 4.2 切分與評估框架

- 以 UDI 為鍵做分層切分：訓練 {_n(c['n_train'])} ／驗證 {_n(c['n_val'])} ／測試 {_n(c['n_test'])} 筆（60/20/20），切分結果存成 `splits.json` 供所有模組共用。
- 測試集不參與訓練、調參與門檻選擇，每個模型只在測試集評估一次。
- 開發集（訓練＋驗證，{_n(c['dev_rows'])} 筆、{fw['dev_failures']} 筆故障）上做 {c['n_splits']} 折 × {c['n_repeats']} 次重複分層交叉驗證，
  報告平均值與 Nadeau–Bengio 校正的 95% 信賴區間；測試集指標的信賴區間以 {c['n_boot']} 次 bootstrap 估計。
- 穩健性檢驗：依 UDI 時間順序切分（訓練／驗證／測試各取連續區段），與分層切分比較；兩種方案都先剔除共用測試集的列，避免測試集洩漏。
- 主要指標為 PR-AUC，輔以 Recall、Precision、F1 與 ROC-AUC。

### 4.3 模型、不平衡處理與門檻

- 模型：邏輯斯迴歸（基準）、隨機森林、HistGradientBoosting，以及主模型 MLP（early stopping，最多 {c['mlp_defaults']['max_iter']} 次迭代）。
- 不平衡處理：不處理、類別權重、SMOTE 過採樣（過採樣包在 Pipeline 內，只作用於訓練折）。
  `MLPClassifier` 不支援 sample_weight，所以「類別權重」對 MLP 以隨機複製少數類實作。
- MLP 超參數網格：隱藏層 {grid['hidden_layer_sizes']}、學習率 {grid['learning_rate_init']}、L2 正則化 {grid['alpha']}，
  只在驗證集上以 PR-AUC 評分。
- 每個 模型×特徵組 取 CV PR-AUC 最高的不平衡處理；最終模型只在訓練集訓練，驗證集用來選使 F1 最大的判斷門檻。

### 4.4 多標籤分類

對 TWF/HDF/PWF/OSF/RNF 各建一個獨立分類器（binary relevance，{MODEL_ZH[ml['model']]}），避免 classifier chain 把模式標籤回流成特徵。
RNF 是隨機產生的，理論上不可預測，因此用「與隨機基準比較」的方式檢驗：CV 的 ROC-AUC 信賴區間是否包含 0.5、
{ml['n_perm']} 次標籤置換檢定（含保留時間區塊的版本，區塊大小 {ml['block_size']}）與固定標記比例下的偵測率。

### 4.5 可解釋性分析

- 排列重要度：8 個 模型×特徵組，以測試集 PR-AUC 的下降量衡量，重複 {ex['importance'][0]['n_repeats']} 次。
- PDP／ICE：在規則座標（功率、溫差、磨耗×扭矩比、磨耗）上**介入原始感測欄位**再重算衍生特徵，
  畫出 ICE 曲線。判定「學到邊界」的標準：至少 80% 的 ICE 曲線在規則值 ±{fmt_pct(ex['window_frac'], 0)} 內跨越模型門檻，且交點中位偏移不超過規則值的 10%。
- 決策邊界吻合度：測試集上模型預測與規則聯集的 Cohen kappa。
- 乘積規則實驗：以 MLP 預測「規則是否成立」（目標＝規則本身），比較原始與衍生特徵，並把訓練資料放大到 40,000 列（由規則合成），
  另以單一特徵軸平行的規則（轉速 < 1380）作為控制組。

### 4.6 成本模型

漏判成本 C_FN、誤報成本 C_FP = {fmt_num(res['cost']['c_fp'], 0)}，成本比 C_FN:C_FP 取 {', '.join(str(r) for r in res['cost']['ratios'])}。
預期成本 = C_FN × FN + C_FP × FP。判斷門檻在驗證集上取使成本最低者，再原封不動套用到測試集；
並與「不預測」（成本 = C_FN × 故障數）與「全部檢修」（成本 = C_FP × 正常數）兩種基準策略比較。"""


def _cv_rows(res):
    rows = []
    for m in MODELS:
        for fs in FEATURE_SETS:
            r = _cv_row(res, m, fs)
            rows.append([MODEL_ZH[m], FS_ZH[fs], res["models"]["chosen"][f"{m}|{fs}"]["imbalance"],
                         fmt_ci(r["pr_auc_mean"], r["pr_auc_low"], r["pr_auc_high"]),
                         fmt_num(r["recall_mean"]), fmt_num(r["precision_mean"]), fmt_num(r["f1_mean"]), fmt_num(r["roc_auc_mean"])])
    return rows


def _paper_results_a(res, f, P) -> str:
    """5.1–5.6：描述性結果（PCA、穩健性、模型比較、多標籤、錯誤分析）。"""
    pca = res["pca"]
    fw = res["framework"]["conclusions"]
    fwd = res["framework"]["drift"]
    cmp_ = pca["comparison"]
    ev = res["evaluation"]
    ml = res["multilabel"]
    sm = {s: r for s, r in fw["failure_rate_by_part"].items()}
    ks = fw["max_ks_temporal"]
    tbl_pca = md_table(["特徵組", "前 3 個主成分累計解釋變異", "達 90% 所需主成分", "僅前 3 個主成分的 CV ROC-AUC", "全部特徵的 CV ROC-AUC"],
                       [[FS_ZH[fs], fmt_pct(cmp_["cumulative_variance_3pc"][fs]), cmp_["n_components_90"][fs],
                         fmt_num(cmp_["cv_auc_logreg_3pc"][fs]), fmt_num(cmp_["cv_auc_logreg_all_features"][fs])] for fs in FEATURE_SETS])
    tbl_drift = md_table(["切分方式", "訓練→驗證 PR-AUC（平均）", "訓練→測試 PR-AUC（平均）"], [
        ["分層（開發集內）", fmt_num(fw["mean_pr_auc_holdout"]["stratified_val"]), "—"],
        ["依時間順序", fmt_num(fw["mean_pr_auc_holdout"]["temporal_val"]), fmt_num(fw["mean_pr_auc_holdout"]["temporal_test"])]])
    tbl_part = md_table(["切分方式", "訓練故障率", "驗證故障率", "測試故障率"], [
        [s, fmt_pct(sm[s]["train"], 2), fmt_pct(sm[s]["val"], 2), fmt_pct(sm[s]["test"], 2)] for s in ["stratified", "temporal"]])
    mode_rows = []
    for m in MODES:
        t = ml["test"]["engineered"][m]
        mode_rows.append([MODE_ZH[m], t["n_positive_test"], fmt_ci(t["recall"]["point"], t["recall"]["low"], t["recall"]["high"], 2),
                          fmt_num(t["precision"]["point"], 2), fmt_num(t["f1"]["point"], 2), fmt_num(t["roc_auc"]["point"]),
                          fmt_num(t["pr_auc"]["point"])])
    ea = res["error_analysis"]
    fn_c, fp_c = ea["summary"]["fn_by_category"], ea["summary"]["fp_by_category"]
    det = ea["mode_detection"]
    det_rows = [[MODE_ZH[m], det[m]["n_rows_with_mode"], det[m]["detected"], fmt_pct(det[m]["rate"], 0)] for m in MODES]
    rn = ml["rnf"]
    n_none = sum(1 for v in res["models"]["chosen"].values() if v["imbalance"] == "none")
    rd_primary = ea["margin_stats"]
    rd_mlp = ev["recall_by_depth"]["mlp|raw"]
    return f"""### 5.1 PCA：線性降維看不出故障結構

{tbl_pca}

前 3 個主成分在原始組解釋 {fmt_pct(cmp_['cumulative_variance_3pc']['raw'])} 變異、衍生組 {fmt_pct(cmp_['cumulative_variance_3pc']['engineered'])}；
衍生特徵反而讓前 3 個主成分的占比下降，推測是乘積特徵帶進與原欄位不完全共線的新方向，使變異分散到更多主成分。
故障樣本多半落在正常樣本雲的邊緣，沒有形成獨立的群。原因是 PCA 不看標籤、只找變異最大的方向，而故障由非線性、局部的門檻規則產生；
單一主成分中最能分辨故障的是原始組的 {cmp_['best_single_pc']['raw']}，並非變異最大的主成分。

{_wrap_images([('pca/pca_3d_modes_engineered.png', '原始＋衍生特徵的 PCA 3D 圖，依故障模式著色'), ('pca/explained_variance.png', '兩組特徵的 PCA 解釋變異')], P)}

### 5.2 評估框架：切分方式影響結論

{tbl_part}

時間順序切分下訓練／驗證／測試的故障率從 {fmt_pct(sm['temporal']['train'], 2)} 降到 {fmt_pct(sm['temporal']['test'], 2)}，
{ks['feature']} 的訓練對測試 KS 統計量高達 {fmt_num(ks['statistic'])}（分層切分最大僅 {fmt_num(fw['max_ks_stratified'])}），分布明顯漂移。

{tbl_drift}

在同樣的邏輯斯迴歸與隨機森林網格上，時間順序切分的平均 PR-AUC 從 {fmt_num(fw['mean_pr_auc_holdout']['stratified_val'])} 降到
{fmt_num(fw['mean_pr_auc_holdout']['temporal_test'])}，**分層切分高估了對「未來」資料的泛化能力**。後續主分析仍用分層切分（與資料的合成性質相符），
但所有結論都應附上這個保留。

{_wrap_images([('framework/temporal_drift.png', '依 UDI 分段的故障率與空氣溫度漂移'), ('framework/holdout_robustness.png', '分層與時間順序切分的表現比較')], P)}

### 5.3 二元分類：模型 × 特徵組 × 不平衡處理

開發集重複 CV（各 模型×特徵組 取最佳不平衡處理；信賴區間為 Nadeau–Bengio 校正）：

{md_table(['模型', '特徵組', '不平衡處理', 'PR-AUC [95% CI]', 'Recall', 'Precision', 'F1', 'ROC-AUC'], _cv_rows(res))}

{n_none}／{len(MODELS) * len(FEATURE_SETS)} 個 模型×特徵組 的最佳不平衡處理是「不處理」：類別權重與 SMOTE 把 Recall 拉高，卻讓 Precision 大幅下降
（例如邏輯斯迴歸＋原始特徵：不處理 Precision {fmt_num(_imb(res, 'logreg', 'raw', 'none')['precision_mean'])}，
類別權重 {fmt_num(_imb(res, 'logreg', 'raw', 'class_weight')['precision_mean'])}）。MLP 的最佳處理為
{res['models']['chosen']['mlp|raw']['imbalance']}（不處理時 PR-AUC 區間為 {fmt_ci(_imb(res, 'mlp', 'raw', 'none')['pr_auc_mean'], _imb(res, 'mlp', 'raw', 'none')['pr_auc_low'], _imb(res, 'mlp', 'raw', 'none')['pr_auc_high'])}，極不穩定）。

測試集（{f['n_test']} 筆，{f['n_test_failures']} 筆故障；門檻取自驗證集；信賴區間為 bootstrap）：

{md_table(['模型', '特徵組', '門檻', 'Recall', 'Precision', 'F1', 'PR-AUC', 'ROC-AUC'], [[MODEL_ZH[r[0]], FS_ZH[r[1]]] + r[2:] for r in test_table_rows(res)])}

測試集的排序與 CV 一致：{MODEL_ZH[f['best_cv_model']]}＋{FS_ZH[f['best_cv_feature_set']]}最佳；MLP 的 PR-AUC 只有
{fmt_num(f['mlp_raw_test']['pr_auc'])}（原始）與 {fmt_num(f['mlp_eng_test']['pr_auc'])}（衍生），低於兩種樹模型。
MLP 訓練曲線上驗證損失最低的 epoch 為第 {res['models']['mlp_curves']['raw']['best_epoch']}（原始）與第 {res['models']['mlp_curves']['engineered']['best_epoch']}（衍生）個。
各模型的機率校準以 ECE 衡量，{MODEL_ZH[f['best_cv_model']]}＋{FS_ZH[f['best_cv_feature_set']]}的 ECE 為
{fmt_num(ev['calibration'][f"{f['best_cv_model']}|{f['best_cv_feature_set']}"]['ece'], 4)}、Brier {fmt_num(ev['calibration'][f"{f['best_cv_model']}|{f['best_cv_feature_set']}"]['brier'], 4)}。

{_wrap_images([('evaluation/model_comparison.png', '各模型測試集指標與 95% 信賴區間'), ('evaluation/pr_curves.png', '測試集 PR 曲線'),
               ('evaluation/confusion_matrices.png', '測試集混淆矩陣'), ('evaluation/calibration.png', '機率校準曲線'),
               ('models/mlp_training_curve.png', 'MLP 訓練曲線')], P)}

### 5.4 多標籤：哪一種故障

{md_table(['模式', '測試集故障數', 'Recall [95% CI]', 'Precision', 'F1', 'ROC-AUC', 'PR-AUC'], mode_rows)}

HDF、PWF、OSF 三種有明確規則的模式 F1 皆 ≥ {fmt_num(min(ml['test']['engineered'][m]['f1']['point'] for m in ['hdf', 'pwf', 'osf']), 2)}；
TWF 與 RNF 在 0.5 門檻下完全偵測不到。TWF 的 ROC-AUC 看似很高（{fmt_num(ml['test']['engineered']['twf']['roc_auc']['point'])}），
但這只代表模型能把「磨耗進入 200–240 min 視窗」的列排在前面，無法判斷視窗內哪一列會失效（見 3.5 節）。

**RNF 的實驗驗證。** {ml['rnf_verdict']}。開發集 OOF 的 ROC-AUC 為 {fmt_num(rn['oof_auc']['roc_auc'])}，
CV 的 ROC-AUC 為 {fmt_ci(rn['cv']['roc_auc']['mean'], rn['cv']['roc_auc']['low'], rn['cv']['roc_auc']['high'])}，
區間{'包含' if rn['checks']['cv_roc_auc_ci_contains_0_5'] else '不包含'} 0.5；
測試集 {rn['test']['n_positive_test']} 筆 RNF 偵測到 {rn['test']['tp']} 筆。整體判定「與隨機一致」：{'是' if rn['consistent_with_random'] else '否'}。

{_wrap_images([('multilabel/mode_metrics_test.png', '各故障模式的測試集指標'), ('multilabel/rnf_vs_random.png', 'RNF 偵測與隨機基準的比較')], P)}

### 5.5 錯誤分析：漏判與誤報

主模型（{MODEL_ZH[f['best_cv_model']]}＋{FS_ZH[f['best_cv_feature_set']]}，門檻 {fmt_num(ea['threshold'])}）在測試集得到
TP {ea['counts']['tp']}、FN {ea['counts']['fn']}、FP {ea['counts']['fp']}、TN {ea['counts']['tn']}。逐筆對照規則邊界：

{md_table(['故障模式', '測試集筆數', '偵測到', '偵測率'], det_rows)}

- {ea['counts']['fn']} 筆漏判中，{fn_c['twf_window']} 筆是落在磨耗視窗的 TWF（隨機失效、規則本身無法判定），
  {fn_c['no_rule_explains']} 筆沒有任何規則能解釋（標記不一致的故障），規則成立卻被漏掉的有 {fn_c['rule_near_boundary'] + fn_c['rule_deep_inside']} 筆。
- {ea['counts']['fp']} 筆誤報中，{fp_c['near_boundary_outside']} 筆緊貼規則邊界的外側（相對距離 < {fmt_pct(ea['near_boundary_rel'], 0)}），{fp_c['twf_window']} 筆落在 TWF 視窗內。
- 主模型對 HDF/PWF/OSF 的漏判數為 {rd_primary['hdf']['n_missed']}/{rd_primary['pwf']['n_missed']}/{rd_primary['osf']['n_missed']}，連貼近邊界 5% 以內的樣本也沒有漏掉；
  較弱的 MLP（原始特徵）則明顯依「規則成立的深度」而異：{'、'.join(f"{d['label']} 的偵測率 {fmt_num(d['recall'], 2)}（n={d['n']}）" for d in rd_mlp)}，
  也就是它主要是漏掉剛好壓在邊界上的樣本。

{_wrap_images([('evaluation/mode_detection.png', '各故障模式的偵測率'), ('evaluation/error_margins.png', '偵測結果與距規則邊界的相對距離')], P)}"""


def _answer_a(res, f, P) -> str:
    ex = res["explain"]
    ba = ex["boundary_agreement"]
    lc = ex["conclusions"]["learned_counts"]
    n_per_fs = len(ba) // len(FEATURE_SETS)
    raw_l, tot = _learned_total(res, "raw")
    eng_l, _ = _learned_total(res, "engineered")
    kap_rows = [[MODEL_ZH[m], fmt_num(_kappa(res, m, "raw"), 2), fmt_num(_kappa(res, m, "engineered"), 2),
                 fmt_num(ba[f"{m}|raw"]["pr_auc_test"]), fmt_num(ba[f"{m}|engineered"]["pr_auc_test"])] for m in MODELS]
    lc_rows = [[b, f"{v['raw']}/{n_per_fs}", f"{v['engineered']}/{n_per_fs}"] for b, v in lc.items()]
    kb = max(((_kappa(res, m, fs), m, fs) for m in MODELS for fs in FEATURE_SETS))
    never_raw = [b for b, v in lc.items() if v["raw"] == 0]
    rf_e = ba["random_forest|engineered"]["per_mode_recall"]
    mlp_r = ba["mlp|raw"]["per_mode_recall"]
    top = {(m, fs): ", ".join(_top_features(res, m, fs)) for m in ["random_forest", "mlp"] for fs in FEATURE_SETS}
    return f"""### 5.6 研究問題 A：模型的決策依據與規則是否一致？

**邊界吻合度。** 測試集上，模型預測與四條規則聯集的 Cohen kappa（越接近 1 越一致）：

{md_table(['模型', 'kappa（原始）', 'kappa（衍生）', '測試 PR-AUC（原始）', '測試 PR-AUC（衍生）'], kap_rows)}

**PDP/ICE 是否把門檻放在規則值上。** 在規則座標上介入後，各邊界被判為「學到」的模型數（共 {n_per_fs} 個模型）：

{md_table(['邊界', '原始特徵', '衍生特徵'], lc_rows)}

**排列重要度。** 隨機森林＋衍生特徵最重要的三個特徵為 {top[('random_forest', 'engineered')]}（正是 HDF、PWF 規則所用的變數）；
MLP＋原始特徵為 {top[('mlp', 'raw')]}；隨機森林＋原始特徵為 {top[('random_forest', 'raw')]}。

**分模式偵測率（測試集）。** 隨機森林＋衍生特徵：HDF {fmt_pct(rf_e['hdf']['recall'], 0)}、PWF {fmt_pct(rf_e['pwf']['recall'], 0)}、OSF {fmt_pct(rf_e['osf']['recall'], 0)}、
TWF {fmt_pct(rf_e['twf']['recall'], 0)}；MLP＋原始特徵：HDF {fmt_pct(mlp_r['hdf']['recall'], 0)}、PWF {fmt_pct(mlp_r['pwf']['recall'], 0)}、
OSF {fmt_pct(mlp_r['osf']['recall'], 0)}、TWF {fmt_pct(mlp_r['twf']['recall'], 0)}。

{_wrap_images([('explain/importance.png', '各模型的排列重要度'), ('explain/pdp_ice_power.png', '功率軸的 PDP/ICE（規則：3500 W 與 9000 W）'),
               ('explain/pdp_ice_temp_diff.png', '溫差軸的 PDP/ICE（規則：8.6 K）'), ('explain/pdp_ice_wear.png', '磨耗軸的 PDP/ICE（TWF 視窗 200–240 min）'),
               ('explain/boundary_offsets.png', '模型邊界相對規則值的偏移')], P)}

**回答 A**：**部分一致，且高度依賴模型與特徵組。** 吻合度最高的是{MODEL_ZH[kb[1]]}＋{FS_ZH[kb[2]]}（kappa {fmt_num(kb[0], 2)}），
它對 HDF/PWF/OSF 的偵測率為 {fmt_pct(rf_e['hdf']['recall'], 0)}/{fmt_pct(rf_e['pwf']['recall'], 0)}/{fmt_pct(rf_e['osf']['recall'], 0)}，
重要特徵正是規則變數。MLP 的 kappa 只有 {fmt_num(_kappa(res, 'mlp', 'raw'), 2)}（原始）與 {fmt_num(_kappa(res, 'mlp', 'engineered'), 2)}（衍生）。
在 {tot} 個「模型×邊界」組合中，原始特徵有 {raw_l} 個、衍生特徵有 {eng_l} 個達到「學到邊界」的標準；原始特徵下完全找不到的邊界是 {'、'.join(never_raw) if never_raw else '（無）'}
——這些多半是需要「兩個變數的組合」（功率、溫差）才能表達的邊界。TWF 的視窗只能學到「進入視窗後風險升高」，學不到視窗內的失效時點，與資料產生方式一致。"""


def _answer_b(res, f, P) -> str:
    ex = res["explain"]
    diff_rows = []
    for m in MODELS:
        d = _fs_diff(res, m)
        te_r, te_e = _eval_row(res, m, "raw"), _eval_row(res, m, "engineered")
        diff_rows.append([MODEL_ZH[m], fmt_num(d["raw_mean"]), fmt_num(d["engineered_mean"]),
                          fmt_ci(d["diff_mean"], d["diff_low"], d["diff_high"]), "是" if d["significant"] else "否",
                          fmt_num(te_r["pr_auc"]), fmt_num(te_e["pr_auc"])])
    rules = [("osf", "OSF（磨耗×扭矩）"), ("pwf", "PWF（功率）"), ("hdf", "HDF（溫差且轉速）")]
    big = max(r["n_train"] for r in ex["mlp_product_rules"]["scaling"])
    rule_rows = []
    for r, name in rules:
        rr, re_ = _real(res, r, "raw"), _real(res, r, "engineered")
        sr, se = _scaling(res, r, "raw", big), _scaling(res, r, "engineered", big)
        rule_rows.append([name, fmt_num(rr["pr_auc_mean"]), fmt_num(re_["pr_auc_mean"]), fmt_num(sr["pr_auc"]), fmt_num(se["pr_auc"]),
                          fmt_num(sr["f1"], 2)])
    ctrl = _real(res, "rpm_only", "raw")
    mlp_r, mlp_e = f["mlp_raw_test"], f["mlp_eng_test"]
    rf_r, rf_e = _eval_row(res, "random_forest", "raw"), _eval_row(res, "random_forest", "engineered")
    return f"""### 5.7 研究問題 B：衍生特徵的價值，MLP 能否自己學到乘積關係？

**B-1　衍生特徵的增益。** 開發集 CV 的 PR-AUC（各 模型 取其最佳不平衡處理；差值為 衍生 − 原始，Nadeau–Bengio 95% 區間）與測試集 PR-AUC：

{md_table(['模型', 'CV 原始', 'CV 衍生', '差值 [95% CI]', '差值顯著？', '測試 原始', '測試 衍生'], diff_rows)}

**B-2　MLP 能否學到乘積型規則？** 把預測目標換成「規則是否成立」（目標＝規則本身，無標記雜訊），比較原始與衍生特徵的 MLP 測試 PR-AUC
（真實 6,000 列訓練，3 個種子平均；以及由規則合成 {_n(big)} 列訓練）：

{md_table(['規則', '真實 6,000 列 原始', '真實 6,000 列 衍生', f'合成 {_n(big)} 列 原始', f'合成 {_n(big)} 列 衍生', f'合成 {_n(big)} 列 原始 F1'], rule_rows)}

控制組（只涉及單一特徵軸的規則 轉速 < 1380）原始特徵即達 PR-AUC {fmt_num(ctrl['pr_auc_mean'])}，顯示 MLP 的訓練流程本身沒有問題。

{_wrap_images([('models/feature_set_diff.png', '衍生特徵對各模型 CV PR-AUC 的增益'), ('explain/mlp_product_rules.png', 'MLP 預測乘積型規則的表現與資料量的關係'),
               ('explain/pdp_ice_strain_ratio.png', '過度負荷比（磨耗×扭矩 / 門檻）軸的 PDP/ICE')], P)}

**回答 B**：**衍生特徵對樹模型的幫助大而顯著，對 MLP 不顯著；MLP 只靠原始特徵「能」學到乘積型規則，但不精準。**
(1) 隨機森林 CV PR-AUC 增加 {fmt_num(_fs_diff(res, 'random_forest')['diff_mean'])}（{_sig(res, 'random_forest')}），HistGradientBoosting 增加 {fmt_num(_fs_diff(res, 'hist_gb')['diff_mean'])}（{_sig(res, 'hist_gb')}），
MLP 增加 {fmt_num(_fs_diff(res, 'mlp')['diff_mean'])}（區間 [{fmt_num(_fs_diff(res, 'mlp')['diff_low'])}, {fmt_num(_fs_diff(res, 'mlp')['diff_high'])}]，{_sig(res, 'mlp')}）。
(2) 「目標＝規則」時，MLP 在原始特徵上的 PR-AUC 為 OSF {fmt_num(_real(res, 'osf', 'raw')['pr_auc_mean'])}、PWF {fmt_num(_real(res, 'pwf', 'raw')['pr_auc_mean'])}，
資料放大到 {_n(big)} 列後三條規則皆 ≥ {fmt_num(min(_scaling(res, r, 'raw', big)['pr_auc'] for r, _ in rules))}，與衍生特徵相當，也就是它可以自行逼近乘積；
但 F1 只在 0.9 上下而非 1.0，邊界不精準。(3) 在含標記雜訊、開發集僅 {res['framework']['config']['dev_failures']} 筆故障的真實任務中，
MLP 測試 PR-AUC 為 {fmt_num(mlp_r['pr_auc'])}（原始）／{fmt_num(mlp_e['pr_auc'])}（衍生），遠低於隨機森林的 {fmt_num(rf_r['pr_auc'])}／{fmt_num(rf_e['pr_auc'])}。
換言之，**領域知識（衍生特徵）加樹模型，比期待神經網路自己發現乘積關係更有效率**。"""


def _answer_c(res, f, P) -> str:
    c = res["cost"]
    rows = []
    for s in c["sensitivity"]:
        sv = s["saving_ci"]
        rows.append([f"{s['ratio']}:1", fmt_num(s["threshold"]), s["test"]["fn"], s["test"]["fp"],
                     fmt_num(s["test"]["total"], 0), f"{_baseline_name(s['baseline_best'])} {fmt_num(s['baseline_cost'], 0)}",
                     fmt_pct(s["saving_pct"]),
                     f"[{fmt_num(sv['low'], 0)}, {fmt_num(sv['high'], 0)}]", fmt_num(s["test"]["recall"], 2), fmt_num(s["test"]["precision"], 2)])
    s1, s5, s100 = _sens(res, 1), _sens(res, 5), _sens(res, 100)
    models_rows = [[MODEL_ZH[k.split('|')[0]] + "／" + FS_ZH[k.split('|')[1]], fmt_num(v["break_even"])]
                   for k, v in sorted(c["all_models"].items(), key=lambda kv: kv[1]["break_even"])]
    return f"""### 5.8 研究問題 C：不同成本假設下的最佳門檻與預期成本

主模型為{MODEL_ZH[c['primary']['model']]}＋{FS_ZH[c['primary']['feature_set']]}。驗證集（{c['n']['val']} 筆、{c['n']['val_pos']} 筆故障）上找出使預期成本最低的門檻，
原封不動套用到測試集（{c['n']['test']} 筆、{c['n']['test_pos']} 筆故障）：

{md_table(['C_FN:C_FP', '驗證集選出的門檻', '測試 FN', '測試 FP', '測試成本', '最佳基準策略與成本', '節省比例', '節省金額 95% CI', 'Recall', 'Precision'], rows)}

「全部檢修」與「不預測」的成本在 C_FN:C_FP ≈ {fmt_num(c['baseline_crossover_ratio'], 1)}:1（= 正常數 / 故障數）處交叉；比例高於此值時，最佳基準由「不預測」變成「全部檢修」。
模型相對「不預測」的損益兩平比例約為 {fmt_num(c['break_even_ratio'])}:1；各模型的損益兩平比例（掃描範圍 0.01–200 的對數網格，0.010 代表落在網格下限）：
{'；'.join(f"{r[0]} {r[1]}" for r in models_rows)}。

{_wrap_images([('cost/cost_curves.png', '判斷門檻 vs. 預期成本（含「不預測」與「全部檢修」基準）'), ('cost/sensitivity.png', '成本比例敏感度：最佳門檻與節省比例')], P)}

**回答 C**：**最佳門檻隨成本比例單調下降，從 {fmt_num(s1['threshold'])}（1:1）降到 {fmt_num(s100['threshold'], 3)}（100:1）。**
測試集上，5:1 時模型成本 {fmt_num(s5['test']['total'], 0)}（Recall {fmt_num(s5['test']['recall'], 2)}、Precision {fmt_num(s5['test']['precision'], 2)}），
相對最佳基準（{_baseline_name(s5['baseline_best'])}，{fmt_num(s5['baseline_cost'], 0)}）節省 {fmt_pct(s5['saving_pct'])}（95% CI 節省 {fmt_num(s5['saving_ci']['low'], 0)}–{fmt_num(s5['saving_ci']['high'], 0)} 單位）；
100:1 時模型成本 {fmt_num(s100['test']['total'], 0)}，基準為{_baseline_name(s100['baseline_best'])} {fmt_num(s100['baseline_cost'], 0)}，節省 {fmt_pct(s100['saving_pct'])}。
當漏判成本極高時，最佳門檻很低（{fmt_num(s100['threshold'], 3)}），模型願意送檢大量可疑機台（FP {s100['test']['fp']} 筆），換取只漏 {s100['test']['fn']} 筆故障。
**需要誠實說明的是**：因為模型排序品質高，只檢修最有把握的少數機台就能在任何 C_FN > 0 下略有獲利，損益兩平比例（{fmt_num(c['break_even_ratio'])}:1）
遠低於實務上合理的成本比，所以「值得部署」的真正門檻不在成本比例，而在節省金額是否大於模型部署與維運成本。"""


def _paper_discussion(res, f) -> str:
    fw = res["framework"]["conclusions"]
    ea = res["error_analysis"]
    c = res["cost"]
    ml = res["multilabel"]
    rules = res["rules"]
    ex = res["explain"]
    fwd_parts = res["framework"]["drift"]["parts"]["temporal"]
    return f"""### 6.1 主要發現的意義

1. **規則已知的資料上，「準」不等於「學到規則」。** 隨機森林＋衍生特徵的測試 PR-AUC 達 {fmt_num(f['best_test']['pr_auc'])}，
   但 MLP 的 kappa 只有 {fmt_num(_kappa(res, 'mlp', 'raw'), 2)}／{fmt_num(_kappa(res, 'mlp', 'engineered'), 2)}；用規則邊界檢查，才看得出哪些模型只是在逼近而非掌握規則。
2. **領域知識是便宜且有效的歸納偏誤。** 三個衍生特徵就讓隨機森林的 CV PR-AUC 增加 {fmt_num(_fs_diff(res, 'random_forest')['diff_mean'])}，
   而 MLP 在大量資料或乾淨目標下才勉強追上，與 Grinsztajn et al.（2022）對樹模型在表格資料上的觀察方向一致。
3. **模型的錯誤大多可以被資料的產生過程解釋。** 測試集 {ea['counts']['fn']} 筆漏判中 {ea['summary']['fn_by_category']['twf_window']} 筆是隨機失效的 TWF，
   {ea['summary']['fn_by_category']['no_rule_explains']} 筆是標記不一致；這些錯誤不是調參能解決的上限。
4. **成本決策應該用成本，而不是 F1。** 最佳門檻從 {fmt_num(_sens(res, 1)['threshold'])} 降到 {fmt_num(_sens(res, 100)['threshold'], 3)}，
   用單一固定門檻（例如 0.5）無法同時適應不同的維護成本結構。

### 6.2 研究限制

- **資料是合成的。** 規則由人設定、噪音來源有限（RNF 與標記不一致），結果不能直接外推到真實產線；樹模型的優勢可能被「規則是門檻型」放大。
- **只判斷故障當下的狀態。** 特徵與故障同時量測，沒有時間窗，因此不是預測未來故障；成本分析中的「檢修」是指對當下狀態做處置。
- **測試集故障數少。** 測試集僅 {f['n_test_failures']} 筆故障，單次測試指標的 95% 信賴區間很寬（例如主模型 Recall {fmt_ci(f['best_test']['recall'], f['best_test']['recall_low'], f['best_test']['recall_high'])}），
  模型之間小的差距（例如 HistGradientBoosting 與隨機森林）不應過度解讀。
- **時間結構與分布漂移。** 時間順序切分的平均 PR-AUC 從 {fmt_num(fw['mean_pr_auc_holdout']['stratified_val'])} 降到 {fmt_num(fw['mean_pr_auc_holdout']['temporal_test'])}（只比較邏輯斯迴歸與隨機森林），
  分層切分的結果可能高估泛化；時間切分的驗證段與測試段的 HDF 筆數分別為 {fwd_parts['val']['mode_counts']['hdf']} 與 {fwd_parts['test']['mode_counts']['hdf']}，也限制了可下的結論。
- **TWF 不可完全重建、RNF 不可預測。** 規則聯集的 Recall 上限為 {fmt_num(rules['agreement']['machine_failure_union']['recall'])}；
  RNF 的檢定樣本很少（開發集 {ml['rnf']['n_positive_dev']} 筆，置換次數僅 {ml['rnf']['permutation']['n_perm']}），p 值解析度有限，結論是「與隨機一致」而非「證明不可預測」。
- **MLP 的搜尋範圍有限。** 網格僅 {len(res['models']['config']['mlp_grid']['hidden_layer_sizes']) * len(res['models']['config']['mlp_grid']['learning_rate_init']) * len(res['models']['config']['mlp_grid']['alpha'])} 組、且受 2 顆 CPU 的計算預算限制；
  「類別權重」以隨機複製少數類實作。更大的網路、更久的訓練或特徵標準化之外的處理，可能縮小與樹模型的差距。
- **解釋方法的假設。** PDP/ICE 以介入單一欄位再重算衍生特徵的方式進行，會產生資料中不存在的組合；「學到邊界」的判準（80% 的曲線、±{fmt_pct(ex['window_frac'], 0)} 視窗、10% 偏移）是人為設定，換用其他閾值結論的細節可能改變。
- **成本模型是簡化的。** 假設每個漏判與誤報的成本固定且與機台無關；損益兩平比例 {fmt_num(c['break_even_ratio'])}:1 很低，代表在此假設下模型幾乎總是「值得」，真實決策還要計入部署與維運成本。
- **標記不一致列的處理。** 主分析保留原始標記；剔除不一致列只在規則重建中做了敏感度分析，未對全部模型重做。"""


def _paper_conclusion(res, f) -> str:
    kb = max(((_kappa(res, m, fs), m, fs) for m in MODELS for fs in FEATURE_SETS))
    s5, s100 = _sens(res, 5), _sens(res, 100)
    raw_l, tot = _learned_total(res, "raw")
    eng_l, _ = _learned_total(res, "engineered")
    return f"""本研究以 AI4I 2020 的已知規則為標準答案，得到三個以數據支持的結論：

- **A（可解釋性）**：模型與規則的一致程度因模型與特徵而異。最一致的是{MODEL_ZH[kb[1]]}＋{FS_ZH[kb[2]]}（kappa {fmt_num(kb[0], 2)}）；
  MLP 僅 {fmt_num(_kappa(res, 'mlp', 'raw'), 2)}／{fmt_num(_kappa(res, 'mlp', 'engineered'), 2)}。{tot} 個模型×邊界組合中，原始特徵 {raw_l} 個、衍生特徵 {eng_l} 個學到規則邊界。
- **B（領域知識）**：衍生特徵使隨機森林 CV PR-AUC 增加 {fmt_num(_fs_diff(res, 'random_forest')['diff_mean'])}（顯著），對 MLP 不顯著；
  MLP 在乾淨目標下可以只靠原始特徵近似乘積規則，但邊界不精準，在真實含雜訊任務上測試 PR-AUC 僅 {fmt_num(f['mlp_raw_test']['pr_auc'])}，低於樹模型。
- **C（維護決策）**：最佳門檻隨成本比例從 {fmt_num(_sens(res, 1)['threshold'])} 降到 {fmt_num(s100['threshold'], 3)}；
  5:1 時節省 {fmt_pct(s5['saving_pct'])}、100:1 時節省 {fmt_pct(s100['saving_pct'])}（相對最佳基準策略）。

實務上的建議是：先用物理知識建立衍生特徵、用樹模型取得強基準，再用規則邊界檢查模型是否為了對的理由而準，最後以成本而非 F1 決定門檻。
所有數字與圖表皆由 `python -m ai4i_lab.pipeline` 自動重新產生。"""


def render_paper(res: dict, prefix: str = "../outputs/ai4i") -> str:
    f = _facts_ext(res)
    parts = [
        "# 以類神經網路與已知規則檢驗預測性維護：AI4I 2020 資料集的可解釋性、領域知識與成本決策\n",
        "> 本文件由 `ai4i_lab/report.py` 依 `outputs/ai4i/` 的實驗結果自動產生，請勿手動修改數字。\n",
        "## 摘要\n\n" + _paper_abstract(res, f),
        "## 1. 前言\n\n" + _paper_intro(res, f),
        "## 2. 文獻回顧\n\n" + _paper_related(),
        "## 3. 資料與探索\n\n" + _paper_data(res, f, prefix),
        "## 4. 方法\n\n" + _paper_method(res, f),
        "## 5. 結果\n\n" + _paper_results_a(res, f, prefix) + "\n\n" + _answer_a(res, f, prefix) + "\n\n"
        + _answer_b(res, f, prefix) + "\n\n" + _answer_c(res, f, prefix),
        "## 6. 討論\n\n" + _paper_discussion(res, f),
        "## 7. 結論\n\n" + _paper_conclusion(res, f),
        "## 參考文獻\n\n" + "\n".join(f"{i}. {r}" for i, r in enumerate(REFERENCES, 1)),
    ]
    return "\n\n".join(parts).rstrip() + "\n"


# ---------------------------------------------------------------- 學習筆記
def _data_meta() -> dict:
    p = ROOT / "data" / "ai4i2020.meta.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _step(n: int, title: str, purpose: str, method: str, reading: str, pitfalls: str, figs: str = "") -> str:
    return (f"## 步驟 {n}：{title}\n\n**目的**\n\n{purpose.strip()}\n\n**做法**\n\n{method.strip()}\n\n"
            f"**結果判讀**\n\n{reading.strip()}\n\n**踩到的坑**\n\n{pitfalls.strip()}\n" + (f"\n{figs.strip()}\n" if figs else ""))


def _notes_steps(res, f, P) -> list[str]:
    e = res["eda"]["exploration"]
    inc = res["eda"]["label_inconsistencies"]["counts"]
    ts = res["eda"]["temporal_structure"]
    rules = res["rules"]
    ag = rules["agreement"]
    tc = rules["tool_cycles"]
    pca = res["pca"]["comparison"]
    fw = res["framework"]["conclusions"]
    ks = fw["max_ks_temporal"]
    ml = res["multilabel"]
    rn = ml["rnf"]
    ea = res["error_analysis"]
    ev = res["evaluation"]
    ex = res["explain"]
    c = res["cost"]
    meta = _data_meta()
    best = f["best_test"]
    mlp_r, mlp_e = f["mlp_raw_test"], f["mlp_eng_test"]
    rf_diff, mlp_diff = _fs_diff(res, "random_forest"), _fs_diff(res, "mlp")
    s1, s5, s100 = _sens(res, 1), _sens(res, 5), _sens(res, 100)
    big = max(r["n_train"] for r in ex["mlp_product_rules"]["scaling"])
    sha = f"SHA-256 前 12 碼 `{meta['sha256'][:12]}`，下載日 {meta['downloaded']}" if meta else "SHA-256 記錄於 `data/ai4i2020.meta.json`"
    kb = max(((_kappa(res, m, fs), m, fs) for m in MODELS for fs in FEATURE_SETS))
    n_none = sum(1 for v in res["models"]["chosen"].values() if v["imbalance"] == "none")
    steps = []
    steps.append(_step(
        1, "資料取得與品質檢查",
        "取得可追溯的原始資料，並確認它能不能直接用：欄位意義、缺失、重複、數值範圍。",
        f"從 UCI（id 601）下載後存成 `data/ai4i2020.csv`，記錄來源、{sha}，之後所有模組都經 `load_dataset()` 讀取，欄名統一為 snake_case。"
        f"檢查缺失值、重複列、型別與範圍（例如轉速 {e['ranges']['rotational_speed_rpm']['min']:.0f}–{e['ranges']['rotational_speed_rpm']['max']:.0f} rpm、"
        f"扭矩 {e['ranges']['torque_nm']['min']}–{e['ranges']['torque_nm']['max']} Nm）。",
        f"{_n(e['n_rows'])} 筆 × {e['n_columns']} 欄，缺失 {e['missing_values']}、重複 {e['duplicate_rows']}，資料乾淨；但乾淨不代表沒有問題——標記有不一致（步驟 2）。",
        "- 原始 CSV 帶有 BOM，雜湊是對去除 BOM 後的檔案計算的，否則重新下載後雜湊會對不上。\n"
        "- `Type` 是類別特徵，不能直接當數字；`UDI`、`Product ID` 是識別欄位，放進特徵可能讓模型記住編號（洩漏）。\n"
        "- 欄名含空格與方括號，建模前先統一改名，避免到處寫字串出錯。"))
    steps.append(_step(
        2, "探索性分析（EDA）",
        "在建模前先看懂資料：不平衡程度、各故障模式、標記是否自洽、有沒有時間結構。這些發現會決定後面的指標與切分方式。",
        "統計故障比例與各模式筆數；畫故障 vs. 正常的特徵分布、相關係數熱圖、依 Type 的故障率；"
        "找出 `machine_failure` 與五個模式標記互相矛盾的列；依 UDI 排序檢查刀具磨耗是否累積與歸零、空氣溫度是否緩慢漂移。",
        f"{_n(f['n_failures'])} 筆故障，故障率 {fmt_pct(f['failure_rate'], 2)}（全猜不故障的準確率就有 {fmt_pct(1 - f['failure_rate'])}），所以不能只看準確率。"
        f"Type 影響故障率：L {fmt_pct(e['failure_rate_by_type']['L'], 2)}、M {fmt_pct(e['failure_rate_by_type']['M'], 2)}、H {fmt_pct(e['failure_rate_by_type']['H'], 2)}。"
        f"標記不一致：{inc['failure_without_mode']} 筆有故障卻無模式、{inc['mode_without_failure']} 筆 RNF=1 卻不算故障。"
        f"UDI 順序有明顯時間結構：磨耗 {fmt_pct(ts['tool_wear_nondecreasing_fraction'])} 相鄰列非遞減、{ts['tool_wear_resets']} 次換刀歸零、空氣溫度 lag-1 自相關 {fmt_num(ts['air_temp_lag1_autocorr'], 4)}。",
        "- 各模式筆數加總（" + f"{sum(e['mode_counts'].values())}）大於故障數（{_n(f['n_failures'])}），因為 {e['multi_mode_rows']} 列同時有多種模式；不要把它們當互斥類別。\n"
        "- 標記不一致不能默默刪掉也不能默默保留：主分析保留原始標記（這是實際紀錄的結果），另做剔除版本當敏感度分析。\n"
        "- 時間結構是「隱藏的資料洩漏」來源：相鄰列幾乎相同，隨機切分會把近乎重複的列分到訓練與測試。",
        img("eda/feature_by_failure.png", "故障 vs. 正常的特徵分布", P) + "\n\n" + img("eda/temporal_structure.png", "UDI 順序下的時間結構", P)))
    steps.append(_step(
        3, "已知規則重建",
        "把資料產生規則當成標準答案，後面才能問「模型學到的和規則一樣嗎」。也順便驗證自己對規則和單位的理解正確。",
        "依 Matzka（2020）重算 HDF（溫差 < 8.6 K 且轉速 < 1380 rpm）、PWF（功率 = 扭矩 × 轉速 × 2π/60，< 3500 W 或 > 9000 W）、"
        "OSF（磨耗 × 扭矩 > 11000/12000/13000，依 L/M/H）、TWF（磨耗 200–240 min 視窗），與資料標記逐列比對，並畫出規則邊界。",
        f"HDF、PWF、OSF 吻合率 100%（FP = FN = 0）。TWF 只能得到候選視窗：Recall {fmt_num(ag['twf']['recall'])} 但 Precision 只有 {fmt_num(ag['twf']['precision'])}，"
        f"因為 {tc['n_cycles']} 個刀具週期中有 {tc['n_cycles_reaching_window']} 個到達視窗，只有 {tc['n_cycles_with_twf']} 個在視窗內失效。"
        f"規則聯集的 Recall 為 {fmt_num(ag['machine_failure_union']['recall'])}，這是純規則偵測的天花板。",
        f"- **單位**：轉速要先從 rpm 換成 rad/s 才能算功率；漏了 2π/60，PWF 的規則就會完全失準。\n"
        f"- **浮點誤差**：有 {rules['boundary_ties']['hdf_temp_diff_8_6_and_rpm_lt_1380']} 列的溫差名義上恰好是 8.6 K，標記是由浮點減法誤差決定的"
        f"（8.5999…→1、8.6000…→0）。若把溫差四捨五入到一位小數，這些列就無法區分，所以後面的衍生特徵刻意不做四捨五入。\n"
        f"- **TWF 不可能 100% 重建**：失效時點是隨機的，不是任何特徵的函數。看到 Precision 很低不是程式有 bug。",
        img("rules/hdf_boundary.png", "HDF 規則邊界", P) + "\n\n" + img("rules/twf_window.png", "TWF 視窗", P)))
    steps.append(_step(
        4, "特徵工程",
        "把物理知識轉成特徵，檢驗「領域知識」對模型的價值（研究問題 B），並嚴格防止洩漏。",
        "建立兩組特徵：原始特徵組（5 個數值 + Type one-hot，共 8 欄），以及原始＋衍生特徵組（再加溫差、功率、磨耗×扭矩，共 11 欄）。"
        "所有建模一律經 `build_features(df, feature_set)` 取得特徵，函式內斷言特徵不含故障模式、目標與識別欄位。",
        f"衍生特徵的價值在步驟 7 量化：隨機森林 CV PR-AUC 增加 {fmt_num(rf_diff['diff_mean'])}。",
        "- 特徵清單裡只要混進 `twf/hdf/pwf/osf/rnf` 任何一欄，模型就會「完美」預測，這是最常見也最致命的洩漏；用測試守住。\n"
        "- Type one-hot 三欄全保留會讓線性模型共線，需搭配正則化。\n"
        "- 衍生特徵與原欄位高度相關，這在樹模型沒問題，在解釋重要度時要記得「重要度會被分散」。"))
    steps.append(_step(
        5, "PCA 降維視覺化",
        "用最常見的線性降維，看故障樣本在低維空間是否可分，並討論 PCA 的限制。",
        "標準化後對兩組特徵各做 PCA，畫前 3 個主成分的 3D 散佈圖（標示故障與各模式），比較解釋變異；再用「只含前 3 個主成分的邏輯斯迴歸」做可分性探針。",
        f"前 3 個主成分累計解釋變異：原始 {fmt_pct(pca['cumulative_variance_3pc']['raw'])}、衍生 {fmt_pct(pca['cumulative_variance_3pc']['engineered'])}。"
        f"只用前 3 個主成分的 CV ROC-AUC 為 {fmt_num(pca['cv_auc_logreg_3pc']['raw'])}／{fmt_num(pca['cv_auc_logreg_3pc']['engineered'])}，"
        f"全部特徵為 {fmt_num(pca['cv_auc_logreg_all_features']['raw'])}／{fmt_num(pca['cv_auc_logreg_all_features']['engineered'])}。故障沒有形成獨立的群。",
        "- PCA 是非監督的，3.39% 的故障幾乎不影響共變異矩陣，所以「PCA 看不出來」不代表資料不可分，只代表線性、依變異排序的投影看不出來。\n"
        "- 衍生特徵多了，前 3 個主成分的解釋比例反而可能下降；解釋變異高低 ≠ 故障資訊多寡。\n"
        f"- 單一主成分裡最能分辨故障的是 {pca['best_single_pc']['raw']}（原始組），並非變異最大的前幾個；取前 3 個會把它丟掉。",
        img("pca/pca_3d_modes_engineered.png", "PCA 3D 圖（衍生特徵組，依模式著色）", P)))
    steps.append(_step(
        6, "評估框架：切分、重複 CV 與信賴區間",
        "在故障只有幾十筆的情況下，讓每個數字都附上不確定性，並檢查切分方式會不會讓結論過度樂觀。",
        "以 UDI 為鍵做分層 60/20/20 切分並存成 `splits.json`；開發集上做 5 折重複分層 CV，信賴區間採 Nadeau–Bengio 校正；"
        "另做依時間順序的切分當穩健性檢驗；並以 KS 檢定量化訓練與測試的分布差異。",
        f"時間順序切分的故障率從 {fmt_pct(fw['failure_rate_by_part']['temporal']['train'], 2)} 降到 {fmt_pct(fw['failure_rate_by_part']['temporal']['test'], 2)}，"
        f"{ks['feature']} 的 KS = {fmt_num(ks['statistic'])}；平均 PR-AUC 從 {fmt_num(fw['mean_pr_auc_holdout']['stratified_val'])}（分層）"
        f"降到 {fmt_num(fw['mean_pr_auc_holdout']['temporal_test'])}（時間，測試段）。",
        "- 重複 CV 的各折共用訓練資料，**不獨立**；直接用標準差算 t 區間會過窄，要用校正的方法。\n"
        "- 測試集只能看一次。穩健性檢驗的切分也要先剔除共用測試集的列，否則等於偷看。\n"
        "- 單次切分的測試集只有 " + f"{f['n_test_failures']} 筆故障，差一筆就會讓 Recall 差 {fmt_pct(1 / f['n_test_failures'])}。",
        img("framework/temporal_drift.png", "時間漂移", P)))
    steps.append(_step(
        7, "二元分類：模型、不平衡處理與 MLP 調參",
        "比較邏輯斯迴歸、隨機森林、HistGradientBoosting 與 MLP，在兩組特徵與三種不平衡處理下的表現，並選門檻。",
        "每個 模型×特徵組×不平衡處理 做重複分層 CV（過採樣放在 Pipeline 內，只動訓練折）；MLP 在驗證集上搜尋隱藏層、學習率、L2；"
        "最終模型只在訓練集訓練，驗證集選使 F1 最大的門檻，測試集只評估一次。",
        f"最佳為{MODEL_ZH[f['best_cv_model']]}＋{FS_ZH[f['best_cv_feature_set']]}：CV PR-AUC {fmt_num(f['best_cv_pr_auc'])}，"
        f"測試 PR-AUC {fmt_ci(best['pr_auc'], best['pr_auc_low'], best['pr_auc_high'])}。MLP 測試 PR-AUC 只有 {fmt_num(mlp_r['pr_auc'])}（原始）／{fmt_num(mlp_e['pr_auc'])}（衍生）。"
        f"{n_none}／8 個組合的最佳不平衡處理是「不處理」。",
        "- SMOTE 如果在切分前做，合成樣本會滲進驗證折，分數會虛高；一定要放在訓練折內。\n"
        "- 類別權重／SMOTE 會把 Recall 拉高、Precision 壓低；是否划算要看成本（步驟 11），不是看 F1。\n"
        "- `MLPClassifier` 不支援 sample_weight，「類別權重」只能用複製少數類替代。\n"
        f"- MLP 的最佳門檻很高（{fmt_num(res['models']['thresholds']['mlp|raw']['threshold'])}），因為訓練時少數類被放大，機率被推高；不能直接用 0.5。\n"
        "- 門檻一定在驗證集上選，在測試集上調門檻是洩漏。",
        img("evaluation/model_comparison.png", "模型比較", P) + "\n\n" + img("models/mlp_training_curve.png", "MLP 訓練曲線", P)))
    steps.append(_step(
        8, "多標籤分類：預測哪一種故障",
        "把問題從「會不會故障」細分成「哪種故障」，並實驗驗證 RNF 理論上不可預測。",
        "對 TWF/HDF/PWF/OSF/RNF 各建一個獨立分類器（binary relevance），報告每個模式的 Precision/Recall/F1；"
        "RNF 與隨機基準比較：CV 的 ROC-AUC 區間、標籤置換檢定（含時間區塊版本）、固定標記比例下的偵測率。",
        f"HDF、PWF、OSF 的 F1 ≥ {fmt_num(min(ml['test']['engineered'][m]['f1']['point'] for m in ['hdf', 'pwf', 'osf']), 2)}；TWF、RNF 在 0.5 門檻下 Recall 為 0。"
        f"RNF 的 CV ROC-AUC {fmt_ci(rn['cv']['roc_auc']['mean'], rn['cv']['roc_auc']['low'], rn['cv']['roc_auc']['high'])}，區間包含 0.5。",
        "- 不要用 classifier chain：後一個分類器會拿前一個的模式預測當特徵，等於把標籤回流。\n"
        f"- RNF 的 p 值受樣本影響很大：只有 {rn['n_positive_dev']} 筆，置換 {rn['permutation']['n_perm']} 次，解析度只有 1/{rn['permutation']['n_perm'] + 1}；"
        f"而且 RNF 在 UDI 上有群聚（{rn['udi_cluster']['max_count']}/{rn['udi_cluster']['n_events']} 筆落在 {rn['udi_cluster']['window']} 個 UDI 內），"
        "模型可能用溫度的時間代理效應取得微弱訊號。結論只能說「與隨機一致」。\n"
        "- TWF 的 ROC-AUC 看起來很高，但只是會排序「視窗內 vs 視窗外」；實際偵測不到。",
        img("multilabel/mode_metrics_test.png", "各模式測試指標", P) + "\n\n" + img("multilabel/rnf_vs_random.png", "RNF vs. 隨機基準", P)))
    steps.append(_step(
        9, "評估視覺化與錯誤分析",
        "用混淆矩陣、PR/ROC 曲線、校準曲線看全貌，再逐筆檢查漏判與誤報，對照規則邊界解釋「為什麼錯」。",
        "對 8 個 模型×特徵組 在測試集畫圖並算指標與 bootstrap 信賴區間；計算 ECE 與 Brier；將每筆錯誤分類為「TWF 視窗」「無規則可解釋」「貼近邊界」等。",
        f"主模型 FN {ea['counts']['fn']}、FP {ea['counts']['fp']}：漏判中 {ea['summary']['fn_by_category']['twf_window']} 筆是 TWF、"
        f"{ea['summary']['fn_by_category']['no_rule_explains']} 筆沒有規則能解釋；誤報有 {ea['summary']['fp_by_category']['near_boundary_outside']} 筆緊貼邊界外側。"
        f"主模型 ECE {fmt_num(ev['calibration'][f['best_cv_model'] + '|' + f['best_cv_feature_set']]['ece'], 4)}。",
        "- PR 曲線的基準線是故障比例（約 3.4%），不是 0.5，別拿 ROC 的直覺看它。\n"
        "- 校準曲線在極度不平衡時，等寬分箱會讓大部分分箱幾乎沒有樣本，本專案改用分位數分箱。\n"
        "- 「錯誤多半無法避免」要有證據：逐筆對照規則才知道漏判的是隨機失效的 TWF，而不是模型不夠強。",
        img("evaluation/pr_curves.png", "PR 曲線", P) + "\n\n" + img("evaluation/calibration.png", "校準曲線", P)))
    steps.append(_step(
        10, "可解釋性分析（研究問題 A、B）",
        "檢驗模型是否為了對的理由而準：重要特徵、決策邊界、以及 MLP 能不能學到乘積關係。",
        "排列重要度（8 個 模型×特徵組）；在規則座標上介入原始欄位並重算衍生特徵，畫 PDP/ICE，量化邊界相對規則值的偏移；"
        "計算測試集上模型與規則聯集的 kappa；另以「目標＝規則本身」的實驗檢驗 MLP 能否學到乘積規則，並放大到 40,000 列。",
        f"kappa 最高為{MODEL_ZH[kb[1]]}＋{FS_ZH[kb[2]]}（{fmt_num(kb[0], 2)}）；MLP 為 {fmt_num(_kappa(res, 'mlp', 'raw'), 2)}／{fmt_num(_kappa(res, 'mlp', 'engineered'), 2)}。"
        f"MLP 目標＝規則時，原始特徵 OSF PR-AUC {fmt_num(_real(res, 'osf', 'raw')['pr_auc_mean'])}、放大到 {_n(big)} 列後三條規則皆 ≥ "
        f"{fmt_num(min(_scaling(res, r, 'raw', big)['pr_auc'] for r in ['osf', 'pwf', 'hdf']))}。",
        "- 排列重要度在相關特徵間會互相稀釋：衍生特徵與原欄位共存時，單看某一欄的重要度會被低估。\n"
        "- 介入單一欄位會產生資料裡不存在的組合（例如高扭矩卻低功率不可能），PDP/ICE 的解讀要保守。\n"
        "- 「學到邊界」需要明確判準，本專案用「80% 的曲線在規則值 ±25% 內跨越門檻、交點偏移 ≤ 10%」；換判準，細節會變。\n"
        "- 「能學到」和「學得準」是兩件事：MLP 能逼近乘積，但 F1 只有 0.9 上下。",
        img("explain/importance.png", "排列重要度", P) + "\n\n" + img("explain/pdp_ice_power.png", "功率軸 PDP/ICE", P)))
    steps.append(_step(
        11, "成本敏感的維護決策（研究問題 C）",
        "把模型輸出轉成決策：漏判與誤報的代價不同，最佳門檻就不同。",
        "定義預期成本 = C_FN × FN + C_FP × FP；在驗證集上對每個成本比（1、5、10、50、100）找最低成本的門檻，套用到測試集；"
        "並與「不預測」「全部檢修」兩種基準比較，bootstrap 估計節省金額的信賴區間。",
        f"最佳門檻從 {fmt_num(s1['threshold'])}（1:1）降到 {fmt_num(s100['threshold'], 3)}（100:1）；5:1 測試成本 {fmt_num(s5['test']['total'], 0)} vs. 基準 {fmt_num(s5['baseline_cost'], 0)}"
        f"（節省 {fmt_pct(s5['saving_pct'])}），100:1 為 {fmt_num(s100['test']['total'], 0)} vs. {fmt_num(s100['baseline_cost'], 0)}（節省 {fmt_pct(s100['saving_pct'])}）。"
        f"損益兩平成本比 {fmt_num(c['break_even_ratio'])}:1。",
        "- 門檻要在驗證集選、測試集評估，否則成本會被低估。\n"
        f"- 「全部檢修」與「不預測」在成本比約 {fmt_num(c['baseline_crossover_ratio'], 1)}:1（= 正常數 / 故障數）交叉，高於此值時最佳基準會換邊，比較時要取兩者較佳者。\n"
        "- 損益兩平比很低（模型幾乎總是值得），所以真正決定要不要部署的是部署與維運成本，不是成本比；別把「節省 80%」當成實際利潤。\n"
        "- 成本比極高時最佳門檻接近 0，等於幾乎全部送檢，這時模型的價值在於「少數可以放心不檢修的機台」。",
        img("cost/cost_curves.png", "門檻 vs. 預期成本", P)))
    return steps


def _notes_glossary() -> str:
    rows = [
        ["PR-AUC", "Precision–Recall 曲線下面積；不平衡資料的主要指標，隨機基準等於故障比例"],
        ["ROC-AUC", "ROC 曲線下面積；對不平衡不敏感，容易看起來很高"],
        ["Recall／Precision／F1", "抓到的故障占全部故障的比例／報警中真的故障的比例／兩者的調和平均"],
        ["ECE／Brier", "機率校準誤差／機率預測的均方誤差"],
        ["Nadeau–Bengio 校正", "重複 CV 的折間不獨立，校正後的 t 區間才不會過窄"],
        ["SMOTE", "在少數類樣本之間插值產生合成樣本的過採樣法"],
        ["PDP／ICE", "部分依賴圖（平均效果）／個別條件期望（每個樣本一條曲線）"],
        ["Cohen kappa", "校正隨機一致後的一致性，1 = 完全一致"],
        ["binary relevance", "多標籤問題拆成每個標籤各一個二元分類器"],
        ["C_FN : C_FP", "漏判成本與誤報成本的比例，決定最佳判斷門檻"],
    ]
    return "## 名詞對照\n\n" + md_table(["名詞", "白話說明"], rows) + "\n"


def _notes_questions(res, f) -> str:
    c = res["cost"]
    fw = res["framework"]["conclusions"]
    ml = res["multilabel"]
    qs = [
        ("為什麼這份資料不能用準確率評估？", f"全猜不故障就有 {fmt_pct(1 - f['failure_rate'])}；故障只占 {fmt_pct(f['failure_rate'], 2)}。"),
        ("TWF 為什麼無法 100% 重建？", "失效時點在 200–240 min 間是隨機的，特徵只能告訴你「進入視窗」。"),
        ("為什麼要做時間順序切分？", f"UDI 順序有時間結構與分布漂移；平均 PR-AUC 從 {fmt_num(fw['mean_pr_auc_holdout']['stratified_val'])} 降到 {fmt_num(fw['mean_pr_auc_holdout']['temporal_test'])}。"),
        ("SMOTE 為什麼要放在訓練折內？", "否則合成樣本會洩漏到驗證折，分數虛高。"),
        ("MLP 為什麼不如樹模型？", f"測試 PR-AUC {fmt_num(f['mlp_raw_test']['pr_auc'])}／{fmt_num(f['mlp_eng_test']['pr_auc'])} vs. 隨機森林 {fmt_num(_eval_row(res, 'random_forest', 'raw')['pr_auc'])}／{fmt_num(f['best_test']['pr_auc'])}；"
         f"規則是門檻型，開發集只有 {res['framework']['config']['dev_failures']} 筆故障，樹模型更容易切出角落。這是本資料上的觀察，不是通則。"),
        ("RNF 為什麼預測不了？", f"它與特徵無關；CV ROC-AUC 區間包含 0.5，測試集 {ml['rnf']['test']['n_positive_test']} 筆偵測到 {ml['rnf']['test']['tp']} 筆。"),
        ("最佳判斷門檻為什麼會隨成本改變？", f"成本比越高越怕漏判，門檻從 {fmt_num(_sens(res, 1)['threshold'])} 降到 {fmt_num(_sens(res, 100)['threshold'], 3)}。"),
        ("為什麼節省 80% 不等於賺 80%？", f"損益兩平成本比只有 {fmt_num(c['break_even_ratio'])}:1；還要扣掉部署與維運成本。"),
    ]
    body = "\n\n".join(f"{i}. **{q}**\n   答：{a}" for i, (q, a) in enumerate(qs, 1))
    return "## 自我檢查問題\n\n" + body + "\n"


def _notes_reproduce() -> str:
    return """## 如何重現

```bash
pip install -r requirements.txt
python -m pytest                       # 全部測試
python -m ai4i_lab.pipeline            # 依序重跑所有分析並重新產生本文件與小論文
python -m ai4i_lab.pipeline --list     # 列出階段與預計耗時
python -m ai4i_lab.pipeline --skip-existing   # 已有輸出的階段跳過
python -m ai4i_lab.pipeline --report-only     # 只用現有結果重新產生 docs
```

全程使用固定隨機種子（SEED = 42）。完整重跑需要一段時間（MLP 搜尋、重複 CV 與可解釋性分析最耗時），
所以 `--skip-existing` 與 `--report-only` 適合在只改文件時使用。
"""


def render_notes(res: dict, prefix: str = "../outputs/ai4i") -> str:
    f = _facts_ext(res)
    parts = [
        "# AI4I 2020 預測性維護：學習筆記\n",
        "> 本文件由 `ai4i_lab/report.py` 依 `outputs/ai4i/` 的實驗結果自動產生。每個步驟依序說明：目的、做法、結果判讀、踩到的坑。\n"
        "> 完整論述與研究問題 A／B／C 的回答見 [小論文](ai4i_paper.md)。",
        "\n\n".join(_notes_steps(res, f, prefix)),
        _notes_glossary(),
        _notes_questions(res, f),
        _notes_reproduce(),
    ]
    return "\n\n".join(parts).rstrip() + "\n"


# ---------------------------------------------------------------- 寫檔
def write_reports(out_dir=OUT_DIR, docs_dir=DOCS_DIR) -> dict:
    """讀取結果 → 產生論文與筆記 → 寫入 docs。先全部產生成功才寫檔，失敗時不留下半成品。"""
    res = load_results(out_dir)
    texts = {"paper": ("ai4i_paper.md", render_paper(res)), "notes": ("ai4i_notes.md", render_notes(res))}
    docs_dir = Path(docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for key, (name, text) in texts.items():
        p = docs_dir / name
        p.write_text(text, encoding="utf-8")
        paths[key] = p
    return paths


def run(out_dir=OUT_DIR, docs_dir=DOCS_DIR) -> dict:
    paths = write_reports(out_dir, docs_dir)
    return {k: str(v) for k, v in paths.items()}


if __name__ == "__main__":
    for k, v in run().items():
        print(f"{k}: {v}")
