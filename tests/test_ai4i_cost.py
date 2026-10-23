"""#12 成本敏感門檻與敏感度分析的測試。"""
import json
import math

import numpy as np
import pandas as pd
import pytest

from ai4i_lab import cost as C
from ai4i_lab.data import ROOT

# 手算範例：y = [1,1,0,0,0,0]，proba 如下
Y = np.array([1, 1, 0, 0, 0, 0])
P = np.array([0.9, 0.4, 0.6, 0.2, 0.1, 0.05])


# ---------- 第 1 輪：成本計算（手算） ----------

def test_expected_cost_hand_computed():
    # 門檻 0.5：判為故障 = {0.9, 0.6} → TP=1, FP=1, FN=1, TN=3；成本 = 1*10 + 1*1 = 11
    r = C.expected_cost(Y, P, 0.5, c_fn=10, c_fp=1)
    assert (r["tp"], r["fp"], r["fn"], r["tn"]) == (1, 1, 1, 3)
    assert r["total"] == 11 and r["per_sample"] == pytest.approx(11 / 6)


def test_expected_cost_threshold_boundary_inclusive():
    # proba >= threshold 判為故障（與 evaluation.binary_metrics 一致）
    assert C.expected_cost(Y, P, 0.9, 5, 1)["tp"] == 1
    assert C.expected_cost(Y, P, 0.91, 5, 1)["tp"] == 0


def test_baselines_hand_computed():
    b = C.baseline_costs(Y, c_fn=10, c_fp=1)
    assert b["never"]["total"] == 20      # 2 個故障全漏判
    assert b["always"]["total"] == 4      # 4 個正常全誤報
    assert b["never"]["per_sample"] == pytest.approx(20 / 6)


def test_baselines_equal_extreme_thresholds():
    assert C.expected_cost(Y, P, math.inf, 7, 3)["total"] == C.baseline_costs(Y, 7, 3)["never"]["total"]
    assert C.expected_cost(Y, P, 0.0, 7, 3)["total"] == C.baseline_costs(Y, 7, 3)["always"]["total"]


def test_invalid_costs_rejected():
    with pytest.raises(ValueError):
        C.expected_cost(Y, P, 0.5, c_fn=-1, c_fp=1)
    with pytest.raises(ValueError):
        C.expected_cost(Y, P[:3], 0.5, 1, 1)


# ---------- 第 2 輪：成本曲線與門檻選擇 ----------

def test_cost_curve_matches_pointwise():
    curve = C.cost_curve(Y, P, c_fn=10, c_fp=1)
    for t, c in zip(curve["threshold"], curve["cost"]):
        assert c == pytest.approx(C.expected_cost(Y, P, t, 10, 1)["total"])
    assert math.isinf(curve["threshold"][0])         # 含「不預測」端點


def test_select_threshold_hand_computed():
    # C_FN=10, C_FP=1：門檻 0.4 → 判 {0.9,0.6,0.4}：TP=2, FP=1 → 成本 1，為最低
    s = C.select_threshold(Y, P, c_fn=10, c_fp=1)
    assert s["threshold"] == 0.4 and s["cost"] == 1
    # C_FN=1, C_FP=10：誤報很貴 → 只判 0.9：成本 = FN 1*1 = 1；不預測成本 2
    s = C.select_threshold(Y, P, c_fn=1, c_fp=10)
    assert s["threshold"] == 0.9 and s["cost"] == 1


def test_select_threshold_ties_take_higher():
    y = np.array([1, 0]); p = np.array([0.8, 0.2])
    s = C.select_threshold(y, p, 1, 1)                # 門檻 0.8 與 0.2 之間皆成本 0，取較高
    assert s["threshold"] == 0.8


def test_higher_fn_cost_lowers_threshold():
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.1).astype(int)
    p = np.clip(0.15 + 0.5 * y + rng.normal(0, 0.2, 2000), 0, 1)
    ths = [C.select_threshold(y, p, r, 1)["threshold"] for r in (1, 5, 20, 100)]
    assert ths == sorted(ths, reverse=True)
    assert ths[0] > ths[-1]


def test_selected_cost_never_worse_than_baselines_on_same_data():
    s = C.select_threshold(Y, P, 3, 2)
    b = C.baseline_costs(Y, 3, 2)
    assert s["cost"] <= min(b["never"]["total"], b["always"]["total"])


# ---------- 第 3 輪：敏感度分析（驗證選門檻、測試只評估） ----------

def test_sensitivity_threshold_chosen_on_val_only():
    yv, pv = Y, P
    yt = np.array([0, 0, 0, 0, 0, 0]); pt = np.array([0.99] * 6)    # 測試集故意與驗證集完全不同
    rows = C.sensitivity(yv, pv, yt, pt, ratios=(10,))
    assert rows[0]["threshold"] == 0.4                              # 與測試資料無關
    assert rows[0]["test"]["fp"] == 6                               # 測試集只做評估


def test_sensitivity_row_fields_and_savings():
    rows = C.sensitivity(Y, P, Y, P, ratios=(1, 10))
    r = rows[1]
    assert r["ratio"] == 10 and r["c_fn"] == 10 and r["c_fp"] == 1
    t = r["test"]
    assert t["total"] == 1                                          # 在同資料上 = 驗證最佳
    assert r["baseline_best"] == "always" and r["baseline_cost"] == 4
    assert r["saving"] == 3 and r["saving_pct"] == pytest.approx(0.75)
    assert {"recall", "precision", "f1"} <= set(t)


def test_break_even_ratio_hand_computed():
    # 模型（門檻 0.4）：FN=0, FP=1；never = 2*r；always = 4；模型 = 1 → 模型贏 always 且贏 never（r>0.5）
    # r=10：never=20、always=4、模型=1 → 值得；r=0（漏判免費）：never=0 已是最低，模型不可能更省 → 不值得
    # r=0.1：模型只判最高分 0.9 → FN=1 成本 0.1，低於 never 的 0.2（手算）→ 仍值得
    assert C.is_worthwhile(Y, P, Y, P, ratio=10)
    assert C.is_worthwhile(Y, P, Y, P, ratio=0.1)
    assert not C.is_worthwhile(Y, P, Y, P, ratio=0)


def test_break_even_search_monotone_region():
    rng = np.random.default_rng(1)
    y = (rng.random(3000) < 0.05).astype(int)
    p = np.clip(0.05 + 0.6 * y + rng.normal(0, 0.15, 3000), 0, 1)
    be = C.break_even_ratio(y, p, y, p, grid=np.arange(1, 201))
    assert be is not None and 1 <= be <= 200
    assert C.is_worthwhile(y, p, y, p, be)
    assert not C.is_worthwhile(y, p, y, p, be - 1) or be == 1


def test_bootstrap_saving_ci_reproducible():
    a = C.bootstrap_saving(Y, P, 0.4, 10, 1, n_boot=200, seed=3)
    b = C.bootstrap_saving(Y, P, 0.4, 10, 1, n_boot=200, seed=3)
    assert a == b and a["low"] <= a["point"] <= a["high"]


# ---------- 第 4 輪：圖表與 run() ----------

def test_plot_cost_curve_writes_png(tmp_path):
    p = tmp_path / "c.png"
    C.plot_cost_curves(Y, P, ratios=(1, 10), path=p)
    assert p.stat().st_size > 1000


def test_plot_sensitivity_writes_png(tmp_path):
    p = tmp_path / "s.png"
    C.plot_sensitivity(C.sensitivity(Y, P, Y, P, ratios=(1, 5, 10)), p)
    assert p.stat().st_size > 1000


def test_load_predictions_filters_model(tmp_path):
    df = pd.DataFrame({"row_id": [1, 2, 3, 4], "model": ["a", "a", "b", "b"],
                       "feature_set": ["raw"] * 4, "y_true": [0, 1, 0, 1], "proba": [.1, .9, .2, .8]})
    f = tmp_path / "v.csv"; df.to_csv(f, index=False)
    y, p = C.load_predictions(f, "b", "raw")
    assert list(y) == [0, 1] and list(p) == [0.2, 0.8]
    with pytest.raises(KeyError):
        C.load_predictions(f, "zzz", "raw")


REAL = ROOT / "outputs" / "ai4i" / "models" / "predictions_test.csv"


@pytest.mark.skipif(not REAL.exists(), reason="需先執行 #8")
def test_run_small(tmp_path):
    r = C.run(out_dir=tmp_path, n_boot=50)
    assert (tmp_path / "cost.json").exists() and (tmp_path / "cost_curves.png").exists()
    assert len(r["sensitivity"]) == 5
    j = json.loads((tmp_path / "cost.json").read_text())
    assert j["primary"]["model"] and j["conclusion"]
    # 測試集只評估：每列門檻皆來自驗證集選擇
    assert all("threshold" in row for row in j["sensitivity"])


# ---------- 第 5 輪：兩種基準策略的交叉點 ----------

def test_baseline_crossover_ratio_hand_computed():
    # never = r·P，always = N → 相等於 r = N/P = 4/2
    assert C.baseline_crossover_ratio(Y) == 2.0
    b = C.baseline_costs(Y, 2, 1)
    assert b["never"]["total"] == b["always"]["total"]


def test_baseline_crossover_no_positives_is_inf():
    assert math.isinf(C.baseline_crossover_ratio(np.zeros(5, int)))
