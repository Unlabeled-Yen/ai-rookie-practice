"""#5 已知故障規則重建的測試。"""
import numpy as np
import pandas as pd
import pytest

from ai4i_lab import rules as R


def _df(**cols):
    n = len(next(iter(cols.values())))
    base = {"type": ["L"] * n, "air_temp_k": [300.0] * n, "process_temp_k": [310.0] * n,
            "rotational_speed_rpm": [1500] * n, "torque_nm": [40.0] * n, "tool_wear_min": [100] * n}
    base.update(cols)
    return pd.DataFrame(base)


# ---------- 第 1 輪：門檻常數與 HDF ----------

def test_threshold_constants_match_literature():
    assert R.HDF_TEMP_DIFF_MAX_K == 8.6
    assert R.HDF_RPM_MAX == 1380
    assert (R.PWF_POWER_MIN_W, R.PWF_POWER_MAX_W) == (3500, 9000)
    assert R.OSF_STRAIN_MAX_MINNM == {"L": 11000, "M": 12000, "H": 13000}
    assert (R.TWF_WEAR_MIN, R.TWF_WEAR_MAX) == (200, 240)


def test_temp_diff_is_process_minus_air():
    d = _df(air_temp_k=[300.0], process_temp_k=[308.5])
    assert R.temp_diff_k(d).iloc[0] == pytest.approx(8.5)


def test_hdf_tie_at_8_6_follows_raw_float_subtraction_like_the_generator():
    # 真實資料 UDI 3237（309.4−300.8 = 8.59999…97）標 HDF=1，UDI 3765（311.0−302.4 = 8.60…02）標 0：
    # 資料產生器以浮點減法直接比較，規則必須照做才能 100% 重現標記。
    d = _df(air_temp_k=[300.8, 302.4], process_temp_k=[309.4, 311.0], rotational_speed_rpm=[1342, 1329])
    assert R.hdf_rule(d).tolist() == [True, False]


def test_hdf_rule_boundaries_both_sides():
    # 溫差 8.5 / 8.6（邊界，不觸發）/ 8.7；轉速 1379 / 1380（邊界，不觸發）
    d = _df(air_temp_k=[300.0, 300.0, 300.0, 300.0, 300.0],
            process_temp_k=[308.5, 308.6, 308.7, 308.5, 308.5],
            rotational_speed_rpm=[1379, 1379, 1379, 1380, 1381])
    assert R.hdf_rule(d).tolist() == [True, False, False, False, False]


# ---------- 第 2 輪：PWF 與 OSF ----------

def test_power_hand_computed():
    # 40 Nm × 1500 rpm × 2π/60 = 6283.185 W
    d = _df(torque_nm=[40.0], rotational_speed_rpm=[1500])
    assert R.power_w(d).iloc[0] == pytest.approx(40 * 1500 * 2 * np.pi / 60)
    assert R.power_w(d).iloc[0] == pytest.approx(6283.185, abs=1e-3)


def test_pwf_rule_boundaries_both_sides():
    w = 2 * np.pi / 60
    rpm = 1000
    torques = [3499 / (rpm * w), 3501 / (rpm * w), 6000 / (rpm * w), 8999 / (rpm * w), 9001 / (rpm * w)]
    d = _df(torque_nm=torques, rotational_speed_rpm=[rpm] * 5)
    assert R.pwf_rule(d).tolist() == [True, False, False, False, True]


def test_osf_strain_and_type_specific_thresholds():
    # 磨耗 × 扭矩：100 × 110 = 11000（L 邊界，不觸發）；100 × 110.1 = 11010 > 11000
    d = _df(type=["L", "L", "M", "M", "H", "H"],
            tool_wear_min=[100] * 6,
            torque_nm=[110.0, 110.1, 120.0, 120.1, 130.0, 130.1])
    assert R.strain_minnm(d).iloc[1] == pytest.approx(11010)
    assert R.osf_rule(d).tolist() == [False, True, False, True, False, True]


def test_osf_same_strain_differs_by_type():
    d = _df(type=["L", "M", "H"], tool_wear_min=[200] * 3, torque_nm=[60.0] * 3)  # 12000
    assert R.osf_rule(d).tolist() == [True, False, False]


# ---------- 第 3 輪：TWF 窗口、吻合率與刀具週期 ----------

def test_twf_window_is_inclusive_on_both_ends():
    d = _df(tool_wear_min=[199, 200, 220, 240, 241])
    assert R.twf_window(d).tolist() == [False, True, True, True, False]


def test_agreement_hand_computed_confusion():
    pred = pd.Series([1, 1, 0, 0, 1], dtype=bool)
    label = pd.Series([1, 0, 0, 1, 1])
    udi = pd.Series([11, 12, 13, 14, 15])
    a = R.agreement(pred, label, udi)
    assert (a["tp"], a["fp"], a["fn"], a["tn"]) == (2, 1, 1, 1)
    assert a["accuracy"] == pytest.approx(3 / 5)
    assert a["precision"] == pytest.approx(2 / 3)
    assert a["recall"] == pytest.approx(2 / 3)
    assert a["false_positive_udi"] == [12]
    assert a["false_negative_udi"] == [14]


def test_agreement_perfect_and_empty_positive_is_safe():
    a = R.agreement(pd.Series([False, False]), pd.Series([0, 0]), pd.Series([1, 2]))
    assert a["accuracy"] == 1.0 and a["precision"] is None and a["recall"] is None


def _labelled(df):
    df = df.copy()
    df["udi"] = range(1, len(df) + 1)
    df["hdf"] = R.hdf_rule(df).astype(int)
    df["pwf"] = R.pwf_rule(df).astype(int)
    df["osf"] = R.osf_rule(df).astype(int)
    df["twf"] = 0
    df["rnf"] = 0
    df["machine_failure"] = df[["hdf", "pwf", "osf"]].max(axis=1)
    return df


def test_rule_agreement_on_rule_generated_labels_is_perfect():
    rng = np.random.default_rng(0)
    n = 400
    d = _df(type=rng.choice(["L", "M", "H"], n).tolist(),
            air_temp_k=rng.uniform(296, 304, n).round(1).tolist(),
            process_temp_k=rng.uniform(305, 313, n).round(1).tolist(),
            rotational_speed_rpm=rng.integers(1170, 2900, n).tolist(),
            torque_nm=rng.uniform(4, 77, n).round(1).tolist(),
            tool_wear_min=rng.integers(0, 254, n).tolist())
    d = _labelled(d)
    out = R.rule_agreement(d)
    for m in ["hdf", "pwf", "osf"]:
        assert out[m]["accuracy"] == 1.0
        assert out[m]["false_positive_udi"] == [] and out[m]["false_negative_udi"] == []
    assert set(out) >= {"hdf", "pwf", "osf", "twf", "machine_failure_union"}


def test_rule_agreement_detects_flipped_label():
    d = _labelled(_df(air_temp_k=[300.0, 300.0], process_temp_k=[308.0, 310.0],
                      rotational_speed_rpm=[1300, 1500]))
    d.loc[1, "hdf"] = 1          # 不符規則卻被標成 HDF
    out = R.rule_agreement(d)
    assert out["hdf"]["false_positive_udi"] == []
    assert out["hdf"]["false_negative_udi"] == [2]


def test_tool_cycles_split_on_wear_reset():
    d = _df(tool_wear_min=[0, 5, 210, 0, 3, 230, 2])
    d["udi"] = range(1, 8)
    d["twf"] = [0, 0, 1, 0, 0, 0, 0]
    cyc = R.tool_cycles(d)
    assert cyc["cycle"].tolist() == [0, 1, 2]
    assert cyc["max_wear_min"].tolist() == [210, 230, 2]
    assert cyc["has_twf"].tolist() == [True, False, False]
    assert cyc["n_rows"].tolist() == [3, 3, 1]


# ---------- 第 4 輪：run() 輸出與真實資料 ----------
import json  # noqa: E402

from ai4i_lab.data import DEFAULT_PATH, FAILURE_MODES, load_dataset  # noqa: E402


def _synthetic(n=300, seed=1):
    rng = np.random.default_rng(seed)
    d = _df(type=rng.choice(["L", "M", "H"], n).tolist(),
            air_temp_k=rng.uniform(296, 304, n).round(1).tolist(),
            process_temp_k=rng.uniform(305, 313, n).round(1).tolist(),
            rotational_speed_rpm=rng.integers(1170, 2900, n).tolist(),
            torque_nm=rng.uniform(4, 77, n).round(1).tolist(),
            tool_wear_min=(np.arange(n) % 230).tolist())
    d = _labelled(d)
    d["twf"] = ((d["tool_wear_min"] == 215)).astype(int)
    d["machine_failure"] = d[["hdf", "pwf", "osf", "twf"]].max(axis=1)
    d["product_id"] = [f"L{i}" for i in range(n)]
    return d


def test_run_writes_json_and_figures(tmp_path):
    res = R.run(out_dir=tmp_path, df=_synthetic())
    js = json.loads((tmp_path / "rule_agreement.json").read_text(encoding="utf-8"))
    assert js["thresholds"]["pwf_power_w"] == [3500, 9000]
    assert js["thresholds"]["osf_strain_max_minnm"]["H"] == 13000
    for m in ["hdf", "pwf", "osf", "twf"]:
        assert m in js["agreement"]
    assert js["agreement"]["hdf"]["accuracy"] == 1.0
    assert "twf_reconstruction_limit" in js
    assert js["tool_cycles"]["n_cycles"] == 2
    for name in R.FIGURES:
        assert (tmp_path / name).exists() and (tmp_path / name).stat().st_size > 5000
    assert res["agreement"]["pwf"]["accuracy"] == 1.0


def test_rules_use_only_raw_features_not_labels():
    d = _synthetic(50)
    stripped = d.drop(columns=FAILURE_MODES + ["machine_failure", "udi", "product_id"])
    for f in [R.hdf_rule, R.pwf_rule, R.osf_rule, R.twf_window]:
        assert f(stripped).equals(f(d))


@pytest.mark.skipif(not DEFAULT_PATH.exists(), reason="需要 data/ai4i2020.csv")
def test_real_data_deterministic_rules_reproduce_labels():
    out = R.rule_agreement(load_dataset())
    for m, n_pos in [("hdf", 115), ("pwf", 95), ("osf", 98)]:
        assert out[m]["accuracy"] == 1.0
        assert out[m]["tp"] == n_pos
    assert out["twf"]["tp"] + out["twf"]["fn"] == 46
    assert out["twf"]["recall"] < 1.0          # 3 筆 TWF 落在 200–240 以外


# ---------- 第 5 輪：圖表樣式約定與邊界情形 ----------

def _capture_figs(monkeypatch):
    figs = []
    monkeypatch.setattr(R, "save", lambda fig, path: figs.append(fig))
    return figs


def test_boundary_lines_use_style_linewidth_2(monkeypatch):
    figs = _capture_figs(monkeypatch)
    d = _synthetic()
    for plot in [R.plot_hdf, R.plot_pwf, R.plot_osf]:
        plot(d, "unused.png")
    assert len(figs) == 3
    for fig in figs:
        lines = [ln for ax in fig.axes for ln in ax.lines]
        assert lines, "每張圖都要畫出規則邊界線"
        assert all(ln.get_linewidth() == 2 for ln in lines)


def test_plot_twf_handles_no_complete_cycle_without_twf(monkeypatch):
    figs = _capture_figs(monkeypatch)
    d = _df(tool_wear_min=[0, 100, 210, 0, 30])     # 唯一完整週期含 TWF；最後一段被截斷
    d["udi"] = range(1, 6)
    d["twf"] = [0, 0, 1, 0, 0]
    R.plot_twf(d, "unused.png")
    assert len(figs) == 1


def test_run_is_reproducible(tmp_path):
    a = R.run(out_dir=tmp_path / "a", df=_synthetic())
    b = R.run(out_dir=tmp_path / "b", df=_synthetic())
    assert a == b
