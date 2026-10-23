import io
import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from ai4i_lab import data as D


def _fake_rows():
    """小型假資料（原始欄名），涵蓋兩種標記不一致的情況。"""
    return pd.DataFrame({
        "UDI": [1, 2, 3, 4, 5, 6],
        "Product ID": ["M1", "L2", "H3", "L4", "M5", "L6"],
        "Type": ["M", "L", "H", "L", "M", "L"],
        "Air temperature [K]": [298.1, 298.2, 298.3, 298.4, 298.5, 298.6],
        "Process temperature [K]": [308.6, 308.7, 308.8, 308.9, 309.0, 309.1],
        "Rotational speed [rpm]": [1551, 1408, 1498, 1433, 1300, 1600],
        "Torque [Nm]": [42.8, 46.3, 49.4, 39.5, 60.0, 30.0],
        "Tool wear [min]": [0, 3, 5, 7, 0, 2],
        "Machine failure": [0, 1, 1, 0, 1, 0],
        "TWF": [0, 0, 0, 0, 0, 0],
        "HDF": [0, 1, 0, 0, 0, 0],
        "PWF": [0, 0, 0, 0, 1, 0],
        "OSF": [0, 0, 0, 0, 1, 0],
        "RNF": [0, 0, 0, 1, 0, 0],
    })
    # row UDI=3：故障但無模式；UDI=4：RNF 但未計故障；UDI=5：多重故障


def _zip_bytes(df):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("ai4i2020.csv", "﻿" + df.to_csv(index=False))
    return buf.getvalue()


@pytest.fixture
def fake_csv(tmp_path):
    path = tmp_path / "ai4i2020.csv"
    D.download_dataset(path, fetch=lambda url: _zip_bytes(_fake_rows()), today="2026-10-23")
    return path


# ---------- 取得資料 ----------

def test_download_writes_csv_and_meta(fake_csv):
    assert fake_csv.exists()
    meta = json.loads(fake_csv.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert meta["source_url"] == D.SOURCE_URL
    assert meta["downloaded"] == "2026-10-23"
    assert meta["sha256"] == D.sha256_of(fake_csv)
    assert meta["n_rows"] == 6
    assert len(meta["source_zip_sha256"]) == 64 and meta["source_zip_sha256"] != meta["sha256"]


def test_download_strips_bom(fake_csv):
    assert not fake_csv.read_bytes().startswith(b"\xef\xbb\xbf")


def test_load_dataset_renames_to_snake_case(fake_csv):
    df = D.load_dataset(fake_csv)
    assert list(df.columns) == D.COLUMNS
    assert df["type"].tolist() == ["M", "L", "H", "L", "M", "L"]


def test_load_dataset_rejects_missing_columns(tmp_path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"UDI": [1]}).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="缺少欄位"):
        D.load_dataset(bad)


def test_load_dataset_verifies_checksum(fake_csv):
    meta_path = fake_csv.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["sha256"] = "0" * 64
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        D.load_dataset(fake_csv)


# ---------- 欄位常數：防止資料洩漏的基礎 ----------

def test_column_groups_are_disjoint():
    groups = [D.ID_COLUMNS, D.FEATURES, [D.TARGET], D.FAILURE_MODES]
    flat = [c for g in groups for c in g]
    assert len(flat) == len(set(flat)) == len(D.COLUMNS)
    assert D.FAILURE_MODES == ["twf", "hdf", "pwf", "osf", "rnf"]
    assert D.TARGET not in D.FEATURES
    assert not set(D.FAILURE_MODES) & set(D.FEATURES)


# ---------- 標記不一致 ----------

def test_find_label_inconsistencies(fake_csv):
    df = D.load_dataset(fake_csv)
    inc = D.find_label_inconsistencies(df)
    assert inc["failure_without_mode"] == [3]
    assert inc["mode_without_failure"] == [4]


def test_label_policy_drop_inconsistent(fake_csv):
    as_is = D.load_dataset(fake_csv)
    dropped = D.load_dataset(fake_csv, label_policy="drop_inconsistent")
    assert len(as_is) == 6 and len(dropped) == 4
    assert set(dropped["udi"]) == {1, 2, 5, 6}


def test_label_policy_unknown_raises(fake_csv):
    with pytest.raises(ValueError):
        D.load_dataset(fake_csv, label_policy="nope")


# ---------- 探索 ----------

def test_explore_basic_checks(fake_csv):
    rep = D.explore(D.load_dataset(fake_csv))
    assert rep["n_rows"] == 6
    assert rep["missing_values"] == 0
    assert rep["duplicate_rows"] == 0
    assert rep["failure_rate"] == pytest.approx(3 / 6)
    assert rep["mode_counts"] == {"twf": 0, "hdf": 1, "pwf": 1, "osf": 1, "rnf": 1}
    assert rep["multi_mode_rows"] == 1
    assert rep["mode_cooccurrence"]["pwf"]["osf"] == 1
    assert rep["failure_rate_by_type"]["H"] == pytest.approx(1.0)
    assert set(rep["ranges"]) == set(D.FEATURES) - {"type"}
    assert set(rep["by_failure"]) == {"0", "1"}


def test_temporal_structure_detects_tool_wear_resets():
    df = pd.DataFrame({"udi": range(1, 9), "tool_wear_min": [0, 2, 5, 9, 0, 3, 6, 1],
                       "air_temp_k": np.linspace(298, 299, 8), "machine_failure": [0] * 8})
    ts = D.temporal_structure(df)
    assert ts["tool_wear_resets"] == 2
    assert ts["tool_wear_nondecreasing_fraction"] == pytest.approx(5 / 7)
    assert ts["air_temp_lag1_autocorr"] > 0.9
    assert isinstance(ts["has_temporal_structure"], bool)


# ---------- 真實資料（data/ai4i2020.csv 已下載時才跑） ----------

real = pytest.mark.skipif(not D.DEFAULT_PATH.exists(), reason="尚未下載 data/ai4i2020.csv")


@real
def test_real_dataset_known_facts():
    df = D.load_dataset()
    assert df.shape == (10000, 14)
    assert int(df[D.TARGET].sum()) == 339
    inc = D.find_label_inconsistencies(df)
    assert len(inc["failure_without_mode"]) == 9
    assert len(inc["mode_without_failure"]) == 18
    assert D.temporal_structure(df)["has_temporal_structure"] is True


@real
def test_run_writes_eda_outputs(tmp_path):
    rep = D.run(out_dir=tmp_path)
    for name in ["feature_by_failure", "correlation", "failure_rate_by_type",
                 "failure_modes", "temporal_structure"]:
        assert (tmp_path / f"{name}.png").exists(), name
    saved = json.loads((tmp_path / "eda.json").read_text(encoding="utf-8"))
    assert saved["label_inconsistencies"]["counts"] == {"failure_without_mode": 9, "mode_without_failure": 18}
    assert saved["temporal_structure"]["has_temporal_structure"] is True
    assert rep["exploration"]["n_rows"] == 10000
