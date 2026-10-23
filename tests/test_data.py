import pandas as pd
import pytest

from iris_lab.data import (
    FEATURES,
    CLASSES,
    TARGET,
    load_iris_df,
    save_dataset,
    load_dataset,
    explore,
)


def test_load_iris_df_shape_and_columns():
    df = load_iris_df()
    assert df.shape == (150, 5)
    assert list(df.columns) == FEATURES + [TARGET]


def test_load_iris_df_has_three_balanced_classes():
    df = load_iris_df()
    assert sorted(df[TARGET].unique()) == sorted(CLASSES)
    assert df[TARGET].value_counts().tolist() == [50, 50, 50]


def test_save_and_load_roundtrip(tmp_path):
    path = tmp_path / "sub" / "iris.csv"
    df = load_iris_df()
    save_dataset(df, path)
    assert path.exists()
    loaded = load_dataset(path)
    pd.testing.assert_frame_equal(df, loaded)


def test_load_dataset_rejects_missing_columns(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"a": [1]}).to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_dataset(path)


def test_explore_reports_basic_checks():
    df = load_iris_df()
    report = explore(df)
    assert report["n_rows"] == 150
    assert report["n_features"] == 4
    assert report["missing_values"] == 0
    assert report["duplicate_rows"] == 1  # Iris 原始資料中有一筆完全重複
    assert report["class_counts"] == {c: 50 for c in CLASSES}
    assert set(report["describe"].keys()) == set(FEATURES)
    assert report["describe"]["sepal_length"]["mean"] == pytest.approx(5.843, abs=1e-3)
    assert set(report["class_means"].keys()) == set(CLASSES)
