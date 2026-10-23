import numpy as np
import pytest

from iris_lab.data import CLASSES, load_iris_df
from iris_lab.model import cross_validate_mlp, evaluate, split_data, train_mlp


@pytest.fixture(scope="module")
def split():
    return split_data(load_iris_df(), test_size=0.2, seed=42)


@pytest.fixture(scope="module")
def trained(split):
    X_train, _, y_train, _ = split
    return train_mlp(X_train, y_train, epochs=200, seed=42)


def test_split_sizes_and_stratification(split):
    X_train, X_test, y_train, y_test = split
    assert X_train.shape == (120, 4) and X_test.shape == (30, 4)
    assert np.bincount(y_test).tolist() == [10, 10, 10]
    assert np.bincount(y_train).tolist() == [40, 40, 40]


def test_split_is_reproducible():
    df = load_iris_df()
    a = split_data(df, seed=7)[1]
    b = split_data(df, seed=7)[1]
    np.testing.assert_array_equal(a, b)


def test_train_records_history_per_epoch(trained):
    _, history = trained
    assert set(history) == {"loss", "train_acc", "val_acc"}
    assert all(len(v) == 200 for v in history.values())


def test_training_reduces_loss_and_learns(trained):
    _, history = trained
    assert history["loss"][-1] < history["loss"][0] / 5
    assert history["train_acc"][-1] > 0.95
    assert history["val_acc"][-1] > 0.9


def test_train_is_reproducible(split):
    X_train, X_test, y_train, _ = split
    m1, _ = train_mlp(X_train, y_train, epochs=50, seed=1)
    m2, _ = train_mlp(X_train, y_train, epochs=50, seed=1)
    np.testing.assert_array_equal(m1.predict_proba(X_test), m2.predict_proba(X_test))


def test_evaluate_on_test_set(trained, split):
    _, X_test, _, y_test = split
    model, _ = trained
    result = evaluate(model, X_test, y_test)
    assert result["accuracy"] > 0.9
    cm = np.array(result["confusion_matrix"])
    assert cm.shape == (3, 3) and cm.sum() == 30
    assert set(result["per_class"]) == set(CLASSES)
    assert {"precision", "recall", "f1"} <= set(result["per_class"]["setosa"])
    assert 0 <= result["macro_f1"] <= 1


def test_evaluate_perfect_predictions():
    class Oracle:
        def predict(self, X):
            return X[:, 0].astype(int)

    X = np.array([[0], [1], [2], [1]])
    y = np.array([0, 1, 2, 1])
    result = evaluate(Oracle(), X, y)
    assert result["accuracy"] == 1.0
    assert result["confusion_matrix"] == [[1, 0, 0], [0, 2, 0], [0, 0, 1]]


def test_cross_validation_scores():
    scores = cross_validate_mlp(load_iris_df(), folds=5, seed=42)
    assert len(scores) == 5
    assert np.mean(scores) > 0.9
