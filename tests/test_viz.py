from iris_lab.data import load_iris_df
from iris_lab.viz import plot_confusion_matrix, plot_feature_distributions, plot_training_history


def _is_png(path):
    return path.exists() and path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_plot_feature_distributions(tmp_path):
    out = tmp_path / "features.png"
    plot_feature_distributions(load_iris_df(), out)
    assert _is_png(out)


def test_plot_training_history(tmp_path):
    out = tmp_path / "history.png"
    history = {"loss": [1.0, 0.5, 0.2], "train_acc": [0.3, 0.8, 0.95], "val_acc": [0.3, 0.7, 0.9]}
    plot_training_history(history, out)
    assert _is_png(out)


def test_plot_confusion_matrix(tmp_path):
    out = tmp_path / "cm.png"
    plot_confusion_matrix([[10, 0, 0], [0, 9, 1], [0, 1, 9]], out)
    assert _is_png(out)
