import json

import pytest

from iris_lab.data import CLASSES
from iris_lab.pipeline import run_experiment
from iris_lab.report import render_paper


@pytest.fixture(scope="module")
def experiment(tmp_path_factory):
    root = tmp_path_factory.mktemp("exp")
    results = run_experiment(root, epochs=60, seed=42)
    return root, results


def test_pipeline_saves_dataset_and_figures(experiment):
    root, results = experiment
    assert (root / "data" / "iris.csv").exists()
    for name in ["feature_distributions", "pca_3d", "training_history", "confusion_matrix"]:
        assert (root / "outputs" / "figures" / f"{name}.png").exists()
    assert set(results["figures"]) == {"feature_distributions", "pca_3d", "training_history", "confusion_matrix"}


def test_pipeline_writes_json_results(experiment):
    root, results = experiment
    saved = json.loads((root / "outputs" / "results.json").read_text(encoding="utf-8"))
    assert saved["test"]["accuracy"] == results["test"]["accuracy"]
    assert {"exploration", "pca", "model", "history", "test", "cv_scores"} <= set(saved)
    assert saved["model"]["n_train"] == 120 and saved["model"]["n_test"] == 30


def test_pipeline_writes_paper(experiment):
    root, _ = experiment
    paper = (root / "docs" / "paper.md").read_text(encoding="utf-8")
    assert paper.startswith("# ")


def test_render_paper_contains_sections_and_numbers(experiment):
    _, results = experiment
    paper = render_paper(results)
    headings = [line for line in paper.splitlines() if line.startswith("## ")]
    for heading in ["摘要", "前言", "資料與探索", "研究方法", "實驗結果", "討論", "結論", "參考文獻"]:
        assert any(heading in h for h in headings), heading
    assert f"{results['test']['accuracy']:.2%}" in paper
    assert f"{sum(results['pca']['explained_variance_ratio']):.2%}" in paper
    for cls in CLASSES:
        assert cls in paper
    assert "../outputs/figures/pca_3d.png" in paper
