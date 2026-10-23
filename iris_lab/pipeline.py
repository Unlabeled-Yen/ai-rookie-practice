"""整個實驗流程：取得資料 → 探索 → PCA → 訓練 MLP → 評估 → 圖表 → 小論文。

執行：python -m iris_lab.pipeline
"""
import json
from pathlib import Path

from iris_lab import model as nn
from iris_lab.data import explore, load_iris_df, save_dataset
from iris_lab.pca import plot_pca_3d
from iris_lab.report import render_paper
from iris_lab.viz import plot_confusion_matrix, plot_feature_distributions, plot_training_history


def run_experiment(root=".", epochs: int = 200, seed: int = 42) -> dict:
    root = Path(root)
    fig_dir = root / "outputs" / "figures"
    figures = {name: f"outputs/figures/{name}.png"
               for name in ["feature_distributions", "pca_3d", "training_history", "confusion_matrix"]}

    df = load_iris_df()
    save_dataset(df, root / "data" / "iris.csv")
    exploration = explore(df)
    plot_feature_distributions(df, fig_dir / "feature_distributions.png")
    ratio = plot_pca_3d(df, fig_dir / "pca_3d.png")

    X_train, X_test, y_train, y_test = nn.split_data(df, test_size=0.2, seed=seed)
    model, history = nn.train_mlp(X_train, y_train, epochs=epochs, seed=seed)
    test = nn.evaluate(model, X_test, y_test)
    plot_training_history(history, fig_dir / "training_history.png")
    plot_confusion_matrix(test["confusion_matrix"], fig_dir / "confusion_matrix.png")

    results = {
        "seed": seed,
        "exploration": exploration,
        "pca": {"explained_variance_ratio": ratio},
        "model": {
            "hidden_layers": list(nn.HIDDEN_LAYERS),
            "activation": "relu",
            "optimizer": "adam",
            "learning_rate": nn.LEARNING_RATE,
            "batch_size": 16,
            "epochs": epochs,
            "n_train": len(X_train),
            "n_test": len(X_test),
        },
        "history": history,
        "test": test,
        "cv_scores": nn.cross_validate_mlp(df, folds=5, seed=seed),
        "figures": figures,
    }
    (root / "outputs").mkdir(parents=True, exist_ok=True)
    (root / "outputs" / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    (root / "docs").mkdir(parents=True, exist_ok=True)
    (root / "docs" / "paper.md").write_text(render_paper(results), encoding="utf-8")
    return results


if __name__ == "__main__":
    r = run_experiment()
    print(f"test accuracy = {r['test']['accuracy']:.2%}, macro F1 = {r['test']['macro_f1']:.3f}")
