import numpy as np
import pytest

from iris_lab.data import FEATURES, load_iris_df
from iris_lab.pca import fit_pca, plot_pca_3d


@pytest.fixture(scope="module")
def df():
    return load_iris_df()


def test_fit_pca_returns_three_components(df):
    coords, ratio = fit_pca(df[FEATURES].to_numpy(), n_components=3)
    assert coords.shape == (150, 3)
    assert len(ratio) == 3


def test_fit_pca_explained_variance_is_sorted_and_high(df):
    _, ratio = fit_pca(df[FEATURES].to_numpy(), n_components=3)
    assert all(ratio[i] >= ratio[i + 1] for i in range(2))
    # 標準化後前三個主成分可解釋超過 99% 的變異
    assert sum(ratio) > 0.99


def test_fit_pca_components_are_centered(df):
    coords, _ = fit_pca(df[FEATURES].to_numpy(), n_components=3)
    np.testing.assert_allclose(coords.mean(axis=0), 0, atol=1e-10)


def test_plot_pca_3d_writes_png(df, tmp_path):
    out = tmp_path / "pca.png"
    ratio = plot_pca_3d(df, out)
    assert out.exists() and out.stat().st_size > 10_000
    assert len(ratio) == 3
