"""
Tests for baseline generative models (MultivariateGaussian and Bootstrap).

These models don't use neural networks but provide important baselines
for comparison with diffusion models.
"""

import pytest
import torch
import numpy as np
from tsgen.models.baselines import MultivariateGaussian, BootstrapGenerativeModel
from tsgen.data.pipeline import load_prices, clean_data, process_prices, create_windows, create_dataloader
from tsgen.data.processor import LogReturnProcessor

pytestmark = pytest.mark.integration


@pytest.fixture
def synthetic_dataloader():
    """Create dataloader from database."""
    # Load and process data
    df = load_prices(['AAPL', 'MSFT'], '2024-01-01', '2024-12-31')
    df_clean = clean_data(df)

    processor = LogReturnProcessor()
    data_scaled = process_prices(df_clean, processor, fit=True)

    sequences = create_windows(data_scaled, sequence_length=64)
    loader = create_dataloader(sequences, batch_size=16, shuffle=True)

    return loader


def test_multivariate_gbm_initialization():
    """Test MultivariateGaussian model can be initialized."""
    # Test with full covariance (default)
    model_full = MultivariateGaussian(features=2)
    assert model_full is not None
    assert model_full.features == 2
    assert model_full.full_covariance == True

    # Test without covariance (independent sampling)
    model_indep = MultivariateGaussian(features=2, full_covariance=False)
    assert model_indep.features == 2
    assert model_indep.full_covariance == False


def test_multivariate_gbm_fit_independent(synthetic_dataloader):
    """Test MultivariateGaussian model fitting (independent mode)."""
    model = MultivariateGaussian(features=2, full_covariance=False)

    # Fit model
    model.fit(synthetic_dataloader)

    # Verify parameters were learned
    assert model.mu is not None
    assert model.sigma is not None
    assert len(model.mu) == 2
    assert len(model.sigma) == 2

    # Verify parameters are reasonable
    assert torch.all(torch.isfinite(model.mu))
    assert torch.all(torch.isfinite(model.sigma))
    assert torch.all(model.sigma > 0), "Sigma should be positive"


def test_multivariate_gbm_sample_independent(synthetic_dataloader):
    """Test MultivariateGaussian model sampling (independent mode)."""
    model = MultivariateGaussian(features=2, full_covariance=False)
    model.fit(synthetic_dataloader)

    # Generate samples
    num_samples = 10
    seq_len = 64
    samples = model.generate(num_samples, seq_len)

    # Verify sample shape
    assert samples.shape == (num_samples, seq_len, 2)

    # Verify samples are finite
    assert torch.all(torch.isfinite(samples))


def test_multivariate_gbm_reproducibility_independent():
    """Test MultivariateGaussian sampling is reproducible with same seed (independent mode)."""
    model = MultivariateGaussian(features=2, full_covariance=False)

    # Set parameters manually
    model.mu = torch.tensor([0.001, 0.002])
    model.sigma = torch.tensor([0.02, 0.03])

    # Generate samples with same seed
    torch.manual_seed(42)
    samples1 = model.generate(5, 32)

    torch.manual_seed(42)
    samples2 = model.generate(5, 32)

    # Verify samples are identical
    assert torch.allclose(samples1, samples2)


def _make_ordered_loader(series: np.ndarray, window_len: int = 32, batch_size: int = 16):
    """Build a stride-1, shuffle=False dataloader of overlapping windows.

    Mirrors what the training pipeline produces for order-dependent baselines.
    """
    from torch.utils.data import DataLoader, TensorDataset
    T, F = series.shape
    n = T - window_len + 1
    windows = np.empty((n, window_len, F), dtype=np.float32)
    for i in range(n):
        windows[i] = series[i:i + window_len]
    ds = TensorDataset(torch.from_numpy(windows))
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def test_bootstrap_initialization():
    """Model constructs with expected attributes and accepts block_p."""
    model = BootstrapGenerativeModel(features=2, sequence_length=64, block_p=0.1)
    assert model.features == 2
    assert model.sequence_length == 64
    assert model.block_p == 0.1
    assert model.history is None  # set by fit()


def test_bootstrap_rejects_invalid_block_p():
    with pytest.raises(ValueError):
        BootstrapGenerativeModel(features=2, sequence_length=64, block_p=0.0)
    with pytest.raises(ValueError):
        BootstrapGenerativeModel(features=2, sequence_length=64, block_p=1.5)


def test_bootstrap_fit_stores_flat_series():
    """Fit reconstructs the chronological series from windowed batches."""
    rng = np.random.default_rng(0)
    series = rng.normal(size=(500, 3)).astype(np.float32)
    loader = _make_ordered_loader(series, window_len=32)

    model = BootstrapGenerativeModel(features=3, sequence_length=32)
    model.fit(loader)

    # History is the flat (T, F) series, not windowed
    assert model.history.ndim == 2
    assert model.history.shape == (500, 3)
    np.testing.assert_allclose(model.history.numpy(), series, atol=1e-6)


def test_bootstrap_respects_requested_seq_len():
    """Unlike the old window-resampler, generate() honors seq_len exactly."""
    rng = np.random.default_rng(1)
    series = rng.normal(size=(400, 2)).astype(np.float32)
    loader = _make_ordered_loader(series, window_len=32)

    model = BootstrapGenerativeModel(features=2, sequence_length=32)
    model.fit(loader)

    for requested_len in [17, 64, 100, 250]:
        out = model.generate(n_samples=4, seq_len=requested_len)
        assert out.shape == (4, requested_len, 2)
        assert torch.isfinite(out).all()


def test_bootstrap_produces_novel_paths():
    """Generated windows must differ from any training window
    (the old window-resampler failed this — it returned exact copies)."""
    rng = np.random.default_rng(2)
    series = rng.normal(size=(500, 1)).astype(np.float32)
    loader = _make_ordered_loader(series, window_len=32)

    model = BootstrapGenerativeModel(features=1, sequence_length=32, block_p=0.3)
    model.fit(loader)

    # Collect all training windows
    all_train_windows = set()
    for i in range(len(series) - 32 + 1):
        all_train_windows.add(tuple(series[i:i + 32, 0].tolist()))

    # Generate many samples and check at least one is novel
    torch.manual_seed(99)
    generated = model.generate(n_samples=50, seq_len=32).numpy()
    novel_count = sum(
        1 for g in generated
        if tuple(g[:, 0].tolist()) not in all_train_windows
    )
    # With block_p=0.3 (avg block length ~3), 32-step sequences will cross
    # multiple block boundaries and almost certainly be novel.
    assert novel_count >= 40, f"Only {novel_count}/50 samples were novel"


def test_bootstrap_block_p_one_is_iid_per_step():
    """block_p=1 means every step starts a new block — equivalent to iid
    resampling of single elements (no temporal structure preserved)."""
    rng = np.random.default_rng(3)
    series = rng.normal(size=(200, 1)).astype(np.float32)
    loader = _make_ordered_loader(series, window_len=32)

    model = BootstrapGenerativeModel(features=1, sequence_length=32, block_p=1.0)
    model.fit(loader)

    torch.manual_seed(0)
    out = model.generate(n_samples=200, seq_len=50).numpy().reshape(-1)
    # Mean should approximate the empirical mean of the series
    np.testing.assert_allclose(out.mean(), series.mean(), atol=0.15)
    np.testing.assert_allclose(out.std(), series.std(), atol=0.15)


def test_bootstrap_preserves_marginal_statistics():
    """On a long enough sample, generated returns should have mean/std
    close to those of the training series."""
    rng = np.random.default_rng(4)
    series = rng.normal(loc=0.01, scale=0.02, size=(2000, 3)).astype(np.float32)
    loader = _make_ordered_loader(series, window_len=64)

    model = BootstrapGenerativeModel(features=3, sequence_length=64, block_p=0.1)
    model.fit(loader)

    torch.manual_seed(1)
    out = model.generate(n_samples=200, seq_len=200).numpy()
    flat = out.reshape(-1, 3)
    np.testing.assert_allclose(flat.mean(axis=0), series.mean(axis=0), atol=0.005)
    np.testing.assert_allclose(flat.std(axis=0), series.std(axis=0), atol=0.003)


def test_bootstrap_reproducibility():
    """Sampling is reproducible under the same seed."""
    rng = np.random.default_rng(5)
    series = rng.normal(size=(300, 2)).astype(np.float32)
    loader = _make_ordered_loader(series, window_len=32)

    model = BootstrapGenerativeModel(features=2, sequence_length=32)
    model.fit(loader)

    torch.manual_seed(42)
    samples1 = model.generate(5, 64)

    torch.manual_seed(42)
    samples2 = model.generate(5, 64)

    assert torch.allclose(samples1, samples2)


def test_multivariate_gbm_vs_bootstrap_different_outputs(synthetic_dataloader):
    """Test that MultivariateGaussian and Bootstrap produce different samples."""
    # Create and fit both models
    gbm = MultivariateGaussian(features=2, full_covariance=False)
    gbm.fit(synthetic_dataloader)

    bootstrap = BootstrapGenerativeModel(features=2, sequence_length=64)
    bootstrap.fit(synthetic_dataloader)

    # Generate samples with same seed
    torch.manual_seed(42)
    gbm_samples = gbm.generate(5, 64)

    torch.manual_seed(42)
    bootstrap_samples = bootstrap.generate(5, 64)

    # Verify samples are different (models work differently)
    assert not torch.allclose(gbm_samples, bootstrap_samples, atol=0.1)


def test_baseline_models_save_load():
    """Test that baseline models can be saved and loaded."""
    import tempfile
    import os

    # Create MultivariateGaussian model (independent mode)
    gbm = MultivariateGaussian(features=2, full_covariance=False)
    gbm.mu = torch.tensor([0.001, 0.002])
    gbm.sigma = torch.tensor([0.02, 0.03])

    # Save model
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
        temp_path = f.name
        torch.save(gbm, temp_path)

    try:
        # Load model
        loaded_gbm = torch.load(temp_path, weights_only=False)

        # Verify parameters are preserved
        assert torch.allclose(loaded_gbm.mu, gbm.mu)
        assert torch.allclose(loaded_gbm.sigma, gbm.sigma)

        # Verify sampling works
        torch.manual_seed(42)
        samples1 = gbm.generate(3, 32)

        torch.manual_seed(42)
        samples2 = loaded_gbm.generate(3, 32)

        assert torch.allclose(samples1, samples2)

    finally:
        os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
