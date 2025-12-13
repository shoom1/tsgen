"""
Tests for baseline generative models (GBM and Bootstrap).

These models don't use neural networks but provide important baselines
for comparison with diffusion models.
"""

import pytest
import torch
import numpy as np
from tsgen.models.baselines import GBMGenerativeModel, BootstrapGenerativeModel
from tsgen.data.pipeline import load_prices, clean_data, process_prices, create_windows, create_dataloader
from tsgen.data.processor import LogReturnProcessor


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


def test_gbm_initialization():
    """Test GBM model can be initialized."""
    model = GBMGenerativeModel(features=2)
    assert model is not None
    assert model.features == 2


def test_gbm_fit(synthetic_dataloader):
    """Test GBM model fitting."""
    model = GBMGenerativeModel(features=2)

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


def test_gbm_sample(synthetic_dataloader):
    """Test GBM model sampling."""
    model = GBMGenerativeModel(features=2)
    model.fit(synthetic_dataloader)

    # Generate samples
    num_samples = 10
    seq_len = 64
    samples = model.sample(num_samples, seq_len)

    # Verify sample shape
    assert samples.shape == (num_samples, seq_len, 2)

    # Verify samples are finite
    assert torch.all(torch.isfinite(samples))


def test_gbm_reproducibility():
    """Test GBM sampling is reproducible with same seed."""
    model = GBMGenerativeModel(features=2)

    # Set parameters manually
    model.mu = torch.tensor([0.001, 0.002])
    model.sigma = torch.tensor([0.02, 0.03])

    # Generate samples with same seed
    torch.manual_seed(42)
    samples1 = model.sample(5, 32)

    torch.manual_seed(42)
    samples2 = model.sample(5, 32)

    # Verify samples are identical
    assert torch.allclose(samples1, samples2)


def test_bootstrap_initialization():
    """Test Bootstrap model can be initialized."""
    model = BootstrapGenerativeModel(features=2, sequence_length=64)
    assert model is not None
    assert model.features == 2
    assert model.sequence_length == 64


def test_bootstrap_fit(synthetic_dataloader):
    """Test Bootstrap model fitting."""
    model = BootstrapGenerativeModel(features=2, sequence_length=64)

    # Fit model (stores historical windows)
    model.fit(synthetic_dataloader)

    # Verify history was stored
    assert model.history is not None
    assert len(model.history) > 0

    # Verify history shape (num_windows, seq_len, features)
    assert model.history.shape[1] == 64  # seq_len
    assert model.history.shape[2] == 2   # features


def test_bootstrap_sample(synthetic_dataloader):
    """Test Bootstrap model sampling."""
    model = BootstrapGenerativeModel(features=2, sequence_length=64)
    model.fit(synthetic_dataloader)

    # Generate samples
    num_samples = 10
    seq_len = 64
    samples = model.sample(num_samples, seq_len)

    # Verify sample shape
    assert samples.shape == (num_samples, seq_len, 2)

    # Verify samples are finite
    assert torch.all(torch.isfinite(samples))


def test_bootstrap_sampling_from_history():
    """Test that bootstrap samples from history."""
    model = BootstrapGenerativeModel(features=1, sequence_length=32)

    # Create simple historical data
    # Store 4 different windows
    history = []
    for i in range(4):
        window = torch.arange(32).float().reshape(1, 32, 1) + i * 10  # Different patterns
        history.append(window)

    model.history = torch.cat(history, dim=0)

    # Generate sample
    torch.manual_seed(42)
    sample = model.sample(1, 32)

    # Verify sample is valid
    assert sample.shape == (1, 32, 1)
    assert torch.all(torch.isfinite(sample))


def test_bootstrap_reproducibility():
    """Test Bootstrap sampling is reproducible with same seed."""
    model = BootstrapGenerativeModel(features=2, sequence_length=64)

    # Set history manually
    model.history = torch.randn(10, 64, 2)

    # Generate samples with same seed
    torch.manual_seed(42)
    samples1 = model.sample(5, 64)

    torch.manual_seed(42)
    samples2 = model.sample(5, 64)

    # Verify samples are identical
    assert torch.allclose(samples1, samples2)


def test_gbm_vs_bootstrap_different_outputs(synthetic_dataloader):
    """Test that GBM and Bootstrap produce different samples."""
    # Create and fit both models
    gbm = GBMGenerativeModel(features=2)
    gbm.fit(synthetic_dataloader)

    bootstrap = BootstrapGenerativeModel(features=2, sequence_length=64)
    bootstrap.fit(synthetic_dataloader)

    # Generate samples with same seed
    torch.manual_seed(42)
    gbm_samples = gbm.sample(5, 64)

    torch.manual_seed(42)
    bootstrap_samples = bootstrap.sample(5, 64)

    # Verify samples are different (models work differently)
    assert not torch.allclose(gbm_samples, bootstrap_samples, atol=0.1)


def test_baseline_models_save_load():
    """Test that baseline models can be saved and loaded."""
    import tempfile
    import os

    # Create GBM model
    gbm = GBMGenerativeModel(features=2)
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
        samples1 = gbm.sample(3, 32)

        torch.manual_seed(42)
        samples2 = loaded_gbm.sample(3, 32)

        assert torch.allclose(samples1, samples2)

    finally:
        os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
