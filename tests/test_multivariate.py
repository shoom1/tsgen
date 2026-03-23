"""
Tests for MultivariateGBM model with full covariance.

This model captures cross-asset correlations via a full covariance matrix
and Cholesky decomposition when full_covariance=True (default).
"""

import pytest
import torch
import numpy as np
from tsgen.models.baselines import MultivariateGBM
from tsgen.data.pipeline import load_prices, clean_data, process_prices, create_windows, create_dataloader
from tsgen.data.processor import LogReturnProcessor


@pytest.fixture
def synthetic_dataloader():
    """Create dataloader from database."""
    # Load and process data
    df = load_prices(['AAPL', 'MSFT', 'GOOG'], '2024-01-01', '2024-12-31')
    df_clean = clean_data(df)

    processor = LogReturnProcessor()
    data_scaled = process_prices(df_clean, processor, fit=True)

    sequences = create_windows(data_scaled, sequence_length=64)
    loader = create_dataloader(sequences, batch_size=16, shuffle=True)

    return loader


def test_multivariate_initialization():
    """Test MultivariateGBM model can be initialized with full covariance."""
    model = MultivariateGBM(features=3)  # full_covariance=True by default
    assert model is not None
    assert model.features == 3
    assert model.full_covariance == True

    # Check buffers are initialized
    assert model.mean.shape == (3,)
    assert model.cholesky_L.shape == (3, 3)

    # Cholesky should start as identity
    assert torch.allclose(model.cholesky_L, torch.eye(3))


def test_multivariate_fit(synthetic_dataloader):
    """Test MultivariateGBM model fitting with full covariance."""
    model = MultivariateGBM(features=3)

    # Fit model
    model.fit(synthetic_dataloader)

    # Verify parameters were learned
    assert model.mean is not None
    assert model.cholesky_L is not None
    assert model.mean.shape == (3,)
    assert model.cholesky_L.shape == (3, 3)

    # Verify parameters are finite
    assert torch.all(torch.isfinite(model.mean))
    assert torch.all(torch.isfinite(model.cholesky_L))

    # Verify Cholesky is lower triangular
    upper_triangle = torch.triu(model.cholesky_L, diagonal=1)
    assert torch.allclose(upper_triangle, torch.zeros_like(upper_triangle), atol=1e-6)

    # Verify diagonal elements are positive (valid Cholesky)
    diagonal = torch.diagonal(model.cholesky_L)
    assert torch.all(diagonal > 0), "Cholesky diagonal should be positive"


def test_multivariate_sample(synthetic_dataloader):
    """Test MultivariateLogNormal model sampling."""
    model = MultivariateGBM(features=3)
    model.fit(synthetic_dataloader)

    # Generate samples
    num_samples = 100
    seq_len = 64
    samples = model.generate(num_samples, seq_len)

    # Verify sample shape
    assert samples.shape == (num_samples, seq_len, 3)

    # Verify samples are finite
    assert torch.all(torch.isfinite(samples))


def test_multivariate_correlation_structure():
    """Test that multivariate model captures correlation structure."""
    model = MultivariateGBM(features=2)

    # Create correlated synthetic data
    # Asset 1 and Asset 2 with correlation 0.7
    torch.manual_seed(42)
    n_samples = 1000
    seq_len = 64

    # Create correlated data using known covariance
    true_mean = torch.tensor([0.0, 0.0])
    true_cov = torch.tensor([[1.0, 0.7],
                             [0.7, 1.0]])
    L_true = torch.linalg.cholesky(true_cov)

    # Generate correlated data
    z = torch.randn(n_samples * seq_len, 2)
    X = true_mean + torch.matmul(z, L_true.T)
    X = X.reshape(n_samples, seq_len, 2)

    # Create simple dataloader
    from torch.utils.data import DataLoader, TensorDataset
    dataset = TensorDataset(X)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Fit model
    model.fit(dataloader)

    # Verify learned mean is close to true mean
    assert torch.allclose(model.mean, true_mean, atol=0.1)

    # Verify learned covariance is close to true covariance
    learned_cov = model.cholesky_L @ model.cholesky_L.T
    assert torch.allclose(learned_cov, true_cov, atol=0.15)

    # Generate samples and verify correlation
    samples = model.generate(10000, 100)
    samples_flat = samples.reshape(-1, 2).numpy()

    # Compute empirical correlation
    empirical_corr = np.corrcoef(samples_flat, rowvar=False)

    # Correlation should be close to 0.7
    assert abs(empirical_corr[0, 1] - 0.7) < 0.05, \
        f"Expected correlation ~0.7, got {empirical_corr[0, 1]:.3f}"


def test_multivariate_vs_gbm_correlation():
    """Test that full covariance mode preserves correlation while independent mode doesn't."""
    # Create correlated data
    torch.manual_seed(42)
    n_samples = 500
    seq_len = 64

    true_cov = torch.tensor([[1.0, 0.8],
                             [0.8, 1.0]])
    L_true = torch.linalg.cholesky(true_cov)

    z = torch.randn(n_samples * seq_len, 2)
    X = torch.matmul(z, L_true.T).reshape(n_samples, seq_len, 2)

    from torch.utils.data import DataLoader, TensorDataset
    dataset = TensorDataset(X)
    dataloader = DataLoader(dataset, batch_size=32)

    # Fit both modes of MultivariateGBM
    multivariate = MultivariateGBM(features=2, full_covariance=True)
    multivariate.fit(dataloader)

    independent = MultivariateGBM(features=2, full_covariance=False)
    independent.fit(dataloader)

    # Generate samples
    mv_samples = multivariate.generate(5000, 100).reshape(-1, 2).numpy()
    indep_samples = independent.generate(5000, 100).reshape(-1, 2).numpy()

    # Compute correlations
    mv_corr = np.corrcoef(mv_samples, rowvar=False)[0, 1]
    indep_corr = np.corrcoef(indep_samples, rowvar=False)[0, 1]

    # Full covariance mode should preserve correlation (~0.8)
    assert abs(mv_corr - 0.8) < 0.1, \
        f"Full covariance correlation should be ~0.8, got {mv_corr:.3f}"

    # Independent mode should have near-zero correlation (samples independently)
    assert abs(indep_corr) < 0.1, \
        f"Independent mode correlation should be ~0, got {indep_corr:.3f}"


def test_multivariate_reproducibility():
    """Test MultivariateGBM sampling is reproducible with same seed."""
    model = MultivariateGBM(features=3)

    # Set parameters manually
    model.mean = torch.tensor([0.001, 0.002, 0.003])
    model.cholesky_L = torch.tensor([[1.0, 0.0, 0.0],
                                     [0.5, 0.9, 0.0],
                                     [0.3, 0.2, 0.8]])

    # Generate samples with same seed
    torch.manual_seed(42)
    samples1 = model.generate(10, 64)

    torch.manual_seed(42)
    samples2 = model.generate(10, 64)

    # Verify samples are identical
    assert torch.allclose(samples1, samples2)


def test_multivariate_save_load():
    """Test that multivariate model can be saved and loaded."""
    import tempfile
    import os

    # Create model
    model = MultivariateGBM(features=2)
    model.mean = torch.tensor([0.01, 0.02])
    model.cholesky_L = torch.tensor([[1.0, 0.0],
                                     [0.7, 0.7]])

    # Save model
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
        temp_path = f.name
        torch.save(model, temp_path)

    try:
        # Load model
        loaded_model = torch.load(temp_path, weights_only=False)

        # Verify parameters are preserved
        assert torch.allclose(loaded_model.mean, model.mean)
        assert torch.allclose(loaded_model.cholesky_L, model.cholesky_L)

        # Verify sampling works and is identical
        torch.manual_seed(42)
        samples1 = model.generate(5, 32)

        torch.manual_seed(42)
        samples2 = loaded_model.generate(5, 32)

        assert torch.allclose(samples1, samples2)

    finally:
        os.unlink(temp_path)


def test_multivariate_is_statistical_model():
    """Test that MultivariateGBM is a StatisticalModel (not DiffusionModel)."""
    from tsgen.models.base_model import StatisticalModel, DiffusionModel

    model = MultivariateGBM(features=2)

    # Should be a StatisticalModel
    assert isinstance(model, StatisticalModel)

    # Should NOT be a DiffusionModel (no forward method)
    assert not isinstance(model, DiffusionModel)

    # Should have fit and generate methods
    assert hasattr(model, 'fit')
    assert hasattr(model, 'generate')
    assert callable(model.fit)
    assert callable(model.generate)


def test_multivariate_cholesky_regularization():
    """Test that regularization prevents singular matrix issues."""
    model = MultivariateGBM(features=2)

    # Create nearly singular data (all same value)
    X = torch.ones(100, 64, 2) * 0.5
    X += torch.randn(100, 64, 2) * 1e-8  # Tiny noise

    from torch.utils.data import DataLoader, TensorDataset
    dataset = TensorDataset(X)
    dataloader = DataLoader(dataset, batch_size=32)

    # Should not raise error due to regularization
    try:
        model.fit(dataloader)
        # Verify Cholesky was computed successfully
        assert torch.all(torch.isfinite(model.cholesky_L))
    except Exception as e:
        pytest.fail(f"Regularization failed: {e}")


def test_multivariate_cov_to_corr():
    """Test covariance to correlation conversion."""
    model = MultivariateGBM(features=2)

    # Create known covariance matrix
    cov = torch.tensor([[4.0, 2.0],
                        [2.0, 9.0]])

    # Expected correlation matrix
    # corr[0,0] = 1.0
    # corr[1,1] = 1.0
    # corr[0,1] = corr[1,0] = 2.0 / (sqrt(4) * sqrt(9)) = 2.0 / 6.0 = 0.333...

    corr = model._cov_to_corr(cov)

    expected_corr = torch.tensor([[1.0, 1/3],
                                  [1/3, 1.0]])

    assert torch.allclose(corr, expected_corr, atol=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
