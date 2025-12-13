"""
Tests for correlation structure metrics.

Tests the comprehensive correlation analysis functions that compare
correlation matrices, eigenvalue spectra, and rolling correlations.
"""

import pytest
import numpy as np
from tsgen.analysis.metrics import (
    compute_correlation_structure_metrics,
    compute_rolling_correlation_stability
)


def test_correlation_metrics_basic():
    """Test basic correlation structure metrics computation."""
    # Create simple correlated data
    np.random.seed(42)
    n_samples = 100
    seq_len = 64
    features = 3

    # Real data with known correlation
    real_returns = np.random.randn(n_samples, seq_len, features)

    # Synthetic data - slightly different correlation
    syn_returns = np.random.randn(n_samples, seq_len, features) * 1.1

    metrics = compute_correlation_structure_metrics(real_returns, syn_returns)

    # Check all expected metrics are present
    assert 'corr_frobenius_norm' in metrics
    assert 'corr_max_diff' in metrics
    assert 'corr_mean_diff' in metrics
    assert 'eigenvalue_mse' in metrics
    assert 'eigenvalue_max_diff' in metrics
    assert 'explained_var_ratio_diff' in metrics

    # Check metrics are finite
    assert np.isfinite(metrics['corr_frobenius_norm'])
    assert np.isfinite(metrics['corr_max_diff'])
    assert np.isfinite(metrics['corr_mean_diff'])
    assert np.isfinite(metrics['eigenvalue_mse'])

    # Check metrics are non-negative
    assert metrics['corr_frobenius_norm'] >= 0
    assert metrics['corr_max_diff'] >= 0
    assert metrics['corr_mean_diff'] >= 0
    assert metrics['eigenvalue_mse'] >= 0


def test_correlation_metrics_identical_data():
    """Test that identical data gives zero difference."""
    np.random.seed(42)
    n_samples = 50
    seq_len = 32
    features = 2

    data = np.random.randn(n_samples, seq_len, features)

    metrics = compute_correlation_structure_metrics(data, data)

    # All differences should be very small (numerical precision)
    assert metrics['corr_frobenius_norm'] < 1e-10
    assert metrics['corr_max_diff'] < 1e-10
    assert metrics['corr_mean_diff'] < 1e-10
    assert metrics['eigenvalue_mse'] < 1e-10


def test_correlation_matrices_stored():
    """Test that correlation matrices are stored for plotting."""
    np.random.seed(42)
    n_samples = 50
    seq_len = 32
    features = 3

    real_returns = np.random.randn(n_samples, seq_len, features)
    syn_returns = np.random.randn(n_samples, seq_len, features)

    metrics = compute_correlation_structure_metrics(real_returns, syn_returns)

    assert 'real_corr_matrix' in metrics
    assert 'syn_corr_matrix' in metrics
    assert metrics['real_corr_matrix'].shape == (features, features)
    assert metrics['syn_corr_matrix'].shape == (features, features)

    # Correlation matrices should be symmetric
    assert np.allclose(metrics['real_corr_matrix'], metrics['real_corr_matrix'].T)
    assert np.allclose(metrics['syn_corr_matrix'], metrics['syn_corr_matrix'].T)

    # Diagonal should be 1
    assert np.allclose(np.diag(metrics['real_corr_matrix']), 1.0)
    assert np.allclose(np.diag(metrics['syn_corr_matrix']), 1.0)


def test_eigenvalue_analysis():
    """Test eigenvalue spectrum comparison."""
    np.random.seed(42)
    n_samples = 100
    seq_len = 64
    features = 4

    real_returns = np.random.randn(n_samples, seq_len, features)
    syn_returns = np.random.randn(n_samples, seq_len, features)

    metrics = compute_correlation_structure_metrics(real_returns, syn_returns)

    assert 'real_eigenvalues' in metrics
    assert 'syn_eigenvalues' in metrics
    assert len(metrics['real_eigenvalues']) == features
    assert len(metrics['syn_eigenvalues']) == features

    # Eigenvalues should be sorted in descending order
    assert np.all(metrics['real_eigenvalues'][:-1] >= metrics['real_eigenvalues'][1:])
    assert np.all(metrics['syn_eigenvalues'][:-1] >= metrics['syn_eigenvalues'][1:])

    # Eigenvalues should be positive (correlation matrix is positive semi-definite)
    assert np.all(metrics['real_eigenvalues'] >= 0)
    assert np.all(metrics['syn_eigenvalues'] >= 0)

    # Sum of eigenvalues should equal number of features
    assert np.abs(metrics['real_eigenvalues'].sum() - features) < 0.01
    assert np.abs(metrics['syn_eigenvalues'].sum() - features) < 0.01


def test_rolling_correlation_stability():
    """Test rolling correlation stability metric."""
    np.random.seed(42)
    n_samples = 50
    seq_len = 64
    features = 2

    real_returns = np.random.randn(n_samples, seq_len, features)
    syn_returns = np.random.randn(n_samples, seq_len, features)

    metrics = compute_rolling_correlation_stability(real_returns, syn_returns, window=20)

    assert 'rolling_corr_stability' in metrics
    assert 'rolling_corr_std_diff' in metrics

    # Should be finite
    assert np.isfinite(metrics['rolling_corr_stability'])
    assert np.isfinite(metrics['rolling_corr_std_diff'])

    # Should be non-negative
    assert metrics['rolling_corr_stability'] >= 0
    assert metrics['rolling_corr_std_diff'] >= 0


def test_rolling_correlation_insufficient_data():
    """Test rolling correlation with insufficient data."""
    # Too few features
    data_1d = np.random.randn(10, 20, 1)
    metrics = compute_rolling_correlation_stability(data_1d, data_1d, window=10)
    assert np.isnan(metrics['rolling_corr_stability'])

    # Sequence too short
    data_short = np.random.randn(10, 10, 2)
    metrics = compute_rolling_correlation_stability(data_short, data_short, window=20)
    assert np.isnan(metrics['rolling_corr_stability'])


def test_rolling_correlation_included_in_full_metrics():
    """Test that rolling correlation is included when data is sufficient."""
    np.random.seed(42)
    n_samples = 50
    seq_len = 64
    features = 3

    real_returns = np.random.randn(n_samples, seq_len, features)
    syn_returns = np.random.randn(n_samples, seq_len, features)

    metrics = compute_correlation_structure_metrics(real_returns, syn_returns)

    # Should have rolling correlation metrics
    assert 'rolling_corr_stability' in metrics
    assert 'rolling_corr_std_diff' in metrics
    assert np.isfinite(metrics['rolling_corr_stability'])


def test_correlation_metrics_single_feature():
    """Test correlation metrics with single feature (edge case)."""
    np.random.seed(42)
    n_samples = 50
    seq_len = 32
    features = 1

    real_returns = np.random.randn(n_samples, seq_len, features)
    syn_returns = np.random.randn(n_samples, seq_len, features)

    metrics = compute_correlation_structure_metrics(real_returns, syn_returns)

    # Should still work, correlation matrix is 1x1
    assert metrics['real_corr_matrix'].shape == (1, 1)
    assert metrics['real_corr_matrix'][0, 0] == pytest.approx(1.0)

    # Rolling correlation should be NaN (need at least 2 features)
    assert np.isnan(metrics['rolling_corr_stability'])


def test_correlation_max_diff_bounds():
    """Test that max difference is bounded correctly."""
    np.random.seed(42)
    n_samples = 100
    seq_len = 64
    features = 3

    real_returns = np.random.randn(n_samples, seq_len, features)
    syn_returns = np.random.randn(n_samples, seq_len, features)

    metrics = compute_correlation_structure_metrics(real_returns, syn_returns)

    # Correlation values are in [-1, 1], so max difference should be in [0, 2]
    assert 0 <= metrics['corr_max_diff'] <= 2
    assert 0 <= metrics['corr_mean_diff'] <= 2


def test_explained_variance_ratio():
    """Test explained variance ratio difference."""
    np.random.seed(42)
    n_samples = 100
    seq_len = 64
    features = 3

    real_returns = np.random.randn(n_samples, seq_len, features)
    syn_returns = np.random.randn(n_samples, seq_len, features)

    metrics = compute_correlation_structure_metrics(real_returns, syn_returns)

    # Explained variance ratio difference should be in [0, 2]
    # (sum of absolute differences of ratios that sum to 1)
    assert 0 <= metrics['explained_var_ratio_diff'] <= 2


def test_correlation_structure_with_perfect_correlation():
    """Test with perfectly correlated data."""
    np.random.seed(42)
    n_samples = 50
    seq_len = 32
    features = 2

    # Create perfectly correlated data
    base = np.random.randn(n_samples, seq_len, 1)
    real_returns = np.concatenate([base, base], axis=2)  # Perfect correlation

    # Synthetic data with different correlation structure
    base2 = np.random.randn(n_samples, seq_len, 1)
    syn_returns = np.concatenate([base2, base2 * 0.9], axis=2)  # Different base

    metrics = compute_correlation_structure_metrics(real_returns, syn_returns)

    # Real data should have correlation close to 1
    assert metrics['real_corr_matrix'][0, 1] > 0.99

    # Synthetic should also have high correlation but from different data
    assert metrics['syn_corr_matrix'][0, 1] > 0.8

    # Metrics should be computed (may or may not have differences depending on sampling)
    assert np.isfinite(metrics['corr_max_diff'])
    assert np.isfinite(metrics['corr_frobenius_norm'])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
