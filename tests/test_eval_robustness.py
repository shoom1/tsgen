"""Robustness tests for evaluation pipeline against degenerate samples.

Exposed by the 2026-04-20 smoke-test run of all experiments:
  - Bootstrap: empty/constant columns in synthetic correlation matrix made
    ``numpy.linalg.eigvalsh`` fail with "Eigenvalues did not converge".
  - Mamba (under-trained in smoke mode): produced NaN samples, similar failure.

The correlation-structure evaluator should produce finite metrics (or
explicit NaNs that propagate cleanly) rather than raising on any degenerate
covariance matrix.
"""

import numpy as np
import pytest

from tsgen.analysis.metrics import compute_correlation_structure_metrics


class TestCorrelationStructureRobustness:
    def test_handles_nan_in_synthetic(self):
        """Synthetic data with NaN entries — evaluator must not crash."""
        np.random.seed(0)
        real = np.random.randn(20, 32, 3)
        synthetic = np.random.randn(20, 32, 3)
        synthetic[0, 5, 1] = np.nan
        synthetic[2, 10, 2] = np.nan

        metrics = compute_correlation_structure_metrics(real, synthetic)
        assert 'corr_frobenius_norm' in metrics
        assert np.isfinite(metrics['eigenvalue_mse'])

    def test_handles_constant_column_in_synthetic(self):
        """Synthetic data with a constant-valued feature column (zero variance)
        produces NaN rows in the correlation matrix. Must not crash."""
        np.random.seed(1)
        real = np.random.randn(20, 32, 3)
        synthetic = np.random.randn(20, 32, 3)
        synthetic[:, :, 1] = 0.0  # Second feature is constant

        metrics = compute_correlation_structure_metrics(real, synthetic)
        assert np.isfinite(metrics['corr_frobenius_norm'])
        assert np.isfinite(metrics['eigenvalue_mse'])

    def test_handles_inf_in_synthetic(self):
        np.random.seed(2)
        real = np.random.randn(20, 32, 3)
        synthetic = np.random.randn(20, 32, 3)
        synthetic[0, 5, 1] = np.inf
        synthetic[3, 12, 2] = -np.inf

        metrics = compute_correlation_structure_metrics(real, synthetic)
        assert np.isfinite(metrics['corr_frobenius_norm'])

    def test_handles_all_nan_synthetic(self):
        """Worst case: every synthetic value is NaN."""
        real = np.random.RandomState(3).randn(20, 32, 3)
        synthetic = np.full_like(real, np.nan)

        metrics = compute_correlation_structure_metrics(real, synthetic)
        # Should produce metrics without raising, even if most end up NaN
        assert 'corr_frobenius_norm' in metrics


class TestCCCGARCHFitRobustness:
    """CCC-GARCH must not crash on data that produces degenerate correlation
    matrices (e.g., pairwise-constant residuals after masking)."""

    def _make_loader(self, series, window_len=64):
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        T, F = series.shape
        n_windows = T - window_len + 1
        windows = np.empty((n_windows, window_len, F), dtype=np.float32)
        for i in range(n_windows):
            windows[i] = series[i:i + window_len]
        ds = TensorDataset(torch.from_numpy(windows))
        return DataLoader(ds, batch_size=64, shuffle=False)

    def test_fit_survives_when_one_ticker_is_flat(self):
        """One ticker has near-zero variance — its standardized residuals will
        be degenerate but fit should complete without raising."""
        from tsgen.models.garch import CCCGARCH

        rng = np.random.default_rng(42)
        T, F = 2000, 3
        series = rng.normal(0, 0.02, size=(T, F)).astype(np.float32)
        series[:, 1] = np.linspace(0, 1e-8, T).astype(np.float32)

        model = CCCGARCH(features=F, distribution='t')
        model.fit(self._make_loader(series))

        R = model.R.numpy()
        assert R.shape == (F, F)
        assert np.isfinite(R).all()
        np.testing.assert_allclose(np.diag(R), 1.0, atol=1e-6)

    def test_fit_survives_when_ticker_starts_with_zeros(self):
        """A ticker that joined mid-series has its pre-IPO positions zero-filled
        by the pipeline's mask strategy. Feeding zeros to GARCH can produce
        NaN/Inf downstream; the fit must tolerate this."""
        from tsgen.models.garch import CCCGARCH

        rng = np.random.default_rng(7)
        T, F = 2000, 4
        series = rng.normal(0, 0.02, size=(T, F)).astype(np.float32)
        # Ticker 2: first 1000 rows all zero (simulates pre-IPO mask-fill)
        series[:1000, 2] = 0.0

        model = CCCGARCH(features=F, distribution='t')
        model.fit(self._make_loader(series))  # must not raise

        R = model.R.numpy()
        assert np.isfinite(R).all()
        np.testing.assert_allclose(np.diag(R), 1.0, atol=1e-6)

    def test_regularized_cholesky_tolerates_nan_matrix(self):
        """Reproduces the real smoke-test crash: R contained NaN, causing the
        diag-fallback to fail because clip(NaN, lo, hi) = NaN."""
        from tsgen.models.garch import CCCGARCH

        F = 4
        R = np.eye(F)
        R[1, 2] = np.nan
        R[2, 1] = np.nan
        R[3, 3] = np.nan
        L = CCCGARCH._regularized_cholesky(R)
        assert L.shape == (F, F)
        assert np.isfinite(L).all()

    def test_regularized_cholesky_tolerates_all_nan(self):
        from tsgen.models.garch import CCCGARCH
        F = 3
        R = np.full((F, F), np.nan)
        L = CCCGARCH._regularized_cholesky(R)
        assert np.isfinite(L).all()

    def test_fit_survives_when_all_tickers_have_partial_zeros(self):
        """Extreme case mimicking 100-ticker universe with many pre-IPO gaps."""
        from tsgen.models.garch import CCCGARCH

        rng = np.random.default_rng(11)
        T, F = 3000, 10
        series = rng.normal(0, 0.02, size=(T, F)).astype(np.float32)
        # Stagger in pre-IPO zero periods across tickers
        for i in range(F):
            zeros_until = rng.integers(0, 1500)
            series[:zeros_until, i] = 0.0

        model = CCCGARCH(features=F, distribution='t')
        model.fit(self._make_loader(series))

        R = model.R.numpy()
        assert np.isfinite(R).all()
        np.testing.assert_allclose(R, R.T, atol=1e-6)
        np.testing.assert_allclose(np.diag(R), 1.0, atol=1e-6)
        L = model.L.numpy()
        assert np.isfinite(L).all()
