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

    # --- Quality tests (not just "doesn't crash") ---
    #
    # When fed real-world-like data with pre-IPO zero runs, the fit should
    # recover plausible parameters: mu small (it's a daily log-return mean),
    # sigma2_last > 0 (otherwise generated samples are constant at mu), and
    # generated samples should have non-zero variance per feature.

    def test_fit_recovers_plausible_mu_with_pre_ipo_zeros(self):
        """Pre-IPO zero-fill should NOT inflate mu into implausible values.

        The real 100-ticker run had mu in [-2.1, 2.5] (log-return units),
        which is physically impossible (daily log-return mean of 2 = 7x
        per day). Caused by leading zeros dominating the mean estimate.
        """
        from tsgen.models.garch import CCCGARCH

        rng = np.random.default_rng(13)
        T, F = 3000, 5
        real_mu_per_day = 0.0005
        series = rng.normal(real_mu_per_day, 0.02, size=(T, F)).astype(np.float32)
        # Ticker 1 and 3 have long pre-IPO zero runs (common in real data)
        series[:1500, 1] = 0.0
        series[:2000, 3] = 0.0

        model = CCCGARCH(features=F, distribution='t')
        model.fit(self._make_loader(series))

        mu = model.mu.numpy()
        # Daily log-return means should be tiny. Anything above 0.01 is wrong.
        assert np.abs(mu).max() < 0.01, (
            f'mu out of plausible range: {mu}; pre-IPO zeros are skewing fits.'
        )

    def test_sigma2_last_never_exactly_zero(self):
        """sigma2_last=0 makes every generated step equal mu (constant sequence)."""
        from tsgen.models.garch import CCCGARCH

        rng = np.random.default_rng(14)
        T, F = 3000, 5
        series = rng.normal(0, 0.02, size=(T, F)).astype(np.float32)
        series[:2500, 2] = 0.0  # ticker 2 only has 500 observations

        model = CCCGARCH(features=F, distribution='t')
        model.fit(self._make_loader(series))

        assert (model.sigma2_last > 0).all(), (
            f'sigma2_last has zeros: {model.sigma2_last}; '
            'generated samples would be constant.'
        )

    def test_generated_samples_have_nonzero_variance_per_feature(self):
        """Canary for degenerate fits: every feature column in generated
        samples must have strictly positive variance."""
        from tsgen.models.garch import CCCGARCH

        rng = np.random.default_rng(15)
        T, F = 3000, 5
        series = rng.normal(0, 0.02, size=(T, F)).astype(np.float32)
        series[:1500, 1] = 0.0
        series[:2000, 3] = 0.0

        model = CCCGARCH(features=F, distribution='t')
        model.fit(self._make_loader(series))

        samples = model.generate(n_samples=20, seq_len=100).numpy()
        # (20, 100, 5) - compute std per feature across all sample/time entries
        flat = samples.reshape(-1, F)
        stds = flat.std(axis=0)
        assert (stds > 1e-6).all(), (
            f'Feature std degenerate: {stds}; fit collapsed for some ticker.'
        )
        assert np.isfinite(samples).all(), 'Generated NaN/Inf samples.'


class TestBootstrapQuality:
    """Bootstrap must skip pre-IPO zero-fill rows when reconstructing history."""

    def _make_loader(self, series, window_len=64):
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        T, F = series.shape
        n = T - window_len + 1
        w = np.empty((n, window_len, F), dtype=np.float32)
        for i in range(n):
            w[i] = series[i:i + window_len]
        ds = TensorDataset(torch.from_numpy(w))
        return DataLoader(ds, batch_size=64, shuffle=False)

    def test_bootstrap_skips_pre_ipo_zero_rows(self):
        """If ticker B has zeros for rows [0:1500], bootstrap should NOT
        include those rows in the resampling pool — otherwise generated
        samples will include constant-zero stretches."""
        from tsgen.models.baselines import BootstrapGenerativeModel

        rng = np.random.default_rng(16)
        T, F = 3000, 3
        series = rng.normal(0, 0.02, size=(T, F)).astype(np.float32)
        series[:1500, 1] = 0.0
        series[:2000, 2] = 0.0

        model = BootstrapGenerativeModel(features=F, sequence_length=64, block_p=0.2)
        model.fit(self._make_loader(series))

        # Every feature column in the history must have non-trivial variance.
        h = model.history.numpy()
        stds = h.std(axis=0)
        assert (stds > 0.005).all(), (
            f'Bootstrap history still has near-constant columns: {stds}. '
            'Pre-IPO zero rows were not filtered.'
        )
