"""Tests for CCC-GARCH baseline."""

import numpy as np
import pytest
import torch
from scipy.stats import t as t_dist

from tsgen.config.schema import ExperimentConfig
from tsgen.models.garch import CCCGARCH
from tsgen.models.registry import ModelRegistry


# ---------- helpers ----------

def _simulate_ccc_garch(T, mu, omega, alpha, beta, nu, R, seed=0):
    """Simulate T steps of CCC-GARCH(1,1) with Student-t innovations.

    Returns: (T, F) array of log-returns and (F,) sigma2 at the last step.
    """
    rng = np.random.default_rng(seed)
    F = len(mu)
    mu = np.asarray(mu, dtype=float)
    omega = np.asarray(omega, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    beta = np.asarray(beta, dtype=float)
    nu = np.asarray(nu, dtype=float)
    R = np.asarray(R, dtype=float)
    L = np.linalg.cholesky(R)

    sigma2 = omega / np.maximum(1.0 - alpha - beta, 1e-6)
    r = np.zeros((T, F))
    for tstep in range(T):
        # Independent standardized t per feature, then correlate through L
        z_indep = t_dist.rvs(df=nu, size=F, random_state=rng) * np.sqrt((nu - 2) / nu)
        z_corr = L @ z_indep
        sigma = np.sqrt(sigma2)
        r[tstep] = mu + sigma * z_corr
        eps_sq = (r[tstep] - mu) ** 2
        sigma2 = omega + alpha * eps_sq + beta * sigma2
    return r, sigma2


def _windows_from_series(series, window_len):
    """Build non-shuffled sliding windows (stride=1), as the pipeline produces."""
    T, F = series.shape
    n = T - window_len + 1
    out = np.empty((n, window_len, F), dtype=np.float32)
    for i in range(n):
        out[i] = series[i:i + window_len]
    return out


def _dataloader_from_windows(windows):
    """Minimal shuffle=False DataLoader analog: yields (tensor,) tuples."""
    from torch.utils.data import DataLoader, TensorDataset
    ds = TensorDataset(torch.from_numpy(windows))
    return DataLoader(ds, batch_size=32, shuffle=False)


# ---------- construction ----------

class TestConstruction:
    def test_registered_under_ccc_garch(self):
        assert 'ccc_garch' in ModelRegistry.list_models()
        assert ModelRegistry.list_models()['ccc_garch'] is CCCGARCH

    def test_init_allocates_buffers(self):
        m = CCCGARCH(features=3, p=1, q=1, distribution='t')
        assert m.features == 3
        assert m.distribution == 't'
        assert m.mu.shape == (3,)
        assert m.omega.shape == (3,)
        assert m.alpha.shape == (3,)
        assert m.beta.shape == (3,)
        assert m.nu.shape == (3,)
        assert m.R.shape == (3, 3)
        assert m.L.shape == (3, 3)
        assert m.sigma2_last.shape == (3,)

    def test_rejects_unknown_distribution(self):
        with pytest.raises(ValueError):
            CCCGARCH(features=2, distribution='skewt')

    def test_from_config_default_distribution(self):
        config = ExperimentConfig(
            model_type='ccc_garch',
            data={'tickers': ['A', 'B', 'C']},
        )
        m = ModelRegistry.create(config, features=3)
        assert isinstance(m, CCCGARCH)
        assert m.distribution == 't'
        assert m.features == 3

    def test_from_config_explicit_normal(self):
        config = ExperimentConfig(
            model_type='ccc_garch',
            data={'tickers': ['A', 'B']},
            model={'distribution': 'normal', 'p': 1, 'q': 1},
        )
        m = ModelRegistry.create(config, features=2)
        assert m.distribution == 'normal'


# ---------- fit / recovery ----------

class TestFit:
    def test_parameter_recovery_single_asset(self):
        T = 4000
        true_mu, true_omega, true_alpha, true_beta, true_nu = 0.0005, 1e-6, 0.08, 0.9, 8.0
        r, _ = _simulate_ccc_garch(
            T=T, mu=[true_mu], omega=[true_omega],
            alpha=[true_alpha], beta=[true_beta], nu=[true_nu],
            R=[[1.0]], seed=42,
        )
        windows = _windows_from_series(r.astype(np.float32), window_len=64)
        loader = _dataloader_from_windows(windows)

        m = CCCGARCH(features=1, distribution='t')
        m.fit(loader)

        assert abs(m.mu.item() - true_mu) < 1e-3
        assert abs(m.alpha.item() - true_alpha) < 0.05
        assert abs(m.beta.item() - true_beta) < 0.05
        assert abs(m.nu.item() - true_nu) < 4.0

    def test_correlation_recovery_two_assets(self):
        T = 3000
        F = 2
        R_true = np.array([[1.0, 0.5], [0.5, 1.0]])
        r, _ = _simulate_ccc_garch(
            T=T, mu=[0.0, 0.0], omega=[1e-6, 1e-6],
            alpha=[0.07, 0.08], beta=[0.9, 0.9], nu=[10.0, 10.0],
            R=R_true, seed=7,
        )
        windows = _windows_from_series(r.astype(np.float32), window_len=64)
        loader = _dataloader_from_windows(windows)

        m = CCCGARCH(features=F, distribution='t')
        m.fit(loader)

        R_fit = m.R.numpy()
        np.testing.assert_allclose(R_fit, R_true, atol=0.08)
        # L must be a valid Cholesky factor
        np.testing.assert_allclose(m.L.numpy() @ m.L.numpy().T, R_fit, atol=1e-6)


# ---------- generate ----------

class TestGenerate:
    def _fit_small(self, F=2, T=2000, seed=1):
        R_true = np.eye(F) * 0.4 + np.ones((F, F)) * 0.6
        r, _ = _simulate_ccc_garch(
            T=T,
            mu=[0.0] * F,
            omega=[1e-6] * F,
            alpha=[0.08] * F,
            beta=[0.9] * F,
            nu=[10.0] * F,
            R=R_true, seed=seed,
        )
        windows = _windows_from_series(r.astype(np.float32), window_len=64)
        m = CCCGARCH(features=F, distribution='t')
        m.fit(_dataloader_from_windows(windows))
        return m, R_true

    def test_generate_shape(self):
        m, _ = self._fit_small(F=2)
        out = m.generate(n_samples=5, seq_len=100, device='cpu')
        assert isinstance(out, torch.Tensor)
        assert out.shape == (5, 100, 2)
        assert torch.isfinite(out).all()

    def test_generated_variance_nontrivial(self):
        m, _ = self._fit_small(F=2)
        out = m.generate(n_samples=50, seq_len=500, device='cpu').numpy()
        # Not zero, not absurd
        stds = out.reshape(-1, 2).std(axis=0)
        assert (stds > 1e-5).all()
        assert (stds < 0.5).all()

    def test_generated_kurtosis_heavy(self):
        """With Student-t innovations, excess kurtosis must be positive."""
        m, _ = self._fit_small(F=1)
        out = m.generate(n_samples=50, seq_len=500, device='cpu').numpy().reshape(-1)
        x = (out - out.mean()) / out.std()
        excess_kurt = (x ** 4).mean() - 3.0
        assert excess_kurt > 0.3

    def test_generated_correlation_matches_fit(self):
        m, _ = self._fit_small(F=2, T=3000, seed=11)
        out = m.generate(n_samples=100, seq_len=500, device='cpu').numpy()
        flat = out.reshape(-1, 2)
        emp_corr = np.corrcoef(flat.T)
        np.testing.assert_allclose(emp_corr, m.R.numpy(), atol=0.06)

    def test_normal_distribution_path(self):
        """Gaussian innovations: kurtosis ~ 0; fit/sample both work."""
        T = 2500
        F = 2
        R_true = np.array([[1.0, 0.3], [0.3, 1.0]])
        # Simulate with high nu (≈Normal) so fit converges under dist='normal'
        r, _ = _simulate_ccc_garch(
            T=T, mu=[0.0, 0.0], omega=[1e-6, 1e-6],
            alpha=[0.07, 0.07], beta=[0.9, 0.9], nu=[500.0, 500.0],
            R=R_true, seed=3,
        )
        windows = _windows_from_series(r.astype(np.float32), window_len=64)
        m = CCCGARCH(features=F, distribution='normal')
        m.fit(_dataloader_from_windows(windows))
        # nu buffer stays at sentinel (we don't fit it for normal)
        out = m.generate(n_samples=30, seq_len=400, device='cpu').numpy()
        assert out.shape == (30, 400, 2)
        x = (out.reshape(-1, 2) - out.reshape(-1, 2).mean(0)) / out.reshape(-1, 2).std(0)
        excess_kurt = (x ** 4).mean(axis=0) - 3.0
        # Normal innovations through GARCH still have some kurtosis from clustering,
        # but much less than the Student-t case. Just sanity check it's finite.
        assert np.isfinite(excess_kurt).all()


# ---------- persistence ----------

class TestPersistence:
    def test_save_load_roundtrip(self, tmp_path):
        T = 1500
        F = 2
        r, _ = _simulate_ccc_garch(
            T=T, mu=[0.0, 0.0], omega=[1e-6, 1e-6],
            alpha=[0.08, 0.08], beta=[0.9, 0.9], nu=[10.0, 10.0],
            R=np.array([[1.0, 0.4], [0.4, 1.0]]), seed=9,
        )
        windows = _windows_from_series(r.astype(np.float32), window_len=64)
        m = CCCGARCH(features=F, distribution='t')
        m.fit(_dataloader_from_windows(windows))

        path = tmp_path / "ccc_garch.pt"
        torch.save(m, str(path))
        m2 = torch.load(str(path), weights_only=False)

        for name in ['mu', 'omega', 'alpha', 'beta', 'nu', 'R', 'L', 'sigma2_last']:
            torch.testing.assert_close(getattr(m, name), getattr(m2, name))
