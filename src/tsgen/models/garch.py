"""CCC-GARCH(1,1) multivariate baseline with Normal or Student-t innovations."""

from __future__ import annotations

import numpy as np
import torch
import torch.distributions as D

from tsgen.models.base_model import StatisticalModel
from tsgen.models.registry import ModelRegistry


# Internal fitting scale. arch's GARCH optimizer is unstable on very small
# magnitudes; financial log-returns (~1e-2) are near the edge. We scale to
# "percent" units for fitting and unscale params back.
_FIT_SCALE = 100.0

_VALID_DISTRIBUTIONS = ('normal', 't')


@ModelRegistry.register('ccc_garch')
class CCCGARCH(StatisticalModel):
    """Constant Conditional Correlation GARCH(1,1).

    Per ticker: r_{i,t} = mu_i + sigma_{i,t} z_{i,t},
                sigma^2_{i,t} = omega_i + alpha_i * eps^2_{i,t-1} + beta_i * sigma^2_{i,t-1}.
    Cross-asset dependence: constant correlation matrix R estimated from
    standardized residuals; conditional covariance Sigma_t = D_t R D_t.

    Innovations are sampled iid in the uncorrelated frame (Student-t or Normal),
    standardized to unit variance, then correlated through L = chol(R).
    """

    def __init__(
        self,
        features: int,
        p: int = 1,
        q: int = 1,
        distribution: str = 't',
    ):
        super().__init__()
        if distribution not in _VALID_DISTRIBUTIONS:
            raise ValueError(
                f"Unknown distribution: {distribution!r}. "
                f"Valid options: {_VALID_DISTRIBUTIONS}"
            )
        if p != 1 or q != 1:
            raise NotImplementedError(
                f"Only GARCH(1,1) is supported for v1 (got p={p}, q={q})."
            )
        self.features = features
        self.p = p
        self.q = q
        self.distribution = distribution

        # Dummy param so optimizer-wrapping code does not choke on a module
        # with no parameters (matches the existing baseline pattern).
        self.dummy = torch.nn.Parameter(torch.zeros(1))

        # GARCH parameters per ticker
        self.register_buffer('mu', torch.zeros(features))
        self.register_buffer('omega', torch.full((features,), 1e-6))
        self.register_buffer('alpha', torch.full((features,), 0.05))
        self.register_buffer('beta', torch.full((features,), 0.9))
        # nu is only meaningful for Student-t; keep a large-ish sentinel for 'normal'
        self.register_buffer('nu', torch.full((features,), 30.0))

        # Correlation structure
        self.register_buffer('R', torch.eye(features))
        self.register_buffer('L', torch.eye(features))

        # Last conditional variance observed in training, used to warm-start sampling
        self.register_buffer('sigma2_last', torch.full((features,), 1e-4))

    # ---------- factory ----------

    @classmethod
    def from_config(cls, config, features: int | None = None):
        params = config.get_model_config()
        p = getattr(params, 'p', 1)
        q = getattr(params, 'q', 1)
        distribution = getattr(params, 'distribution', 't')
        return cls(features=features, p=p, q=q, distribution=distribution)

    # ---------- fitting ----------

    def fit(self, dataloader) -> None:
        """Fit CCC-GARCH from a dataloader yielding windows of raw log-returns.

        Requires ``shuffle=False`` and stride-1 windows, which the training path
        is expected to configure. The original ordered series is reconstructed
        from windows before fitting.
        """
        series = self._reconstruct_series(dataloader)  # (T, F) np.float64

        T, F = series.shape
        if F != self.features:
            raise ValueError(
                f"Feature dimension mismatch: model has {self.features}, data has {F}."
            )
        if T < 200:
            raise ValueError(
                f"CCC-GARCH needs at least ~200 observations (got {T}); "
                f"the optimizer will not converge reliably."
            )
        if not np.isfinite(series).all():
            raise ValueError("Training series contains NaN or inf.")

        mu = np.zeros(F)
        omega = np.zeros(F)
        alpha = np.zeros(F)
        beta = np.zeros(F)
        nu = np.full(F, 30.0)  # sentinel for normal
        sigma2_last = np.zeros(F)
        std_resid = np.zeros_like(series)  # (T, F)

        for i in range(F):
            params_i, sigma_series_i = self._fit_one_series(series[:, i])
            mu[i] = params_i['mu']
            omega[i] = params_i['omega']
            alpha[i] = params_i['alpha']
            beta[i] = params_i['beta']
            if self.distribution == 't':
                nu[i] = params_i['nu']

            # Standardized residuals across the full training window
            # (conditional variance is causal at each timestep by GARCH construction)
            std_resid[:, i] = (series[:, i] - mu[i]) / np.maximum(sigma_series_i, 1e-12)
            sigma2_last[i] = sigma_series_i[-1] ** 2

        # Standardized residuals can contain NaN/Inf if a ticker had zero-
        # filled pre-IPO positions, tiny sigma, or a failed GARCH fit.
        # Replace with zeros before computing correlations — the corresponding
        # row/column of R will be cleaned up below.
        std_resid = np.nan_to_num(std_resid, nan=0.0, posinf=0.0, neginf=0.0)

        # Constant correlation: full-sample sample correlation of standardized residuals.
        # corrcoef emits NaN on constant (zero-variance) columns; sanitize.
        if F == 1:
            R = np.array([[1.0]])
        else:
            R = np.corrcoef(std_resid, rowvar=False)
            R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)
            R = 0.5 * (R + R.T)             # symmetrize away floating-point noise
            np.fill_diagonal(R, 1.0)

        # Regularize to ensure positive-definiteness and compute Cholesky factor
        L = self._regularized_cholesky(R)

        self.mu = torch.as_tensor(mu, dtype=self.mu.dtype)
        self.omega = torch.as_tensor(omega, dtype=self.omega.dtype)
        self.alpha = torch.as_tensor(alpha, dtype=self.alpha.dtype)
        self.beta = torch.as_tensor(beta, dtype=self.beta.dtype)
        self.nu = torch.as_tensor(nu, dtype=self.nu.dtype)
        self.R = torch.as_tensor(R, dtype=self.R.dtype)
        self.L = torch.as_tensor(L, dtype=self.L.dtype)
        self.sigma2_last = torch.as_tensor(sigma2_last, dtype=self.sigma2_last.dtype)

        print(
            f"CCC-GARCH fitted (dist={self.distribution}, F={F}): "
            f"alpha~{alpha.mean():.3f}, beta~{beta.mean():.3f}, "
            f"nu~{nu.mean():.2f}, R_mean_offdiag={(R.sum() - F) / max(F * (F - 1), 1):.3f}"
        )

    def _fit_one_series(self, r: np.ndarray):
        """Fit a single-ticker GARCH(1,1). Returns (params dict, sigma series in raw units)."""
        from arch import arch_model

        r_scaled = r * _FIT_SCALE
        dist_arg = 't' if self.distribution == 't' else 'normal'
        model = arch_model(
            r_scaled,
            mean='Constant',
            vol='GARCH',
            p=self.p,
            q=self.q,
            dist=dist_arg,
            rescale=False,
        )
        res = model.fit(disp='off', show_warning=False)
        p = res.params.to_dict()

        # Unscale: returns-scale vs percent-scale
        # mu scales with s; omega with s^2; alpha, beta, nu unchanged.
        out = {
            'mu': p['mu'] / _FIT_SCALE,
            'omega': p['omega'] / (_FIT_SCALE ** 2),
            'alpha': p.get('alpha[1]', 0.0),
            'beta': p.get('beta[1]', 0.0),
        }
        if self.distribution == 't':
            out['nu'] = p.get('nu', 30.0)

        sigma_series = np.asarray(res.conditional_volatility) / _FIT_SCALE
        return out, sigma_series

    @staticmethod
    def _reconstruct_series(dataloader) -> np.ndarray:
        """Reconstruct a (T, F) ordered series from a stride-1 windowed dataloader.

        The windows are assumed to be chronologically ordered (shuffle=False) and
        to have stride 1, so take column 0 of every window plus the tail of the
        final window.
        """
        all_windows = []
        for batch in dataloader:
            windows = batch[0] if isinstance(batch, (list, tuple)) else batch
            all_windows.append(windows.detach().cpu().numpy())
        arr = np.concatenate(all_windows, axis=0)  # (N, L, F)
        if arr.ndim != 3:
            raise ValueError(
                f"Expected (N, L, F) windows from dataloader, got shape {arr.shape}."
            )
        first_col = arr[:, 0, :]          # (N, F)
        tail = arr[-1, 1:, :]             # (L-1, F)
        return np.concatenate([first_col, tail], axis=0).astype(np.float64)

    @staticmethod
    def _regularized_cholesky(R: np.ndarray) -> np.ndarray:
        """Cholesky of R with incremental diagonal regularization on failure.

        Robust to NaN/Inf entries and severely degenerate matrices: the final
        fallback always succeeds, returning ``I`` (uncorrelated) when every
        other attempt fails.
        """
        F = R.shape[0]
        # Clean NaN/Inf so arithmetic below doesn't propagate them.
        R_clean = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)

        epsilon = 0.0
        for _ in range(6):
            try:
                return np.linalg.cholesky(R_clean + epsilon * np.eye(F))
            except np.linalg.LinAlgError:
                epsilon = max(epsilon * 10, 1e-8)

        # Diagonal fallback — use sanitized diagonal, clip to floor so Cholesky
        # always exists.
        diag = np.nan_to_num(np.diag(R_clean), nan=1.0, posinf=1.0, neginf=1.0)
        diag = np.clip(diag, 1e-6, None)
        try:
            return np.linalg.cholesky(np.diag(diag))
        except np.linalg.LinAlgError:
            # Absolute last resort: identity. Samples will be uncorrelated.
            return np.eye(F)

    # ---------- generation ----------

    def generate(
        self,
        n_samples: int,
        seq_len: int,
        device: str = 'cpu',
        **kwargs,
    ) -> torch.Tensor:
        """Simulate n_samples trajectories of length seq_len under the fitted model."""
        F = self.features
        dtype = self.mu.dtype
        dev = torch.device(device)

        mu = self.mu.to(dev)
        omega = self.omega.to(dev)
        alpha = self.alpha.to(dev)
        beta = self.beta.to(dev)
        L = self.L.to(dev)

        # Warm-start: replicate the last observed sigma^2 across samples
        sigma2 = self.sigma2_last.to(dev).unsqueeze(0).expand(n_samples, -1).clone()

        if self.distribution == 't':
            nu = self.nu.to(dev)
            # Validate: Student-t requires nu > 2 for finite variance standardization.
            # arch should already enforce this, but guard explicitly.
            if torch.any(nu <= 2.0):
                raise ValueError(f"Student-t nu must be > 2 for finite variance; got {nu}")
            # Per-ticker Student-t distribution; sample iid then standardize so Var=1
            t_dist = D.StudentT(df=nu)
            scale_std = torch.sqrt((nu - 2.0) / nu)
        else:
            t_dist = None
            scale_std = None

        out = torch.empty((n_samples, seq_len, F), device=dev, dtype=dtype)

        for tstep in range(seq_len):
            if self.distribution == 't':
                # Sample (n_samples, F) iid standardized-t per ticker
                z_indep = t_dist.sample((n_samples,)) * scale_std  # (n_samples, F)
            else:
                z_indep = torch.randn((n_samples, F), device=dev, dtype=dtype)

            # Correlate across features: rows are iid, Cov(row) = L L^T = R
            z_corr = z_indep @ L.T  # (n_samples, F)

            sigma = torch.sqrt(sigma2)
            r_t = mu.unsqueeze(0) + sigma * z_corr
            out[:, tstep, :] = r_t

            eps_sq = (r_t - mu.unsqueeze(0)) ** 2
            sigma2 = (
                omega.unsqueeze(0)
                + alpha.unsqueeze(0) * eps_sq
                + beta.unsqueeze(0) * sigma2
            )

        return out
