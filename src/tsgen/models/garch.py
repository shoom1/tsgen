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
        from windows before fitting. If the dataloader emits ``(data, mask)``
        tuples (the ``clean_data(strategy='mask')`` path), invalid positions
        are excluded per-ticker so pre-IPO zero-filled rows don't poison
        the GARCH fit.
        """
        series, mask = self._reconstruct_series(dataloader)  # (T, F), (T, F) or None

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

        mu = np.zeros(F)
        omega = np.zeros(F)
        alpha = np.zeros(F)
        beta = np.zeros(F)
        nu = np.full(F, 30.0)  # sentinel for normal
        sigma2_last = np.zeros(F)
        std_resid = np.zeros_like(series)  # (T, F), only filled at valid positions

        valid_mask = np.ones_like(series, dtype=bool) if mask is None else mask.astype(bool)

        for i in range(F):
            col_mask = valid_mask[:, i]
            col = series[:, i]
            # Per-ticker slice: keep only the contiguous valid range. arch fits
            # a single time series; gaps in the middle are uncommon in practice
            # (IPOs stagger at the start, not in the middle) so this is fine.
            if col_mask.any():
                first = int(np.argmax(col_mask))
                last_rev = int(np.argmax(col_mask[::-1]))
                last = len(col_mask) - last_rev
                valid_slice = slice(first, last)
            else:
                valid_slice = slice(0, 0)

            col_valid = col[valid_slice]
            # Also drop any residual NaN/Inf (shouldn't be present after masking,
            # but arch will error on them).
            finite = np.isfinite(col_valid)
            col_valid = col_valid[finite]

            if col_valid.size < 100:
                # Not enough data for a reliable GARCH fit. Fall back to a
                # neutral "low-vol near-Gaussian" ticker so it contributes
                # minimally to the portfolio without breaking evaluation.
                mu[i] = 0.0
                omega[i] = 1e-6
                alpha[i] = 0.05
                beta[i] = 0.9
                nu[i] = 30.0
                sigma_series_i = np.full(col.shape, np.sqrt(1e-6 / max(1e-9, 1 - 0.95)))
                sigma2_last[i] = sigma_series_i[-1] ** 2
                continue

            params_i, sigma_valid = self._fit_one_series(col_valid)

            # Defensive: reject pathologically bad fits (corner solutions).
            # alpha at its bound (~1) and beta~0 is arch's failure mode;
            # fall back to neutral defaults rather than trust garbage params.
            a_raw = params_i['alpha']
            b_raw = params_i['beta']
            mu_raw = params_i['mu']
            omega_raw = params_i['omega']
            bad = (
                a_raw >= 0.999
                or (a_raw + b_raw) >= 0.9999
                or omega_raw <= 0
                or abs(mu_raw) > 0.05  # daily log-return mean > 5% is nonsense
                or not np.isfinite(sigma_valid).all()
            )
            if bad:
                mu[i] = 0.0
                omega[i] = max(float(col_valid.var()) * 0.05, 1e-6)
                alpha[i] = 0.05
                beta[i] = 0.9
                nu[i] = 30.0 if self.distribution == 'normal' else float(params_i.get('nu', 30.0))
                uncond_sigma2 = omega[i] / max(1 - alpha[i] - beta[i], 1e-6)
                sigma2_last[i] = uncond_sigma2
                # std_resid column stays 0 — this ticker won't contribute to R
                continue

            mu[i] = mu_raw
            omega[i] = omega_raw
            alpha[i] = a_raw
            beta[i] = b_raw
            if self.distribution == 't':
                nu[i] = params_i['nu']

            # Place sigma_valid (length = valid_slice) back into the full series;
            # invalid positions remain at 0 in std_resid.
            full_sigma = np.full(col.shape, np.nan)
            # Within the valid_slice, only the 'finite' subset was actually fit.
            # Map sigma_valid back onto those positions.
            valid_indices_in_slice = np.where(finite)[0]
            slice_start = valid_slice.start
            full_sigma[slice_start + valid_indices_in_slice] = sigma_valid

            # std_resid only at valid positions
            resid = (col - mu[i]) / np.maximum(full_sigma, 1e-12)
            std_resid[:, i] = np.where(np.isfinite(resid), resid, 0.0)
            # sigma2_last uses the last finite sigma for that ticker
            last_sigma = sigma_valid[-1] if sigma_valid.size else np.sqrt(1e-6)
            sigma2_last[i] = max(float(last_sigma) ** 2, 1e-10)

        # Safety clamp — every ticker should now have positive sigma2_last;
        # forbid zeros defensively.
        sigma2_last = np.clip(sigma2_last, 1e-10, None)

        # Correlations: use pairwise-complete observations (only positions where
        # both tickers are valid). corrcoef on our zero-filled std_resid would
        # bias estimates toward zero for tickers with long zero-fill prefixes.
        std_resid = np.nan_to_num(std_resid, nan=0.0, posinf=0.0, neginf=0.0)
        if F == 1:
            R = np.array([[1.0]])
        else:
            R = self._pairwise_correlation(std_resid, valid_mask)
            R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)
            R = 0.5 * (R + R.T)
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
    def _reconstruct_series(dataloader):
        """Reconstruct a (T, F) ordered series + optional (T, F) mask from a
        stride-1 windowed dataloader.

        Assumes chronological order (shuffle=False) and stride 1; the series
        is ``windows[:, 0, :]`` concatenated with the tail of the last window.
        If the dataloader yields (data, mask) tuples, the mask is reconstructed
        the same way; otherwise the mask return value is ``None``.
        """
        all_data = []
        all_masks = []
        has_masks = False
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                all_data.append(batch[0].detach().cpu().numpy())
                if len(batch) > 1:
                    all_masks.append(batch[1].detach().cpu().numpy())
                    has_masks = True
            else:
                all_data.append(batch.detach().cpu().numpy())

        def _flatten(chunks):
            arr = np.concatenate(chunks, axis=0)
            if arr.ndim != 3:
                raise ValueError(
                    f"Expected (N, L, F) windows from dataloader, got {arr.shape}."
                )
            first_col = arr[:, 0, :]
            tail = arr[-1, 1:, :]
            return np.concatenate([first_col, tail], axis=0)

        series = _flatten(all_data).astype(np.float64)
        mask = _flatten(all_masks).astype(np.float64) if has_masks else None
        return series, mask

    @staticmethod
    def _pairwise_correlation(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Sample correlation using pairwise-complete observations.

        For each (i, j), compute the correlation over rows where both
        ``mask[:, i]`` and ``mask[:, j]`` are True. Using ``np.corrcoef`` on
        zero-filled data would bias estimates for tickers with long invalid
        prefixes toward zero; this routine gives the correct MLE for the
        constant-correlation assumption under staggered IPO dates.
        """
        T, F = x.shape
        valid = mask.astype(bool)
        R = np.eye(F)
        for i in range(F):
            for j in range(i + 1, F):
                pair = valid[:, i] & valid[:, j]
                n = int(pair.sum())
                if n < 30:
                    R[i, j] = R[j, i] = 0.0
                    continue
                xi = x[pair, i]
                xj = x[pair, j]
                si = xi.std(ddof=1)
                sj = xj.std(ddof=1)
                if si < 1e-12 or sj < 1e-12:
                    R[i, j] = R[j, i] = 0.0
                    continue
                c = np.cov(xi, xj, ddof=1)[0, 1] / (si * sj)
                # Clip to [-1, 1] — numeric drift can push slightly outside
                c = float(np.clip(c, -1.0, 1.0))
                R[i, j] = R[j, i] = c
        return R

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
