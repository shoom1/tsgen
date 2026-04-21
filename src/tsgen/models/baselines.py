import torch
import numpy as np
from tsgen.models.base_model import StatisticalModel
from tsgen.models.registry import ModelRegistry


@ModelRegistry.register('multivariate_gaussian')
class MultivariateGaussian(StatisticalModel):
    """Multivariate Gaussian baseline on scaled log-returns.

    Fits a Gaussian distribution to windowed log-returns (iid across time,
    joint across assets). Captures first and second moments; misses every
    stylized fact except the mean (no volatility clustering, no fat tails).
    Useful as a floor benchmark for multi-asset risk scenarios.

    Named "Gaussian" rather than "GBM" because the model is a static
    multivariate normal fit on z-scored returns — there is no drift term,
    no compounding, nothing that makes it a geometric Brownian motion in
    the continuous-time sense.

    Modes (via ``full_covariance``):
      - True  (default): Fit mean vector + full covariance, sample with
                         Cholesky factor. Captures cross-asset correlation.
      - False          : Fit per-feature mean + std, sample independently.
                         No cross-asset correlation; use only as an ablation.
    """

    @classmethod
    def from_config(cls, config, features=None):
        """Create MultivariateGaussian from ExperimentConfig."""
        params = config.get_model_config()
        full_covariance = getattr(params, 'full_covariance', True)
        return cls(features=features, full_covariance=full_covariance)

    def __init__(self, features, full_covariance=True):
        super().__init__()
        self.features = features
        self.full_covariance = full_covariance

        # Dummy parameter to avoid optimizer errors when used with trainer
        self.dummy = torch.nn.Parameter(torch.zeros(1))

        if full_covariance:
            # Full covariance mode: store mean vector and Cholesky factor
            self.register_buffer('mean', torch.zeros(features))
            self.register_buffer('cholesky_L', torch.eye(features))
        else:
            # Independent mode: store per-feature mean and std
            self.register_buffer('mu', torch.zeros(features))
            self.register_buffer('sigma', torch.ones(features))


    def fit(self, data_loader):
        """
        Fits the model parameters from training data.

        For full_covariance=True:
        - Estimates mean vector and full covariance matrix
        - Performs Cholesky decomposition for efficient sampling

        For full_covariance=False:
        - Estimates per-feature mean and standard deviation

        Supports masked data: When batches contain (data, mask) tuples,
        statistics are computed only using valid (non-masked) values.

        Args:
            data_loader: Yields batches of (Batch, Seq, Features) scaled log-returns,
                        or (data, mask) tuples for masked training
        """
        all_data = []
        all_masks = []
        has_masks = False

        for batch in data_loader:
            # Handle both tuple (from TensorDataset) and raw tensor batches
            if isinstance(batch, (list, tuple)):
                data = batch[0]
                if len(batch) > 1:
                    all_masks.append(batch[1])
                    has_masks = True
            else:
                data = batch
            all_data.append(data)

        # Concatenate: (Total_Samples, Seq, Feat)
        X = torch.cat(all_data, dim=0)

        # Flatten to (N_total, Feat) for computing statistics
        X_flat = X.view(-1, self.features)

        if has_masks:
            M = torch.cat(all_masks, dim=0)  # (N, L, F)
            M_flat = M.view(-1, self.features)  # (N*L, F)

            if self.full_covariance:
                # Full covariance mode with masked data
                # Compute mean per feature using only valid values
                valid_sum = (X_flat * M_flat).sum(dim=0)
                valid_count = M_flat.sum(dim=0).clamp(min=1)
                self.mean = valid_sum / valid_count

                # Compute covariance using pairwise valid observations
                X_centered = X_flat - self.mean
                cov = torch.zeros(self.features, self.features, device=X_flat.device)

                for i in range(self.features):
                    for j in range(i, self.features):  # Only upper triangle + diagonal
                        # Both positions must be valid
                        pair_mask = M_flat[:, i] * M_flat[:, j]
                        pair_count = pair_mask.sum()

                        if pair_count > 1:
                            xi = X_centered[:, i] * pair_mask
                            xj = X_centered[:, j] * pair_mask
                            cov_ij = (xi * xj).sum() / (pair_count - 1)
                        else:
                            # Not enough data, use default
                            cov_ij = 1.0 if i == j else 0.0

                        cov[i, j] = cov_ij
                        cov[j, i] = cov_ij  # Symmetric

                # Regularize and ensure positive-definiteness
                # With masked data, pairwise covariance can be ill-conditioned
                # Use stronger regularization based on diagonal elements
                diag = torch.diagonal(cov)
                diag_mean = diag.mean().item()

                # Start with base regularization
                epsilon = max(1e-4, 0.01 * diag_mean)

                # Try Cholesky with increasing regularization if needed
                for attempt in range(5):
                    try:
                        cov_regularized = cov + epsilon * torch.eye(self.features, device=cov.device)
                        self.cholesky_L = torch.linalg.cholesky(cov_regularized)
                        break
                    except torch._C._LinAlgError:
                        epsilon *= 10
                        print(f"  Increasing regularization to epsilon={epsilon:.6f}")
                else:
                    # Fallback: use diagonal covariance only
                    print("  Warning: Using diagonal covariance (correlation structure lost)")
                    cov_regularized = torch.diag(diag.clamp(min=1e-6))
                    self.cholesky_L = torch.linalg.cholesky(cov_regularized)

                print(f"MultivariateGaussian fitted with masks (full_covariance=True):")
                print(f"  Mean: {self.mean}")
                print(f"  Valid counts per feature: {valid_count}")
                print(f"  Covariance diagonal: {torch.diagonal(cov)}")
                print(f"  Regularization used: epsilon={epsilon:.6f}")
            else:
                # Independent mode with masked data
                valid_sum = (X_flat * M_flat).sum(dim=0)
                valid_count = M_flat.sum(dim=0).clamp(min=1)
                self.mu = valid_sum / valid_count

                # Compute variance using only valid values
                centered_sq = ((X_flat - self.mu) ** 2) * M_flat
                var = centered_sq.sum(dim=0) / (valid_count - 1).clamp(min=1)
                self.sigma = torch.sqrt(var)

                print(f"MultivariateGaussian fitted with masks (full_covariance=False):")
                print(f"  mu={self.mu}")
                print(f"  sigma={self.sigma}")
                print(f"  Valid counts per feature: {valid_count}")
        else:
            # Original non-masked fitting logic
            if self.full_covariance:
                # Full covariance mode
                self.mean = torch.mean(X_flat, dim=0)

                # Compute covariance matrix
                X_centered = X_flat - self.mean
                cov = (X_centered.T @ X_centered) / (X_centered.shape[0] - 1)

                # Add regularization for numerical stability
                epsilon = 1e-6
                cov_regularized = cov + epsilon * torch.eye(self.features, device=cov.device)

                # Cholesky decomposition: cov = L @ L.T
                self.cholesky_L = torch.linalg.cholesky(cov_regularized)

                print(f"MultivariateGaussian fitted (full_covariance=True):")
                print(f"  Mean: {self.mean}")
                print(f"  Covariance diagonal: {torch.diagonal(cov)}")
                print(f"  Correlation matrix:\n{self._cov_to_corr(cov)}")
            else:
                # Independent mode
                self.mu = torch.mean(X_flat, dim=0)
                self.sigma = torch.std(X_flat, dim=0)

                print(f"MultivariateGaussian fitted (full_covariance=False):")
                print(f"  mu={self.mu}")
                print(f"  sigma={self.sigma}")

    def _cov_to_corr(self, cov):
        """Convert covariance matrix to correlation matrix."""
        std = torch.sqrt(torch.diagonal(cov))
        corr = cov / torch.outer(std, std)
        return corr

    def generate(self, n_samples, seq_len, device='cpu', **kwargs):
        """
        Generates synthetic samples.

        For full_covariance=True:
        - Uses Cholesky decomposition: x = mean + z @ L.T where z ~ N(0, I)
        - Preserves cross-asset correlations

        For full_covariance=False:
        - Independent sampling: x ~ N(mu, sigma) per feature
        - No correlations

        Args:
            n_samples: Number of sample sequences to generate
            seq_len: Length of each sequence
            device: Device to generate on
            **kwargs: Additional arguments (unused)

        Returns:
            Tensor of shape (n_samples, seq_len, features)
        """
        if self.full_covariance:
            # Sample from standard normal: (n_samples, seq_len, features)
            z = torch.randn(n_samples, seq_len, self.features,
                           device=self.cholesky_L.device)

            # Transform to correlated samples: x = mean + z @ L.T
            samples = self.mean + torch.matmul(z, self.cholesky_L.T)
        else:
            # Independent sampling per feature
            samples = torch.normal(
                mean=self.mu.expand(n_samples, seq_len, -1),
                std=self.sigma.expand(n_samples, seq_len, -1)
            )

        return samples.to(device)

@ModelRegistry.register('bootstrap')
class BootstrapGenerativeModel(StatisticalModel):
    """Stationary Block Bootstrap (Politis & Romano 1994).

    Fits by reconstructing the chronological training series from windowed
    batches; generates novel paths by concatenating random blocks of random
    (geometrically distributed) length sampled with circular wrap-around.

    This is the canonical bootstrap for stationary time series: within-block
    temporal dependence is preserved (consecutive observations come from
    consecutive training positions), while across blocks the process is
    random, producing samples that are **not** copies of training windows.

    Attributes:
        block_p (float): Probability of ending the current block at each step.
            Expected block length = 1 / block_p. Default 0.1 (avg block = 10).
    """

    @classmethod
    def from_config(cls, config, features=None):
        """Create BootstrapGenerativeModel from ExperimentConfig."""
        data = config.get_data_config()
        params = config.get_model_config()
        block_p = getattr(params, 'block_p', 0.1)
        return cls(
            features=features,
            sequence_length=data.sequence_length,
            block_p=block_p,
        )

    def __init__(self, features, sequence_length, block_p: float = 0.1):
        super().__init__()
        if not (0.0 < block_p <= 1.0):
            raise ValueError(
                f"block_p must be in (0, 1], got {block_p}. "
                "Expected block length = 1 / block_p."
            )
        self.features = features
        self.sequence_length = sequence_length
        self.block_p = block_p
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        # Flat chronological series (T, F); populated by fit()
        self.history = None

    def fit(self, data_loader):
        """Reconstruct the chronological training series from windowed batches.

        Assumes stride-1 windows in chronological order (shuffle=False). Since
        consecutive windows overlap by L-1 positions, the full series is
        windows[:, 0, :] concatenated with windows[-1, 1:, :].

        Masked data: positions where *every* feature is masked are dropped
        from the reconstructed series. Partially-masked positions are kept;
        block resampling will carry their zero-filled values, which matches
        how other models are trained on this data.
        """
        all_windows = []
        all_masks = []
        has_masks = False

        for batch in data_loader:
            if isinstance(batch, (list, tuple)):
                all_windows.append(batch[0])
                if len(batch) > 1:
                    all_masks.append(batch[1])
                    has_masks = True
            else:
                all_windows.append(batch)

        if not all_windows:
            raise ValueError("Empty dataloader.")

        windows = torch.cat(all_windows, dim=0)  # (N, L, F)
        if windows.ndim != 3:
            raise ValueError(
                f"Expected (N, L, F) windows from dataloader, got shape {windows.shape}."
            )

        first_col = windows[:, 0, :]                     # (N, F)
        tail = windows[-1, 1:, :]                        # (L-1, F)
        series = torch.cat([first_col, tail], dim=0)     # (N + L - 1, F) = (T, F)

        if has_masks:
            masks = torch.cat(all_masks, dim=0)          # (N, L, F)
            mask_first = masks[:, 0, :]                  # (N, F)
            mask_tail = masks[-1, 1:, :]                 # (L-1, F)
            mask_series = torch.cat([mask_first, mask_tail], dim=0)  # (T, F)
            # Drop positions where every feature is masked
            any_valid = mask_series.sum(dim=-1) > 0      # (T,)
            series = series[any_valid]

        if series.shape[0] < 2:
            raise ValueError(
                f"Reconstructed training series is too short ({series.shape[0]} steps). "
                "Bootstrap needs at least 2 observations."
            )

        self.history = series
        print(
            f"Bootstrap fitted (stationary block): T={series.shape[0]}, "
            f"F={series.shape[1]}, block_p={self.block_p:.3f} "
            f"(expected block length = {1.0 / self.block_p:.1f})"
        )

    def generate(self, n_samples, seq_len, device='cpu', **kwargs):
        """Generate n_samples novel trajectories of length seq_len.

        Algorithm (fully vectorized):
          1. Decide for each (sample, timestep) whether to start a new block,
             with per-step probability ``block_p``. Force the first timestep
             to start a block.
          2. For each potential block-start position, draw a random starting
             index into the training series (uniform over ``[0, T)``).
          3. For each output position t, find the most recent block-start
             (via cummax of the jump-time indicator) and compute the offset
             within that block.
          4. Source index = (base_index_at_block_start + offset) mod T —
             circular wrap handles blocks that run past the end of the series.
          5. Gather from ``self.history``.
        """
        if self.history is None:
            raise ValueError("Model not fitted.")
        if seq_len < 1:
            raise ValueError(f"seq_len must be positive, got {seq_len}")

        T = self.history.shape[0]
        F = self.features
        N = n_samples
        L = seq_len

        # (N, L) True where a new block starts at this step. First step always True.
        jumps = torch.rand(N, L) < self.block_p
        jumps[:, 0] = True

        # Per-(sample, step) random destination. Only meaningful where jumps=True,
        # but generating everywhere is cheaper than conditional sampling.
        new_indices = torch.randint(0, T, (N, L))

        # For each step t, find t' = most recent jump time at or before t.
        # trick: mark jump times with their index, zero elsewhere, then cummax.
        arange_row = torch.arange(L).unsqueeze(0).expand(N, L)
        marked = torch.where(jumps, arange_row, torch.zeros_like(arange_row))
        block_start_t, _ = torch.cummax(marked, dim=1)           # (N, L)

        # Base index for each (sample, step): new_indices at the block start.
        base_idx = torch.gather(new_indices, dim=1, index=block_start_t)

        # Position within current block.
        offset = arange_row - block_start_t                      # (N, L)

        # Source indices into the training series, with circular wrap.
        src = (base_idx + offset) % T                            # (N, L)

        out = self.history[src]                                  # (N, L, F)
        return out.to(device)
