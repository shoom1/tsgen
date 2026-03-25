import torch
import numpy as np
from tsgen.models.base_model import StatisticalModel
from tsgen.models.registry import ModelRegistry


@ModelRegistry.register('gbm', 'multivariate_gbm', 'multivariate_lognormal')
class MultivariateGBM(StatisticalModel):
    """
    Multivariate Geometric Brownian Motion baseline model.

    This model generates synthetic time series using multivariate Gaussian sampling.
    It supports two modes controlled by the `full_covariance` parameter:

    - full_covariance=True (default): Captures cross-asset correlations via full
      covariance matrix using Cholesky decomposition. Best for multi-asset scenarios
      where correlations matter.

    - full_covariance=False: Treats each feature independently with its own mean
      and standard deviation. Faster but ignores correlations.

    The model operates on scaled log-returns (z-scored data from DataProcessor).
    It's a statistical fit (not gradient-based training), using 'fit' and 'sample'
    lifecycle instead of the diffusion training loop.
    """

    @classmethod
    def from_config(cls, config, features=None):
        """Create MultivariateGBM from ExperimentConfig."""
        # Determine covariance mode based on model_type
        if config.model_type == 'gbm':
            full_covariance = False
        else:
            # 'multivariate_gbm', 'multivariate_lognormal' use full covariance
            full_covariance = True
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

                print(f"MultivariateGBM Fitted with masks (full_covariance=True):")
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

                print(f"MultivariateGBM Fitted with masks (full_covariance=False):")
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

                print(f"MultivariateGBM Fitted (full_covariance=True):")
                print(f"  Mean: {self.mean}")
                print(f"  Covariance diagonal: {torch.diagonal(cov)}")
                print(f"  Correlation matrix:\n{self._cov_to_corr(cov)}")
            else:
                # Independent mode
                self.mu = torch.mean(X_flat, dim=0)
                self.sigma = torch.std(X_flat, dim=0)

                print(f"MultivariateGBM Fitted (full_covariance=False):")
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
    """
    Historical Bootstrap (Block Bootstrap) Generative Model.
    Resamples blocks from historical data.
    """

    @classmethod
    def from_config(cls, config, features=None):
        """Create BootstrapGenerativeModel from ExperimentConfig."""
        data = config.get_data_config()
        return cls(features=features, sequence_length=data.sequence_length)

    def __init__(self, features, sequence_length):
        super().__init__()
        self.features = features
        self.sequence_length = sequence_length
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.history = None  # Will store the pool of windows

    def fit(self, data_loader):
        """
        Stores the historical windows.

        Supports masked data: When batches contain (data, mask) tuples,
        only fully valid windows (all positions are valid) are stored.

        Args:
            data_loader: Yields batches of (Batch, Seq, Features) scaled log-returns,
                        or (data, mask) tuples for masked training
        """
        valid_windows = []
        total_windows = 0
        has_masks = False

        for batch in data_loader:
            # Handle both tuple (from TensorDataset) and raw tensor batches
            if isinstance(batch, (list, tuple)):
                data = batch[0]
                if len(batch) > 1:
                    mask = batch[1]
                    has_masks = True

                    # Keep only fully valid windows: all positions are 1
                    # Sum across seq and features: if sum equals total elements, window is fully valid
                    total_elements = mask.shape[1] * mask.shape[2]
                    fully_valid = (mask.sum(dim=(1, 2)) == total_elements)
                    data = data[fully_valid]
            else:
                data = batch

            total_windows += len(batch[0]) if isinstance(batch, (list, tuple)) else len(batch)
            if len(data) > 0:
                valid_windows.append(data)

        if valid_windows:
            self.history = torch.cat(valid_windows, dim=0)
            if has_masks:
                print(f"Bootstrap Fitted with masks: {len(self.history)} fully valid windows "
                      f"(from {total_windows} total)")
            else:
                print(f"Bootstrap Fitted: History pool size {self.history.shape}")
        else:
            raise ValueError("No fully valid windows found in training data. "
                           "Consider using a different model or adjusting data cleaning strategy.")

    def generate(self, n_samples, seq_len=None, device='cpu', **kwargs):
        """
        Generate samples by resampling from historical windows.

        Args:
            n_samples: Number of samples to generate
            seq_len: Not used (fixed by stored history), kept for API compatibility
            device: Device to generate on
            **kwargs: Additional arguments (unused)

        Returns:
            Tensor of shape (n_samples, sequence_length, features)
        """
        if self.history is None:
            raise ValueError("Model not fitted.")

        # Randomly sample indices
        indices = torch.randint(0, len(self.history), (n_samples,))
        return self.history[indices].to(device)
