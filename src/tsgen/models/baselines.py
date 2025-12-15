import torch
import numpy as np
from tsgen.models.base_model import GenerativeModel

class MultivariateGBM(GenerativeModel):
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
    def __init__(self, features, full_covariance=True):
        super().__init__()
        self.features = features
        self.full_covariance = full_covariance

        # Dummy parameter to avoid optimizer errors
        self.dummy = torch.nn.Parameter(torch.zeros(1))

        if full_covariance:
            # Full covariance mode: store mean vector and Cholesky factor
            self.register_buffer('mean', torch.zeros(features))
            self.register_buffer('cholesky_L', torch.eye(features))
        else:
            # Independent mode: store per-feature mean and std
            self.register_buffer('mu', torch.zeros(features))
            self.register_buffer('sigma', torch.ones(features))

    def forward(self, x, t):
        """
        Not used for baseline models, but required by GenerativeModel interface.
        """
        return torch.zeros_like(x)

    def fit(self, data_loader):
        """
        Fits the model parameters from training data.

        For full_covariance=True:
        - Estimates mean vector and full covariance matrix
        - Performs Cholesky decomposition for efficient sampling

        For full_covariance=False:
        - Estimates per-feature mean and standard deviation

        Args:
            data_loader: Yields batches of (Batch, Seq, Features) scaled log-returns
        """
        all_data = []
        for batch in data_loader:
            # Handle both tuple (from TensorDataset) and raw tensor batches
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            all_data.append(batch)

        # Concatenate: (Total_Samples, Seq, Feat)
        X = torch.cat(all_data, dim=0)

        # Flatten to (N_total, Feat) for computing statistics
        X_flat = X.view(-1, self.features)

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

    def sample(self, n_samples, sequence_length):
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
            sequence_length: Length of each sequence

        Returns:
            Tensor of shape (n_samples, sequence_length, features)
        """
        if self.full_covariance:
            # Sample from standard normal: (n_samples, sequence_length, features)
            z = torch.randn(n_samples, sequence_length, self.features,
                           device=self.cholesky_L.device)

            # Transform to correlated samples: x = mean + z @ L.T
            samples = self.mean + torch.matmul(z, self.cholesky_L.T)
        else:
            # Independent sampling per feature
            samples = torch.normal(
                mean=self.mu.expand(n_samples, sequence_length, -1),
                std=self.sigma.expand(n_samples, sequence_length, -1)
            )

        return samples

class BootstrapGenerativeModel(GenerativeModel):
    """
    Historical Bootstrap (Block Bootstrap) Generative Model.
    Resamples blocks from historical data.
    """
    def __init__(self, features, sequence_length):
        super().__init__()
        self.features = features
        self.sequence_length = sequence_length
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.history = None # Will store the pool of windows

    def forward(self, x, t):
        return torch.zeros_like(x)

    def fit(self, data_loader):
        """
        Stores the historical windows.
        """
        all_data = []
        for batch in data_loader:
            # Handle both tuple (from TensorDataset) and raw tensor batches
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            all_data.append(batch)

        # (Total_Samples, Seq, Feat)
        self.history = torch.cat(all_data, dim=0)
        print(f"Bootstrap Fitted: History pool size {self.history.shape}")

    def sample(self, n_samples, sequence_length=None):
        if self.history is None:
            raise ValueError("Model not fitted.")

        # Randomly sample indices
        indices = torch.randint(0, len(self.history), (n_samples,))
        return self.history[indices]
