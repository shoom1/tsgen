import torch
import numpy as np
from tsgen.models.base_model import GenerativeModel

class GBMGenerativeModel(GenerativeModel):
    """
    A generative model that outputs paths from a Geometric Brownian Motion process.
    It ignores the input 'x' (denoising) and 't' (timesteps) during inference
    because GBM is a direct sampling process, not a diffusion reversal.
    
    However, to fit the 'GenerativeModel' interface which implies a diffusion-like
    training loop, we need to be careful. 
    
    ACTUALLY: GBM is not trained via diffusion. It's a statistical fit.
    For the purpose of the *framework* where we want to 'train' and 'evaluate', 
    we can wrap the fitting logic in 'train' (calculating mu/sigma) and 
    sampling logic in 'sample'.
    
    But 'GenerativeModel' is an nn.Module designed for the Diffusion Loop (predicting noise).
    GBM doesn't fit that interface.
    
    So, we shouldn't force GBM into 'GenerativeModel' if that class assumes forward(x, t).
    Instead, we might need a 'BaselineModel' interface or handle baselines separately 
    in the main loop (skip training, just fit).
    
    For now, let's implement a class that CAN be instantiated by the factory,
    but we will need to adjust 'train.py' to handle non-gradient-based models,
    OR we treat this as a 'dummy' diffusion model that learns nothing but we 
    inject the sampling logic elsewhere.
    
    BETTER APPROACH for Research Framework:
    Baselines shouldn't be "trained" with the diffusion loop. They have their own 
    'fit' and 'sample' lifecycle.
    """
    def __init__(self, features, mu=0.0, sigma=0.02):
        super().__init__()
        self.features = features
        # Parameters (learnable or fixed)
        self.register_buffer('mu', torch.tensor([mu] * features))
        self.register_buffer('sigma', torch.tensor([sigma] * features))
        
        # Dummy parameter to avoid optimizer errors if passed to one
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x, t):
        # GBM doesn't predict noise.
        # If we force it into the training loop, we can just return zeros (no learning)
        # or raise an error.
        return torch.zeros_like(x)
        
    def fit(self, data_loader):
        """
        Fits mu and sigma from the dataloader.
        data_loader yields batches of (Batch, Seq, Feat) log-returns.
        """
        all_data = []
        for batch in data_loader:
            # Handle both tuple (from TensorDataset) and raw tensor batches
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            all_data.append(batch)

        # Concatenate: (Total_Samples, Seq, Feat)
        X = torch.cat(all_data, dim=0)
        
        # Flatten to (N_total, Feat)
        X_flat = X.view(-1, self.features)
        
        # Calculate stats per feature
        # Log-returns are roughly N(mu - 0.5*sigma^2, sigma)
        # But for simple GBM on returns: r_t ~ N(mu_dt, sigma_dt)
        # We model log-returns directly.
        self.mu = torch.mean(X_flat, dim=0)
        self.sigma = torch.std(X_flat, dim=0)
        
        print(f"GBM Fitted: mu={self.mu}, sigma={self.sigma}")

    def sample(self, n_samples, sequence_length):
        """
        Generates (n_samples, sequence_length, features)
        """
        # Sample from N(mu, sigma)
        # Shape: (N, L, F)
        return torch.normal(mean=self.mu.expand(n_samples, sequence_length, -1), 
                            std=self.sigma.expand(n_samples, sequence_length, -1))

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


class MultivariateLogNormalModel(GenerativeModel):
    """
    Multivariate Log-Normal Baseline Model.

    Unlike GBM which samples features independently, this model captures
    cross-asset correlations via a full covariance matrix. Uses Cholesky
    decomposition for efficient correlated sampling.

    The model operates on scaled log-returns (z-scored data from DataProcessor).
    """
    def __init__(self, features):
        super().__init__()
        self.features = features

        # Dummy parameter to avoid optimizer errors
        self.dummy = torch.nn.Parameter(torch.zeros(1))

        # Parameters (will be set during fit)
        self.register_buffer('mean', torch.zeros(features))
        self.register_buffer('cholesky_L', torch.eye(features))

    def forward(self, x, t):
        """
        Not used for baseline models, but required by GenerativeModel interface.
        """
        return torch.zeros_like(x)

    def fit(self, data_loader):
        """
        Fits the multivariate normal distribution from training data.

        Estimates:
        - Mean vector (per feature)
        - Covariance matrix (captures correlations)
        - Cholesky decomposition for efficient sampling

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

        # Compute mean vector and covariance matrix
        self.mean = torch.mean(X_flat, dim=0)

        # Compute covariance matrix
        # Center the data first
        X_centered = X_flat - self.mean
        cov = (X_centered.T @ X_centered) / (X_centered.shape[0] - 1)

        # Add regularization for numerical stability
        # This prevents singular matrix issues during Cholesky decomposition
        epsilon = 1e-6
        cov_regularized = cov + epsilon * torch.eye(self.features, device=cov.device)

        # Cholesky decomposition: cov = L @ L.T
        self.cholesky_L = torch.linalg.cholesky(cov_regularized)

        print(f"MultivariateLogNormal Fitted:")
        print(f"  Mean: {self.mean}")
        print(f"  Covariance diagonal: {torch.diagonal(cov)}")
        print(f"  Correlation matrix:\n{self._cov_to_corr(cov)}")

    def _cov_to_corr(self, cov):
        """Convert covariance matrix to correlation matrix."""
        std = torch.sqrt(torch.diagonal(cov))
        corr = cov / torch.outer(std, std)
        return corr

    def sample(self, n_samples, sequence_length):
        """
        Generates correlated samples using Cholesky decomposition.

        Algorithm:
        1. Sample z ~ N(0, I) - independent standard normals
        2. Transform: x = mean + L @ z.T
           where L is the Cholesky factor (L @ L.T = Cov)

        Args:
            n_samples: Number of sample sequences to generate
            sequence_length: Length of each sequence

        Returns:
            Tensor of shape (n_samples, sequence_length, features)
        """
        # Sample from standard normal: (n_samples, sequence_length, features)
        z = torch.randn(n_samples, sequence_length, self.features,
                       device=self.cholesky_L.device)

        # Transform to correlated samples: x = mean + z @ L.T
        # Broadcasting: (N, L, F) @ (F, F).T -> (N, L, F)
        samples = self.mean + torch.matmul(z, self.cholesky_L.T)

        return samples
