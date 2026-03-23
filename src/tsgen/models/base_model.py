from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class BaseGenerativeModel(nn.Module, ABC):
    """
    Root abstract base class for all generative models in the framework.

    This class establishes the common interface that all generative models
    must implement, regardless of their training paradigm (diffusion, VAE,
    statistical fitting, etc.).
    """

    @property
    def model_type(self) -> str:
        """Return model type identifier for factory/registry lookup."""
        return self.__class__.__name__.lower()

    @classmethod
    @abstractmethod
    def from_config(cls, config, features: int = None):
        """Create model instance from ExperimentConfig."""
        pass

    @abstractmethod
    def generate(self, n_samples: int, seq_len: int, device: str = 'cpu', **kwargs) -> torch.Tensor:
        """
        Generate synthetic samples.

        Args:
            n_samples: Number of samples to generate
            seq_len: Length of each generated sequence
            device: Device to generate on
            **kwargs: Additional model-specific arguments

        Returns:
            torch.Tensor: Generated samples (n_samples, seq_len, features)
        """
        pass


class DiffusionModel(BaseGenerativeModel):
    """
    Abstract base class for diffusion-based generative models.

    These models are trained to predict noise at a given timestep using the
    DDPM framework. They implement forward(x, t, y, mask) for noise prediction.

    Subclasses: UNet1D, DiffusionTransformer, MambaDiffusion

    Class Attributes:
        supports_conditioning (bool): Whether the model supports class conditioning
        supports_masking (bool): Whether the model supports attention masking
    """

    supports_conditioning: bool = False
    supports_masking: bool = False

    # Diffusion sampling attributes (set by from_config)
    _timesteps: int = 1000
    _sampling_method: str = 'ddpm'
    _num_inference_steps: int = 50

    @abstractmethod
    def forward(self, x: torch.Tensor, t: torch.Tensor,
                y: torch.Tensor = None, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Predict noise at timestep t.

        Args:
            x (torch.Tensor): Noisy input tensor (Batch, Seq_Len, Features)
            t (torch.Tensor): Timesteps (Batch,)
            y (torch.Tensor, optional): Class labels for conditional generation (Batch,)
            mask (torch.Tensor, optional): Binary mask indicating valid positions
                (Batch, Seq_Len, Features), 1 = valid, 0 = missing/masked

        Returns:
            torch.Tensor: Predicted noise (Batch, Seq_Len, Features)
        """
        pass

    def generate(self, n_samples: int, seq_len: int, device: str = 'cpu', **kwargs) -> torch.Tensor:
        """
        Generate samples using DDPM or DDIM sampling.

        Args:
            n_samples: Number of samples to generate
            seq_len: Sequence length of each sample
            device: Device to generate on
            **kwargs: Additional arguments (e.g., y for class conditioning)

        Returns:
            torch.Tensor: Generated samples (n_samples, seq_len, features)
        """
        from tsgen.models.diffusion import DiffusionUtils
        y = kwargs.get('y', None)
        # Cache DiffusionUtils to avoid recomputing alpha schedules
        if not hasattr(self, '_diff_utils') or self._diff_utils_device != device:
            self._diff_utils = DiffusionUtils(T=self._timesteps, device=device)
            self._diff_utils_device = device
        image_size = (seq_len, self.features)
        if self._sampling_method == 'ddim':
            return self._diff_utils.ddim_sample(
                self, image_size=image_size, batch_size=n_samples,
                num_inference_steps=self._num_inference_steps, y=y)
        else:
            return self._diff_utils.sample(
                self, image_size=image_size, batch_size=n_samples, y=y)


class VAEModel(BaseGenerativeModel):
    """
    Abstract base class for Variational Autoencoder models.

    VAE models learn a latent representation through variational inference.
    They consist of an encoder (data → latent distribution) and decoder
    (latent sample → reconstruction).

    The forward pass returns reconstruction along with distribution parameters
    for computing the ELBO loss (reconstruction + KL divergence).

    Subclasses: TimeVAE

    Class Attributes:
        latent_dim (int): Dimension of the latent space
    """

    @property
    @abstractmethod
    def latent_dim(self) -> int:
        """Return dimension of the latent space."""
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs) -> tuple:
        """
        Forward pass through encoder and decoder.

        Args:
            x (torch.Tensor): Input tensor (Batch, Seq_Len, Features)
            **kwargs: Additional arguments (e.g., teacher_forcing_ratio)

        Returns:
            tuple: (reconstruction, mu, logvar) where:
                - reconstruction: Reconstructed sequence (Batch, Seq_Len, Features)
                - mu: Mean of latent distribution (Batch, latent_dim)
                - logvar: Log-variance of latent distribution (Batch, latent_dim)
        """
        pass

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to latent representation.

        Args:
            x (torch.Tensor): Input tensor (Batch, Seq_Len, Features)

        Returns:
            torch.Tensor: Latent representation (Batch, latent_dim)
        """
        pass

    @abstractmethod
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to sequence.

        Args:
            z (torch.Tensor): Latent tensor (Batch, latent_dim)

        Returns:
            torch.Tensor: Reconstructed sequence (Batch, Seq_Len, Features)
        """
        pass

    @abstractmethod
    def generate(self, n_samples: int, seq_len: int = None, device: str = 'cpu', **kwargs) -> torch.Tensor:
        """
        Generate samples from prior distribution.

        Args:
            n_samples: Number of samples to generate
            seq_len: Length of sequences (may be fixed for some architectures)
            device: Device to generate on
            **kwargs: Additional model-specific arguments

        Returns:
            torch.Tensor: Generated samples (n_samples, seq_len, features)
        """
        pass

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + eps * sigma

        This allows gradients to flow through the sampling process.

        Args:
            mu: Mean of latent distribution (Batch, latent_dim)
            logvar: Log-variance of latent distribution (Batch, latent_dim)

        Returns:
            torch.Tensor: Sampled latent variable (Batch, latent_dim)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class StatisticalModel(BaseGenerativeModel):
    """
    Abstract base class for statistical/fit-based generative models.

    These models are not trained via gradient descent. Instead, they fit
    statistical parameters from data using the fit() method and generate
    samples using the sample() method.

    Subclasses: MultivariateGBM, BootstrapGenerativeModel
    """

    @abstractmethod
    def fit(self, dataloader) -> None:
        """
        Fit model parameters from training data.

        Args:
            dataloader: PyTorch DataLoader yielding batches of
                (Batch, Seq_Len, Features) tensors or (data, mask) tuples
        """
        pass

    @abstractmethod
    def generate(self, n_samples: int, seq_len: int, device: str = 'cpu', **kwargs) -> torch.Tensor:
        """
        Generate synthetic samples.

        Args:
            n_samples: Number of sample sequences to generate
            seq_len: Length of each generated sequence
            device: Device to generate on
            **kwargs: Additional model-specific arguments

        Returns:
            torch.Tensor: Generated samples (n_samples, seq_len, features)
        """
        pass


# Backward compatibility alias
GenerativeModel = DiffusionModel
