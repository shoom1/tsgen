"""
Models for synthetic time series generation.

This module provides implementations of various generative models:
- UNet1D: 1D convolutional encoder-decoder architecture
- DiffusionTransformer: Transformer-based diffusion model
- MultivariateGaussian: Multivariate Gaussian baseline on scaled log-returns
  (supports both independent and correlated sampling via full_covariance parameter)
- BootstrapGenerativeModel: Historical bootstrap baseline
- TimeVAE: Variational Autoencoder for time series
"""

from tsgen.models.base_model import GenerativeModel
from tsgen.models.embeddings import SinusoidalPositionEmbeddings
from tsgen.models.registry import ModelRegistry
from tsgen.models.unet import UNet1D
from tsgen.models.transformer import DiffusionTransformer
from tsgen.models.mamba import MambaDiffusion
from tsgen.models.diffwave import DiffWave1D
from tsgen.models.dit import DiT1D
from tsgen.models.baselines import MultivariateGaussian, BootstrapGenerativeModel
from tsgen.models.garch import CCCGARCH
from tsgen.models.timevae import TimeVAE
from tsgen.models.diffusion import DiffusionUtils

__all__ = [
    # Base class
    "GenerativeModel",
    # Registry
    "ModelRegistry",
    # Embeddings
    "SinusoidalPositionEmbeddings",
    # Diffusion models
    "UNet1D",
    "DiffusionTransformer",
    "MambaDiffusion",
    "DiffWave1D",
    "DiT1D",
    "DiffusionUtils",
    # VAE models
    "TimeVAE",
    # Baseline models
    "MultivariateGaussian",
    "BootstrapGenerativeModel",
    "CCCGARCH",
]
