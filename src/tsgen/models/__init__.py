"""
Models for synthetic time series generation.

This module provides implementations of various generative models:
- UNet1D: 1D convolutional encoder-decoder architecture
- DiffusionTransformer: Transformer-based diffusion model
- MultivariateGBM: Multivariate Geometric Brownian Motion baseline
  (supports both independent and correlated sampling via full_covariance parameter)
- BootstrapGenerativeModel: Historical bootstrap baseline
"""

from tsgen.models.base_model import GenerativeModel
from tsgen.models.embeddings import SinusoidalPositionEmbeddings
from tsgen.models.unet import UNet1D
from tsgen.models.transformer import DiffusionTransformer
from tsgen.models.baselines import MultivariateGBM, BootstrapGenerativeModel
from tsgen.models.diffusion import DiffusionUtils
from tsgen.models.factory import create_model

__all__ = [
    # Base class
    "GenerativeModel",
    # Embeddings
    "SinusoidalPositionEmbeddings",
    # Diffusion models
    "UNet1D",
    "DiffusionTransformer",
    "DiffusionUtils",
    # Baseline models
    "MultivariateGBM",
    "BootstrapGenerativeModel",
    # Factory
    "create_model",
]
