"""
Models for synthetic time series generation.

This module provides implementations of various generative models:
- UNet1D: 1D convolutional encoder-decoder architecture
- DiffusionTransformer: Transformer-based diffusion model
- GBMGenerativeModel: Geometric Brownian Motion baseline
- BootstrapGenerativeModel: Historical bootstrap baseline
- MultivariateLogNormalModel: Multivariate lognormal baseline with correlation
"""

from tsgen.models.base_model import GenerativeModel
from tsgen.models.embeddings import SinusoidalPositionEmbeddings
from tsgen.models.unet import UNet1D
from tsgen.models.transformer import DiffusionTransformer
from tsgen.models.baselines import GBMGenerativeModel, BootstrapGenerativeModel, MultivariateLogNormalModel
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
    "GBMGenerativeModel",
    "BootstrapGenerativeModel",
    "MultivariateLogNormalModel",
    # Factory
    "create_model",
]
