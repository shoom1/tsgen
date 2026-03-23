"""Tests for base model abstract classes and their interface contracts."""

import pytest
import torch

from tsgen.models.base_model import (
    BaseGenerativeModel,
    DiffusionModel,
    VAEModel,
    StatisticalModel,
)


def test_diffusion_model_has_generate():
    """Test that DiffusionModel subclasses have generate() method."""
    from tsgen.models.unet import UNet1D

    model = UNet1D(sequence_length=16, features=2, base_channels=32)
    assert hasattr(model, 'generate')
    assert callable(model.generate)


def test_diffusion_model_has_features():
    """Test that DiffusionModel subclasses store features attribute."""
    from tsgen.models.unet import UNet1D
    from tsgen.models.transformer import DiffusionTransformer
    from tsgen.models.mamba import MambaDiffusion

    unet = UNet1D(sequence_length=16, features=3, base_channels=32)
    assert unet.features == 3

    transformer = DiffusionTransformer(sequence_length=16, features=5, dim=32)
    assert transformer.features == 5

    mamba = MambaDiffusion(sequence_length=16, features=4, dim=32, depth=2)
    assert mamba.features == 4


def test_base_generative_model_has_from_config():
    """Test that BaseGenerativeModel requires from_config()."""
    assert hasattr(BaseGenerativeModel, 'from_config')


def test_base_generative_model_has_generate():
    """Test that BaseGenerativeModel requires generate()."""
    assert hasattr(BaseGenerativeModel, 'generate')


def test_statistical_model_has_generate():
    """Test that StatisticalModel subclasses have generate()."""
    from tsgen.models.baselines import MultivariateGBM

    model = MultivariateGBM(features=2, full_covariance=False)
    assert hasattr(model, 'generate')
    assert callable(model.generate)


def test_vae_model_has_generate():
    """Test that VAEModel subclasses have generate()."""
    from tsgen.models.timevae import TimeVAE

    model = TimeVAE(features=2, sequence_length=16, latent_dim=4)
    assert hasattr(model, 'generate')
    assert callable(model.generate)


def test_diffusion_model_sampling_attributes():
    """Test that DiffusionModel has diffusion sampling attributes."""
    from tsgen.models.unet import UNet1D

    model = UNet1D(sequence_length=16, features=2, base_channels=32)
    assert model._timesteps == 1000  # default
    assert model._sampling_method == 'ddpm'  # default
    assert model._num_inference_steps == 50  # default


def test_from_config_on_all_models():
    """Test that all registered models implement from_config."""
    from tsgen.models.registry import ModelRegistry
    from tsgen.config.schema import ExperimentConfig

    config = ExperimentConfig(
        model_type='unet',
        tickers=['A', 'B'],
        sequence_length=16,
        base_channels=32,
    )

    model = ModelRegistry.create(config)
    assert hasattr(model, 'from_config')
    assert model.features == 2
