"""Tests for model creation via ModelRegistry (replaces old factory tests)."""

import pytest
import torch
from tsgen.models.registry import ModelRegistry
from tsgen.config.schema import ExperimentConfig
from tsgen.models.unet import UNet1D


def test_registry_unet_creation():
    """Test creating UNet via ModelRegistry."""
    config = ExperimentConfig(
        model_type='unet',
        sequence_length=32,
        tickers=['AAPL', 'GOOG'],
        base_channels=32,
    )

    model = ModelRegistry.create(config)

    assert isinstance(model, UNet1D)

    # Check if params were passed correctly
    # down1 input channels should match features
    assert model.down1.conv1.in_channels == 2
    assert model.features == 2


def test_registry_unknown_type():
    """Test that unknown model_type raises ValueError."""
    config = ExperimentConfig(model_type='unknown_model', tickers=['A'])

    with pytest.raises(ValueError, match="No model registered"):
        ModelRegistry.create(config)
