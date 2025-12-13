import pytest
import torch
from tsgen.models.factory import create_model
from tsgen.models.unet import UNet1D

def test_factory_unet_creation():
    config = {
        'model_type': 'unet',
        'sequence_length': 32,
        'tickers': ['AAPL', 'GOOG'],
        'base_channels': 32
    }
    
    model = create_model(config)
    
    assert isinstance(model, UNet1D)
    
    # Check if params were passed correctly (UNet1D stores features in down1.conv1.in_channels usually, but lets check structure)
    # down1 input channels should match features
    assert model.down1.conv1.in_channels == 2

def test_factory_unknown_type():
    config = {'model_type': 'unknown_model'}
    
    with pytest.raises(ValueError) as excinfo:
        create_model(config)
    assert "Unknown model_type" in str(excinfo.value)

def test_factory_default_fallback():
    # Should default to UNet if model_type not present (backward compat)
    config = {
        'sequence_length': 32,
        'tickers': ['A']
    }
    model = create_model(config)
    assert isinstance(model, UNet1D)
