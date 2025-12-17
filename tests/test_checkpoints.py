"""
Tests for model checkpoint saving and loading.

Ensures that:
- Models can be saved and loaded correctly
- Loaded models produce identical outputs
- Processor state is preserved
- Checkpoints are compatible across sessions
"""

import pytest
import torch
import tempfile
import os
import joblib
from pathlib import Path

from tsgen.models.factory import create_model
from tsgen.models.unet import UNet1D
from tsgen.models.transformer import DiffusionTransformer
from tsgen.models.baselines import MultivariateGBM, BootstrapGenerativeModel
from tsgen.data.processor import LogReturnProcessor
import pandas as pd
import numpy as np


@pytest.fixture
def temp_dir():
    """Create temporary directory for test artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_config():
    """Sample configuration for models."""
    return {
        'sequence_length': 64,
        'tickers': ['AAPL', 'MSFT'],
        'model_type': 'unet',
        'base_channels': 32,
    }


def test_unet_save_load(sample_config, temp_dir):
    """Test UNet model can be saved and loaded."""
    # Create model
    config = sample_config.copy()
    config['model_type'] = 'unet'
    model = create_model(config)

    # Generate test input
    x = torch.randn(2, 64, 2)
    t = torch.randint(0, 100, (2,))

    # Get output before saving
    model.eval()
    with torch.no_grad():
        output_before = model(x, t)

    # Save model
    save_path = os.path.join(temp_dir, 'model.pt')
    torch.save(model.state_dict(), save_path)

    # Load model
    loaded_model = create_model(config)
    loaded_model.load_state_dict(torch.load(save_path, weights_only=True))
    loaded_model.eval()

    # Get output after loading
    with torch.no_grad():
        output_after = loaded_model(x, t)

    # Verify outputs are identical
    assert torch.allclose(output_before, output_after, atol=1e-6)


def test_transformer_save_load(sample_config, temp_dir):
    """Test Transformer model can be saved and loaded."""
    # Create transformer model
    config = sample_config.copy()
    config['model_type'] = 'transformer'
    config['dim'] = 64
    config['depth'] = 2
    config['heads'] = 4
    config['mlp_dim'] = 128

    model = create_model(config)

    # Generate test input
    x = torch.randn(2, 64, 2)
    t = torch.randint(0, 100, (2,))

    # Get output before saving
    model.eval()
    with torch.no_grad():
        output_before = model(x, t)

    # Save model
    save_path = os.path.join(temp_dir, 'transformer.pt')
    torch.save(model.state_dict(), save_path)

    # Load model
    loaded_model = create_model(config)
    loaded_model.load_state_dict(torch.load(save_path, weights_only=True))
    loaded_model.eval()

    # Get output after loading
    with torch.no_grad():
        output_after = loaded_model(x, t)

    # Verify outputs are identical
    assert torch.allclose(output_before, output_after, atol=1e-6)


def test_processor_save_load(temp_dir):
    """Test DataProcessor can be saved and loaded."""
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'AAPL': np.cumsum(np.random.randn(100)) + 100,
        'MSFT': np.cumsum(np.random.randn(100)) + 150,
    }, index=dates)

    # Create and fit processor
    processor = LogReturnProcessor()
    processor.fit(data)

    # Transform data
    transformed_before = processor.transform(data)

    # Save processor
    save_path = os.path.join(temp_dir, 'processor.pkl')
    processor.save(save_path)

    # Load processor
    loaded_processor = LogReturnProcessor.load(save_path)

    # Transform with loaded processor
    transformed_after = loaded_processor.transform(data)

    # Verify transformations are identical (both are numpy arrays)
    np.testing.assert_allclose(
        transformed_before,
        transformed_after,
        rtol=1e-6
    )


def test_processor_inverse_transform_consistency(temp_dir):
    """Test that processor inverse transform is consistent after save/load."""
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'AAPL': np.cumsum(np.random.randn(100)) + 100,
        'MSFT': np.cumsum(np.random.randn(100)) + 150,
    }, index=dates)

    # Create and fit processor
    processor = LogReturnProcessor()
    processor.fit(data)

    # Transform and inverse transform
    transformed = processor.transform(data)
    initial_price = data.iloc[0].values
    reconstructed_before = processor.inverse_transform(transformed, initial_price)

    # Save and load processor
    save_path = os.path.join(temp_dir, 'processor.pkl')
    processor.save(save_path)
    loaded_processor = LogReturnProcessor.load(save_path)

    # Inverse transform with loaded processor
    reconstructed_after = loaded_processor.inverse_transform(transformed, initial_price)

    # Verify reconstructions are identical (both are numpy arrays)
    np.testing.assert_allclose(
        reconstructed_before,
        reconstructed_after,
        rtol=1e-6
    )


def test_checkpoint_directory_structure(temp_dir):
    """Test checkpoint directory is created correctly."""
    checkpoint_dir = Path(temp_dir) / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)

    # Save model checkpoint
    config = {
        'sequence_length': 64,
        'tickers': ['AAPL', 'MSFT'],
        'model_type': 'unet',
        'base_channels': 32,
    }
    model = create_model(config)

    checkpoint_path = checkpoint_dir / 'epoch_10.pt'
    torch.save(model.state_dict(), checkpoint_path)

    # Verify checkpoint exists
    assert checkpoint_path.exists()
    assert checkpoint_path.is_file()


def test_gbm_baseline_save_load(temp_dir):
    """Test GBM baseline model save/load."""
    # Create GBM model
    model = MultivariateGBM(features=2)
    model.mu = torch.tensor([0.001, 0.002])
    model.sigma = torch.tensor([0.02, 0.03])

    # Generate sample
    torch.manual_seed(42)
    sample_before = model.sample(5, 32)

    # Save model (baselines save full object)
    save_path = os.path.join(temp_dir, 'gbm.pt')
    torch.save(model, save_path)

    # Load model
    loaded_model = torch.load(save_path, weights_only=False)

    # Generate sample with same seed
    torch.manual_seed(42)
    sample_after = loaded_model.sample(5, 32)

    # Verify samples are identical
    assert torch.allclose(sample_before, sample_after)


def test_bootstrap_baseline_save_load(temp_dir):
    """Test Bootstrap baseline model save/load."""
    # Create Bootstrap model
    model = BootstrapGenerativeModel(features=2, sequence_length=64)
    model.history = torch.randn(10, 64, 2)

    # Generate sample
    torch.manual_seed(42)
    sample_before = model.sample(5, 64)

    # Save model (baselines save full object)
    save_path = os.path.join(temp_dir, 'bootstrap.pt')
    torch.save(model, save_path)

    # Load model
    loaded_model = torch.load(save_path, weights_only=False)

    # Generate sample with same seed
    torch.manual_seed(42)
    sample_after = loaded_model.sample(5, 64)

    # Verify samples are identical
    assert torch.allclose(sample_before, sample_after)


def test_multiple_checkpoint_management(temp_dir):
    """Test managing multiple checkpoints."""
    checkpoint_dir = Path(temp_dir) / 'checkpoints'
    checkpoint_dir.mkdir()

    config = {
        'sequence_length': 64,
        'tickers': ['AAPL', 'MSFT'],
        'model_type': 'unet',
        'base_channels': 32,
    }

    # Save multiple checkpoints
    for epoch in [10, 20, 30, 40, 50]:
        model = create_model(config)
        checkpoint_path = checkpoint_dir / f'epoch_{epoch}.pt'
        torch.save(model.state_dict(), checkpoint_path)

    # Verify all checkpoints exist
    checkpoints = sorted(checkpoint_dir.glob('epoch_*.pt'))
    assert len(checkpoints) == 5

    # Load latest checkpoint
    latest = checkpoints[-1]
    model = create_model(config)
    model.load_state_dict(torch.load(latest, weights_only=True))

    # Verify model works
    x = torch.randn(2, 64, 2)
    t = torch.randint(0, 100, (2,))
    with torch.no_grad():
        output = model(x, t)
    assert output.shape == (2, 64, 2)


def test_checkpoint_with_optimizer_state(temp_dir):
    """Test saving checkpoint with optimizer state."""
    config = {
        'sequence_length': 64,
        'tickers': ['AAPL', 'MSFT'],
        'model_type': 'unet',
        'base_channels': 32,
    }

    # Create model and optimizer
    model = create_model(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Do one training step
    x = torch.randn(4, 64, 2)
    t = torch.randint(0, 100, (4,))
    target = torch.randn(4, 64, 2)

    output = model(x, t)
    loss = torch.nn.functional.mse_loss(output, target)
    loss.backward()
    optimizer.step()

    # Save checkpoint with optimizer
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': 1,
        'loss': loss.item(),
    }
    save_path = os.path.join(temp_dir, 'checkpoint.pt')
    torch.save(checkpoint, save_path)

    # Load checkpoint
    loaded_checkpoint = torch.load(save_path, weights_only=False)

    # Verify checkpoint contents
    assert 'model_state_dict' in loaded_checkpoint
    assert 'optimizer_state_dict' in loaded_checkpoint
    assert 'epoch' in loaded_checkpoint
    assert 'loss' in loaded_checkpoint

    # Load into new model and optimizer
    new_model = create_model(config)
    new_optimizer = torch.optim.Adam(new_model.parameters(), lr=1e-3)

    new_model.load_state_dict(loaded_checkpoint['model_state_dict'])
    new_optimizer.load_state_dict(loaded_checkpoint['optimizer_state_dict'])

    # Verify loaded model produces same output
    new_model.eval()
    model.eval()
    with torch.no_grad():
        output1 = model(x, t)
        output2 = new_model(x, t)

    assert torch.allclose(output1, output2, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
