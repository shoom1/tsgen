"""Tests for MambaDiffusion model."""

import pytest
import torch
from tsgen.models.registry import ModelRegistry
from tsgen.config.schema import ExperimentConfig
from tsgen.models.mamba import MambaDiffusion, MambaBlock, RMSNorm


def test_rmsnorm():
    """Test RMSNorm layer."""
    d_model = 32
    batch_size = 4
    seq_len = 16

    norm = RMSNorm(d_model)
    x = torch.randn(batch_size, seq_len, d_model)

    out = norm(x)

    assert out.shape == x.shape
    assert torch.allclose(out.pow(2).mean(-1), torch.ones(batch_size, seq_len), atol=1e-3)


def test_mamba_block_forward():
    """Test MambaBlock forward pass."""
    d_model = 64
    batch_size = 2
    seq_len = 8

    block = MambaBlock(d_model=d_model, d_state=16, d_conv=4, expand=2)
    x = torch.randn(batch_size, seq_len, d_model)

    out = block(x)

    assert out.shape == x.shape
    assert torch.all(torch.isfinite(out))


def test_mamba_block_ssm():
    """Test MambaBlock SSM recurrence."""
    d_model = 32
    batch_size = 2
    seq_len = 4

    block = MambaBlock(d_model=d_model, d_state=8, d_conv=4, expand=2)
    # SSM expects input that's already been projected to d_inner
    x = torch.randn(batch_size, seq_len, block.d_inner)

    # Should produce finite outputs
    out = block.ssm(x)

    assert out.shape == (batch_size, seq_len, block.d_inner)
    assert torch.all(torch.isfinite(out))


def test_registry_mamba_creation():
    """Test creating Mamba via ModelRegistry."""
    config = ExperimentConfig(
        model_type='mamba',
        data={'tickers': ['AAPL', 'MSFT'], 'sequence_length': 32},
        model={'dim': 64, 'depth': 2},
    )

    model = ModelRegistry.create(config)

    assert isinstance(model, MambaDiffusion)
    assert model.dim == 64
    assert model.features == 2


def test_mamba_diffusion_forward_shape():
    """Test MambaDiffusion forward pass shape."""
    seq_len = 16
    features = 2
    dim = 64
    depth = 2
    batch_size = 4

    model = MambaDiffusion(
        sequence_length=seq_len,
        features=features,
        dim=dim,
        depth=depth
    )

    x = torch.randn(batch_size, seq_len, features)
    t = torch.randint(0, 1000, (batch_size,))

    out = model(x, t)

    # Should predict noise with same shape as input
    assert out.shape == (batch_size, seq_len, features)
    assert torch.all(torch.isfinite(out))


def test_mamba_diffusion_with_labels():
    """Test MambaDiffusion with class conditioning."""
    seq_len = 16
    features = 2
    dim = 64
    num_classes = 10
    batch_size = 4

    model = MambaDiffusion(
        sequence_length=seq_len,
        features=features,
        dim=dim,
        depth=2,
        num_classes=num_classes
    )

    x = torch.randn(batch_size, seq_len, features)
    t = torch.randint(0, 1000, (batch_size,))
    y = torch.randint(0, num_classes, (batch_size,))

    out = model(x, t, y)

    assert out.shape == (batch_size, seq_len, features)
    assert torch.all(torch.isfinite(out))


def test_mamba_diffusion_reproducibility():
    """Test Mamba sampling is reproducible with same seed."""
    seq_len = 16
    features = 2
    batch_size = 2

    model = MambaDiffusion(
        sequence_length=seq_len,
        features=features,
        dim=32,
        depth=2
    )
    model.eval()

    x = torch.randn(batch_size, seq_len, features)
    t = torch.randint(0, 1000, (batch_size,))

    torch.manual_seed(42)
    out1 = model(x, t)

    torch.manual_seed(42)
    out2 = model(x, t)

    assert torch.allclose(out1, out2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
