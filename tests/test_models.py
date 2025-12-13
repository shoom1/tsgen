import pytest
import torch
import numpy as np
from tsgen.models.unet import UNet1D
from tsgen.models.diffusion import DiffusionUtils
from tsgen.data.processor import LogReturnProcessor
import pandas as pd

def test_processor_fit_transform():
    # Create dummy data: 100 days, 2 tickers
    dates = pd.date_range(start='2022-01-01', periods=100)
    data = pd.DataFrame(np.random.rand(100, 2) * 100 + 100, index=dates, columns=['A', 'B'])

    processor = LogReturnProcessor()
    processor.fit(data)

    # Transform should return scaled log returns
    scaled_returns = processor.transform(data)

    # Expected shape: (100 - 1 for log returns, 2 features) = (99, 2)
    assert scaled_returns.shape == (99, 2)

    # Should have roughly zero mean and unit variance (with some tolerance)
    assert abs(scaled_returns.mean()) < 0.1
    assert abs(scaled_returns.std() - 1.0) < 0.1

def test_unet_output_shape():
    batch_size = 4
    seq_len = 32
    features = 2
    model = UNet1D(sequence_length=seq_len, features=features)
    
    x = torch.randn(batch_size, seq_len, features)
    t = torch.randint(0, 100, (batch_size,))
    
    output = model(x, t)
    
    assert output.shape == (batch_size, seq_len, features)

def test_diffusion_forward_process():
    device = "cpu"
    diff_utils = DiffusionUtils(T=100, device=device)
    x_0 = torch.randn(2, 32, 2) # (Batch, Seq, Feat)
    t = torch.tensor([0, 99])   # Test start and end timesteps
    
    # Deterministic noise for testing
    noise = torch.randn_like(x_0)
    
    x_t = diff_utils.q_sample(x_0, t, noise)
    
    assert x_t.shape == x_0.shape
    assert not torch.allclose(x_t, x_0) # Should be noisy

def test_diffusion_sampling_shape():
    # Mock model outputting zeros (predicting no noise)
    class MockModel(torch.nn.Module):
        def forward(self, x, t, y=None):
            return torch.zeros_like(x)

    model = MockModel()
    seq_len = 16
    features = 2
    diff_utils = DiffusionUtils(T=10, device="cpu")

    samples = diff_utils.sample(model, image_size=(seq_len, features), batch_size=2)

    assert samples.shape == (2, seq_len, features)
