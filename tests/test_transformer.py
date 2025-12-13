import pytest
import torch
from tsgen.models.factory import create_model
from tsgen.models.transformer import DiffusionTransformer

def test_factory_transformer_creation():
    config = {
        'model_type': 'transformer',
        'sequence_length': 32,
        'tickers': ['AAPL'],
        'dim': 32,
        'depth': 2,
        'heads': 2,
        'mlp_dim': 64
    }
    
    model = create_model(config)
    
    assert isinstance(model, DiffusionTransformer)
    assert model.dim == 32

def test_transformer_forward_shape():
    seq_len = 16
    features = 2
    dim = 32
    batch_size = 4
    
    model = DiffusionTransformer(sequence_length=seq_len, features=features, dim=dim)
    
    x = torch.randn(batch_size, seq_len, features)
    t = torch.randint(0, 100, (batch_size,))
    
    out = model(x, t)
    
    assert out.shape == (batch_size, seq_len, features)
