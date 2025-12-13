import pytest
import torch
import numpy as np
from tsgen.analysis.tstr import SimpleLSTM, train_and_evaluate_tstr

def test_simple_lstm_shape():
    batch_size = 4
    seq_len = 10
    features = 3
    hidden_dim = 8
    
    model = SimpleLSTM(input_dim=features, hidden_dim=hidden_dim, output_dim=features)
    x = torch.randn(batch_size, seq_len, features)
    
    output = model(x)
    # Output should be (batch, output_dim)
    assert output.shape == (batch_size, features)

def test_tstr_execution():
    # Smoke test for TSTR loop
    # Create very small dummy data
    # (N, Seq, Feat)
    syn_data = np.random.randn(10, 5, 1).astype(np.float32)
    real_data = np.random.randn(10, 5, 1).astype(np.float32)
    
    # Run for 1 epoch to save time
    mse = train_and_evaluate_tstr(syn_data, real_data, epochs=1, device='cpu')
    
    assert isinstance(mse, float)
    assert mse >= 0.0

def test_tstr_insufficient_data():
    # Sequence length < 2 case
    syn_data = np.random.randn(10, 1, 1).astype(np.float32)
    real_data = np.random.randn(10, 1, 1).astype(np.float32)
    
    mse = train_and_evaluate_tstr(syn_data, real_data, epochs=1)
    assert mse == 0.0
