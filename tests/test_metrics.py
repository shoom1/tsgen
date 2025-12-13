import pytest
import numpy as np
import pandas as pd
from tsgen.analysis.metrics import compute_acf, calculate_stylized_facts

def test_compute_acf():
    # Test with a constant signal (ACF should be 1 everywhere technically, or statsmodels handles it)
    # Test with alternating signal [1, -1, 1, -1] -> High negative correlation at lag 1
    x = np.array([1, -1, 1, -1, 1, -1] * 10)
    acf_vals = compute_acf(x, lags=5)
    
    assert len(acf_vals) == 6 # 0 to 5
    assert acf_vals[0] == 1.0 # Lag 0 is always 1
    assert acf_vals[1] < 0 # Lag 1 should be negative for alternating

def test_calculate_stylized_facts_structure():
    # Create dummy data: (N, Seq, Feat)
    # 10 samples, 50 days, 2 assets (increased from 20 to 50 to avoid ACF truncation)
    real = np.random.randn(10, 50, 2)
    fake = np.random.randn(10, 50, 2)
    
    metrics = calculate_stylized_facts(real, fake)
    
    expected_keys = [
        'real_kurtosis', 'syn_kurtosis', 'kurtosis_diff',
        'real_skew', 'syn_skew', 'skew_diff',
        'corr_matrix_diff_norm',
        'real_acf_ret', 'syn_acf_ret',
        'real_acf_sq_ret', 'syn_acf_sq_ret',
        'acf_ret_diff', 'acf_sq_ret_diff'
    ]
    
    for key in expected_keys:
        assert key in metrics
    
    # Check shapes
    # Kurtosis/Skew: (Features,) -> (2,)
    assert metrics['real_kurtosis'].shape == (2,)
    assert metrics['kurtosis_diff'].shape == (2,)
    
    # ACF: (Features, Lags+1) -> (2, 21) since default lags=20
    assert metrics['real_acf_ret'].shape == (2, 21)

def test_calculate_stylized_facts_logic():
    # If real == fake, diffs should be 0
    data = np.random.randn(5, 10, 1)
    metrics = calculate_stylized_facts(data, data)
    
    assert np.allclose(metrics['kurtosis_diff'], 0)
    assert np.allclose(metrics['skew_diff'], 0)
    assert np.isclose(metrics['corr_matrix_diff_norm'], 0)
    assert np.isclose(metrics['acf_ret_diff'], 0)
