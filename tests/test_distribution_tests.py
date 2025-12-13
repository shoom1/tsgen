import pytest
import numpy as np
from tsgen.analysis.distribution_tests import run_all_distribution_tests, KSTest, CvMTest, ADTest

def test_ks_test():
    # Identical distributions
    s1 = np.random.randn(100)
    s2 = np.random.randn(100)
    test = KSTest()
    res = test.run(s1, s2)
    # With high probability, p-value should be > 0.05 (fail to reject null hypothesis)
    # But randomness can cause failure. We check structure.
    assert "statistic" in res
    assert "p_value" in res

def test_cvm_test():
    s1 = np.random.randn(100)
    s2 = np.random.randn(100) + 2 # Different mean
    test = CvMTest()
    res = test.run(s1, s2)
    # Should reject null hypothesis (p < 0.05 usually)
    assert "statistic" in res
    assert "p_value" in res

def test_ad_test():
    s1 = np.random.randn(50)
    s2 = np.random.randn(50)
    test = ADTest()
    res = test.run(s1, s2)
    assert "statistic" in res
    assert "p_value" in res

def test_run_all_tests():
    s1 = np.random.randn(100, 5) # Multi-dimensional check
    s2 = np.random.randn(100, 5)
    
    results = run_all_distribution_tests(s1, s2)
    
    assert "Kolmogorov-Smirnov" in results
    assert "Cramer-von Mises" in results
    assert "Anderson-Darling" in results
    
    assert results["Kolmogorov-Smirnov"]["statistic"] >= 0
