from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import numpy as np
from scipy import stats

class DistributionTest(ABC):
    """
    Abstract base class for 2-sample distribution tests.
    """
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def run(self, sample1: np.ndarray, sample2: np.ndarray) -> Dict[str, float]:
        """
        Runs the statistical test comparing two samples.
        
        Args:
            sample1: First sample (e.g., Real Returns)
            sample2: Second sample (e.g., Synthetic Returns)
            
        Returns:
            Dict with 'statistic' and 'p_value' (if applicable).
        """
        pass

class KSTest(DistributionTest):
    """
    Kolmogorov-Smirnov test for 2 samples.
    Tests the null hypothesis that 2 independent samples are drawn from the same continuous distribution.
    """
    @property
    def name(self):
        return "Kolmogorov-Smirnov"

    def run(self, sample1, sample2):
        stat, p_val = stats.ks_2samp(sample1, sample2)
        return {"statistic": stat, "p_value": p_val}

class CvMTest(DistributionTest):
    """
    Cramér-von Mises test for 2 samples.
    Generally more powerful than KS for checking if distributions are the same.
    """
    @property
    def name(self):
        return "Cramer-von Mises"

    def run(self, sample1, sample2):
        res = stats.cramervonmises_2samp(sample1, sample2)
        return {"statistic": res.statistic, "p_value": res.pvalue}

class ADTest(DistributionTest):
    """
    Anderson-Darling test for k-samples (here k=2).
    Tests the null hypothesis that k-samples are drawn from the same population without specifying the distribution.
    """
    @property
    def name(self):
        return "Anderson-Darling"

    def run(self, sample1, sample2):
        # anderson_ksamp requires samples to be list of arrays
        # It returns statistic, critical_values, significance_level
        res = stats.anderson_ksamp([sample1, sample2])
        # significance_level is the approximate p-value. 
        # Note: anderson_ksamp caps p-values at 25% (0.25) usually.
        return {"statistic": res.statistic, "p_value": res.significance_level}

def run_all_distribution_tests(real_returns: np.ndarray, synthetic_returns: np.ndarray) -> Dict[str, Dict[str, float]]:
    """
    Runs a suite of distribution tests on flattened return arrays.
    
    Args:
        real_returns: Array of real log-returns (can be multi-dimensional, will be flattened)
        synthetic_returns: Array of synthetic log-returns
        
    Returns:
        Dictionary mapping test name to results.
    """
    # Flatten samples to 1D
    r_flat = real_returns.ravel()
    s_flat = synthetic_returns.ravel()
    
    tests = [KSTest(), CvMTest(), ADTest()]
    results = {}
    
    for test in tests:
        try:
            results[test.name] = test.run(r_flat, s_flat)
        except Exception as e:
            print(f"Test {test.name} failed: {e}")
            results[test.name] = {"error": str(e)}
            
    return results
