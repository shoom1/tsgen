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

def _to_feature_samples(data: np.ndarray) -> np.ndarray:
    """Return a 2D sample matrix (observations, features).

    For windowed data (N, L, F), take every L-th window before flattening.
    This avoids counting the same return in dozens of overlapping stride-1
    windows. The resulting p-values are still screening diagnostics for
    dependent financial returns, but the statistics are much less inflated
    than the old all-window/all-feature flattening.
    """
    arr = np.asarray(data)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        step = max(int(arr.shape[1]), 1)
        reduced = arr[::step]
        return reduced.reshape(-1, arr.shape[-1])
    raise ValueError(f"Expected 1D, 2D, or 3D returns, got shape {arr.shape}")


def _finite_pair(sample1: np.ndarray, sample2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Drop NaN/Inf independently from each sample."""
    s1 = sample1[np.isfinite(sample1)]
    s2 = sample2[np.isfinite(sample2)]
    return s1, s2


def run_all_distribution_tests(real_returns: np.ndarray, synthetic_returns: np.ndarray) -> Dict[str, Dict[str, float]]:
    """
    Runs a suite of distribution tests per feature.

    For 3D windowed inputs, non-overlapping windows are sampled before
    flattening each feature. Results aggregate per-feature test statistics
    instead of mixing all assets into one distribution.
    
    Args:
        real_returns: Array of real log-returns (can be multi-dimensional, will be flattened)
        synthetic_returns: Array of synthetic log-returns
        
    Returns:
        Dictionary mapping test name to results.
    """
    real_samples = _to_feature_samples(real_returns)
    synthetic_samples = _to_feature_samples(synthetic_returns)
    n_features = min(real_samples.shape[1], synthetic_samples.shape[1])
    
    tests = [KSTest(), CvMTest(), ADTest()]
    results = {}
    
    for test in tests:
        stats_out = []
        p_values = []
        sample_sizes = []

        for feature_i in range(n_features):
            r_feature, s_feature = _finite_pair(
                real_samples[:, feature_i],
                synthetic_samples[:, feature_i],
            )
            if len(r_feature) < 2 or len(s_feature) < 2:
                continue

            try:
                res = test.run(r_feature, s_feature)
            except Exception as e:
                print(f"Test {test.name} failed for feature {feature_i}: {e}")
                continue

            if 'statistic' in res and np.isfinite(res['statistic']):
                stats_out.append(float(res['statistic']))
            if 'p_value' in res and np.isfinite(res['p_value']):
                p_values.append(float(res['p_value']))
            sample_sizes.append(min(len(r_feature), len(s_feature)))

        if not stats_out:
            results[test.name] = {
                "error": "No feature had enough finite observations for the test.",
                "n_features": n_features,
            }
            continue

        result = {
            "statistic": float(np.mean(stats_out)),
            "statistic_max": float(np.max(stats_out)),
            "n_features": int(len(stats_out)),
            "sample_size_min": int(np.min(sample_sizes)),
            "sample_method": "per_feature_non_overlapping_windows",
        }
        if p_values:
            result["p_value"] = float(np.median(p_values))
            result["p_value_min"] = float(np.min(p_values))
        results[test.name] = result
            
    return results
