"""
Analysis and evaluation metrics for synthetic time series.

This module provides:
- Stylized facts computation (kurtosis, skewness, volatility clustering, etc.)
- Distribution tests (KS, Cramér-von Mises, Anderson-Darling)
- TSTR (Train on Synthetic, Test on Real) evaluation
"""

from tsgen.analysis.metrics import calculate_stylized_facts, plot_stylized_facts
from tsgen.analysis.distribution_tests import (
    DistributionTest,
    KSTest,
    CvMTest,
    ADTest,
)

__all__ = [
    # Stylized facts
    "calculate_stylized_facts",
    "plot_stylized_facts",
    # Distribution tests
    "DistributionTest",
    "KSTest",
    "CvMTest",
    "ADTest",
]
