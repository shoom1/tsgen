"""
tsgen: Synthetic Financial Time Series Generation using Diffusion Models

A research framework for generating realistic synthetic financial time series
using Denoising Diffusion Probabilistic Models (DDPM).
"""

__version__ = "0.1.0"

# Import key functions for easy access
from tsgen.train import train_model
from tsgen.evaluate import evaluate_model

__all__ = [
    "train_model",
    "evaluate_model",
    "__version__",
]
