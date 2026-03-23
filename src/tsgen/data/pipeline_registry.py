"""
Pipeline registry with metadata for all available pipeline steps.

Maps step names to functions and their type/parameter specifications.
Used by DataPipeline.from_config() to build pipelines from YAML.
"""

import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from .pipeline import (
    load_prices,
    clean_data,
    split_temporal,
    process_prices,
    create_windows,
    create_dataloader
)


# Registry mapping step names to (function, metadata) tuples
# Metadata includes:
#   - input_type: Expected input type (None for first step)
#   - output_type: Output type for next step
#   - required_params: Parameters that must be provided
#   - optional_params: Parameters that have defaults
PIPELINE_REGISTRY = {
    'load_prices': (
        load_prices,
        {
            'input_type': None,  # First step, no input
            'output_type': pd.DataFrame,
            'required_params': ['tickers', 'start_date', 'end_date'],
            'optional_params': ['column', 'db_path']
        }
    ),

    'clean_data': (
        clean_data,
        {
            'input_type': pd.DataFrame,
            'output_type': pd.DataFrame,
            'required_params': [],
            'optional_params': ['strategy']
        }
    ),

    'split_temporal': (
        split_temporal,
        {
            'input_type': pd.DataFrame,
            'output_type': tuple,  # Returns (train_df, test_df)
            'required_params': [],
            'optional_params': ['train_ratio']
        }
    ),

    'process_prices': (
        process_prices,
        {
            'input_type': pd.DataFrame,
            'output_type': np.ndarray,
            'required_params': ['processor'],  # LogReturnProcessor instance
            'optional_params': ['fit']
        }
    ),

    'create_windows': (
        create_windows,
        {
            'input_type': np.ndarray,
            'output_type': np.ndarray,
            'required_params': ['sequence_length'],
            'optional_params': ['stride']
        }
    ),

    'create_dataloader': (
        create_dataloader,
        {
            'input_type': np.ndarray,
            'output_type': DataLoader,
            'required_params': [],
            'optional_params': ['batch_size', 'shuffle']
        }
    )
}


def get_available_steps() -> list:
    """Return list of available pipeline step names.

    Returns:
        Sorted list of step names

    Example:
        >>> steps = get_available_steps()
        >>> print(steps)
        ['clean_data', 'create_dataloader', 'create_windows', 'load_prices', 'process_prices', 'split_temporal']
    """
    return sorted(PIPELINE_REGISTRY.keys())


def get_step_info(step_name: str) -> dict:
    """Get metadata for a specific pipeline step.

    Args:
        step_name: Name of the step

    Returns:
        Dictionary with function and metadata

    Raises:
        KeyError: If step name not found

    Example:
        >>> info = get_step_info('load_prices')
        >>> print(info['required_params'])
        ['tickers', 'start_date', 'end_date']
    """
    if step_name not in PIPELINE_REGISTRY:
        available = ', '.join(get_available_steps())
        raise KeyError(
            f"Unknown pipeline step: '{step_name}'\n"
            f"Available steps: {available}"
        )

    fn, metadata = PIPELINE_REGISTRY[step_name]
    return {
        'function': fn,
        **metadata
    }
