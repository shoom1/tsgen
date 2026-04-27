"""
Data processing and loading utilities.

This module provides composable pipeline functions for:
- Loading data from finbase.DataClient
- Cleaning and preprocessing (NaN handling, scaling)
- Creating sliding windows
- Converting to PyTorch DataLoaders

Example (Manual Pipeline):
    from tsgen.data.pipeline import (
        load_prices, clean_data, process_prices,
        create_windows, create_dataloader
    )
    from tsgen.data.processor import LogReturnProcessor

    df = load_prices(['AAPL'], '2020-01-01', '2024-12-31')
    df_clean = clean_data(df)
    processor = LogReturnProcessor()
    data_scaled = process_prices(df_clean, processor, fit=True)
    sequences = create_windows(data_scaled, sequence_length=64)
    loader = create_dataloader(sequences, batch_size=32)

Example (YAML-Configured Pipeline):
    from tsgen.data.pipeline_builder import DataPipeline
    from tsgen.data.processor import LogReturnProcessor

    config = {
        'pipeline': {
            'steps': [
                {'load_prices': {'column': 'adj_close'}},
                {'clean_data': {'strategy': 'ffill_drop'}},
                {'create_windows': {'sequence_length': 64}},
                {'create_dataloader': {'batch_size': 32}}
            ]
        }
    }

    pipeline = DataPipeline.from_config(config)
    loader = pipeline.execute(
        tickers=['AAPL'],
        start_date='2020-01-01',
        end_date='2024-12-31',
        processor=LogReturnProcessor()
    )
"""

from tsgen.data.processor import DataProcessor, LogReturnProcessor
from tsgen.data.pipeline import (
    load_prices,
    clean_data,
    split_temporal,
    process_prices,
    create_windows,
    to_tensor,
    create_dataloader,
)
from tsgen.data.pipeline_builder import DataPipeline, PipelineStep
from tsgen.data.pipeline_registry import PIPELINE_REGISTRY, get_available_steps, get_step_info

__all__ = [
    # Processors
    "DataProcessor",
    "LogReturnProcessor",
    # Pipeline functions
    "load_prices",
    "clean_data",
    "split_temporal",
    "process_prices",
    "create_windows",
    "to_tensor",
    "create_dataloader",
    # Pipeline builder
    "DataPipeline",
    "PipelineStep",
    "PIPELINE_REGISTRY",
    "get_available_steps",
    "get_step_info",
]
