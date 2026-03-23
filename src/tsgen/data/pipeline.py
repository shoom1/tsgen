"""
Data pipeline functions for loading, cleaning, and processing financial time series.

This module provides composable functions that can be combined to create flexible
data processing pipelines. Each function has a single responsibility and can be
used independently or chained together.

Example usage:
    from tsgen.data.pipeline import (
        load_prices, clean_data, split_temporal,
        process_prices, create_windows, create_dataloader
    )
    from tsgen.data.processor import LogReturnProcessor

    # Load and clean
    df = load_prices(['AAPL', 'MSFT'], '2020-01-01', '2024-12-31')
    df_clean = clean_data(df, strategy='ffill_drop')

    # Split
    train_df, test_df = split_temporal(df_clean, train_ratio=0.8)

    # Process
    processor = LogReturnProcessor()
    train_scaled = process_prices(train_df, processor, fit=True)
    test_scaled = process_prices(test_df, processor, fit=False)

    # Windows
    train_sequences = create_windows(train_scaled, sequence_length=64)

    # DataLoader (optional)
    train_loader = create_dataloader(train_sequences, batch_size=32, shuffle=True)
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

# Import DataClient from findata project
try:
    from findata import DataClient
except ImportError as e:
    raise ImportError(
        f"Could not import findata.DataClient: {e}. "
        "Please ensure findata is installed. "
        "Run: cd ../findata && pip install -e ."
    )


def load_prices(tickers, start_date, end_date, column='adj_close', db_path=None):
    """
    Load price data from database.

    Args:
        tickers: List of ticker symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        column: Price column to load (default: 'adj_close')
            Options: 'open', 'high', 'low', 'close', 'adj_close', 'volume'
        db_path: Optional database path (auto-detects via ~/.findatarc if None)

    Returns:
        pd.DataFrame: Wide format DataFrame with:
            - Index: date (DatetimeIndex)
            - Columns: ticker symbols
            - Values: prices

    Raises:
        ValueError: If no data found for the given tickers and date range
    """
    print(f"Loading data from database: {tickers}...")

    # Initialize DataClient
    client = DataClient(db_path=db_path) if db_path else DataClient()

    # Get data in wide format
    data = client.get_data(
        symbols=tickers,
        start=start_date,
        end=end_date,
        columns=[column],
        format='wide'
    )

    # Handle MultiIndex if present (date, symbol) → pivot to (date x symbol)
    if isinstance(data.index, pd.MultiIndex):
        data = data.reset_index()
        data = data.pivot(index='date', columns='symbol', values=column)

        # Reorder columns to match requested tickers
        available_symbols = [s for s in tickers if s in data.columns]
        if len(available_symbols) < len(tickers):
            missing = set(tickers) - set(available_symbols)
            print(f"Warning: {len(missing)} symbols missing data: {list(missing)[:10]}...")
        data = data[available_symbols]

    # Validate
    if data.empty:
        raise ValueError(
            f"No data found for {tickers} between {start_date} and {end_date}. "
            f"Please use the findata project to load data into the database."
        )

    print(f"Loaded {len(data)} rows for {len(data.columns)} symbols")
    return data


def clean_data(df, strategy='ffill_drop', **kwargs):
    """
    Clean data by handling NaN values.

    Args:
        df: DataFrame with potential NaN values
        strategy: Cleaning strategy:
            - 'ffill_drop': Forward fill, then drop remaining NaNs (default)
            - 'drop_all': Drop any row with NaN in any column
            - 'interpolate': Interpolate missing values
            - 'fillna': Fill with constant value
            - 'none': No cleaning (returns as-is)
            - 'mask': Return (data, mask) tuple for masked training
        **kwargs: Strategy-specific parameters:
            - For 'interpolate': method='linear', limit=None
            - For 'fillna': value=0

    Returns:
        pd.DataFrame: Cleaned data (for most strategies)
        tuple[pd.DataFrame, pd.DataFrame]: (data, mask) for 'mask' strategy
            - data: DataFrame with NaN replaced by 0
            - mask: DataFrame with 1=valid, 0=missing

    Raises:
        ValueError: If unknown strategy provided
    """
    if strategy == 'ffill_drop':
        result = df.ffill().dropna()
    elif strategy == 'drop_all':
        result = df.dropna()
    elif strategy == 'interpolate':
        method = kwargs.get('method', 'linear')
        limit = kwargs.get('limit', None)
        result = df.interpolate(method=method, limit=limit).dropna()
    elif strategy == 'fillna':
        value = kwargs.get('value', 0)
        result = df.fillna(value)
    elif strategy == 'none':
        result = df
    elif strategy == 'mask':
        # Create binary mask: 1 = valid, 0 = missing
        mask = (~df.isna()).astype(float)
        # Replace NaN with 0 in data
        data = df.fillna(0.0)
        print(f"Masking strategy: {mask.sum().sum():.0f} valid values, "
              f"{(~mask.astype(bool)).sum().sum():.0f} masked values")
        return data, mask
    else:
        raise ValueError(
            f"Unknown cleaning strategy: {strategy}. "
            f"Available: 'ffill_drop', 'drop_all', 'interpolate', 'fillna', 'none', 'mask'"
        )

    rows_removed = len(df) - len(result)
    if rows_removed > 0:
        print(f"Cleaning removed {rows_removed} rows ({rows_removed/len(df)*100:.1f}%)")

    return result


def split_temporal(df, train_ratio=0.8):
    """
    Split data temporally (chronologically).

    This preserves time series order by splitting based on position in the index,
    not randomly. The first train_ratio of data is training, the rest is test.

    Args:
        df: DataFrame to split
        train_ratio: Fraction for training (e.g., 0.8 = first 80% for train)

    Returns:
        tuple: (train_df, test_df)
            - train_df: First train_ratio of data
            - test_df: Remaining (1 - train_ratio) of data

    Raises:
        ValueError: If train_ratio not in (0, 1)
    """
    if not 0 < train_ratio < 1:
        raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")

    split_idx = int(len(df) * train_ratio)

    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    print(f"Temporal split: Train={len(train_df)} ({train_df.index[0]} to {train_df.index[-1]}), "
          f"Test={len(test_df)} ({test_df.index[0]} to {test_df.index[-1]})")

    return train_df, test_df


def process_prices(df, processor, fit=True, mask=None):
    """
    Process prices through a processor (e.g., LogReturnProcessor).

    Args:
        df: Price DataFrame (wide format)
        processor: Processor instance (e.g., LogReturnProcessor)
            Must have fit() and transform() methods
        fit: Whether to fit the processor on this data (default: True)
            Set to False when processing test data with a fitted processor
        mask: Optional mask DataFrame (1=valid, 0=missing)
            When provided, processor fits/transforms with masking

    Returns:
        np.ndarray: Processed data (e.g., scaled returns)
            Shape: (time_steps, features)
        tuple[np.ndarray, np.ndarray]: (data, mask) if mask provided

    Example:
        from tsgen.data.processor import LogReturnProcessor

        processor = LogReturnProcessor()
        train_scaled = process_prices(train_df, processor, fit=True)
        test_scaled = process_prices(test_df, processor, fit=False)

        # With mask:
        train_scaled, train_mask = process_prices(train_df, processor, fit=True, mask=price_mask)
    """
    if fit:
        print(f"Fitting processor on {len(df)} price points...")
        if mask is not None:
            processor.fit(df, mask=mask)
        else:
            processor.fit(df)

    if mask is not None:
        data_scaled, mask_out = processor.transform(df, mask=mask)

        # Validate data (only check valid positions)
        valid_data = data_scaled[mask_out.astype(bool)]
        if np.isnan(valid_data).any() or np.isinf(valid_data).any():
            raise ValueError("NaN or Inf values detected in valid positions after processing")

        return data_scaled, mask_out
    else:
        data_scaled = processor.transform(df)

        # Validate
        if np.isnan(data_scaled).any() or np.isinf(data_scaled).any():
            raise ValueError("NaN or Inf values detected after processing")

        return data_scaled


def create_windows(data, sequence_length, stride=1, mask=None):
    """
    Create sliding windows from time series data.

    Args:
        data: np.ndarray with shape (time_steps, features)
        sequence_length: Window size
        stride: Step size for sliding window (default: 1)
            - stride=1: Overlapping windows (each step creates new window)
            - stride=sequence_length: Non-overlapping windows
        mask: Optional np.ndarray with shape (time_steps, features)
            Binary mask where 1=valid, 0=missing

    Returns:
        np.ndarray: Windows with shape (num_windows, sequence_length, features)
        tuple[np.ndarray, np.ndarray]: (data_windows, mask_windows) if mask provided

    Raises:
        ValueError: If data too short for sequence_length
    """
    if len(data) < sequence_length:
        raise ValueError(
            f"Data length ({len(data)}) is less than sequence_length ({sequence_length}). "
            f"Need at least {sequence_length} data points."
        )

    sequences = []
    for i in range(0, len(data) - sequence_length + 1, stride):
        sequences.append(data[i : i + sequence_length])

    result = np.array(sequences)
    print(f"Created {len(result)} windows of length {sequence_length} (stride={stride})")

    if mask is not None:
        mask_sequences = []
        for i in range(0, len(mask) - sequence_length + 1, stride):
            mask_sequences.append(mask[i : i + sequence_length])
        mask_result = np.array(mask_sequences)
        return result, mask_result

    return result


def to_tensor(data):
    """
    Convert numpy array to PyTorch tensor.

    Args:
        data: np.ndarray

    Returns:
        torch.Tensor: FloatTensor of same shape
    """
    return torch.FloatTensor(data)


def create_dataloader(data, batch_size, shuffle=True, mask=None):
    """
    Create PyTorch DataLoader from data.

    This is a convenience function for creating a standard DataLoader.
    For more control, use torch.utils.data.DataLoader directly.

    Args:
        data: np.ndarray or torch.Tensor
            If numpy array, will be converted to tensor
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle data (default: True)
            Set to False for test/validation data
        mask: Optional np.ndarray or torch.Tensor
            Binary mask where 1=valid, 0=missing
            If provided, DataLoader yields (data, mask) tuples

    Returns:
        DataLoader: PyTorch DataLoader yielding batches of tensors
            - Without mask: yields (data,) tuples
            - With mask: yields (data, mask) tuples

    Example:
        train_loader = create_dataloader(train_sequences, batch_size=32, shuffle=True)
        test_loader = create_dataloader(test_sequences, batch_size=32, shuffle=False)

        for batch in train_loader:
            # batch is a list with one tensor: [sequences]
            sequences = batch[0]  # Shape: (batch_size, seq_len, features)

        # With mask:
        train_loader = create_dataloader(train_sequences, batch_size=32, mask=train_mask)
        for data, mask in train_loader:
            # data: (batch_size, seq_len, features)
            # mask: (batch_size, seq_len, features)
    """
    # Convert to tensor if needed
    if isinstance(data, np.ndarray):
        data = to_tensor(data)

    if mask is not None:
        if isinstance(mask, np.ndarray):
            mask = to_tensor(mask)
        dataset = TensorDataset(data, mask)
        print(f"Created DataLoader with masks: {len(dataset)} samples, batch_size={batch_size}, shuffle={shuffle}")
    else:
        dataset = TensorDataset(data)
        print(f"Created DataLoader: {len(dataset)} samples, batch_size={batch_size}, shuffle={shuffle}")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return loader
