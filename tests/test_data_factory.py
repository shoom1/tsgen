import pytest
import pandas as pd
import numpy as np
from tsgen.data.pipeline import (
    load_prices, clean_data, split_temporal,
    process_prices, create_windows, create_dataloader
)
from tsgen.data.processor import LogReturnProcessor


def test_load_prices():
    """Test loading price data."""
    df = load_prices(['AAPL', 'MSFT'], '2024-01-01', '2024-12-31')

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert len(df.columns) > 0


def test_clean_data_ffill_drop():
    """Test data cleaning with ffill_drop strategy."""
    # Create sample data with NaNs
    df = pd.DataFrame({
        'A': [1.0, np.nan, 3.0, 4.0],
        'B': [1.0, 2.0, np.nan, 4.0]
    })

    df_clean = clean_data(df, strategy='ffill_drop')

    assert not df_clean.isnull().any().any()
    assert len(df_clean) <= len(df)


def test_split_temporal():
    """Test temporal splitting."""
    df = pd.DataFrame({
        'A': range(100),
        'B': range(100, 200)
    })

    train_df, test_df = split_temporal(df, train_ratio=0.8)

    assert len(train_df) == 80
    assert len(test_df) == 20
    assert train_df.index[-1] < test_df.index[0]  # Temporal order preserved


def test_process_prices():
    """Test price processing."""
    df = pd.DataFrame({
        'A': [100, 101, 102, 103],
        'B': [200, 202, 204, 206]
    })

    processor = LogReturnProcessor()
    data_scaled = process_prices(df, processor, fit=True)

    assert isinstance(data_scaled, np.ndarray)
    assert data_scaled.shape[1] == 2  # 2 features


def test_create_windows():
    """Test window creation."""
    data = np.random.randn(100, 2)
    windows = create_windows(data, sequence_length=10, stride=1)

    assert windows.shape == (91, 10, 2)  # (100-10+1, 10, 2)


def test_create_dataloader():
    """Test DataLoader creation."""
    data = np.random.randn(50, 10, 2)
    loader = create_dataloader(data, batch_size=8, shuffle=True)

    assert loader is not None
    batch = next(iter(loader))
    assert len(batch) == 1  # TensorDataset returns tuple
    assert batch[0].shape[0] <= 8  # Batch size


def test_full_pipeline():
    """Test complete pipeline."""
    # Load
    df = load_prices(['AAPL'], '2024-01-01', '2024-12-31')

    # Clean
    df_clean = clean_data(df)

    # Split
    train_df, test_df = split_temporal(df_clean, train_ratio=0.8)

    # Process
    processor = LogReturnProcessor()
    train_scaled = process_prices(train_df, processor, fit=True)
    test_scaled = process_prices(test_df, processor, fit=False)

    # Windows
    train_windows = create_windows(train_scaled, sequence_length=10)
    test_windows = create_windows(test_scaled, sequence_length=10)

    # DataLoader
    train_loader = create_dataloader(train_windows, batch_size=16)

    # Verify
    assert train_loader is not None
    assert len(train_windows) > 0
    assert len(test_windows) > 0
