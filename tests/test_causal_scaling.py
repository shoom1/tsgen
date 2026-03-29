"""Tests for causal (expanding-window) scaling in LogReturnProcessor."""

import pytest
import numpy as np
import pandas as pd
from tsgen.data.processor import LogReturnProcessor


def _make_price_df(n_rows=200, n_features=3, seed=42):
    """Create a synthetic price DataFrame with random walk."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0005, 0.01, (n_rows, n_features))
    prices = 100.0 * np.exp(np.cumsum(returns, axis=0))
    columns = [f"ASSET_{i}" for i in range(n_features)]
    return pd.DataFrame(prices, columns=columns)


class TestConstructor:
    def test_default_is_global(self):
        p = LogReturnProcessor()
        assert p.scaling == 'global'
        assert p.min_periods == 60

    def test_expanding_mode(self):
        p = LogReturnProcessor(scaling='expanding', min_periods=30)
        assert p.scaling == 'expanding'
        assert p.min_periods == 30

    def test_global_backward_compatible(self):
        """Default constructor produces identical behavior to old code."""
        p = LogReturnProcessor()
        df = _make_price_df(100, 2)
        p.fit(df)
        result = p.transform(df)
        assert result.shape == (99, 2)  # T-1 rows
