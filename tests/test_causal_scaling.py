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


class TestExpandingFit:
    def test_fit_stores_expanding_arrays(self):
        p = LogReturnProcessor(scaling='expanding', min_periods=10)
        df = _make_price_df(100, 2)
        p.fit(df)

        assert hasattr(p, 'expanding_means_')
        assert hasattr(p, 'expanding_stds_')
        assert p.expanding_means_.shape == (99, 2)  # T-1 rows, 2 features
        assert p.expanding_stds_.shape == (99, 2)

    def test_fit_sets_scaler_to_final_stats(self):
        p = LogReturnProcessor(scaling='expanding', min_periods=10)
        df = _make_price_df(100, 2)
        p.fit(df)

        np.testing.assert_array_almost_equal(
            p.scaler.mean_, p.expanding_means_[-1]
        )
        np.testing.assert_array_almost_equal(
            p.scaler.scale_, p.expanding_stds_[-1]
        )

    def test_expanding_stats_converge_to_global(self):
        """For stationary data, final expanding stats ≈ global stats."""
        df = _make_price_df(500, 2)

        p_global = LogReturnProcessor(scaling='global')
        p_global.fit(df)

        p_expand = LogReturnProcessor(scaling='expanding', min_periods=10)
        p_expand.fit(df)

        np.testing.assert_allclose(
            p_expand.scaler.mean_, p_global.scaler.mean_, atol=1e-6
        )
        np.testing.assert_allclose(
            p_expand.scaler.scale_, p_global.scaler.scale_, atol=1e-3
        )

    def test_fit_raises_if_min_periods_too_large(self):
        p = LogReturnProcessor(scaling='expanding', min_periods=200)
        df = _make_price_df(100, 2)  # Only 99 returns
        with pytest.raises(ValueError, match="min_periods"):
            p.fit(df)

    def test_fit_sets_expanding_flag(self):
        p = LogReturnProcessor(scaling='expanding', min_periods=10)
        df = _make_price_df(100, 2)
        p.fit(df)
        assert p._fitted_expanding is True

    def test_global_fit_unchanged(self):
        """Global mode should not create expanding arrays."""
        p = LogReturnProcessor(scaling='global')
        df = _make_price_df(100, 2)
        p.fit(df)
        assert not hasattr(p, 'expanding_means_') or p.expanding_means_ is None
