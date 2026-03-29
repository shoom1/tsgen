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


class TestExpandingTransform:
    def test_output_shorter_by_min_periods(self):
        df = _make_price_df(100, 2)
        p = LogReturnProcessor(scaling='expanding', min_periods=10)
        p.fit(df)
        result = p.transform(df)
        # Global would be 99 rows; expanding drops first 10
        assert result.shape == (89, 2)

    def test_output_is_causal(self):
        """Each z-score uses only past data — verify first valid row."""
        df = _make_price_df(100, 2)
        p = LogReturnProcessor(scaling='expanding', min_periods=10)
        p.fit(df)
        result = p.transform(df)

        # Manually compute what the first output row should be
        log_returns = np.log(df.values[1:] / df.values[:-1])
        # Row index 10 in log_returns (first row after burn-in)
        r = log_returns[10]
        mu = np.mean(log_returns[:11], axis=0)  # mean of rows 0..10
        sigma = np.std(log_returns[:11], axis=0, ddof=0)  # population std to match StandardScaler
        expected = (r - mu) / sigma
        np.testing.assert_allclose(result[0], expected, atol=1e-10)

    def test_test_data_falls_back_to_global(self):
        """transform() on test data (after fit) uses converged stats."""
        df = _make_price_df(200, 2)
        train_df = df.iloc[:160]
        test_df = df.iloc[160:]

        p = LogReturnProcessor(scaling='expanding', min_periods=10)
        p.fit(train_df)
        train_result = p.transform(train_df)  # Uses expanding arrays

        # Now transform test data — should use converged global stats
        test_result = p.transform(test_df)
        assert test_result.shape == (39, 2)  # 40 rows - 1 for log return

        # Verify it matches what global scaling would produce
        log_returns_test = np.log(test_df.values[1:] / test_df.values[:-1])
        expected = (log_returns_test - p.scaler.mean_) / p.scaler.scale_
        np.testing.assert_allclose(test_result, expected, atol=1e-10)

    def test_flag_reset_after_transform(self):
        """_fitted_expanding flag resets after training transform."""
        df = _make_price_df(100, 2)
        p = LogReturnProcessor(scaling='expanding', min_periods=10)
        p.fit(df)
        assert p._fitted_expanding is True
        p.transform(df)
        assert p._fitted_expanding is False

    def test_global_mode_unchanged(self):
        """Global transform produces same output as before."""
        df = _make_price_df(100, 2)

        p_old = LogReturnProcessor()
        p_old.fit(df)
        result_old = p_old.transform(df)

        p_new = LogReturnProcessor(scaling='global')
        p_new.fit(df)
        result_new = p_new.transform(df)

        np.testing.assert_array_equal(result_old, result_new)


class TestEdgeCases:
    def test_single_feature(self):
        df = _make_price_df(100, 1)
        p = LogReturnProcessor(scaling='expanding', min_periods=10)
        p.fit(df)
        result = p.transform(df)
        assert result.shape == (89, 1)
        assert not np.any(np.isnan(result))

    def test_min_periods_equals_data_length_raises(self):
        df = _make_price_df(50, 2)  # 49 returns
        p = LogReturnProcessor(scaling='expanding', min_periods=49)
        with pytest.raises(ValueError, match="min_periods"):
            p.fit(df)

    def test_masked_expanding_output_shape(self):
        """Masked expanding transform should trim by min_periods."""
        df = _make_price_df(100, 2)
        mask = pd.DataFrame(np.ones_like(df.values), columns=df.columns)
        p = LogReturnProcessor(scaling='expanding', min_periods=10)
        p.fit(df, mask=mask)
        result, mask_out = p.transform(df, mask=mask)
        assert result.shape == (89, 2)  # 99 returns - 10 burn-in
        assert mask_out.shape == (89, 2)

    def test_masked_expanding_all_nan_column(self):
        """Feature with all NaN should fall back to mean=0, std=1."""
        df = pd.DataFrame({
            'A': np.exp(np.cumsum(np.random.default_rng(42).normal(0, 0.01, 100))),
            'B': np.exp(np.cumsum(np.random.default_rng(43).normal(0, 0.01, 100))),
        })
        mask = pd.DataFrame({
            'A': np.ones(100),
            'B': np.zeros(100),  # All masked
        })
        p = LogReturnProcessor(scaling='expanding', min_periods=10)
        p.fit(df, mask=mask)
        result, mask_out = p.transform(df, mask=mask)

        # Column B should be all zeros (masked out) — no NaN/Inf
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))


class TestSerialization:
    def test_save_load_preserves_expanding_mode(self, tmp_path):
        df = _make_price_df(100, 2)
        p = LogReturnProcessor(scaling='expanding', min_periods=10)
        p.fit(df)
        train_result = p.transform(df)

        # Save and load
        path = tmp_path / "processor.pkl"
        p.save(str(path))
        p2 = LogReturnProcessor.load(str(path))

        assert p2.scaling == 'expanding'
        assert p2.min_periods == 10
        np.testing.assert_array_equal(p2.expanding_means_, p.expanding_means_)
        np.testing.assert_array_equal(p2.expanding_stds_, p.expanding_stds_)

        # Loaded processor should use global fallback for new data
        test_df = _make_price_df(50, 2, seed=99)
        result = p2.transform(test_df)
        assert result.shape == (49, 2)


class TestRoundTrip:
    def test_global_exact_roundtrip(self):
        df = _make_price_df(100, 2)
        p = LogReturnProcessor(scaling='global')
        p.fit(df)
        scaled = p.transform(df)
        initial = df.values[0]
        prices = p.inverse_transform(scaled, initial)
        # Should reconstruct original prices closely
        np.testing.assert_allclose(prices[0, :, :], df.values[:], rtol=1e-10)

    def test_expanding_approximate_roundtrip(self):
        """Expanding round-trip is approximate; test shape and sanity."""
        df = _make_price_df(300, 2)
        p = LogReturnProcessor(scaling='expanding', min_periods=10)
        p.fit(df)
        scaled = p.transform(df)

        # Use the final converged inverse transform
        initial = df.values[0]
        prices = p.inverse_transform(scaled, initial)

        assert prices.shape[1] == scaled.shape[0] + 1  # seq + initial
        assert not np.any(np.isnan(prices))
        assert not np.any(np.isinf(prices))

        # All reconstructed prices should be positive (log-return model invariant)
        assert np.all(prices > 0)

        # Prices should be in the same order of magnitude as the original
        # (within 10x, not exact due to expanding stats differing from converged)
        actual_last = df.values[-1]
        reconstructed_last = prices[0, -1, :]
        ratio = reconstructed_last / actual_last
        assert np.all(ratio > 0.1) and np.all(ratio < 10.0)
