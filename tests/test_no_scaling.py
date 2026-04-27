"""Tests for scaling='none' mode in LogReturnProcessor (raw log-returns)."""

import numpy as np
import pandas as pd
import pytest

from tsgen.data.processor import LogReturnProcessor


def _make_price_df(n=200, n_features=3, seed=0):
    rng = np.random.default_rng(seed)
    returns = rng.normal(0, 0.02, size=(n, n_features))
    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    cols = [f"T{i}" for i in range(n_features)]
    return pd.DataFrame(prices, index=dates, columns=cols)


class TestNoScaling:
    def test_scaling_none_stored_on_instance(self):
        p = LogReturnProcessor(scaling='none')
        assert p.scaling == 'none'

    def test_transform_returns_raw_log_returns(self):
        df = _make_price_df()
        p = LogReturnProcessor(scaling='none')
        p.fit(df)
        out = p.transform(df)

        expected = np.log(df / df.shift(1)).dropna().values
        np.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-12)

    def test_fit_does_not_rescale(self):
        df = _make_price_df()
        p = LogReturnProcessor(scaling='none')
        p.fit(df)
        # Scaler should be identity: mean=0, scale=1
        np.testing.assert_array_equal(p.scaler.mean_, np.zeros(df.shape[1]))
        np.testing.assert_array_equal(p.scaler.scale_, np.ones(df.shape[1]))

    def test_roundtrip_recovers_prices(self):
        df = _make_price_df()
        p = LogReturnProcessor(scaling='none')
        p.fit(df)
        returns = p.transform(df)
        initial = df.iloc[0].values
        recovered = p.inverse_transform(returns, initial)  # (1, Seq+1, F)
        np.testing.assert_allclose(recovered[0], df.values, rtol=1e-10, atol=1e-10)

    def test_masked_path_returns_raw_log_returns(self):
        df = _make_price_df()
        mask_df = pd.DataFrame(
            np.ones_like(df.values), index=df.index, columns=df.columns
        )
        # Poke a few missing values (not at the very first row)
        mask_df.iloc[5, 0] = 0.0
        mask_df.iloc[10, 1] = 0.0

        p = LogReturnProcessor(scaling='none')
        p.fit(df, mask=mask_df)
        data, mask_out = p.transform(df, mask=mask_df)

        # At valid positions, values must equal raw log-returns (no scaling)
        raw_logret = np.log(df / df.shift(1)).iloc[1:].values
        valid = mask_out.astype(bool)
        np.testing.assert_allclose(
            data[valid], raw_logret[valid], rtol=1e-12, atol=1e-12
        )

    def test_save_load_roundtrip(self, tmp_path):
        df = _make_price_df()
        p = LogReturnProcessor(scaling='none')
        p.fit(df)
        out_before = p.transform(df)

        path = tmp_path / "proc.pkl"
        p.save(str(path))
        p2 = LogReturnProcessor.load(str(path))
        assert p2.scaling == 'none'
        out_after = p2.transform(df)
        np.testing.assert_array_equal(out_before, out_after)

    def test_unknown_scaling_still_raises(self):
        with pytest.raises((ValueError, KeyError)):
            p = LogReturnProcessor(scaling='bogus')
            p.fit(_make_price_df())
