import pandas as pd
import pytest

import tsgen.data.pipeline as pipeline


class _FakeClient:
    def get_data(self, **kwargs):
        idx = pd.date_range("2024-01-01", periods=3)
        return pd.DataFrame({"AAA": [1.0, 2.0, 3.0]}, index=idx)


def test_load_prices_raises_when_requested_ticker_missing(monkeypatch):
    monkeypatch.setattr(pipeline, "DataClient", lambda *args, **kwargs: _FakeClient())

    with pytest.raises(ValueError, match="Missing data"):
        pipeline.load_prices(["AAA", "BBB"], "2024-01-01", "2024-01-03")


def test_load_prices_can_allow_partial_universe(monkeypatch):
    monkeypatch.setattr(pipeline, "DataClient", lambda *args, **kwargs: _FakeClient())

    df = pipeline.load_prices(
        ["AAA", "BBB"],
        "2024-01-01",
        "2024-01-03",
        allow_missing=True,
    )

    assert list(df.columns) == ["AAA"]

