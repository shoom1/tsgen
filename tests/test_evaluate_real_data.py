import numpy as np
import pandas as pd
import pytest

import tsgen.evaluate as evaluate_mod
from tsgen.config.schema import ExperimentConfig
from tsgen.data.pipeline import clean_data
from tsgen.data.processor import LogReturnProcessor
from tsgen.tracking.base import NoOpTracker


def _prices(n=100):
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "AAA": 100.0 + np.arange(n),
        "BBB": 200.0 + 2.0 * np.arange(n),
    }, index=idx)


def test_prepare_real_evaluation_data_uses_heldout_split(monkeypatch):
    df = _prices()
    monkeypatch.setattr(evaluate_mod, "load_prices", lambda *args, **kwargs: df)

    processor = LogReturnProcessor()
    processor.fit(df.iloc[:80])
    config = ExperimentConfig(
        model_type="bootstrap",
        data={
            "tickers": ["AAA", "BBB"],
            "start_date": "2020-01-01",
            "end_date": "2020-04-09",
            "sequence_length": 5,
        },
        data_pipeline=[
            {"load_prices": {"column": "adj_close"}},
            {"clean_data": {"strategy": "ffill_drop"}},
            {"split_temporal": {"train_ratio": 0.8}},
            {"process_prices": {"fit": True}},
            {"create_windows": {"sequence_length": 5}},
        ],
    )

    windows, eval_df, meta = evaluate_mod._prepare_real_evaluation_data(
        config, processor, ["AAA", "BBB"]
    )

    assert meta["heldout"] is True
    assert eval_df.index[0] == df.index[80]
    assert len(eval_df) == 20
    assert windows.shape == (15, 5, 2)


def test_prepare_real_evaluation_data_filters_masked_heldout_rows(monkeypatch):
    df = _prices()
    df.loc[df.index[:85], "BBB"] = np.nan
    monkeypatch.setattr(evaluate_mod, "load_prices", lambda *args, **kwargs: df)

    filled, mask = clean_data(df, strategy="mask")
    processor = LogReturnProcessor()
    processor.fit(filled.iloc[:80], mask=mask.iloc[:80])
    config = ExperimentConfig(
        model_type="bootstrap",
        data={
            "tickers": ["AAA", "BBB"],
            "start_date": "2020-01-01",
            "end_date": "2020-04-09",
            "sequence_length": 3,
        },
        data_pipeline=[
            {"load_prices": {"column": "adj_close"}},
            {"clean_data": {"strategy": "mask"}},
            {"split_temporal": {"train_ratio": 0.8}},
            {"process_prices": {"fit": True}},
            {"create_windows": {"sequence_length": 3}},
        ],
    )

    windows, eval_df, meta = evaluate_mod._prepare_real_evaluation_data(
        config, processor, ["AAA", "BBB"]
    )

    assert meta["heldout"] is True
    assert eval_df.index[0] == df.index[85]
    assert eval_df["BBB"].ne(0).all()
    assert windows.shape[-1] == 2


def test_baseline_load_uses_weights_only_false(monkeypatch):
    calls = {}

    def fake_load(path, **kwargs):
        calls.update(kwargs)
        raise FileNotFoundError(path)

    class Tracker(NoOpTracker):
        def get_artifact_path(self, artifact_name, artifact_type):
            return artifact_name

    monkeypatch.setattr(evaluate_mod.torch, "load", fake_load)
    config = ExperimentConfig(
        model_type="bootstrap",
        data={"tickers": ["AAA"], "sequence_length": 4},
    )

    with pytest.raises(FileNotFoundError):
        evaluate_mod.evaluate_model(config, Tracker())

    assert calls["weights_only"] is False

