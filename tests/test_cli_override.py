"""Tests for the CLI --override flag.

The --override flag accepts dotted-path assignments like
``--override training.epochs=3`` and applies them to the loaded Pydantic
``ExperimentConfig`` before training/evaluation. It exists primarily for
smoke-testing experiments without editing their YAML.
"""

import pytest

from tsgen.cli.main import (
    apply_overrides,
    parse_override_pair,
)
from tsgen.config.schema import ExperimentConfig


class TestParseOverridePair:
    def test_simple_key_value(self):
        assert parse_override_pair("training.epochs=3") == (["training", "epochs"], "3")

    def test_nested_path(self):
        assert parse_override_pair("evaluation.num_samples=500") == (
            ["evaluation", "num_samples"],
            "500",
        )

    def test_top_level_key(self):
        assert parse_override_pair("output_dir=/tmp/foo") == (["output_dir"], "/tmp/foo")

    def test_value_with_equals_in_it(self):
        """Only the first '=' delimits key from value; later '=' are part of value."""
        assert parse_override_pair("output_dir=/tmp/a=b") == (["output_dir"], "/tmp/a=b")

    def test_missing_equals_raises(self):
        with pytest.raises(ValueError, match="must be key=value"):
            parse_override_pair("training.epochs")

    def test_empty_key_raises(self):
        with pytest.raises(ValueError, match="empty key"):
            parse_override_pair("=value")


class TestApplyOverrides:
    def _base_config(self):
        return ExperimentConfig(
            model_type="unet",
            data={"tickers": ["AAPL", "MSFT"], "sequence_length": 64},
            training={"epochs": 200, "batch_size": 32, "learning_rate": 1e-3, "timesteps": 500},
            model={"base_channels": 128},
        )

    def test_override_nested_int(self):
        c = self._base_config()
        apply_overrides(c, ["training.epochs=3"])
        assert c.get_training_config().epochs == 3

    def test_override_nested_float(self):
        c = self._base_config()
        apply_overrides(c, ["training.learning_rate=0.0005"])
        assert c.get_training_config().learning_rate == pytest.approx(0.0005)

    def test_override_top_level_str(self):
        c = self._base_config()
        apply_overrides(c, ["output_dir=smoke_test/foo"])
        assert c.output_dir == "smoke_test/foo"

    def test_override_bool_true(self):
        c = ExperimentConfig(
            model_type="timevae",
            data={"tickers": ["A"], "sequence_length": 32},
            training={"epochs": 10, "use_annealing": False},
        )
        apply_overrides(c, ["training.use_annealing=true"])
        assert c.get_training_config().use_annealing is True

    def test_override_bool_false(self):
        c = ExperimentConfig(
            model_type="timevae",
            data={"tickers": ["A"], "sequence_length": 32},
            training={"epochs": 10, "use_annealing": True},
        )
        apply_overrides(c, ["training.use_annealing=false"])
        assert c.get_training_config().use_annealing is False

    def test_override_model_section(self):
        c = self._base_config()
        apply_overrides(c, ["model.base_channels=32"])
        assert c.get_model_config().base_channels == 32

    def test_override_evaluation_section(self):
        c = self._base_config()
        apply_overrides(c, ["evaluation.num_samples=50"])
        assert c.get_evaluation_config().num_samples == 50

    def test_multiple_overrides_applied_in_order(self):
        c = self._base_config()
        apply_overrides(
            c,
            [
                "training.epochs=5",
                "evaluation.num_samples=42",
                "output_dir=smoke/experiment",
            ],
        )
        assert c.get_training_config().epochs == 5
        assert c.get_evaluation_config().num_samples == 42
        assert c.output_dir == "smoke/experiment"

    def test_unknown_key_raises(self):
        c = self._base_config()
        with pytest.raises((KeyError, ValueError, AttributeError)):
            apply_overrides(c, ["training.nonexistent_field=999"])

    def test_empty_overrides_is_noop(self):
        c = self._base_config()
        original_epochs = c.get_training_config().epochs
        apply_overrides(c, [])
        assert c.get_training_config().epochs == original_epochs


class TestOverridesRespectValidation:
    """Overrides must go through Pydantic validation so typos aren't silent."""

    def test_invalid_value_type_raises(self):
        c = ExperimentConfig(
            model_type="unet",
            data={"tickers": ["A"], "sequence_length": 64},
            training={"epochs": 200},
        )
        # epochs must be a positive int; setting to -5 should fail validation
        with pytest.raises(Exception):
            apply_overrides(c, ["training.epochs=-5"])

    def test_invalid_literal_raises(self):
        """scaling is a Literal['global', 'expanding', 'none'] — other values must fail."""
        c = ExperimentConfig(
            model_type="unet",
            data={"tickers": ["A"], "sequence_length": 64, "scaling": "global"},
            training={"epochs": 200},
        )
        with pytest.raises(Exception):
            apply_overrides(c, ["data.scaling=bogus"])
