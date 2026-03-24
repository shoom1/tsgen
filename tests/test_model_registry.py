"""Tests for Model Registry Pattern."""

import pytest
import torch
from tsgen.models.registry import ModelRegistry
from tsgen.models.base_model import BaseGenerativeModel


class DummyModel(BaseGenerativeModel):
    """Minimal model for testing the registry."""

    def __init__(self, features, sequence_length):
        super().__init__()
        self.features = features
        self.sequence_length = sequence_length
        self.linear = torch.nn.Linear(features, features)

    def forward(self, x):
        return self.linear(x)

    def generate(self, n_samples):
        return torch.randn(n_samples, self.sequence_length, self.features)

    @classmethod
    def from_config(cls, config, features=None):
        data = config.get_data_config()
        return cls(features=features or 1, sequence_length=data.sequence_length)


@pytest.fixture(autouse=True)
def _save_restore_registry():
    """Save and restore registry state around each test."""
    saved = ModelRegistry._models.copy()
    yield
    ModelRegistry._models = saved


def test_register_and_create():
    """Register a dummy model via decorator and create it."""
    @ModelRegistry.register('dummy')
    class TestModel(DummyModel):
        pass

    from tsgen.config.schema import ExperimentConfig
    config = ExperimentConfig(model_type='dummy', tickers=['A'], sequence_length=32)

    model = ModelRegistry.create(config, features=3)

    assert isinstance(model, TestModel)
    assert model.features == 3
    assert model.sequence_length == 32


def test_register_multiple_types():
    """Register one model for multiple type strings."""
    @ModelRegistry.register('alias_a', 'alias_b', 'alias_c')
    class MultiAliasModel(DummyModel):
        pass

    registry = ModelRegistry.list_models()
    assert registry['alias_a'] is MultiAliasModel
    assert registry['alias_b'] is MultiAliasModel
    assert registry['alias_c'] is MultiAliasModel


def test_unknown_model_type_raises():
    """Unknown model_type raises ValueError with helpful message."""
    from tsgen.config.schema import ExperimentConfig
    config = ExperimentConfig(model_type='nonexistent', tickers=['A'])

    with pytest.raises(ValueError, match="No model registered for model type 'nonexistent'"):
        ModelRegistry.create(config, features=1)


def test_unknown_model_type_lists_available():
    """Error message includes available model types."""
    @ModelRegistry.register('registered_one')
    class RegModel(DummyModel):
        pass

    from tsgen.config.schema import ExperimentConfig
    config = ExperimentConfig(model_type='bad_type', tickers=['A'])

    with pytest.raises(ValueError, match="registered_one"):
        ModelRegistry.create(config, features=1)


def test_registry_state_populated():
    """Registry state is correctly populated after registration."""
    assert 'my_test_model' not in ModelRegistry._models

    @ModelRegistry.register('my_test_model')
    class MyTestModel(DummyModel):
        pass

    models = ModelRegistry.list_models()
    assert 'my_test_model' in models
    assert models['my_test_model'] is MyTestModel


def test_list_models_returns_copy():
    """list_models() returns a copy, not the internal dict."""
    models = ModelRegistry.list_models()
    models['injected'] = DummyModel
    assert 'injected' not in ModelRegistry._models
