"""Tests for ExperimentConfig accessor methods and field completeness."""

import pytest
from tsgen.config.schema import (
    ExperimentConfig,
    DataConfig,
    DiffusionConfig,
    EvaluationConfig,
    ModelParamsConfig,
)


class TestGetDataConfigFlat:
    """get_data_config() should include column, db_path, train_test_split from flat fields."""

    def test_includes_column(self):
        cfg = ExperimentConfig(model_type='unet', column='close')
        data = cfg.get_data_config()
        assert data.column == 'close'

    def test_column_default(self):
        cfg = ExperimentConfig(model_type='unet')
        data = cfg.get_data_config()
        assert data.column == 'adj_close'

    def test_includes_db_path(self):
        cfg = ExperimentConfig(model_type='unet', db_path='/tmp/test.db')
        data = cfg.get_data_config()
        assert data.db_path == '/tmp/test.db'

    def test_db_path_default_none(self):
        cfg = ExperimentConfig(model_type='unet')
        data = cfg.get_data_config()
        assert data.db_path is None

    def test_includes_train_test_split(self):
        cfg = ExperimentConfig(model_type='unet', train_test_split=0.8)
        data = cfg.get_data_config()
        assert data.train_test_split == 0.8

    def test_train_test_split_default_none(self):
        cfg = ExperimentConfig(model_type='unet')
        data = cfg.get_data_config()
        assert data.train_test_split is None

    def test_all_flat_fields_present(self):
        cfg = ExperimentConfig(
            model_type='unet',
            tickers=['AAPL', 'MSFT'],
            start_date='2020-01-01',
            end_date='2024-01-01',
            sequence_length=128,
            column='high',
            db_path='/data/fin.db',
            train_test_split=0.75,
        )
        data = cfg.get_data_config()
        assert data.tickers == ['AAPL', 'MSFT']
        assert data.start_date == '2020-01-01'
        assert data.end_date == '2024-01-01'
        assert data.sequence_length == 128
        assert data.column == 'high'
        assert data.db_path == '/data/fin.db'
        assert data.train_test_split == 0.75


class TestGetDataConfigNested:
    """get_data_config() should return nested data config when provided."""

    def test_returns_nested_data(self):
        cfg = ExperimentConfig(
            model_type='unet',
            data=DataConfig(
                tickers=['GOOG'],
                column='volume',
                db_path='/nested/path.db',
                train_test_split=0.9,
            ),
        )
        data = cfg.get_data_config()
        assert data.tickers == ['GOOG']
        assert data.column == 'volume'
        assert data.db_path == '/nested/path.db'
        assert data.train_test_split == 0.9

    def test_nested_takes_precedence_over_flat(self):
        cfg = ExperimentConfig(
            model_type='unet',
            tickers=['AAPL'],
            column='close',
            data=DataConfig(tickers=['GOOG'], column='volume'),
        )
        data = cfg.get_data_config()
        assert data.tickers == ['GOOG']
        assert data.column == 'volume'


class TestGetModelParamsConfigFlat:
    """get_model_params_config() should return ModelParamsConfig from flat config fields."""

    def test_returns_model_params_from_flat(self):
        cfg = ExperimentConfig(
            model_type='unet',
            base_channels=128,
        )
        params = cfg.get_model_params_config()
        assert isinstance(params, ModelParamsConfig)
        assert params.base_channels == 128

    def test_flat_transformer_params(self):
        cfg = ExperimentConfig(
            model_type='transformer',
            dim=256,
            depth=6,
            heads=8,
        )
        params = cfg.get_model_params_config()
        assert params.dim == 256
        assert params.depth == 6
        assert params.heads == 8

    def test_flat_num_classes(self):
        cfg = ExperimentConfig(
            model_type='unet',
            num_classes=5,
        )
        params = cfg.get_model_params_config()
        assert params.num_classes == 5

    def test_defaults(self):
        cfg = ExperimentConfig(model_type='unet')
        params = cfg.get_model_params_config()
        assert params.base_channels == 64
        assert params.dim == 64
        assert params.depth == 4
        assert params.heads == 4
        assert params.num_classes == 0


class TestGetModelParamsConfigNested:
    """get_model_params_config() should return nested model config when provided."""

    def test_returns_nested_model_params(self):
        cfg = ExperimentConfig(
            model_type='unet',
            model=ModelParamsConfig(base_channels=256, dropout=0.1),
        )
        params = cfg.get_model_params_config()
        assert params.base_channels == 256
        assert params.dropout == 0.1

    def test_nested_takes_precedence_over_flat(self):
        cfg = ExperimentConfig(
            model_type='unet',
            base_channels=64,
            model=ModelParamsConfig(base_channels=256),
        )
        params = cfg.get_model_params_config()
        assert params.base_channels == 256


class TestGetDiffusionConfig:
    """get_diffusion_config() returns correct defaults and values."""

    def test_defaults(self):
        cfg = ExperimentConfig(model_type='unet')
        diff = cfg.get_diffusion_config()
        assert isinstance(diff, DiffusionConfig)
        assert diff.time_steps == 1000
        assert diff.sampling_method == 'ddpm'

    def test_flat_timesteps(self):
        cfg = ExperimentConfig(model_type='unet', timesteps=500)
        diff = cfg.get_diffusion_config()
        assert diff.time_steps == 500

    def test_nested(self):
        cfg = ExperimentConfig(
            model_type='unet',
            diffusion=DiffusionConfig(time_steps=200, sampling_method='ddim'),
        )
        diff = cfg.get_diffusion_config()
        assert diff.time_steps == 200
        assert diff.sampling_method == 'ddim'


class TestGetEvaluationConfig:
    """get_evaluation_config() returns correct defaults."""

    def test_defaults(self):
        cfg = ExperimentConfig(model_type='unet')
        ev = cfg.get_evaluation_config()
        assert isinstance(ev, EvaluationConfig)
        assert ev.num_samples == 500
        assert ev.discriminator_epochs == 20

    def test_nested(self):
        cfg = ExperimentConfig(
            model_type='unet',
            evaluation=EvaluationConfig(num_samples=1000),
        )
        ev = cfg.get_evaluation_config()
        assert ev.num_samples == 1000


class TestFromDictRoundTrip:
    """from_dict() and to_dict() round-trip with flat format including new fields."""

    def test_round_trip_flat_with_column_and_db_path(self):
        raw = {
            'model_type': 'unet',
            'tickers': ['AAPL'],
            'column': 'close',
            'db_path': '/tmp/test.db',
            'train_test_split': 0.8,
        }
        cfg = ExperimentConfig.from_dict(raw)
        assert cfg.column == 'close'
        assert cfg.db_path == '/tmp/test.db'
        assert cfg.train_test_split == 0.8

        # to_dict should include these fields
        d = cfg.to_dict()
        assert d['column'] == 'close'
        assert d['db_path'] == '/tmp/test.db'
        assert d['train_test_split'] == 0.8

    def test_round_trip_flat_defaults(self):
        raw = {'model_type': 'unet'}
        cfg = ExperimentConfig.from_dict(raw)
        assert cfg.column == 'adj_close'
        assert cfg.db_path is None
        assert cfg.train_test_split is None
