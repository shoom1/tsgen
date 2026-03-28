"""Tests for config schema: per-model configs, per-paradigm training configs, and ExperimentConfig."""

import pytest
from pydantic import ValidationError
from tsgen.config.schema import (
    DataConfig, TrainingConfig, DiffusionTrainingConfig, VAETrainingConfig,
    BaselineTrainingConfig, EvaluationConfig, TrackerConfig,
    UNetModelConfig, TransformerModelConfig, MambaModelConfig,
    VAEModelConfig, BaselineModelConfig, ExperimentConfig,
)


class TestDataConfig:
    def test_defaults(self):
        c = DataConfig()
        assert c.sequence_length == 64
        assert c.column == 'adj_close'
        assert c.tickers == []

    def test_custom(self):
        c = DataConfig(tickers=['AAPL'], sequence_length=128)
        assert c.sequence_length == 128

    def test_invalid_sequence_length(self):
        with pytest.raises(ValidationError):
            DataConfig(sequence_length=-1)

    def test_invalid_split(self):
        with pytest.raises(ValidationError):
            DataConfig(train_test_split=1.5)

    def test_scaling_default(self):
        c = DataConfig()
        assert c.scaling == 'global'
        assert c.min_periods == 60

    def test_scaling_expanding(self):
        c = DataConfig(scaling='expanding', min_periods=30)
        assert c.scaling == 'expanding'
        assert c.min_periods == 30

    def test_invalid_scaling(self):
        with pytest.raises(ValidationError):
            DataConfig(scaling='invalid')

    def test_invalid_min_periods(self):
        with pytest.raises(ValidationError):
            DataConfig(min_periods=1)

    def test_invalid_min_periods_zero(self):
        with pytest.raises(ValidationError):
            DataConfig(min_periods=0)


class TestTrainingConfigs:
    def test_base_defaults(self):
        c = TrainingConfig()
        assert c.epochs == 100
        assert c.batch_size == 32
        assert c.start_epoch == 0

    def test_diffusion_defaults(self):
        c = DiffusionTrainingConfig()
        assert c.timesteps == 1000
        assert c.sampling_method == 'ddpm'
        assert c.epochs == 100
        assert c.classifier_free_guidance_probability == 0.0

    def test_diffusion_custom(self):
        c = DiffusionTrainingConfig(epochs=50, timesteps=500)
        assert c.epochs == 50
        assert c.timesteps == 500

    def test_diffusion_rejects_typo(self):
        with pytest.raises(ValidationError):
            DiffusionTrainingConfig(timestepz=500)

    def test_vae_defaults(self):
        c = VAETrainingConfig()
        assert c.beta == 0.5
        assert c.use_annealing is True
        assert c.epochs == 100

    def test_vae_rejects_typo(self):
        with pytest.raises(ValidationError):
            VAETrainingConfig(betta=0.5)

    def test_baseline_defaults(self):
        c = BaselineTrainingConfig()
        assert c.epochs == 100


class TestModelConfigs:
    def test_unet(self):
        c = UNetModelConfig(base_channels=32)
        assert c.base_channels == 32
        assert c.num_classes == 0

    def test_unet_rejects_typo(self):
        with pytest.raises(ValidationError):
            UNetModelConfig(base_chanels=32)

    def test_transformer(self):
        c = TransformerModelConfig(dim=128, depth=6)
        assert c.dim == 128
        assert c.heads == 4

    def test_mamba(self):
        c = MambaModelConfig(dim=64)
        assert c.d_state == 16

    def test_vae_model(self):
        c = VAEModelConfig(latent_dim=8)
        assert c.hidden_dim == 64

    def test_baseline_model(self):
        c = BaselineModelConfig()
        assert c.model_dump() == {}


class TestExperimentConfig:
    def test_minimal(self):
        c = ExperimentConfig(model_type='unet')
        assert c.data.sequence_length == 64
        assert c.evaluation.num_samples == 500

    def test_with_nested_sections(self):
        c = ExperimentConfig(
            model_type='unet',
            data={'tickers': ['AAPL'], 'sequence_length': 32},
            model={'base_channels': 32},
            training={'epochs': 10, 'timesteps': 500},
        )
        assert c.data.tickers == ['AAPL']

    def test_with_data_pipeline(self):
        c = ExperimentConfig(
            model_type='unet',
            data_pipeline=[{'load_prices': {}}, {'clean_data': {}}],
        )
        assert len(c.data_pipeline) == 2

    def test_get_model_config_unet(self):
        c = ExperimentConfig(
            model_type='unet',
            model={'base_channels': 32},
        )
        mc = c.get_model_config()
        assert isinstance(mc, UNetModelConfig)
        assert mc.base_channels == 32

    def test_get_model_config_transformer(self):
        c = ExperimentConfig(
            model_type='transformer',
            model={'dim': 128, 'heads': 8},
        )
        mc = c.get_model_config()
        assert isinstance(mc, TransformerModelConfig)
        assert mc.dim == 128

    def test_get_model_config_defaults(self):
        c = ExperimentConfig(model_type='unet')
        mc = c.get_model_config()
        assert mc.base_channels == 64

    def test_get_training_config_diffusion(self):
        c = ExperimentConfig(
            model_type='unet',
            training={'epochs': 50, 'timesteps': 500},
        )
        tc = c.get_training_config()
        assert isinstance(tc, DiffusionTrainingConfig)
        assert tc.timesteps == 500
        assert tc.epochs == 50

    def test_get_training_config_vae(self):
        c = ExperimentConfig(
            model_type='timevae',
            training={'epochs': 50, 'beta': 0.3},
        )
        tc = c.get_training_config()
        assert isinstance(tc, VAETrainingConfig)
        assert tc.beta == 0.3

    def test_get_training_config_baseline(self):
        c = ExperimentConfig(model_type='gbm')
        tc = c.get_training_config()
        assert isinstance(tc, BaselineTrainingConfig)

    def test_get_training_config_defaults(self):
        c = ExperimentConfig(model_type='unet')
        tc = c.get_training_config()
        assert tc.epochs == 100
        assert tc.timesteps == 1000

    def test_unknown_model_type_model_config(self):
        c = ExperimentConfig(model_type='nonexistent')
        with pytest.raises(ValueError, match="Unknown model_type"):
            c.get_model_config()

    def test_to_dict_roundtrip(self):
        c = ExperimentConfig(
            model_type='unet',
            data={'tickers': ['AAPL']},
            training={'epochs': 10},
        )
        d = c.to_dict()
        assert d['model_type'] == 'unet'
        assert d['data']['tickers'] == ['AAPL']

    def test_get_data_config_returns_nested(self):
        """get_data_config() returns the nested data section."""
        c = ExperimentConfig(
            model_type='unet',
            data={'tickers': ['AAPL'], 'sequence_length': 32},
        )
        data = c.get_data_config()
        assert data.tickers == ['AAPL']
        assert data.sequence_length == 32

    def test_get_evaluation_config_returns_nested(self):
        """get_evaluation_config() returns the nested evaluation section."""
        c = ExperimentConfig(
            model_type='unet',
            evaluation={'num_samples': 200},
        )
        ev = c.get_evaluation_config()
        assert ev.num_samples == 200
