"""Tests for Trainer Registry Pattern."""

import pytest
import torch
from tsgen.training import (
    TrainerRegistry,
    BaseTrainer,
    DiffusionTrainer,
    VAETrainer,
    BaselineTrainer
)
from tsgen.tracking.base import NoOpTracker
from tsgen.config.schema import ExperimentConfig


def test_registry_lists_all_trainers():
    """Test that registry contains all expected trainers."""
    trainers = TrainerRegistry.list_trainers()

    # Check model types are registered
    assert 'unet' in trainers
    assert 'transformer' in trainers
    assert 'timevae' in trainers
    assert 'gbm' in trainers
    assert 'bootstrap' in trainers
    assert 'multivariate_lognormal' in trainers
    assert 'multivariate_gbm' in trainers


def test_registry_get_diffusion_trainer():
    """Test getting diffusion trainer from registry."""
    model = torch.nn.Linear(10, 10)
    config = ExperimentConfig(
        model_type='unet',
        epochs=1,
        timesteps=10,
        learning_rate=1e-3,
        sequence_length=10,
        tickers=['A']
    )
    tracker = NoOpTracker()

    trainer = TrainerRegistry.get_trainer('unet', model, config, tracker, 'cpu')
    assert isinstance(trainer, DiffusionTrainer)
    assert isinstance(trainer, BaseTrainer)


def test_registry_get_vae_trainer():
    """Test getting VAE trainer from registry."""
    from tsgen.models.timevae import TimeVAE
    model = TimeVAE(features=2, sequence_length=32, latent_dim=8, hidden_dim=16)
    config = ExperimentConfig(model_type='timevae', epochs=1, learning_rate=1e-3)
    tracker = NoOpTracker()

    trainer = TrainerRegistry.get_trainer('timevae', model, config, tracker, 'cpu')
    assert isinstance(trainer, VAETrainer)


def test_registry_get_baseline_trainer():
    """Test getting baseline trainer from registry."""
    from tsgen.models.baselines import MultivariateGBM
    model = MultivariateGBM(features=2)
    config = ExperimentConfig(model_type='gbm', epochs=1)
    tracker = NoOpTracker()

    trainer = TrainerRegistry.get_trainer('gbm', model, config, tracker, 'cpu')
    assert isinstance(trainer, BaselineTrainer)


def test_registry_unknown_model_type_raises():
    """Test that unknown model type raises ValueError."""
    model = torch.nn.Linear(10, 10)
    config = ExperimentConfig(model_type='unknown')
    tracker = NoOpTracker()

    with pytest.raises(ValueError, match="No trainer registered"):
        TrainerRegistry.get_trainer('unknown', model, config, tracker, 'cpu')


def test_trainer_decorator_registers_multiple_types():
    """Test that decorator can register multiple model types."""
    trainers = TrainerRegistry.list_trainers()

    # DiffusionTrainer registered for both
    assert trainers['unet'] == DiffusionTrainer
    assert trainers['transformer'] == DiffusionTrainer

    # BaselineTrainer registered for all
    assert trainers['gbm'] == BaselineTrainer
    assert trainers['bootstrap'] == BaselineTrainer
    assert trainers['multivariate_lognormal'] == BaselineTrainer
    assert trainers['multivariate_gbm'] == BaselineTrainer


def test_trainer_has_common_interface():
    """Test that all trainers implement the BaseTrainer interface."""
    # Create simple models
    diff_model = torch.nn.Linear(10, 10)
    from tsgen.models.timevae import TimeVAE
    vae_model = TimeVAE(features=2, sequence_length=32, latent_dim=8, hidden_dim=16)
    from tsgen.models.baselines import MultivariateGBM
    baseline_model = MultivariateGBM(features=2)

    tracker = NoOpTracker()
    device = 'cpu'

    # Create trainers
    diff_config = ExperimentConfig(
        model_type='unet', epochs=1, timesteps=10,
        learning_rate=1e-3, sequence_length=10, tickers=['A']
    )
    vae_config = ExperimentConfig(model_type='timevae', epochs=1, learning_rate=1e-3)
    baseline_config = ExperimentConfig(model_type='gbm', epochs=1)

    diff_trainer = TrainerRegistry.get_trainer('unet', diff_model, diff_config, tracker, device)
    vae_trainer = TrainerRegistry.get_trainer('timevae', vae_model, vae_config, tracker, device)
    baseline_trainer = TrainerRegistry.get_trainer('gbm', baseline_model, baseline_config, tracker, device)

    # All should have train() method
    assert hasattr(diff_trainer, 'train')
    assert hasattr(vae_trainer, 'train')
    assert hasattr(baseline_trainer, 'train')

    # All should have save_model() method
    assert hasattr(diff_trainer, 'save_model')
    assert hasattr(vae_trainer, 'save_model')
    assert hasattr(baseline_trainer, 'save_model')


def test_trainer_stores_config_and_model():
    """Test that trainers store references to model and config."""
    model = torch.nn.Linear(10, 10)
    config = ExperimentConfig(
        model_type='unet',
        epochs=1,
        timesteps=10,
        learning_rate=1e-3,
        sequence_length=10,
        tickers=['A']
    )
    tracker = NoOpTracker()

    trainer = TrainerRegistry.get_trainer('unet', model, config, tracker, 'cpu')

    # Trainer should store references
    assert trainer.model is model
    assert trainer.config is config
    assert trainer.tracker is tracker
    assert trainer.device == 'cpu'
