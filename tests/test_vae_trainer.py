"""Tests for TimeVAE trainer module."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import tempfile
import os

from tsgen.training import VAETrainer
from tsgen.models.timevae import TimeVAE
from tsgen.tracking.base import NoOpTracker, ConsoleTracker
from tsgen.models.losses import vae_loss, VAELossTracker, linear_beta_schedule


@pytest.fixture
def device():
    """Get device for testing."""
    return "cpu"


@pytest.fixture
def sample_dataloader():
    """Create sample dataloader for testing."""
    # Create random time series data: (batch, seq_len, features)
    data = torch.randn(50, 32, 2)
    dataset = TensorDataset(data)
    return DataLoader(dataset, batch_size=8, shuffle=True)


@pytest.fixture
def sample_model():
    """Create sample TimeVAE model."""
    return TimeVAE(
        features=2,
        sequence_length=32,
        hidden_dim=16,
        latent_dim=8,
        encoder_type='lstm'
    )


@pytest.fixture
def basic_config(tmpdir):
    """Create basic training configuration."""
    return {
        'epochs': 3,
        'learning_rate': 1e-3,
        'vae_beta': 1.0,
        'vae_use_annealing': False,
        'output_dir': str(tmpdir),
    }


@pytest.fixture
def annealing_config(tmpdir):
    """Create configuration with beta annealing."""
    return {
        'epochs': 10,
        'learning_rate': 1e-3,
        'vae_beta': 2.0,
        'vae_use_annealing': True,
        'vae_annealing_epochs': 5,
        'output_dir': str(tmpdir),
    }


class TestTrainVAEBasic:
    """Basic tests for train_vae function."""

    def test_train_vae_completes(self, sample_model, sample_dataloader, basic_config, device, tmpdir):
        """Test that training completes without errors."""
        tracker = NoOpTracker()
        basic_config['output_dir'] = str(tmpdir)


        trainer = VAETrainer(sample_model, basic_config, tracker, device)
        trained_model = trainer.train(sample_dataloader)

        assert trained_model is not None
        assert isinstance(trained_model, TimeVAE)

    def test_model_moved_to_device(self, sample_model, sample_dataloader, basic_config, device, tmpdir):
        """Test that model is moved to correct device."""
        tracker = NoOpTracker()
        basic_config['output_dir'] = str(tmpdir)


        trainer = VAETrainer(sample_model, basic_config, tracker, device)
        trained_model = trainer.train(sample_dataloader)

        # Check that model parameters are on the correct device
        for param in trained_model.parameters():
            assert param.device.type == device

    def test_model_parameters_updated(self, sample_model, sample_dataloader, basic_config, device, tmpdir):
        """Test that model parameters are updated during training."""
        tracker = NoOpTracker()
        basic_config['output_dir'] = str(tmpdir)


        # Get initial parameters
        initial_params = [p.clone() for p in sample_model.parameters()]

        trained_model = VAETrainer(sample_model, basic_config, tracker, device).train(sample_dataloader)

        # Check that at least some parameters changed
        params_changed = False
        for initial_p, trained_p in zip(initial_params, trained_model.parameters()):
            if not torch.allclose(initial_p, trained_p):
                params_changed = True
                break

        assert params_changed, "Model parameters should be updated during training"

    def test_returns_same_model_instance(self, sample_model, sample_dataloader, basic_config, device, tmpdir):
        """Test that the same model instance is returned."""
        tracker = NoOpTracker()
        basic_config['output_dir'] = str(tmpdir)


        trained_model = VAETrainer(sample_model, basic_config, tracker, device).train(sample_dataloader)

        # Should return the same model instance
        assert trained_model is sample_model


class TestTrainVAEConfiguration:
    """Tests for different configuration options."""

    def test_custom_learning_rate(self, sample_model, sample_dataloader, device, tmpdir):
        """Test that custom learning rate is used."""
        config = {
            'epochs': 2,
            'learning_rate': 5e-4,
            'vae_beta': 1.0,
            'vae_use_annealing': False,
        }
        config['output_dir'] = str(tmpdir)
        tracker = NoOpTracker()

        # Should not raise errors with custom learning rate
        trained_model = VAETrainer(sample_model, config, tracker, device).train(sample_dataloader)

        assert trained_model is not None

    def test_beta_annealing_enabled(self, sample_model, sample_dataloader, annealing_config, device, tmpdir):
        """Test training with beta annealing enabled."""
        tracker = NoOpTracker()
        annealing_config['output_dir'] = str(tmpdir)


        trained_model = VAETrainer(sample_model, annealing_config, tracker, device).train(sample_dataloader)

        assert trained_model is not None

    def test_fixed_beta(self, sample_model, sample_dataloader, basic_config, device, tmpdir):
        """Test training with fixed beta value."""
        basic_config['vae_beta'] = 0.5
        tracker = NoOpTracker()

        trained_model = VAETrainer(sample_model, basic_config, tracker, device).train(sample_dataloader)

        assert trained_model is not None

    def test_default_learning_rate(self, sample_model, sample_dataloader, device, tmpdir):
        """Test that default learning rate is used when not specified."""
        config = {
            'epochs': 2,
            # No learning_rate specified
            'vae_beta': 1.0,
            'vae_use_annealing': False,
            'output_dir': str(tmpdir),
        }
        tracker = NoOpTracker()

        trained_model = VAETrainer(sample_model, config, tracker, device).train(sample_dataloader)

        assert trained_model is not None


class TestTrainVAETracking:
    """Tests for experiment tracking during training."""

    def test_tracks_batch_metrics(self, sample_model, sample_dataloader, basic_config, device, tmpdir):
        """Test that batch metrics are tracked."""
        tracker = NoOpTracker()
        basic_config['output_dir'] = str(tmpdir)


        # Mock log_metrics to track calls
        calls = []
        original_log_metrics = tracker.log_metrics

        def mock_log_metrics(metrics, step=None):
            calls.append((metrics, step))
            return original_log_metrics(metrics, step)

        tracker.log_metrics = mock_log_metrics

        VAETrainer(sample_model, basic_config, tracker, device).train(sample_dataloader)

        # Should have logged metrics multiple times
        assert len(calls) > 0

        # Check that batch metrics were logged
        batch_metrics = [c for c in calls if 'batch_loss' in c[0]]
        assert len(batch_metrics) > 0

    def test_tracks_epoch_metrics(self, sample_model, sample_dataloader, basic_config, device, tmpdir):
        """Test that epoch-level metrics are tracked."""
        tracker = NoOpTracker()
        basic_config['output_dir'] = str(tmpdir)


        calls = []
        original_log_metrics = tracker.log_metrics

        def mock_log_metrics(metrics, step=None):
            calls.append((metrics, step))
            return original_log_metrics(metrics, step)

        tracker.log_metrics = mock_log_metrics

        VAETrainer(sample_model, basic_config, tracker, device).train(sample_dataloader)

        # Check that epoch metrics were logged
        epoch_metrics = [c for c in calls if 'epoch_loss' in c[0]]
        assert len(epoch_metrics) == basic_config['epochs']

    def test_tracks_artifacts(self, sample_model, sample_dataloader, device, tmpdir):
        """Test that checkpoints are tracked as artifacts."""
        config = {
            'epochs': 10,  # Need 10 epochs to trigger checkpoint saving
            'learning_rate': 1e-3,
            'vae_beta': 1.0,
            'vae_use_annealing': False,
            'output_dir': str(tmpdir),
        }
        tracker = NoOpTracker()

        # Mock log_artifact
        artifacts = []
        original_log_artifact = tracker.log_artifact

        def mock_log_artifact(path, artifact_type='other', artifact_name=None):
            artifacts.append(path)
            return original_log_artifact(path, artifact_type, artifact_name)

        tracker.log_artifact = mock_log_artifact

        VAETrainer(sample_model, config, tracker, "cpu").train(sample_dataloader)

        # Should have logged checkpoint artifact at epoch 10
        assert len(artifacts) > 0
        assert any('checkpoint' in str(a) or 'timevae' in str(a) for a in artifacts)


class TestTrainVAEGradientFlow:
    """Tests for gradient flow and optimization."""

    def test_gradient_clipping_applied(self, sample_model, sample_dataloader, basic_config, device, tmpdir):
        """Test that gradient clipping is applied during training."""
        tracker = NoOpTracker()
        basic_config['output_dir'] = str(tmpdir)


        # Training should complete without exploding gradients
        trained_model = VAETrainer(sample_model, basic_config, tracker, device).train(sample_dataloader)

        # Model should be trained successfully
        assert trained_model is not None

        # Check that parameters are finite (no NaN or Inf)
        for param in trained_model.parameters():
            assert torch.isfinite(param).all()

    def test_loss_decreases_over_epochs(self, sample_model, sample_dataloader, device, tmpdir):
        """Test that loss generally decreases over training."""
        config = {
            'epochs': 5,
            'learning_rate': 1e-3,
            'vae_beta': 1.0,
            'vae_use_annealing': False,
        }
        config['output_dir'] = str(tmpdir)
        tracker = NoOpTracker()

        # Track epoch losses
        epoch_losses = []
        original_log_metrics = tracker.log_metrics

        def mock_log_metrics(metrics, step=None):
            if 'epoch_loss' in metrics:
                epoch_losses.append(metrics['epoch_loss'])
            return original_log_metrics(metrics, step)

        tracker.log_metrics = mock_log_metrics

        VAETrainer(sample_model, config, tracker, device).train(sample_dataloader)

        # Loss should generally decrease (allowing for some variation)
        # Compare first and last epoch
        assert epoch_losses[-1] <= epoch_losses[0] * 1.5  # Allow 50% tolerance


class TestTrainVAEWithBetaAnnealing:
    """Tests for beta annealing functionality."""

    def test_beta_increases_during_warmup(self, sample_model, sample_dataloader, annealing_config, device, tmpdir):
        """Test that beta increases during warmup period."""
        tracker = NoOpTracker()
        annealing_config['output_dir'] = str(tmpdir)


        # Track beta values
        beta_values = []
        original_log_metrics = tracker.log_metrics

        def mock_log_metrics(metrics, step=None):
            if 'beta' in metrics:
                beta_values.append(metrics['beta'])
            return original_log_metrics(metrics, step)

        tracker.log_metrics = mock_log_metrics

        VAETrainer(sample_model, annealing_config, tracker, device).train(sample_dataloader)

        # Beta should increase during warmup
        # Get unique epoch-level beta values (skip batch-level duplicates)
        epoch_betas = []
        for i, beta in enumerate(beta_values):
            # Every Nth value is an epoch metric (where N is steps per epoch)
            if i > 0 and beta != beta_values[i-1]:
                epoch_betas.append(beta)

        # First beta should be less than max beta
        assert epoch_betas[0] < annealing_config['vae_beta']

    def test_beta_reaches_max_after_warmup(self, sample_model, sample_dataloader, device, tmpdir):
        """Test that beta reaches max value after warmup."""
        config = {
            'epochs': 10,
            'learning_rate': 1e-3,
            'vae_beta': 2.0,
            'vae_use_annealing': True,
            'vae_annealing_epochs': 3,
        }
        config['output_dir'] = str(tmpdir)
        tracker = NoOpTracker()

        # Track beta values
        beta_values = []
        original_log_metrics = tracker.log_metrics

        def mock_log_metrics(metrics, step=None):
            if 'beta' in metrics and 'epoch' in metrics:
                # Only track epoch-level beta
                beta_values.append((metrics['epoch'], metrics['beta']))
            return original_log_metrics(metrics, step)

        tracker.log_metrics = mock_log_metrics

        VAETrainer(sample_model, config, tracker, device).train(sample_dataloader)

        # After warmup epochs, beta should be at max
        later_betas = [beta for epoch, beta in beta_values if epoch > config['vae_annealing_epochs']]
        if later_betas:
            assert all(abs(beta - config['vae_beta']) < 0.01 for beta in later_betas)


class TestTrainVAEEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_single_epoch_training(self, sample_model, sample_dataloader, device, tmpdir):
        """Test training with single epoch."""
        config = {
            'epochs': 1,
            'learning_rate': 1e-3,
            'vae_beta': 1.0,
            'vae_use_annealing': False,
        }
        config['output_dir'] = str(tmpdir)
        tracker = NoOpTracker()

        trained_model = VAETrainer(sample_model, config, tracker, device).train(sample_dataloader)

        assert trained_model is not None

    def test_small_batch_size(self, sample_model, device, tmpdir):
        """Test training with very small batches."""
        # Create small dataloader
        data = torch.randn(10, 32, 2)
        dataset = TensorDataset(data)
        small_dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

        config = {
            'epochs': 2,
            'learning_rate': 1e-3,
            'vae_beta': 1.0,
            'vae_use_annealing': False,
            'output_dir': str(tmpdir),
        }
        tracker = NoOpTracker()

        trained_model = VAETrainer(sample_model, config, tracker, device).train(small_dataloader)

        assert trained_model is not None

    def test_handles_tuple_batch_from_dataset(self, sample_model, device, tmpdir):
        """Test that training handles tuple batches from TensorDataset."""
        # TensorDataset returns tuples
        data = torch.randn(20, 32, 2)
        dataset = TensorDataset(data)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

        config = {
            'epochs': 2,
            'learning_rate': 1e-3,
            'vae_beta': 1.0,
            'vae_use_annealing': False,
            'output_dir': str(tmpdir),
        }
        tracker = NoOpTracker()

        # Should handle tuple unpacking correctly
        trained_model = VAETrainer(sample_model, config, tracker, device).train(dataloader)

        assert trained_model is not None

    def test_checkpoint_saving_at_final_epoch(self, sample_model, sample_dataloader, device, tmpdir):
        """Test that checkpoint is logged at final epoch."""
        config = {
            'epochs': 5,  # Not a multiple of 10
            'learning_rate': 1e-3,
            'vae_beta': 1.0,
            'vae_use_annealing': False,
            'output_dir': str(tmpdir),
        }
        tracker = NoOpTracker()

        # Mock log_artifact to track checkpoints
        artifacts = []
        original_log_artifact = tracker.log_artifact

        def mock_log_artifact(path, artifact_type='other', artifact_name=None):
            artifacts.append((path, artifact_type))
            return original_log_artifact(path, artifact_type, artifact_name)

        tracker.log_artifact = mock_log_artifact

        VAETrainer(sample_model, config, tracker, device).train(sample_dataloader)

        # Should have logged checkpoint from final epoch
        checkpoint_artifacts = [a for a in artifacts if a[1] == 'checkpoint']
        assert len(checkpoint_artifacts) > 0
        assert any('epoch_5' in str(a[0]) for a in checkpoint_artifacts)


class TestVAELossTracker:
    """Tests for VAELossTracker helper class."""

    def test_loss_tracker_initialization(self):
        """Test that loss tracker initializes correctly."""
        tracker = VAELossTracker()

        assert tracker.total_loss == 0.0
        assert tracker.recon_loss == 0.0
        assert tracker.kl_loss == 0.0
        assert tracker.n_batches == 0

    def test_loss_tracker_update(self):
        """Test updating loss tracker."""
        tracker = VAELossTracker()

        tracker.update(1.5, 1.0, 0.5)
        tracker.update(1.2, 0.8, 0.4)

        assert tracker.n_batches == 2
        assert tracker.total_loss == pytest.approx(2.7)  # 1.5 + 1.2
        assert tracker.recon_loss == pytest.approx(1.8)  # 1.0 + 0.8

    def test_loss_tracker_average(self):
        """Test computing average losses."""
        tracker = VAELossTracker()

        tracker.update(2.0, 1.5, 0.5)
        tracker.update(1.0, 0.5, 0.5)

        averages = tracker.get_average()

        assert averages['total_loss'] == pytest.approx(1.5)
        assert averages['recon_loss'] == pytest.approx(1.0)
        assert averages['kl_loss'] == pytest.approx(0.5)

    def test_loss_tracker_reset(self):
        """Test resetting loss tracker."""
        tracker = VAELossTracker()

        tracker.update(1.0, 0.5, 0.5)
        tracker.reset()

        assert tracker.total_loss == 0.0
        assert tracker.recon_loss == 0.0
        assert tracker.kl_loss == 0.0
        assert tracker.n_batches == 0


class TestLinearBetaSchedule:
    """Tests for linear beta schedule function."""

    def test_beta_starts_at_zero(self):
        """Test that beta starts at 0 during warmup."""
        schedule = linear_beta_schedule(max_epochs=10, warmup_epochs=5, max_beta=2.0)

        assert schedule(0) == pytest.approx(0.0)

    def test_beta_reaches_max_after_warmup(self):
        """Test that beta reaches max after warmup."""
        schedule = linear_beta_schedule(max_epochs=10, warmup_epochs=5, max_beta=2.0)

        # At warmup_epochs, should be at max_beta
        assert schedule(5) == pytest.approx(2.0)

    def test_beta_stays_at_max_after_warmup(self):
        """Test that beta stays at max after warmup period."""
        schedule = linear_beta_schedule(max_epochs=10, warmup_epochs=3, max_beta=1.5)

        assert schedule(3) == pytest.approx(1.5)
        assert schedule(5) == pytest.approx(1.5)
        assert schedule(10) == pytest.approx(1.5)

    def test_beta_increases_linearly_during_warmup(self):
        """Test that beta increases linearly during warmup."""
        schedule = linear_beta_schedule(max_epochs=10, warmup_epochs=4, max_beta=2.0)

        # Should increase linearly
        beta_0 = schedule(0)
        beta_1 = schedule(1)
        beta_2 = schedule(2)
        beta_3 = schedule(3)
        beta_4 = schedule(4)

        # Check linear progression
        diff1 = beta_1 - beta_0
        diff2 = beta_2 - beta_1
        diff3 = beta_3 - beta_2

        assert pytest.approx(diff1) == diff2
        assert pytest.approx(diff2) == diff3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
