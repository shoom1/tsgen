"""Tests for checkpoint resumption functionality."""

import pytest
import torch
import tempfile
import os
from pathlib import Path

from tsgen.training.base import BaseTrainer
from tsgen.training.diffusion_trainer import DiffusionTrainer
from tsgen.training.checkpoint_utils import (
    find_latest_checkpoint,
    list_checkpoints,
    get_checkpoint_path,
    extract_epoch_from_checkpoint
)
from tsgen.models.unet import UNet1D
from tsgen.tracking.base import NoOpTracker


class DummyTrainer(BaseTrainer):
    """Simple trainer for testing checkpoint functionality."""

    def __init__(self, model, config, tracker, device):
        super().__init__(model, config, tracker, device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def train(self, dataloader):
        return self.model


def test_save_and_load_checkpoint():
    """Test basic checkpoint save and load."""
    model = UNet1D(sequence_length=16, features=2, base_channels=32)
    config = {'test': 'value'}
    tracker = NoOpTracker()
    device = 'cpu'

    trainer = DummyTrainer(model, config, tracker, device)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, 'checkpoint_epoch_10.pt')

        # Save checkpoint
        trainer.save_checkpoint(ckpt_path, epoch=10, optimizer=trainer.optimizer)

        assert os.path.exists(ckpt_path)

        # Create new model and trainer
        model2 = UNet1D(sequence_length=16, features=2, base_channels=32)
        trainer2 = DummyTrainer(model2, config, tracker, device)

        # Load checkpoint
        checkpoint = trainer2.load_checkpoint(ckpt_path, optimizer=trainer2.optimizer)

        assert checkpoint['epoch'] == 10
        assert 'model_state_dict' in checkpoint
        assert 'optimizer_state_dict' in checkpoint


def test_checkpoint_preserves_model_weights():
    """Test that checkpoint preserves exact model weights."""
    model = UNet1D(sequence_length=16, features=2, base_channels=32)
    config = {}
    tracker = NoOpTracker()
    device = 'cpu'

    trainer = DummyTrainer(model, config, tracker, device)

    # Get initial weights
    initial_weights = {k: v.clone() for k, v in model.state_dict().items()}

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, 'checkpoint_epoch_5.pt')

        # Save checkpoint
        trainer.save_checkpoint(ckpt_path, epoch=5)

        # Modify model weights
        for param in model.parameters():
            param.data.fill_(0.0)

        # Load checkpoint
        trainer.load_checkpoint(ckpt_path)

        # Verify weights are restored
        loaded_weights = model.state_dict()
        for key in initial_weights:
            assert torch.allclose(initial_weights[key], loaded_weights[key])


def test_checkpoint_preserves_optimizer_state():
    """Test that checkpoint preserves optimizer state."""
    model = UNet1D(sequence_length=16, features=2, base_channels=32)
    config = {}
    tracker = NoOpTracker()
    device = 'cpu'

    trainer = DummyTrainer(model, config, tracker, device)

    # Take a training step to create optimizer state
    x = torch.randn(2, 16, 2)
    loss = model(x, torch.tensor([0, 1])).mean()
    loss.backward()
    trainer.optimizer.step()

    # Get optimizer state
    initial_opt_state = trainer.optimizer.state_dict()

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, 'checkpoint_epoch_1.pt')

        # Save checkpoint
        trainer.save_checkpoint(ckpt_path, epoch=1, optimizer=trainer.optimizer)

        # Create new optimizer
        model2 = UNet1D(sequence_length=16, features=2, base_channels=32)
        trainer2 = DummyTrainer(model2, config, tracker, device)

        # Load checkpoint
        trainer2.load_checkpoint(ckpt_path, optimizer=trainer2.optimizer)

        # Verify optimizer state keys match
        loaded_opt_state = trainer2.optimizer.state_dict()
        assert loaded_opt_state.keys() == initial_opt_state.keys()


def test_find_latest_checkpoint():
    """Test finding latest checkpoint in directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create some checkpoint files
        Path(tmpdir, 'checkpoint_epoch_10.pt').touch()
        Path(tmpdir, 'checkpoint_epoch_20.pt').touch()
        Path(tmpdir, 'checkpoint_epoch_15.pt').touch()
        Path(tmpdir, 'other_file.txt').touch()

        latest = find_latest_checkpoint(tmpdir)

        assert latest is not None
        assert 'checkpoint_epoch_20.pt' in latest


def test_find_latest_checkpoint_empty_dir():
    """Test finding checkpoint in empty directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        latest = find_latest_checkpoint(tmpdir)
        assert latest is None


def test_list_checkpoints():
    """Test listing all checkpoints."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create checkpoint files
        Path(tmpdir, 'checkpoint_epoch_10.pt').touch()
        Path(tmpdir, 'checkpoint_epoch_30.pt').touch()
        Path(tmpdir, 'checkpoint_epoch_20.pt').touch()

        checkpoints = list_checkpoints(tmpdir)

        assert len(checkpoints) == 3
        # Should be sorted descending by epoch
        assert checkpoints[0][0] == 30
        assert checkpoints[1][0] == 20
        assert checkpoints[2][0] == 10


def test_get_checkpoint_path_latest():
    """Test getting latest checkpoint path from experiment dir."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create experiment structure
        ckpt_dir = Path(tmpdir) / 'artifacts' / 'checkpoints'
        ckpt_dir.mkdir(parents=True)

        (ckpt_dir / 'checkpoint_epoch_10.pt').touch()
        (ckpt_dir / 'checkpoint_epoch_20.pt').touch()

        path = get_checkpoint_path(tmpdir)

        assert path is not None
        assert 'checkpoint_epoch_20.pt' in path


def test_get_checkpoint_path_specific_epoch():
    """Test getting checkpoint for specific epoch."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create experiment structure
        ckpt_dir = Path(tmpdir) / 'artifacts' / 'checkpoints'
        ckpt_dir.mkdir(parents=True)

        (ckpt_dir / 'checkpoint_epoch_10.pt').touch()
        (ckpt_dir / 'checkpoint_epoch_20.pt').touch()

        path = get_checkpoint_path(tmpdir, epoch=10)

        assert path is not None
        assert 'checkpoint_epoch_10.pt' in path


def test_extract_epoch_from_checkpoint():
    """Test extracting epoch number from checkpoint filename."""
    assert extract_epoch_from_checkpoint('checkpoint_epoch_42.pt') == 42
    assert extract_epoch_from_checkpoint('/path/to/checkpoint_epoch_100.pt') == 100
    assert extract_epoch_from_checkpoint('other_file.pt') is None


def test_checkpoint_with_extra_state():
    """Test saving and loading checkpoint with extra state."""
    model = UNet1D(sequence_length=16, features=2, base_channels=32)
    config = {}
    tracker = NoOpTracker()
    device = 'cpu'

    trainer = DummyTrainer(model, config, tracker, device)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, 'checkpoint_epoch_5.pt')

        # Save with extra state
        trainer.save_checkpoint(
            ckpt_path,
            epoch=5,
            step_count=1000,
            best_loss=0.123,
            custom_metric=42.0
        )

        # Load and verify
        checkpoint = trainer.load_checkpoint(ckpt_path)

        assert checkpoint['epoch'] == 5
        assert checkpoint['step_count'] == 1000
        assert checkpoint['best_loss'] == 0.123
        assert checkpoint['custom_metric'] == 42.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
