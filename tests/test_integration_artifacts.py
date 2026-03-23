"""
Integration tests for artifact organization and storage.

Tests the complete artifact management system to ensure:
- No duplication (files stored in ONE location only)
- Correct typed subdirectories (models/, plots/, checkpoints/, data/)
- Tracker-managed storage works end-to-end
- Backward compatibility maintained
"""

import pytest
import os
import tempfile
from pathlib import Path

from tsgen.train import train_model
from tsgen.evaluate import evaluate_model
from tsgen.tracking.base import FileTracker
from tsgen.config.schema import ExperimentConfig


@pytest.fixture
def minimal_config():
    """Minimal config for fast testing."""
    return ExperimentConfig(
        model_type='unet',
        tickers=['AAPL', 'MSFT'],
        start_date='2023-01-01',
        end_date='2023-12-31',
        sequence_length=64,
        batch_size=16,
        epochs=3,
        timesteps=100,
        learning_rate=1e-3,
        in_channels=2,
        channels=32,
        tracker='file',
        data_pipeline=[
            {'load_prices': {'column': 'adj_close'}},
            {'clean_data': {'strategy': 'ffill_drop'}},
            {'process_prices': {'fit': True}},
            {'create_windows': {'sequence_length': 64}},
            {'create_dataloader': {'batch_size': 16, 'shuffle': True}}
        ]
    )


def test_no_artifact_duplication(minimal_config):
    """Verify artifacts stored in ONE location only."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = FileTracker(experiment_dir=tmpdir)

        # Train model
        train_model(minimal_config, tracker)

        # Check correct locations exist
        artifacts_dir = Path(tmpdir) / 'artifacts'
        assert (artifacts_dir / 'models' / 'model_final.pt').exists(), \
            "Model should be in artifacts/models/"
        assert (artifacts_dir / 'data' / 'processor.pkl').exists(), \
            "Processor should be in artifacts/data/"

        # Verify no duplication in old flat locations
        assert not (artifacts_dir / 'model_final.pt').exists(), \
            "Model should NOT be in flat artifacts/ directory"
        assert not (Path(tmpdir) / 'model_final.pt').exists(), \
            "Model should NOT be in root directory"
        assert not (artifacts_dir / 'processor.pkl').exists(), \
            "Processor should NOT be in flat artifacts/ directory"
        assert not (Path(tmpdir) / 'processor.pkl').exists(), \
            "Processor should NOT be in root directory"


def test_checkpoint_organization(minimal_config):
    """Verify checkpoints are organized correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Use more epochs to trigger checkpoint saving
        config = ExperimentConfig(**{**minimal_config.to_dict(), 'epochs': 10})

        tracker = FileTracker(experiment_dir=tmpdir)

        # Train model
        train_model(config, tracker)

        # Check checkpoint location
        checkpoints_dir = Path(tmpdir) / 'artifacts' / 'checkpoints'
        assert checkpoints_dir.exists(), "Checkpoints directory should exist"

        # Should have checkpoint at epoch 10
        checkpoint_files = list(checkpoints_dir.glob('*.pt'))
        assert len(checkpoint_files) > 0, "Should have at least one checkpoint"

        # Check that checkpoints are not in old locations
        assert not (Path(tmpdir) / 'checkpoints').exists(), \
            "Checkpoints should NOT be in root checkpoints/ directory"
        assert not (Path(tmpdir) / 'artifacts' / 'model_final.pt').exists(), \
            "Checkpoints should NOT be mixed with final model"


def test_plot_organization(minimal_config):
    """Verify plots are organized correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = FileTracker(experiment_dir=tmpdir)

        # Train and evaluate to generate plots
        train_model(minimal_config, tracker)

        eval_dict = {**minimal_config.to_dict(), 'num_samples': 50}
        evaluate_model(eval_dict, tracker)

        # Check plot locations
        plots_dir = Path(tmpdir) / 'artifacts' / 'plots'
        assert plots_dir.exists(), "Plots directory should exist"

        # Should have generated plots
        plot_files = list(plots_dir.glob('*.png'))
        assert len(plot_files) > 0, "Should have at least one plot"

        # Verify expected plots exist
        expected_plots = ['stylized_facts.png', 'correlation_structure.png', 'synthetic_comparison.png']
        for plot_name in expected_plots:
            assert (plots_dir / plot_name).exists(), f"Should have {plot_name}"

        # Check that plots are not in old locations
        assert not (Path(tmpdir) / 'plots').exists(), \
            "Plots should NOT be in root plots/ directory"


def test_typed_subdirectories_complete(minimal_config):
    """Verify all artifact types are in correct subdirectories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = FileTracker(experiment_dir=tmpdir)

        # Train model (creates models, data, checkpoints)
        config = ExperimentConfig(**{**minimal_config.to_dict(), 'epochs': 10})
        train_model(config, tracker)

        # Evaluate model (creates plots) - evaluate_model still expects dict (Task 5)
        evaluate_model({**config.to_dict(), 'num_samples': 50}, tracker)

        artifacts_dir = Path(tmpdir) / 'artifacts'

        # Verify typed subdirectories exist
        assert (artifacts_dir / 'models').exists(), "models/ subdirectory should exist"
        assert (artifacts_dir / 'data').exists(), "data/ subdirectory should exist"
        assert (artifacts_dir / 'checkpoints').exists(), "checkpoints/ subdirectory should exist"
        assert (artifacts_dir / 'plots').exists(), "plots/ subdirectory should exist"

        # Verify files are in correct subdirectories
        assert (artifacts_dir / 'models' / 'model_final.pt').exists()
        assert (artifacts_dir / 'data' / 'processor.pkl').exists()
        assert len(list((artifacts_dir / 'checkpoints').glob('*.pt'))) > 0
        assert len(list((artifacts_dir / 'plots').glob('*.png'))) > 0

        # Verify no files in flat artifacts/ directory (except subdirs)
        flat_files = [f for f in artifacts_dir.iterdir() if f.is_file()]
        assert len(flat_files) == 0, \
            "No files should be in flat artifacts/ directory, only subdirectories"


def test_artifact_retrieval_via_tracker(minimal_config):
    """Verify artifacts can be retrieved via tracker.get_artifact_path()."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = FileTracker(experiment_dir=tmpdir)

        # Train model
        train_model(minimal_config, tracker)

        # Retrieve artifact paths
        model_path = tracker.get_artifact_path('model_final.pt', artifact_type='model')
        processor_path = tracker.get_artifact_path('processor.pkl', artifact_type='data')

        # Verify paths are correct
        assert model_path is not None, "Should return model path"
        assert processor_path is not None, "Should return processor path"

        assert os.path.exists(model_path), "Model path should exist"
        assert os.path.exists(processor_path), "Processor path should exist"

        # Verify paths point to typed subdirectories
        assert 'artifacts/models' in model_path, "Model should be in models/ subdirectory"
        assert 'artifacts/data' in processor_path, "Processor should be in data/ subdirectory"


def test_multiple_training_runs_no_interference(minimal_config):
    """Verify multiple training runs in same directory don't interfere."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = FileTracker(experiment_dir=tmpdir)

        # First training run
        train_model(minimal_config, tracker)

        # Verify first run artifacts exist
        artifacts_dir = Path(tmpdir) / 'artifacts'
        assert (artifacts_dir / 'models' / 'model_final.pt').exists()

        # Get file size of first model
        first_model_size = (artifacts_dir / 'models' / 'model_final.pt').stat().st_size

        # Second training run (should overwrite)
        config = ExperimentConfig(**{**minimal_config.to_dict(), 'epochs': 5})
        train_model(config, tracker)

        # Verify second run artifacts exist
        assert (artifacts_dir / 'models' / 'model_final.pt').exists()

        # Should still only be ONE model file (not duplicated)
        model_files = list((artifacts_dir / 'models').glob('model_final.pt'))
        assert len(model_files) == 1, "Should have only one model_final.pt"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
