"""Tests for tracking modules."""

import pytest
import tempfile
import os
from pathlib import Path

from tsgen.tracking.base import (
    ExperimentTracker,
    NoOpTracker,
    ConsoleTracker,
    FileTracker,
    ArtifactType
)


@pytest.fixture
def temp_dir():
    """Create temporary directory for test artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestNoOpTracker:
    """Tests for NoOpTracker."""

    def test_log_params(self):
        """Test that log_params does nothing."""
        tracker = NoOpTracker()
        # Should not raise any exceptions
        tracker.log_params({'param1': 'value1', 'param2': 123})

    def test_log_metrics(self):
        """Test that log_metrics does nothing."""
        tracker = NoOpTracker()
        tracker.log_metrics({'loss': 0.5, 'accuracy': 0.9}, step=100)

    def test_log_artifact(self):
        """Test that log_artifact does nothing."""
        tracker = NoOpTracker()
        tracker.log_artifact('/fake/path/artifact.txt')

    def test_end_run(self):
        """Test that end_run does nothing."""
        tracker = NoOpTracker()
        tracker.end_run()


class TestConsoleTracker:
    """Tests for ConsoleTracker."""

    def test_log_params(self, capsys):
        """Test that log_params prints to console."""
        tracker = ConsoleTracker()
        params = {'learning_rate': 0.001, 'batch_size': 32}

        tracker.log_params(params)

        captured = capsys.readouterr()
        assert 'Params:' in captured.out
        assert 'learning_rate' in captured.out
        assert '0.001' in captured.out
        assert 'batch_size' in captured.out
        assert '32' in captured.out

    def test_log_metrics(self, capsys):
        """Test that log_metrics prints to console."""
        tracker = ConsoleTracker()
        metrics = {'loss': 0.123, 'accuracy': 0.95}

        tracker.log_metrics(metrics, step=10)

        captured = capsys.readouterr()
        assert 'Metrics' in captured.out
        assert 'loss' in captured.out
        assert '0.123' in captured.out

    def test_log_metrics_without_step(self, capsys):
        """Test that log_metrics works without step parameter."""
        tracker = ConsoleTracker()
        metrics = {'loss': 0.456}

        tracker.log_metrics(metrics)

        captured = capsys.readouterr()
        assert 'Metrics' in captured.out
        assert 'loss' in captured.out

    def test_log_artifact(self, capsys):
        """Test that log_artifact prints to console."""
        tracker = ConsoleTracker()
        artifact_path = '/path/to/model.pt'

        tracker.log_artifact(artifact_path)

        captured = capsys.readouterr()
        assert 'Saving artifact' in captured.out
        assert artifact_path in captured.out

    def test_end_run(self, capsys):
        """Test that end_run prints to console."""
        tracker = ConsoleTracker()

        tracker.end_run()

        captured = capsys.readouterr()
        assert 'Run ended' in captured.out


class TestFileTracker:
    """Tests for FileTracker."""

    def test_initialization_creates_log_file(self, temp_dir):
        """Test that FileTracker creates log file."""
        log_path = os.path.join(temp_dir, 'test.log')
        tracker = FileTracker(log_file=log_path)

        assert os.path.exists(log_path)

    def test_initialization_with_experiment_dir(self, temp_dir):
        """Test FileTracker with experiment directory."""
        tracker = FileTracker(log_file='training.log', experiment_dir=temp_dir)

        expected_path = os.path.join(temp_dir, 'training.log')
        assert os.path.exists(expected_path)

    def test_log_params_writes_to_file(self, temp_dir):
        """Test that log_params writes to log file."""
        log_path = os.path.join(temp_dir, 'params.log')
        tracker = FileTracker(log_file=log_path)

        params = {'learning_rate': 0.001, 'epochs': 100}
        tracker.log_params(params)

        with open(log_path, 'r') as f:
            content = f.read()

        assert 'Params:' in content
        assert 'learning_rate' in content
        assert '0.001' in content
        assert 'epochs' in content
        assert '100' in content

    def test_log_metrics_writes_to_file(self, temp_dir):
        """Test that log_metrics writes to log file."""
        log_path = os.path.join(temp_dir, 'metrics.log')
        tracker = FileTracker(log_file=log_path)

        metrics = {'loss': 0.123, 'accuracy': 0.95}
        tracker.log_metrics(metrics, step=5)

        with open(log_path, 'r') as f:
            content = f.read()

        assert 'Metrics' in content
        assert 'Step: 5' in content
        assert 'loss' in content
        assert '0.123' in content

    def test_log_metrics_without_step_writes_to_file(self, temp_dir):
        """Test log_metrics without step parameter."""
        log_path = os.path.join(temp_dir, 'metrics2.log')
        tracker = FileTracker(log_file=log_path)

        metrics = {'loss': 0.456}
        tracker.log_metrics(metrics)

        with open(log_path, 'r') as f:
            content = f.read()

        assert 'Metrics' in content
        assert 'loss' in content

    def test_log_artifact_writes_to_file(self, temp_dir):
        """Test that log_artifact writes to log file."""
        log_path = os.path.join(temp_dir, 'artifacts.log')
        tracker = FileTracker(log_file=log_path)

        artifact_path = '/path/to/checkpoint.pt'
        tracker.log_artifact(artifact_path)

        with open(log_path, 'r') as f:
            content = f.read()

        assert 'Logging artifact' in content
        assert artifact_path in content

    def test_end_run_writes_to_file(self, temp_dir):
        """Test that end_run writes to log file."""
        log_path = os.path.join(temp_dir, 'run.log')
        tracker = FileTracker(log_file=log_path)

        tracker.end_run()

        with open(log_path, 'r') as f:
            content = f.read()

        assert 'Run ended' in content

    def test_multiple_operations(self, temp_dir):
        """Test multiple tracker operations in sequence."""
        log_path = os.path.join(temp_dir, 'full.log')
        tracker = FileTracker(log_file=log_path)

        # Log params
        tracker.log_params({'lr': 0.001})

        # Log multiple metrics
        for step in range(3):
            tracker.log_metrics({'loss': 1.0 / (step + 1)}, step=step)

        # Log artifact
        tracker.log_artifact('/model/final.pt')

        # End run
        tracker.end_run()

        # Verify all entries are in log
        with open(log_path, 'r') as f:
            content = f.read()

        assert 'Params:' in content
        assert 'lr' in content
        assert 'Metrics' in content
        assert 'Step: 0' in content
        assert 'Step: 1' in content
        assert 'Step: 2' in content
        assert 'Logging artifact' in content
        assert '/model/final.pt' in content
        assert 'Run ended' in content

    def test_file_tracker_creates_parent_directory(self, temp_dir):
        """Test that FileTracker creates parent directories if needed."""
        log_path = os.path.join(temp_dir, 'subdir', 'nested', 'test.log')
        tracker = FileTracker(log_file=log_path)

        assert os.path.exists(log_path)
        assert os.path.isfile(log_path)

    def test_log_large_params_dict(self, temp_dir):
        """Test logging a large dictionary of parameters."""
        log_path = os.path.join(temp_dir, 'large_params.log')
        tracker = FileTracker(log_file=log_path)

        params = {f'param_{i}': i * 0.1 for i in range(50)}
        tracker.log_params(params)

        with open(log_path, 'r') as f:
            content = f.read()

        # Check that all params are logged
        assert 'param_0' in content
        assert 'param_25' in content
        assert 'param_49' in content

    def test_log_various_metric_types(self, temp_dir):
        """Test logging metrics with different data types."""
        log_path = os.path.join(temp_dir, 'metric_types.log')
        tracker = FileTracker(log_file=log_path)

        metrics = {
            'int_metric': 42,
            'float_metric': 3.14159,
            'string_metric': 'test_value',
            'bool_metric': True,
        }
        tracker.log_metrics(metrics, step=0)

        with open(log_path, 'r') as f:
            content = f.read()

        assert '42' in content
        assert '3.14159' in content
        assert 'test_value' in content
        assert 'True' in content

    def test_log_artifact_with_type_creates_subdirectory(self, temp_dir):
        """Test that log_artifact with artifact_type creates typed subdirectory."""
        # Create a temporary artifact file
        artifact_path = os.path.join(temp_dir, 'model.pt')
        with open(artifact_path, 'w') as f:
            f.write('model data')

        # Create tracker with experiment dir
        exp_dir = os.path.join(temp_dir, 'experiment')
        tracker = FileTracker(experiment_dir=exp_dir)

        # Log artifact with type
        tracker.log_artifact(artifact_path, artifact_type='model')

        # Verify file was copied to typed subdirectory
        expected_path = os.path.join(exp_dir, 'artifacts', 'models', 'model.pt')
        assert os.path.exists(expected_path), f"Artifact not found at {expected_path}"

        # Verify content
        with open(expected_path, 'r') as f:
            content = f.read()
        assert content == 'model data'

    def test_log_artifact_all_types(self, temp_dir):
        """Test that all artifact types create correct subdirectories."""
        exp_dir = os.path.join(temp_dir, 'experiment')
        tracker = FileTracker(experiment_dir=exp_dir)

        artifact_types: list[ArtifactType] = ['model', 'plot', 'checkpoint', 'data', 'other']
        expected_subdirs = {
            'model': 'models',
            'plot': 'plots',
            'checkpoint': 'checkpoints',
            'data': 'data',
            'other': ''  # No subdirectory for 'other'
        }

        for artifact_type in artifact_types:
            # Create temporary artifact
            artifact_file = os.path.join(temp_dir, f'{artifact_type}_test.txt')
            with open(artifact_file, 'w') as f:
                f.write(f'test {artifact_type}')

            # Log with type
            tracker.log_artifact(artifact_file, artifact_type=artifact_type)

            # Verify location
            subdir = expected_subdirs[artifact_type]
            if subdir:
                expected_path = os.path.join(exp_dir, 'artifacts', subdir, f'{artifact_type}_test.txt')
            else:
                expected_path = os.path.join(exp_dir, 'artifacts', f'{artifact_type}_test.txt')

            assert os.path.exists(expected_path), f"Artifact not found for type {artifact_type} at {expected_path}"

    def test_log_artifact_with_custom_name(self, temp_dir):
        """Test that artifact_name parameter works."""
        # Create temporary artifact
        artifact_path = os.path.join(temp_dir, 'temp_model.pt')
        with open(artifact_path, 'w') as f:
            f.write('model data')

        exp_dir = os.path.join(temp_dir, 'experiment')
        tracker = FileTracker(experiment_dir=exp_dir)

        # Log with custom name
        tracker.log_artifact(artifact_path, artifact_type='model', artifact_name='final_model.pt')

        # Verify file saved with custom name
        expected_path = os.path.join(exp_dir, 'artifacts', 'models', 'final_model.pt')
        assert os.path.exists(expected_path)

        # Original filename should NOT exist
        wrong_path = os.path.join(exp_dir, 'artifacts', 'models', 'temp_model.pt')
        assert not os.path.exists(wrong_path)

    def test_get_artifact_path(self, temp_dir):
        """Test that get_artifact_path returns correct paths."""
        exp_dir = os.path.join(temp_dir, 'experiment')
        tracker = FileTracker(experiment_dir=exp_dir)

        # Create and log an artifact
        artifact_file = os.path.join(temp_dir, 'model.pt')
        with open(artifact_file, 'w') as f:
            f.write('model')

        tracker.log_artifact(artifact_file, artifact_type='model')

        # Get artifact path
        retrieved_path = tracker.get_artifact_path('model.pt', artifact_type='model')

        # Should return the correct path
        expected_path = os.path.join(exp_dir, 'artifacts', 'models', 'model.pt')
        assert retrieved_path == expected_path
        assert os.path.exists(retrieved_path)

    def test_get_artifact_path_returns_none_for_missing(self, temp_dir):
        """Test that get_artifact_path returns None for non-existent artifacts."""
        exp_dir = os.path.join(temp_dir, 'experiment')
        tracker = FileTracker(experiment_dir=exp_dir)

        # Try to get non-existent artifact
        path = tracker.get_artifact_path('nonexistent.pt', artifact_type='model')

        assert path is None

    def test_log_artifact_backward_compatible(self, temp_dir):
        """Test that log_artifact works without artifact_type (backward compatible)."""
        artifact_path = os.path.join(temp_dir, 'artifact.txt')
        with open(artifact_path, 'w') as f:
            f.write('test')

        exp_dir = os.path.join(temp_dir, 'experiment')
        tracker = FileTracker(experiment_dir=exp_dir)

        # Call without artifact_type (should default to 'other')
        tracker.log_artifact(artifact_path)

        # Should be saved to artifacts root (no subdirectory for 'other')
        expected_path = os.path.join(exp_dir, 'artifacts', 'artifact.txt')
        assert os.path.exists(expected_path)


class TestExperimentTracker:
    """Tests for ExperimentTracker abstract base class."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that ExperimentTracker cannot be instantiated directly."""
        with pytest.raises(TypeError):
            ExperimentTracker()

    def test_subclass_must_implement_methods(self):
        """Test that subclasses must implement all abstract methods."""

        # Create incomplete subclass (missing methods)
        class IncompleteTracker(ExperimentTracker):
            def log_params(self, params):
                pass
            # Missing other methods

        with pytest.raises(TypeError):
            IncompleteTracker()

    def test_complete_subclass_can_be_instantiated(self):
        """Test that complete subclass can be instantiated."""

        class CompleteTracker(ExperimentTracker):
            def start_run(self, run_name=None):
                pass

            def log_params(self, params):
                pass

            def log_metrics(self, metrics, step=None):
                pass

            def log_artifact(self, local_path, artifact_type='other', artifact_name=None):
                pass

            def end_run(self):
                pass

        # Should not raise an error
        tracker = CompleteTracker()
        assert isinstance(tracker, ExperimentTracker)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
