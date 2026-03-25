"""Tests for CLI modules."""

import pytest
import tempfile
import os
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock

from tsgen.cli.main import load_config, setup_experiment
from tsgen.tracking.factory import create_tracker
from tsgen.tracking.base import ConsoleTracker, NoOpTracker, FileTracker
from tsgen.config.schema import ExperimentConfig


@pytest.fixture
def temp_dir():
    """Create temporary directory for test artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_config_file(temp_dir):
    """Create a sample YAML config file."""
    config = {
        'experiment_name': 'test_experiment',
        'model_type': 'unet',
        'data': {
            'sequence_length': 64,
            'tickers': ['AAPL', 'MSFT'],
        },
        'tracker': 'console',
    }
    config_path = os.path.join(temp_dir, 'test_config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    return config_path


def test_load_config(sample_config_file):
    """Test loading YAML configuration returns ExperimentConfig."""
    config = load_config(sample_config_file)

    assert config is not None
    assert isinstance(config, ExperimentConfig)
    assert config.experiment_name == 'test_experiment'
    assert config.model_type == 'unet'
    assert config.data.sequence_length == 64
    assert config.data.tickers == ['AAPL', 'MSFT']


def test_load_config_invalid_path():
    """Test loading config with invalid path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_config('nonexistent_config.yaml')


def test_create_tracker_console():
    """Test creating console tracker."""
    config = {'tracker': 'console', 'experiment_name': 'test'}
    tracker = create_tracker(config)

    assert isinstance(tracker, ConsoleTracker)


def test_create_tracker_noop():
    """Test creating no-op tracker."""
    config = {'tracker': 'noop', 'experiment_name': 'test'}
    tracker = create_tracker(config)

    assert isinstance(tracker, NoOpTracker)


def test_create_tracker_file(temp_dir):
    """Test creating file tracker."""
    config = {
        'tracker': 'file',
        'experiment_name': 'test',
        'log_file': os.path.join(temp_dir, 'test.log')
    }
    tracker = create_tracker(config)

    assert isinstance(tracker, FileTracker)


def test_create_tracker_file_with_experiment_dir(temp_dir):
    """Test creating file tracker with experiment directory."""
    config = {'tracker': 'file', 'experiment_name': 'test'}
    tracker = create_tracker(config, experiment_dir=temp_dir)

    assert isinstance(tracker, FileTracker)


def test_create_tracker_unknown_defaults_to_console():
    """Test that unknown tracker type defaults to console."""
    config = {'tracker': 'unknown_tracker', 'experiment_name': 'test'}

    with patch('builtins.print') as mock_print:
        tracker = create_tracker(config)

        # Should print warning
        mock_print.assert_called_once()
        assert 'Unknown tracker' in str(mock_print.call_args)

    # Should default to ConsoleTracker
    assert isinstance(tracker, ConsoleTracker)


def test_create_tracker_default():
    """Test tracker defaults to console when not specified."""
    config = {'experiment_name': 'test'}  # No tracker specified
    tracker = create_tracker(config)

    assert isinstance(tracker, ConsoleTracker)


def test_setup_experiment_no_number():
    """Test setup_experiment returns None when no experiment number provided."""
    config = ExperimentConfig(model_type='unet', experiment_name='test')
    exp_dir, model_name = setup_experiment(config, None, None)

    assert exp_dir is None
    assert model_name is None


def test_setup_experiment_with_mock_manager(temp_dir):
    """Test setup_experiment with mocked ExperimentManager."""
    config = ExperimentConfig(model_type='unet', experiment_name='test_experiment')

    # Mock the ExperimentManager
    with patch('tsgen.cli.main.ExperimentManager') as MockManager:
        mock_manager = MockManager.return_value
        mock_manager.get_experiment_path.return_value = None
        mock_manager.create_experiment.return_value = temp_dir

        exp_dir, model_name = setup_experiment(config, 1, 'unet_v1')

        # Should call create_experiment if path doesn't exist
        mock_manager.create_experiment.assert_called_once()

        assert exp_dir == temp_dir
        assert model_name == 'unet_v1'


def test_create_tracker_mlflow():
    """Test creating MLflow tracker."""
    config = {'tracker': 'mlflow', 'experiment_name': 'test'}

    # Mock MLFlowTracker to avoid MLflow dependencies in tests
    with patch('tsgen.tracking.mlflow_tracker.MLFlowTracker') as MockMLFlow:
        mock_tracker = MagicMock()
        MockMLFlow.return_value = mock_tracker

        tracker = create_tracker(config)

        MockMLFlow.assert_called_once_with(
            experiment_name='test',
            tracking_uri=None,
            artifact_location=None
        )
        assert tracker == mock_tracker


def test_create_tracker_output_dir_fallback(temp_dir):
    """Test that output_dir config key still works as fallback."""
    config = {'tracker': 'file', 'output_dir': temp_dir}

    tracker = create_tracker(config)

    # Should still work (backward compatibility)
    assert isinstance(tracker, FileTracker)


class TestMainCLI:
    """Tests for main CLI function."""

    def test_main_requires_config(self):
        """Test that main requires --config argument."""
        with patch('sys.argv', ['tsgen']):
            with patch('tsgen.cli.main.argparse.ArgumentParser.parse_args') as mock_parse:
                mock_parse.side_effect = SystemExit(2)

                with pytest.raises(SystemExit):
                    from tsgen.cli.main import main
                    main()

    def test_main_with_train_mode(self, sample_config_file, temp_dir):
        """Test main CLI with train mode."""
        # Change to temp directory
        original_dir = os.getcwd()
        os.chdir(temp_dir)

        try:
            with patch('sys.argv', ['tsgen', '--config', sample_config_file, '--mode', 'train']):
                with patch('tsgen.cli.main.train_model') as mock_train:
                    with patch('tsgen.cli.main.create_tracker') as mock_create_tracker:
                        mock_tracker = MagicMock()
                        mock_create_tracker.return_value = mock_tracker
                        mock_train.return_value = (MagicMock(), MagicMock())

                        from tsgen.cli.main import main
                        main()

                        # Should call train_model
                        mock_train.assert_called_once()
        finally:
            os.chdir(original_dir)

    def test_main_with_eval_mode(self, sample_config_file, temp_dir):
        """Test main CLI with eval mode."""
        original_dir = os.getcwd()
        os.chdir(temp_dir)

        try:
            with patch('sys.argv', ['tsgen', '--config', sample_config_file, '--mode', 'eval']):
                with patch('tsgen.cli.main.evaluate_model') as mock_eval:
                    with patch('tsgen.cli.main.create_tracker') as mock_create_tracker:
                        mock_tracker = MagicMock()
                        mock_create_tracker.return_value = mock_tracker
                        mock_eval.return_value = MagicMock()

                        from tsgen.cli.main import main
                        main()

                        # Should call evaluate_model
                        mock_eval.assert_called_once()
        finally:
            os.chdir(original_dir)

    def test_main_with_train_eval_mode(self, sample_config_file, temp_dir):
        """Test main CLI with train_eval mode."""
        original_dir = os.getcwd()
        os.chdir(temp_dir)

        try:
            with patch('sys.argv', ['tsgen', '--config', sample_config_file, '--mode', 'train_eval']):
                with patch('tsgen.cli.main.train_model') as mock_train:
                    with patch('tsgen.cli.main.evaluate_model') as mock_eval:
                        with patch('tsgen.cli.main.create_tracker') as mock_create_tracker:
                            mock_tracker = MagicMock()
                            mock_create_tracker.return_value = mock_tracker
                            mock_train.return_value = (MagicMock(), MagicMock())
                            mock_eval.return_value = MagicMock()

                            from tsgen.cli.main import main
                            main()

                            # Should call both train and evaluate
                            mock_train.assert_called_once()
                            mock_eval.assert_called_once()
        finally:
            os.chdir(original_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
