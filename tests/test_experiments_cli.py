"""Tests for the experiments CLI (tsgen-experiments)."""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
from argparse import Namespace

from tsgen.cli.experiments import list_experiments, show_info, create_experiment, main


@pytest.fixture
def temp_experiments_dir():
    """Create a temp dir with fake experiment structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a fake experiment
        exp_dir = Path(tmpdir) / "experiments" / "0001_test_exp"
        exp_dir.mkdir(parents=True)

        # Write README
        (exp_dir / "README.md").write_text("# Test Experiment\nA test.")

        # Write config
        (exp_dir / "config.yaml").write_text("model_type: unet\n")

        yield tmpdir, exp_dir


class TestListExperiments:
    def test_list_with_experiments(self, temp_experiments_dir, capsys):
        tmpdir, exp_dir = temp_experiments_dir
        mock_manager = MagicMock()
        mock_manager.list_experiments.return_value = [
            {'number': '0001', 'name': 'test_exp', 'model': 'unet', 'completed': True}
        ]

        with patch('tsgen.cli.experiments.ExperimentManager', return_value=mock_manager):
            list_experiments(Namespace())

        output = capsys.readouterr().out
        assert '0001' in output
        assert 'test_exp' in output

    def test_list_no_experiments(self, capsys):
        mock_manager = MagicMock()
        mock_manager.list_experiments.return_value = []

        with patch('tsgen.cli.experiments.ExperimentManager', return_value=mock_manager):
            list_experiments(Namespace())

        output = capsys.readouterr().out
        assert 'No experiments found' in output


class TestShowInfo:
    def test_show_info_found(self, temp_experiments_dir, capsys):
        tmpdir, exp_dir = temp_experiments_dir
        mock_manager = MagicMock()
        mock_manager.get_experiment_path.return_value = exp_dir

        with patch('tsgen.cli.experiments.ExperimentManager', return_value=mock_manager):
            show_info(Namespace(experiment_id='1'))

        output = capsys.readouterr().out
        assert 'Test Experiment' in output

    def test_show_info_not_found(self, capsys):
        mock_manager = MagicMock()
        mock_manager.get_experiment_path.return_value = None

        with patch('tsgen.cli.experiments.ExperimentManager', return_value=mock_manager):
            show_info(Namespace(experiment_id='999'))

        output = capsys.readouterr().out
        assert 'not found' in output


class TestCreateExperiment:
    def test_create(self, capsys):
        mock_manager = MagicMock()
        mock_manager.get_next_experiment_number.return_value = 7
        mock_manager.create_experiment.return_value = Path("/tmp/0007_new_exp")

        with patch('tsgen.cli.experiments.ExperimentManager', return_value=mock_manager):
            create_experiment(Namespace(
                name='new_exp', model='unet', description='A new experiment'
            ))

        mock_manager.create_experiment.assert_called_once()
        output = capsys.readouterr().out
        assert '0007' in output


class TestMainEntrypoint:
    def test_no_command_shows_help(self, capsys):
        with patch('sys.argv', ['tsgen-experiments']):
            main()
        # Should not crash; prints help

    def test_list_command(self):
        mock_manager = MagicMock()
        mock_manager.list_experiments.return_value = []

        with patch('tsgen.cli.experiments.ExperimentManager', return_value=mock_manager):
            with patch('sys.argv', ['tsgen-experiments', 'list']):
                main()

        mock_manager.list_experiments.assert_called_once()
