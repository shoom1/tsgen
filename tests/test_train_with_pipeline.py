"""
Tests for train.py with YAML-configured pipeline.

Tests the DataPipeline-based training approach.
"""

import pytest
import torch
import os
import tempfile
from pathlib import Path

from tsgen.train import train_model
from tsgen.tracking.base import FileTracker
from tsgen.config.schema import ExperimentConfig


@pytest.fixture
def pipeline_config():
    """Minimal config with DataPipeline."""
    return ExperimentConfig(
        experiment_name='test_pipeline_training',
        model_type='unet',
        tickers=['AAPL', 'MSFT'],
        start_date='2024-01-01',
        end_date='2024-12-31',
        sequence_length=64,
        batch_size=16,
        epochs=2,  # Fast
        timesteps=50,  # Fewer timesteps
        learning_rate=1e-3,
        base_channels=32,
        tracker='file',

        # data_pipeline configuration
        data_pipeline=[
            {'load_prices': {'column': 'adj_close'}},
            {'clean_data': {'strategy': 'ffill_drop'}},
            {'process_prices': {'fit': True}},
            {'create_windows': {'sequence_length': 64}},
            {'create_dataloader': {'batch_size': 16, 'shuffle': True}}
        ]
    )


class TestTrainWithPipeline:
    """Tests for training with YAML-configured pipeline."""

    def test_train_with_pipeline_config(self, pipeline_config):
        """Test that training works with data_pipeline config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = FileTracker(experiment_dir=tmpdir)

            # Run training with pipeline
            model, processor = train_model(pipeline_config, tracker)

            # Verify model was created
            assert model is not None
            assert isinstance(model, torch.nn.Module)

            # Verify processor was created
            assert processor is not None

            # Verify artifacts were saved
            assert os.path.exists(os.path.join(tmpdir, 'artifacts', 'models', 'model_final.pt'))
            assert os.path.exists(os.path.join(tmpdir, 'artifacts', 'data', 'processor.pkl'))

    def test_pipeline_with_split(self):
        """Test pipeline with train/test split."""
        config = ExperimentConfig(
            experiment_name='test_split',
            model_type='unet',
            tickers=['AAPL', 'MSFT'],
            start_date='2024-01-01',
            end_date='2024-12-31',
            sequence_length=64,
            batch_size=16,
            epochs=1,
            timesteps=50,
            learning_rate=1e-3,
            base_channels=32,

            # data_pipeline with split_temporal step
            data_pipeline=[
                {'load_prices': {'column': 'adj_close'}},
                {'clean_data': {'strategy': 'ffill_drop'}},
                {'split_temporal': {'train_ratio': 0.8}},
                {'process_prices': {'fit': True}},
                {'create_windows': {'sequence_length': 64}},
                {'create_dataloader': {'batch_size': 16, 'shuffle': True}}
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = FileTracker(experiment_dir=tmpdir)

            # Should work with split_temporal in pipeline
            # Note: split_temporal returns tuple, but process_prices handles it
            model, processor = train_model(config, tracker)

            assert model is not None

    def test_pipeline_creates_model_with_correct_features(self, pipeline_config):
        """Test that the model is created with the correct feature count."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = FileTracker(experiment_dir=tmpdir)

            # Run training
            model, processor = train_model(pipeline_config, tracker)

            # Verify the model has the correct number of features
            # (2 tickers = 2 features)
            assert hasattr(model, 'features')
            assert model.features == 2  # AAPL, MSFT


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
