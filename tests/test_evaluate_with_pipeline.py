"""
Tests for evaluate.py with DataPipeline configuration.
"""

import pytest
import torch
import tempfile

from tsgen.train import train_model
from tsgen.evaluate import evaluate_model
from tsgen.evaluation import EvaluationResult
from tsgen.tracking.base import FileTracker
from tsgen.config.schema import ExperimentConfig


@pytest.fixture
def pipeline_config():
    """Config with DataPipeline for both training and evaluation."""
    return ExperimentConfig(
        experiment_name='test_eval_pipeline',
        model_type='unet',
        tickers=['AAPL', 'MSFT'],
        start_date='2024-01-01',
        end_date='2024-12-31',
        sequence_length=64,
        batch_size=16,
        epochs=2,
        timesteps=50,
        learning_rate=1e-3,
        base_channels=32,
        num_samples=50,  # For evaluation

        # data_pipeline configuration
        data_pipeline=[
            {'load_prices': {'column': 'adj_close'}},
            {'clean_data': {'strategy': 'ffill_drop'}},
            {'process_prices': {'fit': True}},
            {'create_windows': {'sequence_length': 64}},
            {'create_dataloader': {'batch_size': 16, 'shuffle': True}}
        ]
    )


class TestEvaluateWithPipeline:
    """Tests for evaluation with data_pipeline configuration."""

    def test_evaluate_with_pipeline_config(self, pipeline_config):
        """Test that evaluation works with data_pipeline config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = FileTracker(experiment_dir=tmpdir)

            # First train a model
            model, processor = train_model(pipeline_config, tracker)
            assert model is not None

            # Now evaluate with ExperimentConfig directly
            result = evaluate_model(pipeline_config, tracker)

            # Verify metrics were computed
            assert result is not None
            assert isinstance(result, EvaluationResult)
            assert 'discriminator_accuracy' in result.metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
