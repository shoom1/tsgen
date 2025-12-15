"""
Main training module with Trainer Registry Pattern.

Uses Strategy + Registry + Factory patterns to dispatch to appropriate
trainer based on model type.

Data pipeline is configured via YAML using the DataPipeline key.
"""

import torch
import tempfile
import os
from tsgen.models.factory import create_model
from tsgen.data.pipeline_builder import DataPipeline
from tsgen.data.processor import LogReturnProcessor
from tsgen.tracking.base import ExperimentTracker
from tsgen.training.registry import TrainerRegistry

# Import trainers to trigger registration
from tsgen.training.diffusion_trainer import DiffusionTrainer
from tsgen.training.vae_trainer import VAETrainer
from tsgen.training.baseline_trainer import BaselineTrainer


def train_model(config, tracker: ExperimentTracker):
    """
    Main training entry point.

    Uses Trainer Registry Pattern to dispatch to appropriate trainer
    based on model type. Supports diffusion models (UNet, Transformer),
    VAE models (TimeVAE), and baseline models (GBM, Bootstrap, etc.).

    Data pipeline is configured via YAML using config['DataPipeline'].

    Args:
        config: Configuration dictionary containing hyperparameters
        tracker: Experiment tracker for logging metrics and artifacts

    Returns:
        tuple: (trained_model, data_processor)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}")

    # Log parameters
    tracker.log_params(config)

    # Create processor
    processor = LogReturnProcessor()

    # Build and execute YAML-configured data pipeline
    print("Using YAML-configured data pipeline")
    pipeline = DataPipeline.from_config(config)

    # Execute pipeline with runtime parameters
    dataloader = pipeline.execute(
        tickers=config['tickers'],
        start_date=config['start_date'],
        end_date=config['end_date'],
        column=config.get('column', 'adj_close'),
        db_path=config.get('db_path'),
        processor=processor
    )

    # Get feature count from config
    config['num_features'] = len(config['tickers'])

    # Create model
    model = create_model(config)

    # Get appropriate trainer via registry (Factory Pattern)
    model_type = config['model_type']
    print(f"Using trainer for model type: {model_type}")
    trainer = TrainerRegistry.get_trainer(model_type, model, config, tracker, device)

    # Train (Strategy Pattern)
    print(f"Trainer: {trainer.__class__.__name__}")
    model = trainer.train(dataloader)

    # Save final model and processor to temporary files
    # Tracker will handle final storage location
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_model = os.path.join(tmpdir, "model_final.pt")
        tmp_processor = os.path.join(tmpdir, "processor.pkl")

        trainer.save_model(tmp_model)
        processor.save(tmp_processor)

        # Log artifacts with types - tracker manages final storage
        tracker.log_artifact(tmp_model, artifact_type='model')
        tracker.log_artifact(tmp_processor, artifact_type='data')

    print("Training complete. Artifacts logged to tracker.")
    return model, processor
