"""
Main training module with Trainer Registry Pattern.

Uses Strategy + Registry + Factory patterns to dispatch to appropriate
trainer based on model type.

Data pipeline is configured via YAML using the data_pipeline key.
"""

import torch
import tempfile
import os
from tsgen.models.registry import ModelRegistry
from tsgen.config.schema import ExperimentConfig
from tsgen.data.pipeline_builder import DataPipeline
from tsgen.data.processor import LogReturnProcessor
from tsgen.tracking.base import ExperimentTracker
from tsgen.training.registry import TrainerRegistry
from tsgen.training.checkpoint_utils import extract_epoch_from_checkpoint

# Import models package to trigger ModelRegistry registration
import tsgen.models

# Import trainers to trigger TrainerRegistry registration
from tsgen.training.diffusion_trainer import DiffusionTrainer
from tsgen.training.vae_trainer import VAETrainer
from tsgen.training.baseline_trainer import BaselineTrainer


def train_model(config: ExperimentConfig, tracker: ExperimentTracker):
    """
    Main training entry point.

    Uses Trainer Registry Pattern to dispatch to appropriate trainer
    based on model type. Supports diffusion models (UNet, Transformer),
    VAE models (TimeVAE), and baseline models (GBM, Bootstrap, etc.).

    Data pipeline is configured via YAML using config.data_pipeline.

    Args:
        config: ExperimentConfig with validated hyperparameters
        tracker: Experiment tracker for logging metrics and artifacts

    Returns:
        tuple: (trained_model, data_processor)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}")

    # Log parameters
    tracker.log_params(config.to_dict())

    # Create processor
    processor = LogReturnProcessor()

    # Resolve data configuration
    data_conf = config.get_data_config()

    # Build and execute YAML-configured data pipeline
    print("Using YAML-configured data pipeline")
    pipeline = DataPipeline.from_config(config)

    # Execute pipeline with runtime parameters
    dataloader = pipeline.execute(
        tickers=data_conf.tickers if data_conf.tickers else None,
        start_date=data_conf.start_date,
        end_date=data_conf.end_date,
        column=data_conf.column,
        db_path=data_conf.db_path,
        processor=processor
    )

    # Get feature count from config or processor
    tickers = data_conf.tickers
    if tickers:
        features = len(tickers)
    elif hasattr(processor, 'n_features_'):
        features = processor.n_features_
    else:
        features = None

    # Create model via ModelRegistry
    model = ModelRegistry.create(config, features=features)

    # Get appropriate trainer via registry (Factory Pattern)
    model_type = config.model_type
    print(f"Using trainer for model type: {model_type}")
    trainer = TrainerRegistry.get_trainer(model_type, model, config, tracker, device)

    # Load checkpoint if resuming
    checkpoint_path = getattr(config, 'resume_from_checkpoint', None)
    if checkpoint_path:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print(f"\n{'='*80}")
        print(f"Loading checkpoint from: {checkpoint_path}")

        # Load checkpoint and restore training state
        checkpoint = trainer.load_checkpoint(
            checkpoint_path,
            optimizer=trainer.optimizer if hasattr(trainer, 'optimizer') else None
        )

        # Set start epoch from checkpoint
        start_epoch = checkpoint.get('epoch', 0)
        config.start_epoch = start_epoch

        print(f"Resuming from epoch {start_epoch}")
        if 'step_count' in checkpoint:
            print(f"Step count: {checkpoint['step_count']}")
        print(f"{'='*80}\n")

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
