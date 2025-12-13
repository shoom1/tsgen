"""
Main training module with Trainer Registry Pattern.

Uses Strategy + Registry + Factory patterns to dispatch to appropriate
trainer based on model type.
"""

import torch
import tempfile
import os
from tsgen.models.factory import create_model
from tsgen.data.pipeline import (
    load_prices, clean_data, split_temporal,
    process_prices, create_windows, create_dataloader
)
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

    # Data Pipeline
    # 1. Load prices
    df = load_prices(
        config['tickers'],
        config['start_date'],
        config['end_date'],
        column=config.get('column', 'adj_close'),
        db_path=config.get('db_path')
    )

    # 2. Clean data
    df_clean = clean_data(df, strategy='ffill_drop')

    # 3. Split if requested (temporal split for proper out-of-sample evaluation)
    train_test_split = config.get('train_test_split')
    if train_test_split:
        train_df, _ = split_temporal(df_clean, train_ratio=train_test_split)
    else:
        train_df = df_clean

    # 4. Process (fit processor on training data only)
    processor = LogReturnProcessor()
    train_scaled = process_prices(train_df, processor, fit=True)

    # 5. Create windows
    train_sequences = create_windows(train_scaled, sequence_length=config['sequence_length'])

    # 6. Create DataLoader
    dataloader = create_dataloader(
        train_sequences,
        batch_size=config['batch_size'],
        shuffle=True
    )

    # Store feature count in config for trainers that need it
    ticker_names = df_clean.columns.tolist()
    config['num_features'] = len(ticker_names)

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
