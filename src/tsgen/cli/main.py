#!/usr/bin/env python3
"""Main CLI entry point for tsgen."""

import argparse
import yaml
import os
import warnings
from pathlib import Path

from pydantic import ValidationError

from tsgen.train import train_model
from tsgen.evaluate import evaluate_model
from tsgen.tracking.factory import create_tracker
from tsgen.tracking.base import FileTracker
from tsgen.experiments.manager import ExperimentManager
from tsgen.training.checkpoint_utils import get_checkpoint_path, list_checkpoints
from tsgen.config import validate_config
from tsgen.config.schema import ExperimentConfig


def load_config(config_path, validate=True):
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file
        validate: Whether to validate config with Pydantic (default: True)

    Returns:
        ExperimentConfig: Validated configuration object

    Raises:
        ValidationError: If config validation fails
    """
    with open(config_path, 'r') as f:
        raw_config = yaml.safe_load(f)

    if validate:
        try:
            return validate_config(raw_config)
        except ValidationError as e:
            print(f"\nConfiguration validation failed for: {config_path}")
            print("=" * 60)
            for error in e.errors():
                loc = ' -> '.join(str(x) for x in error['loc'])
                print(f"  {loc}: {error['msg']}")
            print("=" * 60)
            raise

    # Unvalidated path: wrap raw dict in ExperimentConfig
    return ExperimentConfig(**raw_config)


def setup_experiment(config, experiment_number, model_name):
    """
    Set up experiment folder structure for model-specific runs.

    Args:
        config: ExperimentConfig or dict
        experiment_number: Experiment number or None
        model_name: Model name for this run or None

    Returns:
        Tuple of (experiment_dir, model_name) where:
        - experiment_dir: Path to experiment folder or None
        - model_name: Resolved model name or None
    """
    if experiment_number is None:
        # No experiment management
        return None, None

    manager = ExperimentManager()

    # Check if experiment folder exists
    exp_path = manager.get_experiment_path(f"{experiment_number:04d}")

    if exp_path is None:
        # Create new experiment
        exp_name = getattr(config, 'experiment_name', None) or 'unnamed'
        short_name = exp_name.lower().replace('-', '_').replace(' ', '_')
        parts = short_name.split('_')
        short_name = '_'.join(parts[:3]) if len(parts) > 3 else short_name

        description = getattr(config, 'experiment_description', '')
        exp_path = manager.create_experiment(
            name=short_name,
            config=None,  # Don't save single config for multi-model experiments
            description=description,
            experiment_number=experiment_number
        )

        print(f"\n{'='*80}")
        print(f"Created Experiment {experiment_number:04d}: {short_name}")
        print(f"Directory: {exp_path}")
        print(f"{'='*80}\n")
    else:
        print(f"\n{'='*80}")
        print(f"Using Existing Experiment {experiment_number:04d}")
        print(f"Directory: {exp_path}")
        print(f"{'='*80}\n")

    # Determine model name
    if model_name is None:
        model_conf = getattr(config, 'model', None)
        if model_conf is not None and hasattr(model_conf, 'name'):
            model_name = model_conf.name
        else:
            model_name = getattr(config, 'model_type', 'default')

    # Add model config to experiment (manager expects dict)
    config_dict = config.to_dict() if hasattr(config, 'to_dict') else config
    manager.add_model_config(exp_path, model_name, config_dict)

    return str(exp_path), model_name


def main():
    parser = argparse.ArgumentParser(description="Financial Time Series Generation")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--mode", type=str, default="train_eval", choices=["train", "eval", "train_eval"], help="Mode to run")
    parser.add_argument("--experiment-number", type=int, help="Experiment number (creates or adds to experiment folder)")
    parser.add_argument("--model-name", type=str, help="Model name within experiment (e.g., baseline, timevae)")
    parser.add_argument("--resume-from-checkpoint", type=str, help="Path to checkpoint file to resume training from")
    parser.add_argument("--resume-latest", action="store_true", help="Resume from latest checkpoint in experiment directory")
    parser.add_argument("--list-checkpoints", action="store_true", help="List available checkpoints and exit")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Setup experiment structure
    experiment_dir, model_name = setup_experiment(config, args.experiment_number, args.model_name)

    # Handle checkpoint listing
    if args.list_checkpoints:
        if not experiment_dir:
            print("Error: --list-checkpoints requires --experiment-number")
            return

        checkpoint_dir = os.path.join(experiment_dir, 'artifacts', 'checkpoints')
        checkpoints = list_checkpoints(checkpoint_dir)

        if not checkpoints:
            print(f"No checkpoints found in {checkpoint_dir}")
        else:
            print(f"\nAvailable checkpoints in experiment {args.experiment_number}:")
            print(f"{'='*60}")
            for epoch, path in checkpoints:
                print(f"  Epoch {epoch:3d}: {os.path.basename(path)}")
            print(f"{'='*60}")
            print(f"\nTo resume from a checkpoint, use:")
            print(f"  --resume-from-checkpoint {checkpoints[0][1]}")
            print(f"Or use --resume-latest to automatically use the latest checkpoint")
        return

    # Handle checkpoint resumption
    if args.resume_latest or args.resume_from_checkpoint:
        if args.resume_latest:
            if not experiment_dir:
                print("Error: --resume-latest requires --experiment-number")
                return

            # Find latest checkpoint
            checkpoint_path = get_checkpoint_path(experiment_dir)
            if not checkpoint_path:
                print(f"Error: No checkpoints found in {experiment_dir}/artifacts/checkpoints")
                return

            print(f"\n{'='*80}")
            print(f"Auto-resuming from latest checkpoint: {os.path.basename(checkpoint_path)}")
            print(f"{'='*80}\n")
        else:
            checkpoint_path = args.resume_from_checkpoint
            if not os.path.exists(checkpoint_path):
                print(f"Error: Checkpoint not found: {checkpoint_path}")
                return

        # Add checkpoint path to config for train.py to use
        config.resume_from_checkpoint = checkpoint_path

    if experiment_dir:
        # Experiment mode: use model-specific tracker and paths
        tracker = FileTracker(
            log_file=f"training_{model_name}.log",
            experiment_dir=experiment_dir
        )

        # Add experiment and model info to config for use in training/evaluation
        config._experiment_path = experiment_dir
        config._model_name = model_name

        print(f"Model: {model_name}")
        print(f"Log file: {os.path.join(experiment_dir, f'training_{model_name}.log')}\n")
    else:
        # Standalone mode: use regular tracker
        tracker = create_tracker(config)

    try:
        exp_name = getattr(config, 'experiment_name', None) or 'default'
        tracker.start_run(run_name=f"Run_{exp_name}_{model_name or 'default'}")

        if "train" in args.mode:
            train_model(config, tracker)

        if "eval" in args.mode:
            evaluate_model(config, tracker)

        tracker.end_run()

        print(f"\n{'='*80}")
        if experiment_dir:
            print(f"Experiment complete!")
            print(f"  Model: {model_name}")
            print(f"  Log: {os.path.join(experiment_dir, f'training_{model_name}.log')}")
            print(f"  Artifacts: {os.path.join(experiment_dir, 'artifacts', model_name)}")
            print(f"  Plots: {os.path.join(experiment_dir, 'plots', model_name)}")
            print(f"  Results: {os.path.join(experiment_dir, 'results.md')}")
        else:
            print("Training complete!")
        print(f"{'='*80}\n")

    except Exception as e:
        print(f"\nError during execution: {e}")
        raise


if __name__ == "__main__":
    main()
