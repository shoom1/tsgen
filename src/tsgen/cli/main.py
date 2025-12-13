#!/usr/bin/env python3
"""Main CLI entry point for tsgen."""

import argparse
import yaml
import os
import warnings
from pathlib import Path

from tsgen.train import train_model
from tsgen.evaluate import evaluate_model
from tsgen.tracking.mlflow_tracker import MLFlowTracker
from tsgen.tracking.base import ConsoleTracker, NoOpTracker, FileTracker
from tsgen.experiments.manager import ExperimentManager

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_tracker(config, experiment_dir=None):
    """
    Create tracker based on config.

    Args:
        config: Configuration dictionary
        experiment_dir: Optional experiment directory for file-based tracking

    Returns:
        ExperimentTracker instance
    """
    # Warn about deprecated config keys
    if 'output_dir' in config:
        warnings.warn(
            "Config key 'output_dir' is deprecated. "
            "Use 'experiment_dir' instead, or let trackers manage artifact paths automatically. "
            "For experiment management, use --experiment-number CLI flag.",
            DeprecationWarning,
            stacklevel=2
        )

    tracker_type = config.get('tracker', 'console').lower()
    exp_name = config.get('experiment_name', 'Default_Experiment')

    if tracker_type == 'mlflow':
        # Support optional MLflow configuration
        return MLFlowTracker(
            experiment_name=exp_name,
            tracking_uri=config.get('mlflow_tracking_uri'),
            artifact_location=config.get('mlflow_artifact_location')
        )
    elif tracker_type == 'console':
        return ConsoleTracker()
    elif tracker_type == 'file':
        # Priority 1: Use experiment_dir parameter (multi-model experiments)
        if experiment_dir:
            return FileTracker(log_file="training.log", experiment_dir=experiment_dir)
        # Priority 2: Use output_dir from config (multi-run experiments)
        elif 'output_dir' in config:
            return FileTracker(log_file="training.log", experiment_dir=config['output_dir'])
        # Priority 3: Use experiment_dir from config (if specified)
        elif 'experiment_dir' in config:
            return FileTracker(log_file="training.log", experiment_dir=config['experiment_dir'])
        # Priority 4: Standalone log file
        else:
            log_file = config.get('log_file', f"logs/{exp_name}.log")
            return FileTracker(log_file=log_file)
    elif tracker_type == 'noop':
        return NoOpTracker()
    else:
        print(f"Warning: Unknown tracker '{tracker_type}'. Defaulting to Console.")
        return ConsoleTracker()


def setup_experiment(config, experiment_number, model_name):
    """
    Set up experiment folder structure for model-specific runs.

    Args:
        config: Configuration dictionary
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
        exp_name = config.get('experiment_name', 'unnamed')
        short_name = exp_name.lower().replace('-', '_').replace(' ', '_')
        parts = short_name.split('_')
        short_name = '_'.join(parts[:3]) if len(parts) > 3 else short_name

        description = config.get('experiment_description', '')
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
        model_name = config.get('model_type', 'default')

    # Add model config to experiment
    manager.add_model_config(exp_path, model_name, config)

    return str(exp_path), model_name


def main():
    parser = argparse.ArgumentParser(description="Financial Time Series Generation")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--mode", type=str, default="train_eval", choices=["train", "eval", "train_eval"], help="Mode to run")
    parser.add_argument("--experiment-number", type=int, help="Experiment number (creates or adds to experiment folder)")
    parser.add_argument("--model-name", type=str, help="Model name within experiment (e.g., baseline, timevae)")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Setup experiment structure
    experiment_dir, model_name = setup_experiment(config, args.experiment_number, args.model_name)

    if experiment_dir:
        # Experiment mode: use model-specific tracker and paths
        tracker = FileTracker(
            log_file=f"training_{model_name}.log",
            experiment_dir=experiment_dir
        )

        # Add experiment and model info to config for use in training/evaluation
        config['_experiment_path'] = experiment_dir
        config['_model_name'] = model_name

        print(f"Model: {model_name}")
        print(f"Log file: {os.path.join(experiment_dir, f'training_{model_name}.log')}\n")
    else:
        # Standalone mode: use regular tracker
        tracker = get_tracker(config)

    try:
        tracker.start_run(run_name=f"Run_{config.get('experiment_name', 'default')}_{model_name or 'default'}")

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
