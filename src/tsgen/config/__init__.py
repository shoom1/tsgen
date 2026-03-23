"""
Configuration management with Pydantic validation.

This module provides structured configuration schemas with validation,
defaults, and backward compatibility for both flat and nested config formats.

Example usage:
    from tsgen.config import validate_config, ExperimentConfig

    # Validate raw config dict
    config = validate_config(raw_config)

    # Access typed config sections
    training = config.get_training_config()
    print(training.epochs)

    # Load and validate from file
    from tsgen.config import load_and_validate_config
    config = load_and_validate_config('config.yaml')
"""

from tsgen.config.schema import (
    # Config classes
    DataConfig,
    DiffusionConfig,
    TrainingConfig,
    VAEConfig,
    ModelParamsConfig,
    EvaluationConfig,
    TrackerConfig,
    ExperimentConfig,
    # Functions
    validate_config,
    load_and_validate_config,
)

__all__ = [
    'DataConfig',
    'DiffusionConfig',
    'TrainingConfig',
    'VAEConfig',
    'ModelParamsConfig',
    'EvaluationConfig',
    'TrackerConfig',
    'ExperimentConfig',
    'validate_config',
    'load_and_validate_config',
]
