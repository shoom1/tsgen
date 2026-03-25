"""
Configuration management with Pydantic validation.

This module provides structured configuration schemas with validation
and defaults for nested config format.

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
    TrainingConfig,
    DiffusionTrainingConfig,
    VAETrainingConfig,
    BaselineTrainingConfig,
    UNetModelConfig,
    TransformerModelConfig,
    MambaModelConfig,
    VAEModelConfig,
    BaselineModelConfig,
    EvaluationConfig,
    TrackerConfig,
    ExperimentConfig,
    # Mapping dicts
    MODEL_CONFIG_MAP,
    TRAINING_CONFIG_MAP,
    # Functions
    validate_config,
    load_and_validate_config,
)

__all__ = [
    'DataConfig',
    'TrainingConfig',
    'DiffusionTrainingConfig',
    'VAETrainingConfig',
    'BaselineTrainingConfig',
    'UNetModelConfig',
    'TransformerModelConfig',
    'MambaModelConfig',
    'VAEModelConfig',
    'BaselineModelConfig',
    'EvaluationConfig',
    'TrackerConfig',
    'ExperimentConfig',
    'MODEL_CONFIG_MAP',
    'TRAINING_CONFIG_MAP',
    'validate_config',
    'load_and_validate_config',
]
