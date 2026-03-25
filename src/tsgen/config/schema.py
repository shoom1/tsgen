"""
Pydantic schemas for configuration validation.

Provides structured config classes with validation and defaults
for nested config format.
"""

from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import List, Optional, Dict, Any, Literal, Union


class DataConfig(BaseModel):
    """Data loading and processing configuration."""

    tickers: List[str] = Field(default_factory=list)
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    sequence_length: int = 64
    column: str = 'adj_close'
    db_path: Optional[str] = None
    train_test_split: Optional[float] = None

    @field_validator('sequence_length')
    @classmethod
    def validate_sequence_length(cls, v):
        if v < 1:
            raise ValueError('sequence_length must be positive')
        return v

    @field_validator('train_test_split')
    @classmethod
    def validate_split_ratio(cls, v):
        if v is not None and not (0 < v < 1):
            raise ValueError('train_test_split must be between 0 and 1')
        return v


class TrainingConfig(BaseModel):
    """Training hyperparameters (base class for paradigm-specific configs)."""

    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    gradient_clip: float = 1.0
    checkpoint_interval: int = 10
    validation_interval: int = 0
    num_validation_samples: int = 100
    start_epoch: int = 0

    @field_validator('epochs', 'batch_size')
    @classmethod
    def validate_positive(cls, v):
        if v < 1:
            raise ValueError('Value must be positive')
        return v

    @field_validator('learning_rate')
    @classmethod
    def validate_learning_rate(cls, v):
        if v <= 0:
            raise ValueError('learning_rate must be positive')
        return v


# ---------------------------------------------------------------------------
# Per-paradigm training configs
# ---------------------------------------------------------------------------

class DiffusionTrainingConfig(TrainingConfig):
    """Training config for diffusion models (UNet, Transformer, Mamba)."""

    timesteps: int = 1000
    sampling_method: Literal['ddpm', 'ddim'] = 'ddpm'
    num_inference_steps: int = 50
    beta_start: float = 1e-4
    beta_end: float = 0.02
    classifier_free_guidance_probability: float = 0.0

    model_config = ConfigDict(extra='forbid')


class VAETrainingConfig(TrainingConfig):
    """Training config for VAE models."""

    beta: float = 0.5
    use_annealing: bool = True
    annealing_epochs: int = 50
    use_free_bits: bool = True
    free_bits: float = 0.5
    teacher_forcing_ratio: float = 0.5

    model_config = ConfigDict(extra='forbid')


class BaselineTrainingConfig(TrainingConfig):
    """Training config for baseline models (GBM, Bootstrap, etc.)."""

    model_config = ConfigDict(extra='forbid')


# ---------------------------------------------------------------------------
# Per-model config classes
# ---------------------------------------------------------------------------

class UNetModelConfig(BaseModel):
    """UNet model architecture parameters."""

    base_channels: int = 64
    num_classes: int = 0

    model_config = ConfigDict(extra='forbid')


class TransformerModelConfig(BaseModel):
    """Transformer model architecture parameters."""

    dim: int = 64
    depth: int = 4
    heads: int = 4
    mlp_dim: int = 128
    dropout: float = 0.0
    num_classes: int = 0

    model_config = ConfigDict(extra='forbid')


class MambaModelConfig(BaseModel):
    """Mamba model architecture parameters."""

    dim: int = 128
    depth: int = 4
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    num_classes: int = 0

    model_config = ConfigDict(extra='forbid')


class VAEModelConfig(BaseModel):
    """VAE model architecture parameters."""

    hidden_dim: int = 64
    latent_dim: int = 16
    encoder_type: str = 'lstm'
    num_layers: int = 2

    model_config = ConfigDict(extra='forbid')


class BaselineModelConfig(BaseModel):
    """Baseline model parameters (empty - no architecture params needed)."""

    model_config = ConfigDict(extra='forbid')


# ---------------------------------------------------------------------------
# Mapping dicts: model_type -> config class
# ---------------------------------------------------------------------------

MODEL_CONFIG_MAP: Dict[str, type] = {
    'unet': UNetModelConfig,
    'transformer': TransformerModelConfig,
    'mamba': MambaModelConfig,
    'timevae': VAEModelConfig,
    'gbm': BaselineModelConfig,
    'bootstrap': BaselineModelConfig,
    'multivariate_gbm': BaselineModelConfig,
    'multivariate_lognormal': BaselineModelConfig,
}

TRAINING_CONFIG_MAP: Dict[str, type] = {
    'unet': DiffusionTrainingConfig,
    'transformer': DiffusionTrainingConfig,
    'mamba': DiffusionTrainingConfig,
    'timevae': VAETrainingConfig,
    'gbm': BaselineTrainingConfig,
    'bootstrap': BaselineTrainingConfig,
    'multivariate_gbm': BaselineTrainingConfig,
    'multivariate_lognormal': BaselineTrainingConfig,
}


class EvaluationConfig(BaseModel):
    """Evaluation configuration."""

    num_samples: int = 500
    discriminator_epochs: int = 20
    discriminator_hidden_dim: int = 64
    stylized_facts_lags: int = 20
    var_alpha: float = 0.05
    tstr_epochs: int = 10

    @field_validator('num_samples')
    @classmethod
    def validate_num_samples(cls, v):
        if v < 1:
            raise ValueError('num_samples must be positive')
        return v


class TrackerConfig(BaseModel):
    """Experiment tracking configuration."""

    output_type: Literal['console', 'file', 'mlflow', 'noop'] = 'console'
    output_dir: Optional[str] = None
    mlflow_tracking_uri: Optional[str] = None
    mlflow_artifact_location: Optional[str] = None


class ExperimentConfig(BaseModel):
    """
    Root configuration schema with validation.

    Nested format:
        model_type: 'unet'
        data:
          tickers: ['AAPL', 'MSFT']
          sequence_length: 64
        model:
          base_channels: 32
        training:
          epochs: 100
          timesteps: 500
    """

    # Required
    model_type: str

    # Nested config sections
    data: DataConfig = Field(default_factory=DataConfig)
    model: Optional[Dict[str, Any]] = None
    training: Optional[Dict[str, Any]] = None
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    tracker: Optional[Union[TrackerConfig, str]] = None

    # Experiment metadata
    experiment_name: Optional[str] = None
    output_dir: Optional[str] = None

    # Pipeline config (passed through)
    data_pipeline: Optional[List[Dict[str, Any]]] = None

    # Allow extra fields for extensibility
    model_config = {'extra': 'allow'}

    # ------------------------------------------------------------------
    # Typed config accessors
    # ------------------------------------------------------------------

    def get_data_config(self) -> DataConfig:
        """Return the nested DataConfig."""
        return self.data

    def get_model_config(self) -> BaseModel:
        """Resolve model dict to the appropriate per-model config class.

        Returns:
            Typed model config (UNetModelConfig, TransformerModelConfig, etc.)

        Raises:
            ValueError: If model_type is not in MODEL_CONFIG_MAP
        """
        if hasattr(self, '_cached_model_config'):
            return self._cached_model_config
        config_cls = MODEL_CONFIG_MAP.get(self.model_type)
        if config_cls is None:
            raise ValueError(
                f"Unknown model_type '{self.model_type}'. "
                f"Known types: {list(MODEL_CONFIG_MAP.keys())}"
            )
        kwargs = self.model if isinstance(self.model, dict) else {}
        result = config_cls(**(kwargs or {}))
        object.__setattr__(self, '_cached_model_config', result)
        return result

    def get_training_config(self) -> TrainingConfig:
        """Get unified TrainingConfig, resolved to paradigm-specific subclass.

        If self.training is already a TrainingConfig instance, return it.
        If self.training is a dict (or None), resolve via TRAINING_CONFIG_MAP.
        When no training section is provided, returns defaults for the paradigm.
        """
        if hasattr(self, '_cached_training_config'):
            return self._cached_training_config

        # Already resolved
        if isinstance(self.training, TrainingConfig):
            result = self.training
        else:
            # Resolve via mapping
            config_cls = TRAINING_CONFIG_MAP.get(self.model_type)
            if config_cls is not None:
                if isinstance(self.training, dict):
                    result = config_cls(**self.training)
                else:
                    result = config_cls()
            elif isinstance(self.training, dict):
                result = TrainingConfig(**self.training)
            else:
                result = TrainingConfig()

        object.__setattr__(self, '_cached_training_config', result)
        return result

    def get_evaluation_config(self) -> EvaluationConfig:
        """Return the nested EvaluationConfig."""
        return self.evaluation

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'ExperimentConfig':
        """
        Load config from dictionary with validation.

        Args:
            config: Raw config dictionary from YAML

        Returns:
            Validated ExperimentConfig

        Raises:
            ValidationError: If config is invalid
        """
        return cls(**config)

    def to_dict(self) -> Dict[str, Any]:
        """Convert back to dictionary for backward compatibility."""
        return self.model_dump(exclude_none=True)


def validate_config(config: Dict[str, Any]) -> ExperimentConfig:
    """
    Validate configuration dictionary.

    Args:
        config: Raw config dictionary from YAML

    Returns:
        Validated ExperimentConfig

    Raises:
        pydantic.ValidationError: If config is invalid with detailed error messages
    """
    return ExperimentConfig.from_dict(config)


def load_and_validate_config(path: str) -> ExperimentConfig:
    """
    Load YAML config file and validate.

    Args:
        path: Path to YAML config file

    Returns:
        Validated ExperimentConfig

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
        pydantic.ValidationError: If config is invalid
    """
    import yaml

    with open(path, 'r') as f:
        raw_config = yaml.safe_load(f)

    return validate_config(raw_config)
