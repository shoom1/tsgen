"""
Pydantic schemas for configuration validation.

Provides structured config classes with validation, defaults, and
backward compatibility for both flat and nested config formats.
"""

from pydantic import BaseModel, Field, field_validator, model_validator
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


class DiffusionConfig(BaseModel):
    """Diffusion process configuration."""

    time_steps: int = Field(default=1000, alias='timesteps')
    sampling_method: Literal['ddpm', 'ddim'] = 'ddpm'
    num_inference_steps: int = 50
    beta_start: float = 1e-4
    beta_end: float = 0.02

    model_config = {'populate_by_name': True}

    @field_validator('time_steps')
    @classmethod
    def validate_timesteps(cls, v):
        if v < 1:
            raise ValueError('time_steps must be positive')
        return v


class TrainingConfig(BaseModel):
    """Training hyperparameters."""

    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    gradient_clip: float = 1.0
    checkpoint_interval: int = 10
    validation_interval: int = 0
    num_validation_samples: int = 100

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

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'TrainingConfig':
        """
        Extract training config from flat or nested dict.

        Supports both:
            training:
              epochs: 100
        And:
            epochs: 100
        """
        # Check for nested 'training' section first
        training_section = config.get('training', {})
        if isinstance(training_section, dict) and training_section:
            return cls(**training_section)

        # Fall back to flat config
        return cls(
            epochs=config.get('epochs', 100),
            batch_size=config.get('batch_size', 32),
            learning_rate=config.get('learning_rate', 1e-3),
            gradient_clip=config.get('gradient_clip', 1.0),
            checkpoint_interval=config.get('checkpoint_interval', 10),
            validation_interval=config.get('validation_interval', 0),
            num_validation_samples=config.get('num_validation_samples', 100),
        )


class ModelParamsConfig(BaseModel):
    """Model-specific parameters."""

    # UNet params
    base_channels: int = 64

    # Transformer params
    dim: int = 64
    depth: int = 4
    heads: int = 4
    mlp_dim: int = 128
    dropout: float = 0.0

    # Mamba params
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2

    # VAE params
    hidden_dim: int = 64
    latent_dim: int = 16
    encoder_type: str = 'lstm'
    num_layers: int = 2

    # Conditioning
    num_classes: int = 0

    model_config = {'extra': 'allow'}


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

    Supports both nested and flat config formats for backward compatibility.

    Nested format:
        model_type: 'unet'
        data:
          tickers: ['AAPL', 'MSFT']
          sequence_length: 64
        training:
          epochs: 100

    Flat format:
        model_type: 'unet'
        tickers: ['AAPL', 'MSFT']
        sequence_length: 64
        epochs: 100
    """

    # Required
    model_type: str

    # Nested config sections (optional)
    data: Optional[DataConfig] = None
    training: Optional[TrainingConfig] = None
    diffusion: Optional[DiffusionConfig] = None
    model: Optional[ModelParamsConfig] = None
    evaluation: Optional[EvaluationConfig] = None
    tracker: Optional[Union[TrackerConfig, str]] = None

    # Flat config fields (backward compatible)
    tickers: Optional[List[str]] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    sequence_length: int = 64
    column: str = 'adj_close'
    db_path: Optional[str] = None
    train_test_split: Optional[float] = None
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 1e-3
    timesteps: int = 1000

    # Model params at root level
    base_channels: int = 64
    dim: int = 64
    depth: int = 4
    heads: int = 4
    num_classes: int = 0

    # Experiment metadata
    experiment_name: Optional[str] = None
    output_dir: Optional[str] = None

    # Pipeline config (passed through)
    data_pipeline: Optional[List[Dict[str, Any]]] = None

    # Allow extra fields for extensibility
    model_config = {'extra': 'allow'}

    @model_validator(mode='after')
    def validate_data_source(self):
        """Ensure data source is specified either in data section or flat."""
        has_nested_tickers = self.data and self.data.tickers
        has_flat_tickers = self.tickers and len(self.tickers) > 0

        if not has_nested_tickers and not has_flat_tickers:
            # This is okay - tickers might come from pipeline config
            pass

        return self

    def get_data_config(self) -> DataConfig:
        """Get unified DataConfig from nested or flat fields."""
        if self.data:
            return self.data

        return DataConfig(
            tickers=self.tickers or [],
            start_date=self.start_date,
            end_date=self.end_date,
            sequence_length=self.sequence_length,
            column=self.column,
            db_path=self.db_path,
            train_test_split=self.train_test_split,
        )

    def get_training_config(self) -> TrainingConfig:
        """Get unified TrainingConfig from nested or flat fields."""
        if self.training:
            return self.training

        return TrainingConfig(
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
        )

    def get_diffusion_config(self) -> DiffusionConfig:
        """Get unified DiffusionConfig from nested or flat fields."""
        if self.diffusion:
            return self.diffusion

        return DiffusionConfig(time_steps=self.timesteps)

    def get_model_params_config(self) -> ModelParamsConfig:
        """Get unified ModelParamsConfig from nested or flat fields."""
        if self.model:
            return self.model

        return ModelParamsConfig(
            base_channels=self.base_channels,
            dim=self.dim,
            depth=self.depth,
            heads=self.heads,
            num_classes=self.num_classes,
        )

    def get_evaluation_config(self) -> EvaluationConfig:
        """Get EvaluationConfig with defaults."""
        if self.evaluation:
            return self.evaluation
        return EvaluationConfig()

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
