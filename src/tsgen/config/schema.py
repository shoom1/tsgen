"""
Pydantic schemas for configuration validation.

Provides structured config classes with validation, defaults, and
backward compatibility for both flat and nested config formats.
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


class DiffusionConfig(BaseModel):
    """Diffusion process configuration (deprecated, use DiffusionTrainingConfig)."""

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


class VAEConfig(BaseModel):
    """VAE-specific training hyperparameters (deprecated, use VAETrainingConfig)."""

    beta: float = 0.5
    use_annealing: bool = True
    annealing_epochs: int = 50
    use_free_bits: bool = True
    free_bits: float = 0.5
    teacher_forcing_ratio: float = 0.5


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
    validation_interval: int = 0
    num_validation_samples: int = 100
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


# ---------------------------------------------------------------------------
# Deprecated shims (kept for backward compatibility)
# ---------------------------------------------------------------------------

class ModelParamsConfig(BaseModel):
    """Model-specific parameters (deprecated, use per-model config classes)."""

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

    Nested format (preferred):
        model_type: 'unet'
        data:
          tickers: ['AAPL', 'MSFT']
          sequence_length: 64
        model:
          base_channels: 32
        training:
          epochs: 100
          timesteps: 500

    Flat format (deprecated, kept for backward compatibility):
        model_type: 'unet'
        tickers: ['AAPL', 'MSFT']
        sequence_length: 64
        epochs: 100
    """

    # Required
    model_type: str

    # Nested config sections
    data: DataConfig = Field(default_factory=DataConfig)
    model: Optional[Dict[str, Any]] = None
    training: Optional[Dict[str, Any]] = None
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    # Deprecated nested sections (kept for backward compatibility)
    diffusion: Optional[DiffusionConfig] = None
    vae: Optional[VAEConfig] = None
    tracker: Optional[Union[TrackerConfig, str]] = None

    # Flat config fields (deprecated, kept for backward compatibility)
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

    # Model params at root level (deprecated)
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

    # ------------------------------------------------------------------
    # New typed config accessors
    # ------------------------------------------------------------------

    def get_model_config(self) -> BaseModel:
        """Resolve model dict to the appropriate per-model config class.

        Returns:
            Typed model config (UNetModelConfig, TransformerModelConfig, etc.)

        Raises:
            ValueError: If model_type is not in MODEL_CONFIG_MAP
        """
        config_cls = MODEL_CONFIG_MAP.get(self.model_type)
        if config_cls is None:
            raise ValueError(
                f"Unknown model_type '{self.model_type}'. "
                f"Known types: {list(MODEL_CONFIG_MAP.keys())}"
            )
        kwargs = self.model if isinstance(self.model, dict) else {}
        return config_cls(**(kwargs or {}))

    def get_training_config(self) -> TrainingConfig:
        """Get unified TrainingConfig, resolved to paradigm-specific subclass.

        If self.training is already a TrainingConfig instance, return it.
        If self.training is a dict (or None), resolve via TRAINING_CONFIG_MAP.
        Falls back to flat fields when no nested training section is provided.
        """
        # Already resolved
        if isinstance(self.training, TrainingConfig):
            return self.training

        # Resolve via mapping
        config_cls = TRAINING_CONFIG_MAP.get(self.model_type)
        if config_cls is not None:
            if isinstance(self.training, dict):
                return config_cls(**self.training)
            else:
                # No training section provided - build from flat fields
                return config_cls(
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    learning_rate=self.learning_rate,
                )

        # Unknown model_type - fall back to base TrainingConfig
        if isinstance(self.training, dict):
            return TrainingConfig(**self.training)
        return TrainingConfig(
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
        )

    # ------------------------------------------------------------------
    # Deprecated shim accessors (kept for backward compatibility)
    # ------------------------------------------------------------------

    def get_data_config(self) -> DataConfig:
        """Get unified DataConfig from nested or flat fields (deprecated shim)."""
        if self.data and (self.data.tickers or self.data.start_date or self.data.end_date
                          or self.data.sequence_length != 64 or self.data.column != 'adj_close'
                          or self.data.db_path or self.data.train_test_split):
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

    def get_diffusion_config(self) -> DiffusionConfig:
        """Get unified DiffusionConfig from nested or flat fields (deprecated shim)."""
        if self.diffusion:
            return self.diffusion

        return DiffusionConfig(time_steps=self.timesteps)

    def get_model_params_config(self) -> ModelParamsConfig:
        """Get unified ModelParamsConfig from nested or flat fields (deprecated shim)."""
        if isinstance(self.model, dict) and self.model:
            return ModelParamsConfig(**self.model)

        return ModelParamsConfig(
            base_channels=self.base_channels,
            dim=self.dim,
            depth=self.depth,
            heads=self.heads,
            num_classes=self.num_classes,
        )

    def get_vae_config(self) -> VAEConfig:
        """Get VAEConfig from nested section or flat fields (deprecated shim)."""
        if self.vae:
            return self.vae

        kwargs = {}
        for field_name in VAEConfig.model_fields:
            # Check flat fields with vae_ prefix (backward compat)
            val = getattr(self, f'vae_{field_name}', None)
            if val is not None:
                kwargs[field_name] = val
        return VAEConfig(**kwargs)

    def get_evaluation_config(self) -> EvaluationConfig:
        """Get EvaluationConfig from nested section or flat fields (deprecated shim)."""
        if self.evaluation:
            return self.evaluation

        # Build from flat/extra fields if present
        kwargs = {}
        for field_name in EvaluationConfig.model_fields:
            val = getattr(self, field_name, None)
            if val is not None:
                kwargs[field_name] = val
        return EvaluationConfig(**kwargs)

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
