"""
Model Registry for mapping model type strings to model classes.

Implements Registry Pattern + Factory Pattern for model instantiation.
"""

from __future__ import annotations

from typing import Dict, Type, TYPE_CHECKING

from tsgen.models.base_model import BaseGenerativeModel

if TYPE_CHECKING:
    from tsgen.config.schema import ExperimentConfig


class ModelRegistry:
    """
    Registry for mapping model type strings to model classes.

    This class implements two design patterns:
    1. Registry Pattern: Maps model type strings to model classes
    2. Factory Pattern: Creates model instances via create() method

    Usage:
        # Register a model (typically done with decorator)
        @ModelRegistry.register('unet')
        class UNet1D(DiffusionModel):
            pass

        # Create a model instance (factory method)
        model = ModelRegistry.create(config, features=5)
    """

    _models: Dict[str, Type[BaseGenerativeModel]] = {}

    @classmethod
    def register(cls, *model_types: str):
        """
        Decorator to register a model class for one or more model type strings.

        This allows declarative registration of models at the class definition.
        Multiple model type strings can map to the same model class.

        Args:
            *model_types: One or more model type strings to register

        Returns:
            Decorator function that registers the model class

        Example:
            @ModelRegistry.register('gbm', 'multivariate_lognormal')
            class MultivariateGBM(StatisticalModel):
                pass
        """
        def decorator(model_class: Type[BaseGenerativeModel]):
            for model_type in model_types:
                cls._models[model_type] = model_class
            return model_class
        return decorator

    @classmethod
    def create(
        cls,
        config: ExperimentConfig,
        features: int | None = None,
    ) -> BaseGenerativeModel:
        """
        Factory method: Create model instance from config.

        Looks up the model class by config.model_type and delegates
        construction to the class's from_config() classmethod.

        Args:
            config: Validated ExperimentConfig
            features: Number of input features (e.g., number of tickers)

        Returns:
            Model instance of the appropriate type

        Raises:
            ValueError: If model_type is not registered in the registry

        Example:
            model = ModelRegistry.create(config, features=5)
        """
        model_type = config.model_type
        model_class = cls._models.get(model_type)

        if model_class is None:
            raise ValueError(
                f"No model registered for model type '{model_type}'. "
                f"Available: {list(cls._models.keys())}"
            )

        return model_class.from_config(config, features=features)

    @classmethod
    def list_models(cls) -> Dict[str, Type[BaseGenerativeModel]]:
        """
        List all registered models.

        Returns:
            Dictionary mapping model type strings to model classes
        """
        return cls._models.copy()
