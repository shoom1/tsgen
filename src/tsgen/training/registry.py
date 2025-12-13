"""
Trainer Registry for mapping model types to trainer classes.

Implements Registry Pattern + Factory Pattern for trainer instantiation.
"""

from typing import Dict, Type
from tsgen.training.base import BaseTrainer


class TrainerRegistry:
    """
    Registry for mapping model types to trainer classes.

    This class implements two design patterns:
    1. Registry Pattern: Maps model type strings to trainer classes
    2. Factory Pattern: Creates trainer instances via get_trainer() method

    Usage:
        # Register a trainer (typically done with decorator)
        @TrainerRegistry.register('unet', 'transformer')
        class DiffusionTrainer(BaseTrainer):
            pass

        # Get a trainer instance (factory method)
        trainer = TrainerRegistry.get_trainer('unet', model, config, tracker, device)
    """

    _trainers: Dict[str, Type[BaseTrainer]] = {}

    @classmethod
    def register(cls, *model_types: str):
        """
        Decorator to register a trainer for one or more model types.

        This allows declarative registration of trainers at the class definition.
        Multiple model types can share the same trainer class.

        Args:
            *model_types: One or more model type strings to register

        Returns:
            Decorator function that registers the trainer class

        Example:
            @TrainerRegistry.register('unet', 'transformer')
            class DiffusionTrainer(BaseTrainer):
                def train(self, dataloader):
                    # diffusion training logic
                    pass
        """
        def decorator(trainer_class: Type[BaseTrainer]):
            for model_type in model_types:
                cls._trainers[model_type] = trainer_class
            return trainer_class
        return decorator

    @classmethod
    def get_trainer(
        cls,
        model_type: str,
        model,
        config,
        tracker,
        device
    ) -> BaseTrainer:
        """
        Factory method: Get trainer instance for model type.

        This is the factory method that creates the appropriate trainer instance
        based on the model type. Raises an error if the model type is not registered.

        Args:
            model_type: Type of model (e.g., 'unet', 'timevae', 'gbm')
            model: Model instance to train
            config: Configuration dictionary
            tracker: Experiment tracker
            device: Training device ('cuda' or 'cpu')

        Returns:
            Trainer instance of the appropriate type

        Raises:
            ValueError: If model_type is not registered in the registry

        Example:
            trainer = TrainerRegistry.get_trainer(
                'timevae', model, config, tracker, 'cpu'
            )
            trained_model = trainer.train(dataloader)
        """
        trainer_class = cls._trainers.get(model_type)

        if trainer_class is None:
            raise ValueError(
                f"No trainer registered for model type '{model_type}'. "
                f"Available: {list(cls._trainers.keys())}"
            )

        return trainer_class(model, config, tracker, device)

    @classmethod
    def list_trainers(cls) -> Dict[str, Type[BaseTrainer]]:
        """
        List all registered trainers.

        Returns:
            Dictionary mapping model types to trainer classes
        """
        return cls._trainers.copy()
