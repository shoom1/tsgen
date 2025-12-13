"""
Base trainer interface for all model training strategies.

Implements the Strategy Pattern where each trainer encapsulates
a different training algorithm.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict
import torch
from torch.utils.data import DataLoader
from tsgen.tracking.base import ExperimentTracker


class BaseTrainer(ABC):
    """
    Base trainer interface for all model training strategies.

    This class implements the Strategy Pattern, where each concrete trainer
    encapsulates a specific training algorithm (e.g., diffusion, VAE, baseline).
    All trainers share common infrastructure while implementing their own
    training logic.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        config: Dict[str, Any],
        tracker: ExperimentTracker,
        device: str
    ):
        """
        Initialize trainer.

        Args:
            model: Model instance to train
            config: Configuration dictionary containing hyperparameters
            tracker: Experiment tracker for logging metrics and artifacts
            device: Device to train on ('cuda' or 'cpu')
        """
        self.model = model
        self.config = config
        self.tracker = tracker
        self.device = device

    @abstractmethod
    def train(self, dataloader: DataLoader) -> torch.nn.Module:
        """
        Train the model.

        This is the main method that must be implemented by all concrete trainers.
        Each trainer implements its own training algorithm here.

        Args:
            dataloader: Training data loader

        Returns:
            Trained model
        """
        pass

    def save_model(self, path: str):
        """
        Save model to path.

        Default implementation saves only the state_dict. Override this method
        in concrete trainers if special saving logic is needed (e.g., baseline
        models that need to save the full object).

        Args:
            path: Path where model should be saved
        """
        torch.save(self.model.state_dict(), path)
