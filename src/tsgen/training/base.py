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
        config,
        tracker: ExperimentTracker,
        device: str
    ):
        """
        Initialize trainer.

        Args:
            model: Model instance to train
            config: ExperimentConfig (or dict for backward compatibility)
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

    def save_checkpoint(self, path: str, epoch: int, optimizer=None, **extra_state):
        """
        Save full checkpoint with training state.

        Args:
            path: Path where checkpoint should be saved
            epoch: Current epoch number
            optimizer: Optimizer instance (optional)
            **extra_state: Additional state to save (e.g., scheduler, metrics)
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'config': self.config.to_dict() if hasattr(self.config, 'to_dict') else self.config,
        }

        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        # Add any extra state (e.g., scheduler, best_loss, etc.)
        checkpoint.update(extra_state)

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str, optimizer=None, strict=True):
        """
        Load checkpoint and restore training state.

        Args:
            path: Path to checkpoint file
            optimizer: Optimizer instance to restore state into (optional)
            strict: Whether to strictly enforce state_dict keys match

        Returns:
            dict: Checkpoint dictionary containing epoch and any extra state
        """
        checkpoint = torch.load(path, map_location=self.device)

        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=strict)

        # Load optimizer state if provided
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return checkpoint
