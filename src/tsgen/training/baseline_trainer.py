"""
Baseline model trainer for statistical models.

Handles training for models that use .fit() method rather than
gradient-based optimization (GBM, Bootstrap, Multivariate LogNormal).
"""

import torch
from torch.utils.data import DataLoader

from tsgen.training.base import BaseTrainer
from tsgen.training.registry import TrainerRegistry


@TrainerRegistry.register('gbm', 'bootstrap', 'multivariate_lognormal')
class BaselineTrainer(BaseTrainer):
    """
    Trainer for baseline models that use .fit() method.

    Baseline models don't use gradient descent - they just fit
    statistical parameters from the data. This trainer simply
    calls the model's .fit() method.
    """

    def train(self, dataloader: DataLoader) -> torch.nn.Module:
        """
        Train baseline model (calls .fit() method).

        Baseline models don't use gradient descent - they just
        fit statistical parameters from the data.

        Args:
            dataloader: Training data loader

        Returns:
            Fitted model
        """
        print(f"Fitting baseline model: {self.config.get('model_type')}")
        self.model.fit(dataloader)
        return self.model

    def save_model(self, path: str):
        """
        Save full model object for baselines.

        Baselines need full object saved to preserve buffers
        (e.g., Bootstrap's history buffer). Unlike neural network
        models that only save state_dict, baseline models save
        the entire object.

        Args:
            path: Path where model should be saved
        """
        torch.save(self.model, path)
