from abc import ABC, abstractmethod
import torch.nn as nn

class GenerativeModel(nn.Module, ABC):
    """
    Abstract base class for all generative models in the framework.
    Enforces a common interface for the forward pass and configuration.
    """
    @abstractmethod
    def forward(self, x, t):
        """
        Args:
            x (torch.Tensor): Input tensor (Batch, Seq_Len, Features)
            t (torch.Tensor): Timesteps (Batch,)
        Returns:
            torch.Tensor: Predicted noise (Batch, Seq_Len, Features)
        """
        pass
