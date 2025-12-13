"""
Embedding modules for time series models.

This module provides various embedding layers used across different model architectures.
"""

import torch
import torch.nn as nn
import math


class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal position embeddings for time step encoding.

    Uses sine and cosine functions of different frequencies to encode
    timesteps, similar to the positional encoding in "Attention is All You Need".

    Args:
        dim (int): Dimension of the embedding
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        """
        Generate sinusoidal embeddings for given timesteps.

        Args:
            time (torch.Tensor): Timesteps of shape (batch_size,)

        Returns:
            torch.Tensor: Embeddings of shape (batch_size, dim)
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
