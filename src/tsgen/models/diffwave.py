"""DiffWave-style 1D dilated-conv diffusion backbone.

Adapted from Kong et al. (2020), "DiffWave: A Versatile Diffusion Model for
Audio Synthesis". The key property for financial time series is that the
dilated-conv stack preserves temporal resolution throughout the network —
unlike a pooling UNet, no high-frequency information is destroyed by
downsampling. This matters for log-returns because volatility clustering
lives at the highest frequencies.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsgen.models.base_model import DiffusionModel
from tsgen.models.embeddings import SinusoidalPositionEmbeddings
from tsgen.models.registry import ModelRegistry


class ResidualBlock(nn.Module):
    """Gated dilated residual block.

    x (B, C, L) -> (residual_out (B, C, L), skip_out (B, C, L))

    The conditioning embedding (time + optional class) is projected to 2*C
    and added before the tanh/sigmoid gate.
    """

    def __init__(self, residual_channels: int, dilation: int, kernel_size: int, cond_dim: int):
        super().__init__()
        self.residual_channels = residual_channels
        padding = (kernel_size - 1) // 2 * dilation
        # Output 2*C: one half drives tanh (filter), other half drives sigmoid (gate)
        self.dilated_conv = nn.Conv1d(
            residual_channels,
            2 * residual_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )
        self.cond_proj = nn.Linear(cond_dim, 2 * residual_channels)
        # Splits the gated output into (residual_update, skip_update)
        self.out_proj = nn.Conv1d(residual_channels, 2 * residual_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        # x: (B, C, L), cond: (B, cond_dim)
        h = self.dilated_conv(x)  # (B, 2C, L)
        # Broadcast conditioning along sequence dim
        cond_term = self.cond_proj(cond).unsqueeze(-1)  # (B, 2C, 1)
        h = h + cond_term

        filter_out, gate_out = h.chunk(2, dim=1)  # each (B, C, L)
        h = torch.tanh(filter_out) * torch.sigmoid(gate_out)

        residual_update, skip_update = self.out_proj(h).chunk(2, dim=1)
        # Residual path: scale by 1/sqrt(2) per DiffWave paper
        residual_out = (x + residual_update) * (1.0 / math.sqrt(2.0))
        return residual_out, skip_update


@ModelRegistry.register('diffwave')
class DiffWave1D(DiffusionModel):
    """DiffWave-style diffusion model for multivariate time series.

    Architecture (input at top, output at bottom):
      (B, L, F)
        -> transpose                                   (B, F, L)
        -> Conv1d(F -> C, 1x1) + ReLU                  (B, C, L)
        -> stack of N ResidualBlocks with dilations
           cycling through [1, 2, 4, ..., 2^(cycle-1)]
        -> sum(skip connections) * 1/sqrt(N)
        -> ReLU + Conv1d(C -> C, 1x1)
        -> ReLU + Conv1d(C -> F, 1x1)
        -> transpose back                              (B, L, F)

    Conditioning: time (required) + optional class label. Both are summed
    into a single `cond_dim`-dimensional vector that's injected at every
    residual block.

    Sequence length is *not* a constraint on the architecture; any positive
    integer L works. Receptive field ≈ 1 + 2*(sum of dilations) per cycle.
    """

    supports_conditioning = True
    supports_masking = False  # Plain convs; mask accepted but not applied

    def __init__(
        self,
        sequence_length,                # unused; kept for API parity with other DiffusionModels
        features: int,
        residual_channels: int = 64,
        num_blocks: int = 10,
        dilation_cycle_length: int = 5,
        kernel_size: int = 3,
        num_classes: int = 0,
    ):
        super().__init__()
        if residual_channels <= 0:
            raise ValueError(f"residual_channels must be positive, got {residual_channels}")
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}")
        if dilation_cycle_length <= 0:
            raise ValueError(f"dilation_cycle_length must be positive, got {dilation_cycle_length}")

        self.features = features
        self.residual_channels = residual_channels
        self.num_blocks = num_blocks
        self.dilation_cycle_length = dilation_cycle_length
        self.num_classes = num_classes

        cond_dim = residual_channels * 4

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(residual_channels),
            nn.Linear(residual_channels, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        # Optional class embedding (same dim as time for additive combination)
        if num_classes > 0:
            self.label_emb = nn.Embedding(num_classes, cond_dim)
        else:
            self.label_emb = None

        # Input projection
        self.input_proj = nn.Conv1d(features, residual_channels, kernel_size=1)

        # Residual block stack with cycled dilations
        self.blocks = nn.ModuleList([
            ResidualBlock(
                residual_channels=residual_channels,
                dilation=2 ** (i % dilation_cycle_length),
                kernel_size=kernel_size,
                cond_dim=cond_dim,
            )
            for i in range(num_blocks)
        ])

        # Output head
        self.skip_proj = nn.Conv1d(residual_channels, residual_channels, kernel_size=1)
        self.output_proj = nn.Conv1d(residual_channels, features, kernel_size=1)

        self._initialize_weights()

    def _initialize_weights(self):
        # DiffWave initializes the final conv to zero so training starts
        # predicting zero noise — stabilizes early epochs.
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    @classmethod
    def _model_kwargs_from_config(cls, params) -> dict:
        return {
            'residual_channels': params.residual_channels,
            'num_blocks': params.num_blocks,
            'dilation_cycle_length': params.dilation_cycle_length,
            'kernel_size': params.kernel_size,
            'num_classes': params.num_classes,
        }

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                y: torch.Tensor = None, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Noisy input (B, L, F)
            t: Timesteps (B,)
            y: Optional class labels (B,)
            mask: Accepted for API parity; ignored (see supports_masking)

        Returns:
            Predicted noise (B, L, F)
        """
        # (B, L, F) -> (B, F, L)
        x = x.transpose(1, 2)

        cond = self.time_mlp(t)                      # (B, cond_dim)
        if self.num_classes > 0 and y is not None:
            cond = cond + self.label_emb(y)

        h = F.relu(self.input_proj(x))               # (B, C, L)

        skip_accum = 0.0
        for block in self.blocks:
            h, skip = block(h, cond)
            skip_accum = skip_accum + skip

        # Normalize skip sum per DiffWave paper
        skip_accum = skip_accum * (1.0 / math.sqrt(self.num_blocks))

        out = F.relu(self.skip_proj(skip_accum))
        out = self.output_proj(out)                  # (B, F, L)

        return out.transpose(1, 2)                   # (B, L, F)
