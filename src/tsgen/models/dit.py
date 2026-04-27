"""DiT1D — Diffusion Transformer for 1D time series, with adaLN-Zero conditioning.

Adapted from Peebles & Xie (2023), "Scalable Diffusion Models with
Transformers". The key architectural difference from the additive-
conditioning baseline in ``DiffusionTransformer`` is **adaLN-Zero**: instead
of adding the time/class embedding to the token stream, each transformer
block consumes the conditioning vector via an MLP that produces per-block
(shift, scale, gate) modulation parameters for both the self-attention and
MLP sublayers. The modulation MLPs and the final output projection are
zero-initialized so the network starts as an identity map that predicts
zero noise — a standard stability trick.

Positional encoding is sinusoidal (computed per forward) rather than a
learned parameter, so the same checkpoint can sample variable-length
windows.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsgen.models.base_model import DiffusionModel
from tsgen.models.embeddings import SinusoidalPositionEmbeddings
from tsgen.models.registry import ModelRegistry


def _sinusoidal_positional_encoding(seq_len: int, dim: int, device, dtype) -> torch.Tensor:
    """Standard sinusoidal positional encoding, computed per-call.

    Returns a (1, seq_len, dim) tensor so it broadcasts against (B, seq_len, dim).
    """
    pos = torch.arange(seq_len, device=device, dtype=dtype)
    div = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=dtype)
        * (-math.log(10000.0) / dim)
    )
    pe = torch.zeros(seq_len, dim, device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(pos[:, None] * div[None, :])
    pe[:, 1::2] = torch.cos(pos[:, None] * div[None, :])
    return pe.unsqueeze(0)


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply adaLN modulation: x * (1 + scale) + shift.

    x: (B, L, D); shift, scale: (B, D). Broadcast over L.
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """Single DiT block: adaLN-Zero modulated self-attention + MLP.

    Per block, the conditioning vector is projected to 6 modulation signals
    (shift/scale/gate for attn, shift/scale/gate for mlp). The modulation MLP
    is zero-initialized so that at init gate_msa = gate_mlp = 0 and the block
    reduces to an identity map.
    """

    def __init__(self, dim: int, heads: int, mlp_ratio: float, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True,
        )
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(approximate='tanh'),
            nn.Linear(mlp_hidden, dim),
        )
        # Produces 6 modulation signals from the conditioning vector
        self.ada_ln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim),
        )
        # Zero-init the final linear in ada_ln so modulation is identity at init
        nn.init.zeros_(self.ada_ln[-1].weight)
        nn.init.zeros_(self.ada_ln[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor,
                key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (B, L, dim)
            cond: (B, dim) — time + optional class embedding, post-MLP
            key_padding_mask: (B, L) boolean, True = masked position
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.ada_ln(cond).chunk(6, dim=-1)
        )

        # Self-attention sublayer with adaptive modulation
        h = _modulate(self.norm1(x), shift_msa, scale_msa)
        h, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + gate_msa.unsqueeze(1) * h

        # MLP sublayer with adaptive modulation
        h = _modulate(self.norm2(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        x = x + gate_mlp.unsqueeze(1) * h

        return x


class DiTFinalLayer(nn.Module):
    """Final adaLN-Zero layer: modulated LayerNorm + linear projection to features."""

    def __init__(self, dim: int, features: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, features)
        self.ada_ln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 2 * dim),
        )
        # Zero-init both the modulation MLP and the output linear —
        # network starts predicting exactly zero noise.
        nn.init.zeros_(self.ada_ln[-1].weight)
        nn.init.zeros_(self.ada_ln[-1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift, scale = self.ada_ln(cond).chunk(2, dim=-1)
        x = _modulate(self.norm(x), shift, scale)
        return self.linear(x)


@ModelRegistry.register('dit')
class DiT1D(DiffusionModel):
    """Diffusion Transformer (DiT) for 1D time series.

    Structurally a transformer encoder with adaLN-Zero conditioning and
    sinusoidal positional encoding. Supports class conditioning and masking.
    """

    supports_conditioning = True
    supports_masking = True

    def __init__(
        self,
        features: int,
        dim: int = 128,
        depth: int = 4,
        heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        num_classes: int = 0,
        sequence_length=None,     # unused; kept for API parity
    ):
        super().__init__()
        self.features = features
        self.dim = dim
        self.num_classes = num_classes

        # Input projection: features -> dim
        self.input_proj = nn.Linear(features, dim)

        # Time conditioning: sinusoidal -> MLP -> dim
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

        # Class embedding (optional), same dim as time for additive fusion
        if num_classes > 0:
            self.label_emb = nn.Embedding(num_classes, dim)
        else:
            self.label_emb = None

        # Transformer blocks with adaLN-Zero
        self.blocks = nn.ModuleList([
            DiTBlock(dim=dim, heads=heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(depth)
        ])

        # Final adaLN-Zero layer + linear to features
        self.final_layer = DiTFinalLayer(dim=dim, features=features)

        # Exposed for tests that need to turn off the zero-init stability trick
        self.final_proj = self.final_layer.linear

    @classmethod
    def _model_kwargs_from_config(cls, params) -> dict:
        return {
            'dim': params.dim,
            'depth': params.depth,
            'heads': params.heads,
            'mlp_ratio': params.mlp_ratio,
            'dropout': params.dropout,
            'num_classes': params.num_classes,
        }

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                y: torch.Tensor = None, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (B, L, features) noisy input
            t: (B,) diffusion timesteps
            y: (B,) class labels (optional)
            mask: (B, L, F) binary mask, 1 = valid (optional). A position is
                considered masked if ALL features at that position are masked.

        Returns:
            Predicted noise (B, L, features)
        """
        B, L, F_in = x.shape

        # Input projection
        x = self.input_proj(x)                                   # (B, L, dim)

        # Sinusoidal positional encoding, computed per-call so any L works
        x = x + _sinusoidal_positional_encoding(
            L, self.dim, device=x.device, dtype=x.dtype,
        )

        # Conditioning vector c = time + optional class
        cond = self.time_mlp(t)                                  # (B, dim)
        if self.num_classes > 0 and y is not None:
            cond = cond + self.label_emb(y)

        # Attention padding mask from feature mask: position is padded iff
        # every feature at that position is masked out.
        key_padding_mask = None
        if mask is not None:
            positions_valid = mask.sum(dim=-1) > 0               # (B, L)
            key_padding_mask = ~positions_valid                  # True = padded

        for block in self.blocks:
            x = block(x, cond, key_padding_mask=key_padding_mask)

        return self.final_layer(x, cond)                         # (B, L, features)
