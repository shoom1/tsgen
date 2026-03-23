import torch
import torch.nn as nn
from tsgen.models.base_model import DiffusionModel
from tsgen.models.embeddings import SinusoidalPositionEmbeddings

class DiffusionTransformer(DiffusionModel):
    """
    A simplified Diffusion Transformer (DiT) for 1D time series.
    Treats the sequence time-steps as tokens.
    """

    supports_conditioning = True
    supports_masking = True
    def __init__(self, sequence_length, features, dim=128, depth=4, heads=4, mlp_dim=256, dropout=0.0, num_classes=0):
        super().__init__()
        self.dim = dim
        self.num_classes = num_classes
        
        # Input projection: Features -> Embedding Dim
        self.input_proj = nn.Linear(features, dim)
        
        # Time Embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        
        # Class Embedding (for conditional generation)
        if num_classes > 0:
            self.label_emb = nn.Embedding(num_classes, dim)
        
        # Positional Embedding for sequence order
        self.pos_embedding = nn.Parameter(torch.randn(1, sequence_length, dim))
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # Output projection: Embedding Dim -> Features
        self.output_proj = nn.Linear(dim, features)

    def forward(self, x, t, y=None, mask=None):
        """
        Forward pass for DiffusionTransformer.

        Args:
            x: Input tensor (Batch, Seq_Len, Features)
            t: Timesteps (Batch,)
            y: Optional class labels (Batch,) for conditional generation
            mask: Optional binary mask (Batch, Seq_Len, Features) where 1=valid, 0=masked
                  Used to create attention mask for the transformer

        Returns:
            Predicted noise (Batch, Seq_Len, Features)
        """
        # x: (Batch, Seq_Len, Features)
        # t: (Batch,)
        # y: (Batch,) - optional class labels

        # 1. Project Input
        x = self.input_proj(x) # (B, L, Dim)

        # 2. Add Positional Embedding
        x = x + self.pos_embedding

        # 3. Add Time Embedding
        t_emb = self.time_mlp(t) # (B, Dim)

        # 4. Add Class Embedding if provided
        if self.num_classes > 0 and y is not None:
            y_emb = self.label_emb(y) # (B, Dim)
            # Combine time and class embeddings (e.g., by summing)
            t_emb = t_emb + y_emb

        x = x + t_emb.unsqueeze(1) # (B, L, Dim) + (B, 1, Dim)

        # 5. Create attention mask from feature mask
        # A sequence position is masked if ALL features at that position are masked
        src_key_padding_mask = None
        if mask is not None:
            # mask: (B, L, F) -> src_key_padding_mask: (B, L)
            # PyTorch TransformerEncoder expects True = masked (ignored), False = valid
            # Sum across features: if any feature is valid, the position is valid
            positions_valid = mask.sum(dim=-1) > 0  # (B, L), True if any feature is valid
            src_key_padding_mask = ~positions_valid  # True = masked, False = valid

        # 6. Transformer Layers
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        # 7. Output Projection
        return self.output_proj(x)
