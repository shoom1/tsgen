import torch
import torch.nn as nn
from tsgen.models.base_model import GenerativeModel
from tsgen.models.embeddings import SinusoidalPositionEmbeddings

class DiffusionTransformer(GenerativeModel):
    """
    A simplified Diffusion Transformer (DiT) for 1D time series.
    Treats the sequence time-steps as tokens.
    """
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

    def forward(self, x, t, y=None):
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
        
        # 5. Transformer Layers
        x = self.transformer(x)
        
        # 6. Output Projection
        return self.output_proj(x)
