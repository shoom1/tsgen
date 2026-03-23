import torch
import torch.nn as nn
import torch.nn.functional as F
from tsgen.models.base_model import DiffusionModel
from tsgen.models.embeddings import SinusoidalPositionEmbeddings
from tsgen.models.registry import ModelRegistry

class Block1D(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, kernel_size=3):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.GroupNorm(8, out_channels)
        self.bn2 = nn.GroupNorm(8, out_channels)
        self.relu = nn.SiLU()

    def forward(self, x, t):
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.relu(h)
        time_emb = self.relu(self.time_mlp(t))
        h = h + time_emb.unsqueeze(-1)
        h = self.conv2(h)
        h = self.bn2(h)
        h = self.relu(h)
        return h

@ModelRegistry.register('unet')
class UNet1D(DiffusionModel):
    def __init__(self, sequence_length, features, base_channels=64):
        super().__init__()
        self.features = features
        self.time_dim = base_channels * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_channels),
            nn.Linear(base_channels, self.time_dim),
            nn.SiLU(),
            nn.Linear(self.time_dim, self.time_dim),
        )
        self.down1 = Block1D(features, base_channels, self.time_dim)
        self.down2 = Block1D(base_channels, base_channels * 2, self.time_dim)
        self.down_sample = nn.MaxPool1d(2)
        self.bot1 = Block1D(base_channels * 2, base_channels * 4, self.time_dim)
        self.bot2 = Block1D(base_channels * 4, base_channels * 4, self.time_dim)
        self.up_sample = nn.Upsample(scale_factor=2, mode='nearest')
        self.up1 = Block1D(base_channels * 4 + base_channels * 2, base_channels * 2, self.time_dim)
        self.up2 = Block1D(base_channels * 2 + base_channels, base_channels, self.time_dim)
        self.final_conv = nn.Conv1d(base_channels, features, kernel_size=1)

    @classmethod
    def from_config(cls, config, features=None):
        """Create UNet1D from ExperimentConfig."""
        data = config.get_data_config()
        params = config.get_model_params_config()
        diff = config.get_diffusion_config()
        features = features or len(data.tickers)
        model = cls(
            sequence_length=data.sequence_length,
            features=features,
            base_channels=params.base_channels,
        )
        model._apply_diffusion_config(diff)
        return model

    def forward(self, x, t, y=None, mask=None):
        """
        Forward pass for UNet1D.

        Args:
            x: Input tensor (Batch, Seq_Len, Features)
            t: Timesteps (Batch,)
            y: Optional class labels (Batch,) - not used by UNet, for API compatibility
            mask: Optional binary mask (Batch, Seq_Len, Features) - not used by UNet
                  (UNet uses convolutions which don't support attention-style masking)

        Returns:
            Predicted noise (Batch, Seq_Len, Features)
        """
        x = x.transpose(1, 2)
        t = self.time_mlp(t)
        x1 = self.down1(x, t)
        x2 = self.down_sample(x1)
        x2 = self.down2(x2, t)
        x3 = self.down_sample(x2)
        x3 = self.bot1(x3, t)
        x3 = self.bot2(x3, t)
        x_up = self.up_sample(x3)
        x_up = torch.cat((x_up, x2), dim=1)
        x_up = self.up1(x_up, t)
        x_up = self.up_sample(x_up)
        x_up = torch.cat((x_up, x1), dim=1)
        x_up = self.up2(x_up, t)
        out = self.final_conv(x_up)
        return out.transpose(1, 2)
