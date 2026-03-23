"""
TimeVAE: Variational Autoencoder for Time Series Generation.

TimeVAE learns a latent representation of time series data using variational
inference. It consists of an encoder that maps sequences to a latent distribution
and a decoder that reconstructs sequences from latent samples.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsgen.models.base_model import VAEModel
from tsgen.models.registry import ModelRegistry


class TimeVAEEncoder(nn.Module):
    """
    Temporal encoder using LSTM or Transformer.

    Maps input sequences to latent distribution parameters (mean, log-variance).
    """
    def __init__(self, features, hidden_dim, latent_dim, encoder_type='lstm', num_layers=2):
        """
        Args:
            features: Number of input features (tickers)
            hidden_dim: Hidden dimension for LSTM/Transformer
            latent_dim: Dimension of latent space
            encoder_type: 'lstm' or 'transformer'
            num_layers: Number of encoder layers
        """
        super().__init__()
        self.encoder_type = encoder_type
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        if encoder_type == 'lstm':
            self.encoder = nn.LSTM(
                features,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True
            )
            # Bidirectional LSTM outputs hidden_dim * 2
            self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
            self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)

        elif encoder_type == 'transformer':
            # Project input to hidden_dim
            self.input_projection = nn.Linear(features, hidden_dim)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dim_feedforward=hidden_dim * 4,
                batch_first=True,
                dropout=0.1
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

            self.fc_mu = nn.Linear(hidden_dim, latent_dim)
            self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

    def forward(self, x):
        """
        Encode input sequence to latent distribution parameters.

        Args:
            x: Input tensor (Batch, Seq, Features)

        Returns:
            mu: Mean of latent distribution (Batch, latent_dim)
            logvar: Log-variance of latent distribution (Batch, latent_dim)
        """
        if self.encoder_type == 'lstm':
            # LSTM encoding
            _, (h_n, _) = self.encoder(x)
            # Concatenate forward and backward hidden states from last layer
            # h_n shape: (num_layers * 2, Batch, hidden_dim)
            h_forward = h_n[-2]  # Last layer forward
            h_backward = h_n[-1]  # Last layer backward
            h = torch.cat([h_forward, h_backward], dim=-1)  # (Batch, hidden_dim * 2)

        else:  # transformer
            # Project input
            x_proj = self.input_projection(x)  # (Batch, Seq, hidden_dim)
            # Encode
            encoded = self.encoder(x_proj)  # (Batch, Seq, hidden_dim)
            # Aggregate over sequence (mean pooling)
            h = encoded.mean(dim=1)  # (Batch, hidden_dim)

        # Map to latent distribution parameters
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar


class TimeVAEDecoder(nn.Module):
    """
    Improved temporal decoder with autoregressive structure.

    Reconstructs sequences from latent representations using:
    - Autoregressive generation (uses previous timesteps)
    - Teacher forcing during training
    - Latent conditioning at each timestep
    - Batch normalization for stability
    - Residual connections

    This addresses posterior collapse by making the decoder actually
    use the latent information effectively.
    """
    def __init__(self, latent_dim, hidden_dim, features, sequence_length, num_layers=2):
        """
        Args:
            latent_dim: Dimension of latent space
            hidden_dim: Hidden dimension for LSTM
            features: Number of output features (tickers)
            sequence_length: Length of output sequences
            num_layers: Number of LSTM layers
        """
        super().__init__()
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.features = features
        self.num_layers = num_layers

        # Project latent to initial hidden and cell states
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim * num_layers)
        self.latent_to_cell = nn.Linear(latent_dim, hidden_dim * num_layers)

        # Project latent for conditioning at each timestep
        self.latent_projection = nn.Linear(latent_dim, hidden_dim)

        # Autoregressive LSTM decoder
        # Input: previous output + latent conditioning
        self.decoder = nn.LSTM(
            features + hidden_dim,  # Concatenate previous output with latent conditioning
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0
        )

        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(hidden_dim)

        # Output projection with residual
        self.pre_output = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, features)

    def forward(self, z, x=None, teacher_forcing_ratio=0.5):
        """
        Decode latent representation to sequence with autoregressive structure.

        Args:
            z: Latent tensor (Batch, latent_dim)
            x: Ground truth for teacher forcing (Batch, Seq, Features) [optional]
            teacher_forcing_ratio: Probability of using teacher forcing (training only)

        Returns:
            Reconstructed sequence (Batch, Seq, Features)
        """
        batch_size = z.size(0)
        device = z.device

        # Initialize hidden and cell states from latent
        h0 = self.latent_to_hidden(z).view(self.num_layers, batch_size, self.hidden_dim)
        c0 = self.latent_to_cell(z).view(self.num_layers, batch_size, self.hidden_dim)

        # Project latent for conditioning at each timestep
        z_projected = self.latent_projection(z)  # (Batch, hidden_dim)
        z_seq = z_projected.unsqueeze(1).repeat(1, self.sequence_length, 1)  # (Batch, Seq, hidden_dim)

        # Start with zeros
        decoder_input = torch.zeros(batch_size, self.features, device=device)

        outputs = []
        hidden = (h0, c0)

        for t in range(self.sequence_length):
            # Combine previous output with latent conditioning
            # (Batch, features) + (Batch, hidden_dim) -> (Batch, features + hidden_dim)
            combined_input = torch.cat([decoder_input, z_seq[:, t, :]], dim=-1)
            combined_input = combined_input.unsqueeze(1)  # (Batch, 1, features + hidden_dim)

            # LSTM step
            lstm_out, hidden = self.decoder(combined_input, hidden)  # (Batch, 1, hidden_dim)
            lstm_out = lstm_out.squeeze(1)  # (Batch, hidden_dim)

            # Batch norm (apply on hidden_dim dimension)
            if self.training and batch_size > 1:
                lstm_out = self.batch_norm(lstm_out)

            # Residual connection through pre-output
            pre_out = self.pre_output(lstm_out)
            pre_out = F.relu(pre_out)
            output_t = self.output(pre_out + lstm_out)  # Residual

            outputs.append(output_t)

            # Teacher forcing: use ground truth or predicted output
            if x is not None and self.training and torch.rand(1).item() < teacher_forcing_ratio:
                decoder_input = x[:, t, :]  # Use ground truth
            else:
                decoder_input = output_t  # Use prediction

        # Stack outputs
        recon = torch.stack(outputs, dim=1)  # (Batch, Seq, Features)

        return recon


@ModelRegistry.register('timevae')
class TimeVAE(VAEModel):
    """
    TimeVAE: Variational Autoencoder for Time Series.

    Combines encoder and decoder with reparameterization trick for
    learning latent representations of time series data.

    Inherits from VAEModel which provides the standard VAE interface:
    - forward(x) -> (reconstruction, mu, logvar)
    - encode(x) -> latent representation
    - decode(z) -> reconstruction
    - generate(n) -> generated samples from prior
    """

    def __init__(
        self,
        features,
        sequence_length,
        hidden_dim=64,
        latent_dim=16,
        encoder_type='lstm',
        num_layers=2
    ):
        """
        Args:
            features: Number of features (tickers)
            sequence_length: Length of sequences
            hidden_dim: Hidden dimension for encoder/decoder
            latent_dim: Dimension of latent space
            encoder_type: 'lstm' or 'transformer'
            num_layers: Number of layers in encoder/decoder
        """
        super().__init__()
        self.features = features
        self.sequence_length = sequence_length
        self._latent_dim = latent_dim  # Store as private attribute
        self.encoder_type = encoder_type

        self.encoder = TimeVAEEncoder(
            features,
            hidden_dim,
            latent_dim,
            encoder_type,
            num_layers
        )

        self.decoder = TimeVAEDecoder(
            latent_dim,
            hidden_dim,
            features,
            sequence_length,
            num_layers
        )

    @classmethod
    def from_config(cls, config, features=None):
        """Create TimeVAE from ExperimentConfig."""
        data = config.get_data_config()
        params = config.get_model_params_config()
        features = features or len(data.tickers)
        return cls(
            features=features,
            sequence_length=data.sequence_length,
            hidden_dim=params.hidden_dim,
            latent_dim=params.latent_dim,
            encoder_type=params.encoder_type,
            num_layers=params.num_layers,
        )

    @property
    def latent_dim(self) -> int:
        """Return dimension of the latent space."""
        return self._latent_dim

    def forward(self, x, teacher_forcing_ratio=0.5):
        """
        Forward pass through encoder and decoder.

        Args:
            x: Input tensor (Batch, Seq, Features)
            teacher_forcing_ratio: Probability of using teacher forcing in decoder

        Returns:
            recon: Reconstructed sequence (Batch, Seq, Features)
            mu: Mean of latent distribution (Batch, latent_dim)
            logvar: Log-variance of latent distribution (Batch, latent_dim)
        """
        # Encode to latent distribution
        mu, logvar = self.encoder(x)

        # Sample from latent distribution
        z = self.reparameterize(mu, logvar)

        # Decode to reconstruction (pass x for teacher forcing)
        recon = self.decoder(z, x=x, teacher_forcing_ratio=teacher_forcing_ratio)

        return recon, mu, logvar

    def generate(self, n_samples, seq_len=None, device='cpu', **kwargs):
        """
        Generate samples from prior distribution.

        Args:
            n_samples: Number of samples to generate
            seq_len: Length of sequences (uses self.sequence_length if None)
            device: Device to generate on
            **kwargs: Additional arguments (unused)

        Returns:
            Generated sequences (n_samples, sequence_length, features)
        """
        if seq_len is not None and seq_len != self.sequence_length:
            raise ValueError(
                f"TimeVAE decoder is fixed to sequence_length={self.sequence_length}. "
                f"Cannot generate sequences of length {seq_len}."
            )

        model_device = next(self.parameters()).device

        # Sample from prior N(0, I)
        z = torch.randn(n_samples, self.latent_dim, device=model_device)

        # Decode
        with torch.no_grad():
            samples = self.decoder(z)

        return samples.to(device)

    def encode(self, x):
        """
        Encode input to latent representation (using mean, not sampling).

        Args:
            x: Input tensor (Batch, Seq, Features)

        Returns:
            Latent representation (Batch, latent_dim)
        """
        mu, _ = self.encoder(x)
        return mu

    def decode(self, z):
        """
        Decode latent representation to sequence.

        Args:
            z: Latent tensor (Batch, latent_dim)

        Returns:
            Reconstructed sequence (Batch, Seq, Features)
        """
        return self.decoder(z)
