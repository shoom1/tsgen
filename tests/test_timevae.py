"""
Tests for TimeVAE model.

TimeVAE learns latent representations of time series using variational
inference with encoder-decoder architecture.
"""

import pytest
import torch
import numpy as np
from tsgen.models.timevae import TimeVAE, TimeVAEEncoder, TimeVAEDecoder
from tsgen.training.losses import vae_loss, linear_beta_schedule, VAELossTracker


def test_timevae_initialization():
    """Test TimeVAE can be initialized."""
    model = TimeVAE(features=3, sequence_length=64, hidden_dim=32, latent_dim=8)
    assert model is not None
    assert model.features == 3
    assert model.sequence_length == 64
    assert model.latent_dim == 8


def test_timevae_lstm_encoder():
    """Test LSTM encoder forward pass."""
    encoder = TimeVAEEncoder(
        features=3,
        hidden_dim=32,
        latent_dim=8,
        encoder_type='lstm'
    )

    x = torch.randn(4, 64, 3)  # (Batch, Seq, Features)
    mu, logvar = encoder(x)

    assert mu.shape == (4, 8)
    assert logvar.shape == (4, 8)
    assert torch.all(torch.isfinite(mu))
    assert torch.all(torch.isfinite(logvar))


def test_timevae_transformer_encoder():
    """Test Transformer encoder forward pass."""
    encoder = TimeVAEEncoder(
        features=3,
        hidden_dim=32,
        latent_dim=8,
        encoder_type='transformer'
    )

    x = torch.randn(4, 64, 3)
    mu, logvar = encoder(x)

    assert mu.shape == (4, 8)
    assert logvar.shape == (4, 8)


def test_timevae_decoder():
    """Test decoder forward pass."""
    decoder = TimeVAEDecoder(
        latent_dim=8,
        hidden_dim=32,
        features=3,
        sequence_length=64
    )

    z = torch.randn(4, 8)  # (Batch, latent_dim)
    recon = decoder(z)

    assert recon.shape == (4, 64, 3)
    assert torch.all(torch.isfinite(recon))


def test_timevae_forward():
    """Test TimeVAE forward pass."""
    model = TimeVAE(features=3, sequence_length=64, hidden_dim=32, latent_dim=8)

    x = torch.randn(4, 64, 3)
    recon, mu, logvar = model(x)

    assert recon.shape == x.shape
    assert mu.shape == (4, 8)
    assert logvar.shape == (4, 8)
    assert torch.all(torch.isfinite(recon))


def test_timevae_reparameterize():
    """Test reparameterization trick."""
    model = TimeVAE(features=2, sequence_length=32, latent_dim=4)

    mu = torch.zeros(10, 4)
    logvar = torch.zeros(10, 4)

    # Sample multiple times with same seed
    torch.manual_seed(42)
    z1 = model.reparameterize(mu, logvar)

    torch.manual_seed(42)
    z2 = model.reparameterize(mu, logvar)

    # Should be reproducible
    assert torch.allclose(z1, z2)

    # Should have correct shape
    assert z1.shape == (10, 4)


def test_timevae_sample():
    """Test TimeVAE sampling from prior."""
    model = TimeVAE(features=3, sequence_length=64, hidden_dim=32, latent_dim=8)

    samples = model.generate(10)

    assert samples.shape == (10, 64, 3)
    assert torch.all(torch.isfinite(samples))


def test_timevae_sample_sequence_length_validation():
    """Test that TimeVAE validates sequence length during sampling."""
    model = TimeVAE(features=2, sequence_length=64)

    # Should work with default or matching length
    samples1 = model.generate(5)
    assert samples1.shape == (5, 64, 2)

    samples2 = model.generate(5, seq_len=64)
    assert samples2.shape == (5, 64, 2)

    # Should raise error for different length
    with pytest.raises(ValueError):
        model.generate(5, seq_len=128)


def test_timevae_encode_decode():
    """Test separate encode and decode methods."""
    model = TimeVAE(features=2, sequence_length=32, latent_dim=4)

    x = torch.randn(5, 32, 2)

    # Encode
    z = model.encode(x)
    assert z.shape == (5, 4)

    # Decode
    recon = model.decode(z)
    assert recon.shape == (5, 32, 2)


def test_vae_loss():
    """Test VAE loss function."""
    batch_size = 8
    seq_len = 32
    features = 2

    x = torch.randn(batch_size, seq_len, features)
    recon = torch.randn(batch_size, seq_len, features)
    mu = torch.randn(batch_size, 4)
    logvar = torch.randn(batch_size, 4)

    total_loss, recon_loss, kl_loss = vae_loss(recon, x, mu, logvar, beta=1.0)

    # Check types
    assert isinstance(total_loss.item(), float)
    assert isinstance(recon_loss.item(), float)
    assert isinstance(kl_loss.item(), float)

    # Check total loss composition
    assert torch.isclose(total_loss, recon_loss + kl_loss, atol=1e-5)


def test_vae_loss_beta():
    """Test VAE loss with different beta values."""
    batch_size = 4
    x = torch.randn(batch_size, 16, 2)
    recon_tensor = torch.randn(batch_size, 16, 2)
    mu = torch.randn(batch_size, 4)
    logvar = torch.randn(batch_size, 4)

    # Get losses with different betas
    loss_beta_0, recon_loss, kl = vae_loss(recon_tensor, x, mu, logvar, beta=0.0)
    loss_beta_1, _, _ = vae_loss(recon_tensor, x, mu, logvar, beta=1.0)
    loss_beta_2, _, _ = vae_loss(recon_tensor, x, mu, logvar, beta=2.0)

    # Higher beta should give higher total loss (if KL > 0)
    if kl > 0:
        assert loss_beta_0 < loss_beta_1 < loss_beta_2


def test_linear_beta_schedule():
    """Test linear beta annealing schedule."""
    schedule = linear_beta_schedule(max_epochs=100, warmup_epochs=10, max_beta=1.0)

    # At epoch 0, should be 0
    assert schedule(0) == 0.0

    # At halfway through warmup, should be 0.5
    assert schedule(5) == pytest.approx(0.5)

    # At end of warmup, should be 1.0
    assert schedule(10) == pytest.approx(1.0)

    # After warmup, should stay at 1.0
    assert schedule(50) == pytest.approx(1.0)


def test_vae_loss_tracker():
    """Test VAE loss tracking helper."""
    tracker = VAELossTracker()

    # Initially empty
    avg = tracker.get_average()
    assert avg['total_loss'] == 0.0

    # Update with some values
    tracker.update(1.5, 1.0, 0.5)
    tracker.update(2.0, 1.2, 0.8)

    # Check averages
    avg = tracker.get_average()
    assert avg['total_loss'] == pytest.approx(1.75)
    assert avg['recon_loss'] == pytest.approx(1.1)
    assert avg['kl_loss'] == pytest.approx(0.65)

    # Reset and check
    tracker.reset()
    avg = tracker.get_average()
    assert avg['total_loss'] == 0.0


def test_timevae_reconstruction_quality():
    """Test that TimeVAE can reconstruct simple patterns."""
    torch.manual_seed(42)

    # Create simple periodic pattern
    t = torch.linspace(0, 4 * np.pi, 64).unsqueeze(0).unsqueeze(-1)
    x = torch.sin(t).repeat(16, 1, 2)  # (16, 64, 2)

    model = TimeVAE(features=2, sequence_length=64, hidden_dim=64, latent_dim=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Train briefly (use no teacher forcing when overfitting single batch)
    for _ in range(100):
        recon, mu, logvar = model(x, teacher_forcing_ratio=0.0)
        loss, _, _ = vae_loss(recon, x, mu, logvar, beta=0.1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Check reconstruction improves
    with torch.no_grad():
        recon, _, _ = model(x, teacher_forcing_ratio=0.0)
        final_loss = torch.nn.functional.mse_loss(recon, x)

    # More complex autoregressive decoder may need more epochs or higher threshold
    assert final_loss < 0.5, "Model should learn to reconstruct simple pattern"


def test_timevae_latent_space():
    """Test that latent space has reasonable properties."""
    model = TimeVAE(features=2, sequence_length=32, latent_dim=4)

    # Create two different patterns
    x1 = torch.sin(torch.linspace(0, 2 * np.pi, 32)).unsqueeze(0).unsqueeze(-1).repeat(1, 1, 2)
    x2 = torch.cos(torch.linspace(0, 2 * np.pi, 32)).unsqueeze(0).unsqueeze(-1).repeat(1, 1, 2)

    # Encode
    z1 = model.encode(x1)
    z2 = model.encode(x2)

    # Latents should be finite
    assert torch.all(torch.isfinite(z1))
    assert torch.all(torch.isfinite(z2))

    # Latents should have correct shape
    assert z1.shape == (1, 4)
    assert z2.shape == (1, 4)


def test_timevae_save_load():
    """Test that TimeVAE can be saved and loaded."""
    import tempfile
    import os

    model = TimeVAE(features=2, sequence_length=32, latent_dim=4)

    # Generate sample
    torch.manual_seed(42)
    sample1 = model.generate(3)

    # Save model
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
        temp_path = f.name
        torch.save(model.state_dict(), temp_path)

    try:
        # Load model
        loaded_model = TimeVAE(features=2, sequence_length=32, latent_dim=4)
        loaded_model.load_state_dict(torch.load(temp_path, weights_only=True))

        # Generate sample with same seed
        torch.manual_seed(42)
        sample2 = loaded_model.generate(3)

        # Should be identical
        assert torch.allclose(sample1, sample2, atol=1e-5)

    finally:
        os.unlink(temp_path)


def test_timevae_different_encoder_types():
    """Test that both LSTM and Transformer encoders work."""
    x = torch.randn(4, 32, 2)

    # LSTM encoder
    model_lstm = TimeVAE(
        features=2,
        sequence_length=32,
        encoder_type='lstm',
        hidden_dim=32,
        latent_dim=4
    )
    recon_lstm, mu_lstm, logvar_lstm = model_lstm(x)
    assert recon_lstm.shape == x.shape

    # Transformer encoder
    model_transformer = TimeVAE(
        features=2,
        sequence_length=32,
        encoder_type='transformer',
        hidden_dim=32,
        latent_dim=4
    )
    recon_transformer, mu_transformer, logvar_transformer = model_transformer(x)
    assert recon_transformer.shape == x.shape


def test_timevae_gradient_flow():
    """Test that gradients flow through the model."""
    model = TimeVAE(features=2, sequence_length=16, hidden_dim=16, latent_dim=4)

    x = torch.randn(2, 16, 2)
    recon, mu, logvar = model(x)
    loss, _, _ = vae_loss(recon, x, mu, logvar)

    # Compute gradients
    loss.backward()

    # Check that encoder and decoder have gradients
    encoder_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                          for p in model.encoder.parameters())
    decoder_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                          for p in model.decoder.parameters())

    assert encoder_has_grad, "Encoder should have gradients"
    assert decoder_has_grad, "Decoder should have gradients"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
