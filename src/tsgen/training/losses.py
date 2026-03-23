"""
Loss functions for training generative models.

This module contains all loss functions used during training:
- Generic masked losses for handling missing data
- VAE losses with KL divergence and various regularization strategies
- Loss tracking and diagnostic utilities

Loss functions are training concerns (not model architecture) so they
belong in the training module rather than models.
"""

import torch
import torch.nn.functional as F


# =============================================================================
# Generic Masked Losses (for missing data handling)
# =============================================================================

def masked_mse_loss(pred, target, mask=None):
    """
    Mean Squared Error loss computed only on valid (non-masked) positions.

    This is essential for training on data with missing values. The loss
    is computed only where the mask is 1, ignoring positions where data
    is missing (mask is 0).

    Args:
        pred: (Batch, Seq_Len, Features) predicted values
        target: (Batch, Seq_Len, Features) target values
        mask: (Batch, Seq_Len, Features) binary mask, 1=valid, 0=masked
              If None, computes regular MSE loss

    Returns:
        Scalar loss value

    Example:
        >>> pred = model(x_t, t, mask=mask)
        >>> loss = masked_mse_loss(pred, noise, mask)
        >>> loss.backward()
    """
    if mask is None:
        return F.mse_loss(pred, target)

    # Element-wise squared error
    sq_error = (pred - target) ** 2

    # Apply mask: only keep errors at valid positions
    masked_error = sq_error * mask

    # Compute mean over valid positions only
    # Use clamp to avoid division by zero when mask is all zeros
    num_valid = mask.sum().clamp(min=1)
    return masked_error.sum() / num_valid


def masked_l1_loss(pred, target, mask=None):
    """
    L1 (Mean Absolute Error) loss computed only on valid positions.

    Args:
        pred: (Batch, Seq_Len, Features) predicted values
        target: (Batch, Seq_Len, Features) target values
        mask: (Batch, Seq_Len, Features) binary mask, 1=valid, 0=masked
              If None, computes regular L1 loss

    Returns:
        Scalar loss value
    """
    if mask is None:
        return F.l1_loss(pred, target)

    abs_error = torch.abs(pred - target)
    masked_error = abs_error * mask
    num_valid = mask.sum().clamp(min=1)
    return masked_error.sum() / num_valid


def masked_huber_loss(pred, target, mask=None, delta=1.0):
    """
    Huber loss (smooth L1) computed only on valid positions.

    Huber loss is less sensitive to outliers than MSE.

    Args:
        pred: (Batch, Seq_Len, Features) predicted values
        target: (Batch, Seq_Len, Features) target values
        mask: (Batch, Seq_Len, Features) binary mask, 1=valid, 0=masked
        delta: Threshold for switching between L1 and L2 loss

    Returns:
        Scalar loss value
    """
    if mask is None:
        return F.smooth_l1_loss(pred, target, beta=delta)

    # Compute element-wise Huber loss
    diff = torch.abs(pred - target)
    huber = torch.where(
        diff < delta,
        0.5 * diff ** 2,
        delta * (diff - 0.5 * delta)
    )

    masked_loss = huber * mask
    num_valid = mask.sum().clamp(min=1)
    return masked_loss.sum() / num_valid


# =============================================================================
# VAE Losses
# =============================================================================

def vae_loss(recon, x, mu, logvar, beta=1.0):
    """
    Variational Autoencoder (VAE) loss function.

    Combines reconstruction loss and KL divergence regularization.
    Uses beta-VAE formulation for controlling regularization strength.

    Loss = Reconstruction_Loss + beta * KL_Divergence

    Args:
        recon: Reconstructed data (Batch, Seq, Features)
        x: Original data (Batch, Seq, Features)
        mu: Mean of latent distribution (Batch, latent_dim)
        logvar: Log-variance of latent distribution (Batch, latent_dim)
        beta: Weight for KL divergence term (beta-VAE)
              beta=1.0: Standard VAE
              beta<1.0: Less regularization (better reconstruction)
              beta>1.0: More regularization (better disentanglement)

    Returns:
        total_loss: Combined loss (scalar)
        recon_loss: Reconstruction loss component (scalar)
        kl_loss: KL divergence component (scalar)
    """
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(recon, x, reduction='mean')

    # KL divergence: KL(q(z|x) || p(z))
    # where q(z|x) = N(mu, exp(logvar)) and p(z) = N(0, I)
    # KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # Using logvar: KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    # Total loss
    total_loss = recon_loss + beta * kl_loss

    return total_loss, recon_loss, kl_loss


def vae_loss_with_free_bits(recon, x, mu, logvar, beta=1.0, free_bits=0.5):
    """
    VAE loss with free bits constraint to prevent posterior collapse.

    Free bits: Don't penalize KL divergence if it's already below a threshold.
    This prevents the encoder from completely collapsing to the prior.

    Args:
        recon: Reconstructed data (Batch, Seq, Features)
        x: Original data (Batch, Seq, Features)
        mu: Mean of latent distribution (Batch, latent_dim)
        logvar: Log-variance of latent distribution (Batch, latent_dim)
        beta: Weight for KL divergence term
        free_bits: Minimum KL bits per dimension (e.g., 0.5 bits)

    Returns:
        total_loss: Combined loss (scalar)
        recon_loss: Reconstruction loss component (scalar)
        kl_loss: KL divergence component (scalar)
        kl_per_dim_mean: Mean KL per dimension for monitoring (latent_dim,)
    """
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(recon, x, reduction='mean')

    # Per-dimension KL divergence
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # (Batch, latent_dim)

    # Apply free bits constraint per dimension
    # Only penalize KL if it's above the free bits threshold
    kl_per_dim_clamped = torch.clamp(kl_per_dim, min=free_bits)

    # Average over batch and dimensions
    kl_loss = torch.mean(kl_per_dim_clamped)

    # Total loss
    total_loss = recon_loss + beta * kl_loss

    # Return per-dim KL for monitoring
    kl_per_dim_mean = torch.mean(kl_per_dim, dim=0)  # (latent_dim,)

    return total_loss, recon_loss, kl_loss, kl_per_dim_mean


# =============================================================================
# Beta Schedules for VAE Training
# =============================================================================

def linear_beta_schedule(max_epochs, warmup_epochs=50, max_beta=0.5):
    """
    Creates a linear beta annealing schedule.

    Beta increases linearly from 0 to max_beta over warmup_epochs,
    then stays at max_beta.

    Default parameters are more conservative to prevent posterior collapse:
    - warmup_epochs=50 (slower warmup)
    - max_beta=0.5 (lower max value)

    Args:
        max_epochs: Total number of training epochs
        warmup_epochs: Number of epochs for warmup (default: 50, increased from 10)
        max_beta: Maximum beta value (default: 0.5, reduced from 1.0)

    Returns:
        Function that maps epoch -> beta
    """
    def schedule(epoch):
        if epoch < warmup_epochs:
            return (epoch / warmup_epochs) * max_beta
        else:
            return max_beta

    return schedule


def cyclical_beta_schedule(cycle_length=10, n_cycles=4, max_beta=1.0):
    """
    Creates a cyclical beta schedule.

    Beta oscillates between 0 and max_beta in cycles. This can help
    avoid posterior collapse while still providing strong regularization.

    Args:
        cycle_length: Number of epochs per cycle
        n_cycles: Number of cycles
        max_beta: Maximum beta value

    Returns:
        Function that maps epoch -> beta
    """
    def schedule(epoch):
        cycle = (epoch % cycle_length) / cycle_length
        return cycle * max_beta

    return schedule


# =============================================================================
# Training Utilities
# =============================================================================

class VAELossTracker:
    """
    Helper class for tracking VAE loss components during training.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all tracked metrics."""
        self.total_loss = 0.0
        self.recon_loss = 0.0
        self.kl_loss = 0.0
        self.n_batches = 0

    def update(self, total_loss, recon_loss, kl_loss):
        """
        Update tracked metrics with batch losses.

        Args:
            total_loss: Total loss for batch
            recon_loss: Reconstruction loss for batch
            kl_loss: KL divergence for batch
        """
        self.total_loss += total_loss
        self.recon_loss += recon_loss
        self.kl_loss += kl_loss
        self.n_batches += 1

    def get_average(self):
        """
        Get average losses over all batches.

        Returns:
            Dictionary with average losses
        """
        if self.n_batches == 0:
            return {
                'total_loss': 0.0,
                'recon_loss': 0.0,
                'kl_loss': 0.0
            }

        return {
            'total_loss': self.total_loss / self.n_batches,
            'recon_loss': self.recon_loss / self.n_batches,
            'kl_loss': self.kl_loss / self.n_batches
        }


class VAEDiagnostics:
    """
    Track detailed diagnostics to detect posterior collapse early.

    Monitors per-dimension KL, mu statistics, and active dimensions to
    detect if the VAE posterior is collapsing to the prior.
    """
    def __init__(self, latent_dim):
        """
        Args:
            latent_dim: Dimension of latent space
        """
        self.latent_dim = latent_dim
        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.kl_per_dim_history = []
        self.mu_std_history = []
        self.logvar_mean_history = []
        self.recon_loss_history = []
        self.active_dims_history = []

    def update(self, mu, logvar, recon_loss, kl_per_dim):
        """
        Update diagnostics with batch statistics.

        Args:
            mu: (Batch, latent_dim)
            logvar: (Batch, latent_dim)
            recon_loss: scalar
            kl_per_dim: (latent_dim,)
        """
        # Track per-dimension KL
        self.kl_per_dim_history.append(kl_per_dim.detach().cpu())

        # Track mu std (should not be zero)
        mu_std = mu.std(dim=0).mean().item()
        self.mu_std_history.append(mu_std)

        # Track logvar mean (should not be exactly 0)
        logvar_mean = logvar.mean().item()
        self.logvar_mean_history.append(logvar_mean)

        # Track reconstruction loss
        self.recon_loss_history.append(recon_loss)

        # Count active dimensions (KL > 0.1)
        active_dims = (kl_per_dim > 0.1).sum().item()
        self.active_dims_history.append(active_dims)

    def is_collapsed(self, window=10):
        """
        Detect if posterior has collapsed.

        Args:
            window: Number of recent epochs to check

        Returns:
            bool: True if collapse is detected
        """
        if len(self.active_dims_history) < window:
            return False

        recent_active = self.active_dims_history[-window:]
        recent_mu_std = self.mu_std_history[-window:]

        # Collapse indicators:
        # 1. Very few active dimensions
        # 2. Mu has very low variance
        avg_active = sum(recent_active) / len(recent_active)
        avg_mu_std = sum(recent_mu_std) / len(recent_mu_std)

        if avg_active < self.latent_dim * 0.2:  # Less than 20% dimensions active
            return True

        if avg_mu_std < 0.01:  # Mu has collapsed
            return True

        return False

    def get_status(self):
        """
        Get current diagnostic status as dictionary.

        Returns:
            Dictionary with current metrics
        """
        if not self.kl_per_dim_history:
            return {
                'active_dims': 0,
                'mean_kl': 0.0,
                'mu_std': 0.0,
                'recon_loss': 0.0,
                'collapsed': False
            }

        recent_kl = self.kl_per_dim_history[-1]
        recent_active = self.active_dims_history[-1]
        recent_mu_std = self.mu_std_history[-1]
        recent_recon = self.recon_loss_history[-1]

        return {
            'active_dims': recent_active,
            'total_dims': self.latent_dim,
            'mean_kl': recent_kl.mean().item(),
            'mu_std': recent_mu_std,
            'recon_loss': recent_recon,
            'collapsed': self.is_collapsed()
        }
