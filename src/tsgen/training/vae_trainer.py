"""
Training loop for TimeVAE model.

Implements VAE training with reconstruction loss and KL divergence,
beta annealing, and posterior collapse detection.
"""

import torch
import tempfile
import os
from tqdm import tqdm
from torch.utils.data import DataLoader

from tsgen.training.base import BaseTrainer
from tsgen.training.registry import TrainerRegistry
from tsgen.models.losses import (
    vae_loss,
    vae_loss_with_free_bits,
    VAELossTracker,
    VAEDiagnostics,
    linear_beta_schedule
)


@TrainerRegistry.register('timevae')
class VAETrainer(BaseTrainer):
    """
    Trainer for TimeVAE with beta annealing and collapse detection.

    Implements VAE-specific training with:
    - Reconstruction + KL divergence loss
    - Beta annealing to prevent posterior collapse
    - Free bits constraint
    - Teacher forcing for sequential generation
    - Posterior collapse diagnostics
    """

    def __init__(self, model, config, tracker, device):
        super().__init__(model, config, tracker, device)

        # Resolve configuration sections
        training_conf = config.get('training', config)

        # Optimizer
        learning_rate = training_conf.get('learning_rate', 1e-3)
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(learning_rate)
        )

        # VAE-specific hyperparameters
        self.beta = config.get('vae_beta', 0.5)
        self.use_annealing = config.get('vae_use_annealing', True)
        self.annealing_epochs = config.get('vae_annealing_epochs', 50)
        self.use_free_bits = config.get('vae_use_free_bits', True)
        self.free_bits = config.get('vae_free_bits', 0.5)
        self.teacher_forcing_ratio = config.get('vae_teacher_forcing_ratio', 0.5)

        # Beta annealing schedule
        epochs = training_conf.get('epochs', 10)
        if self.use_annealing:
            self.beta_schedule = linear_beta_schedule(
                max_epochs=epochs,
                warmup_epochs=self.annealing_epochs,
                max_beta=self.beta
            )
            tqdm.write(
                f"Using beta annealing: warmup over {self.annealing_epochs} "
                f"epochs to beta={self.beta}"
            )
        else:
            self.beta_schedule = lambda epoch: self.beta
            tqdm.write(f"Using fixed beta={self.beta}")

        # Diagnostics
        self.diagnostics = VAEDiagnostics(latent_dim=model.latent_dim)

        if self.use_free_bits:
            tqdm.write(f"Using free bits constraint: {self.free_bits} bits per dimension")
        tqdm.write(f"Teacher forcing ratio: {self.teacher_forcing_ratio}")

    def train(self, dataloader: DataLoader) -> torch.nn.Module:
        """
        Train TimeVAE model.

        Args:
            dataloader: Training data loader

        Returns:
            Trained model
        """
        tqdm.write(f"Training TimeVAE on {self.device}")

        self.model.to(self.device)
        self.model.train()

        step_count = 0

        # Get epochs from training section
        training_conf = self.config.get('training', self.config)
        epochs = training_conf.get('epochs', 10)
        start_epoch = self.config.get('start_epoch', 0)

        # Load checkpoint if resuming
        if start_epoch > 0:
            print(f"Resuming training from epoch {start_epoch}")

        with tempfile.TemporaryDirectory() as tmpdir:
            for epoch in range(start_epoch, epochs):
                loss_tracker = VAELossTracker()
                current_beta = self.beta_schedule(epoch)

                pbar = tqdm(
                    dataloader,
                    desc=f"Epoch {epoch+1}/{epochs}",
                    leave=False
                )

                for batch in pbar:
                    # Extract data
                    x = self._extract_batch(batch)

                    # Forward pass with teacher forcing
                    recon, mu, logvar = self.model(x, teacher_forcing_ratio=self.teacher_forcing_ratio)

                    # Compute VAE loss
                    if self.use_free_bits:
                        total_loss, recon_loss, kl_loss, kl_per_dim = vae_loss_with_free_bits(
                            recon, x, mu, logvar, beta=current_beta, free_bits=self.free_bits
                        )
                    else:
                        total_loss, recon_loss, kl_loss = vae_loss(
                            recon, x, mu, logvar, beta=current_beta
                        )
                        kl_per_dim = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=0)

                    # Backward pass
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                    # Track losses
                    loss_tracker.update(total_loss.item(), recon_loss.item(), kl_loss.item())

                    # Update progress bar
                    pbar.set_postfix({
                        'loss': total_loss.item(),
                        'recon': recon_loss.item(),
                        'kl': kl_loss.item(),
                        'beta': current_beta
                    })

                    # Log batch metrics
                    if step_count % 10 == 0:
                        self.tracker.log_metrics({
                            "batch_loss": total_loss.item(),
                            "batch_recon_loss": recon_loss.item(),
                            "batch_kl_loss": kl_loss.item(),
                            "beta": current_beta
                        }, step=step_count)

                    step_count += 1

                # Update diagnostics
                self.diagnostics.update(mu, logvar, recon_loss.item(), kl_per_dim)

                # Epoch-level metrics
                avg_losses = loss_tracker.get_average()
                diag_status = self.diagnostics.get_status()

                tqdm.write(
                    f"Epoch {epoch+1} - "
                    f"Loss: {avg_losses['total_loss']:.5f}, "
                    f"Recon: {avg_losses['recon_loss']:.5f}, "
                    f"KL: {avg_losses['kl_loss']:.5f}, "
                    f"Beta: {current_beta:.3f}, "
                    f"Active dims: {diag_status['active_dims']}/{diag_status['total_dims']}"
                )

                # Warn if collapsed
                if diag_status['collapsed']:
                    tqdm.write("⚠️  WARNING: Posterior collapse detected! Consider:")
                    tqdm.write("   - Reducing max_beta or increasing warmup_epochs")
                    tqdm.write("   - Increasing free_bits constraint")
                    tqdm.write("   - Reducing learning rate")

                self.tracker.log_metrics({
                    "epoch_loss": avg_losses['total_loss'],
                    "epoch_recon_loss": avg_losses['recon_loss'],
                    "epoch_kl_loss": avg_losses['kl_loss'],
                    "epoch": epoch + 1,
                    "beta": current_beta,
                    "active_dimensions": diag_status['active_dims'],
                    "mean_kl_per_dim": diag_status['mean_kl'],
                    "mu_std": diag_status['mu_std'],
                    "collapsed": diag_status['collapsed']
                }, step=step_count)

                # Checkpointing - save full checkpoint with optimizer state
                if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
                    ckpt_filename = f"checkpoint_epoch_{epoch+1}.pt"
                    ckpt_path = os.path.join(tmpdir, ckpt_filename)
                    self.save_checkpoint(
                        ckpt_path,
                        epoch=epoch + 1,
                        optimizer=self.optimizer,
                        step_count=step_count,
                        beta=current_beta
                    )
                    self.tracker.log_artifact(ckpt_path, artifact_type='checkpoint')

            tqdm.write("TimeVAE Training Complete.")
            return self.model

    def _extract_batch(self, batch):
        """
        Extract tensor from batch.

        Args:
            batch: Batch from dataloader

        Returns:
            Tensor moved to device
        """
        if isinstance(batch, (list, tuple)):
            return batch[0].to(self.device)
        return batch.to(self.device)
