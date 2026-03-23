"""
Diffusion model trainer for UNet and Transformer architectures.

Implements training with noise prediction, conditional generation support,
and optional periodic validation using stylized facts.
"""

import torch
import torch.nn as nn
import numpy as np
import tempfile
import os
from tqdm import tqdm
from torch.utils.data import DataLoader

from tsgen.training.base import BaseTrainer
from tsgen.training.registry import TrainerRegistry
from tsgen.training.losses import masked_mse_loss
from tsgen.models.diffusion import DiffusionUtils
from tsgen.analysis.metrics import calculate_stylized_facts


@TrainerRegistry.register('unet', 'transformer', 'mamba')
class DiffusionTrainer(BaseTrainer):
    """
    Trainer for diffusion models (UNet, Transformer).

    Trains models to predict noise at random timesteps using the DDPM framework.
    Supports conditional generation via classifier-free guidance and optional
    periodic validation during training.
    """

    def __init__(self, model, config, tracker, device):
        super().__init__(model, config, tracker, device)

        # Parse configuration using typed config accessors
        self.training_config = config.get_training_config()
        self.diffusion_config = config.get_diffusion_config()

        # Diffusion utilities
        self.diff_utils = DiffusionUtils(T=self.diffusion_config.time_steps, device=device)

        # Optimizer with config-driven learning rate
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.training_config.learning_rate
        )
        self.loss_fn = nn.MSELoss()

        # Gradient clipping threshold from config
        self.gradient_clip = self.training_config.gradient_clip

        # Conditional generation settings
        self.num_classes = config.get_model_params_config().num_classes
        # Probability of dropping the conditioning during training for Classifier-Free Guidance
        self.cfg_probability = getattr(config, 'classifier_free_guidance_probability', 0.0)

        # Validation settings from config
        self.validation_interval = self.training_config.validation_interval
        self.num_validation_samples = self.training_config.num_validation_samples

    def train(self, dataloader: DataLoader) -> torch.nn.Module:
        """
        Train diffusion model with noise prediction.

        Args:
            dataloader: Training data loader

        Returns:
            Trained model
        """
        self.model.to(self.device)
        self.model.train()

        step_count = 0

        # Get feature count from model
        features = self.model.features if hasattr(self.model, 'features') else len(self.config.get_data_config().tickers)

        # Use typed training config
        epochs = self.training_config.epochs
        start_epoch = getattr(self.config, 'start_epoch', 0)
        checkpoint_interval = self.training_config.checkpoint_interval

        # Load checkpoint if resuming
        if start_epoch > 0:
            print(f"Resuming training from epoch {start_epoch}")

        with tempfile.TemporaryDirectory() as tmpdir:
            for epoch in range(start_epoch, epochs):
                pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
                total_loss = 0

                for batch in pbar:
                    # Extract data and optional mask from batch
                    x_0, mask = self._extract_batch(batch)

                    # Sample timestep and add noise using typed config
                    t = torch.randint(
                        0, self.diffusion_config.time_steps,
                        (x_0.shape[0],), device=self.device
                    ).long()
                    noise = torch.randn_like(x_0)

                    # If mask provided, zero out noise at masked positions
                    # This ensures we don't add noise to missing data
                    if mask is not None:
                        noise = noise * mask

                    x_t = self.diff_utils.q_sample(x_0, t, noise)

                    # Conditional generation
                    y_conditional = self._get_conditional_labels(x_0.shape[0])

                    # Forward pass with mask
                    predicted_noise = self.model(x_t, t, y_conditional, mask=mask)

                    # Use masked loss if mask provided, otherwise regular MSE
                    loss = masked_mse_loss(predicted_noise, noise, mask)

                    # Backward pass
                    self.optimizer.zero_grad()
                    loss.backward()
                    # Gradient clipping to prevent exploding gradients (configurable)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clip)
                    self.optimizer.step()

                    # Logging
                    current_loss = loss.item()
                    total_loss += current_loss
                    pbar.set_postfix({'loss': current_loss})

                    if step_count % 10 == 0:
                        self.tracker.log_metrics({"batch_loss": current_loss}, step=step_count)
                    step_count += 1

                # Epoch metrics
                avg_loss = total_loss / len(dataloader)
                print(f"Epoch {epoch+1} Average Loss: {avg_loss:.5f}")
                self.tracker.log_metrics({
                    "epoch_loss": avg_loss,
                    "epoch": epoch+1
                }, step=step_count)

                # Periodic validation
                if self.validation_interval > 0 and (epoch + 1) % self.validation_interval == 0:
                    self._run_validation(epoch, features, step_count, dataloader)

                # Checkpointing - save full checkpoint with optimizer state
                if (epoch + 1) % checkpoint_interval == 0 or epoch == epochs - 1:
                    ckpt_filename = f"checkpoint_epoch_{epoch+1}.pt"
                    ckpt_path = os.path.join(tmpdir, ckpt_filename)
                    self.save_checkpoint(
                        ckpt_path,
                        epoch=epoch + 1,
                        optimizer=self.optimizer,
                        step_count=step_count
                    )
                    self.tracker.log_artifact(ckpt_path, artifact_type='checkpoint')

        return self.model

    def _extract_batch(self, batch):
        """
        Extract data and optional mask from batch (handles TensorDataset tuples).

        Args:
            batch: Batch from dataloader. Can be:
                - Single tensor
                - Tuple of (data,)
                - Tuple of (data, mask) for masked training

        Returns:
            tuple: (data_tensor, mask_tensor or None) moved to device
        """
        if isinstance(batch, (list, tuple)):
            data = batch[0].to(self.device)
            if len(batch) > 1:
                # Batch contains (data, mask)
                mask = batch[1].to(self.device)
                return data, mask
            return data, None
        return batch.to(self.device), None

    def _get_conditional_labels(self, batch_size):
        """
        Generate conditional labels with classifier-free guidance.

        Args:
            batch_size: Size of the current batch

        Returns:
            Conditional labels tensor or None if not using conditioning
        """
        if self.num_classes == 0:
            return None

        # Generate random labels for the batch
        y = torch.randint(0, self.num_classes, (batch_size,), device=self.device).long()

        # Apply Classifier-Free Guidance by randomly dropping the condition
        if self.cfg_probability > 0:
            keep_condition = torch.rand(batch_size, device=self.device) > self.cfg_probability
            # Use num_classes as a special token for unconditional generation
            return torch.where(keep_condition, y, torch.full_like(y, self.num_classes)).long()

        return y

    def _run_validation(self, epoch, features, step_count, dataloader):
        """
        Run validation during training.

        Generates synthetic samples and compares stylized facts with real data.

        Args:
            epoch: Current epoch number
            features: Number of features in data
            step_count: Current step count
            dataloader: Training dataloader (used to get real data sample)
        """
        print(f"Running validation at epoch {epoch+1}...")
        self.model.eval()

        with torch.no_grad():
            # 1. Generate synthetic samples
            y_val_sampling = None
            if self.num_classes > 0:
                # For validation, generate random classes for samples
                y_val_sampling = torch.randint(
                    0, self.num_classes,
                    (self.num_validation_samples,),
                    device=self.device
                ).long()

            # Use model.generate() which handles DDPM/DDIM dispatch internally
            seq_len = self.config.get_data_config().sequence_length
            gen_seqs = self.model.generate(
                self.num_validation_samples,
                seq_len,
                device=self.device,
                y=y_val_sampling
            )
            gen_seqs_np = gen_seqs.cpu().numpy()  # Scaled returns

            # 2. Prepare real data for comparison
            # We need to get some real data. For simplicity, we'll take a batch from the dataloader.
            # In a real validation scenario, you'd use a dedicated validation set.
            batch = next(iter(dataloader))
            if isinstance(batch, (list, tuple)):
                real_batch_for_val = batch[0].cpu().numpy()
            else:
                real_batch_for_val = batch.cpu().numpy()
            limit = min(self.num_validation_samples, real_batch_for_val.shape[0])
            real_sample_scaled = real_batch_for_val[:limit]

            # 3. Calculate Stylized Facts
            sf_metrics = calculate_stylized_facts(real_sample_scaled, gen_seqs_np[:limit])

            # 4. Log validation metrics
            self.tracker.log_metrics({
                "val_kurtosis_diff_mean": np.mean(sf_metrics['kurtosis_diff']),
                "val_skew_diff_mean": np.mean(sf_metrics['skew_diff']),
                "val_acf_ret_diff_mse": sf_metrics['acf_ret_diff'],
                "val_acf_sq_ret_diff_mse": sf_metrics['acf_sq_ret_diff'],
                "val_corr_matrix_norm_diff": sf_metrics['corr_matrix_diff_norm'],
                "val_var_diff_mean": np.mean(sf_metrics['var_diff']),
                "val_es_diff_mean": np.mean(sf_metrics['es_diff'])
            }, step=step_count)
            print(
                f"Validation Metrics: "
                f"Kurtosis Diff Mean={np.mean(sf_metrics['kurtosis_diff']):.4f}, "
                f"ES Diff Mean={np.mean(sf_metrics['es_diff']):.4f}"
            )

        self.model.train()  # Set model back to training mode
