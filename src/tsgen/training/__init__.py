"""
Training module with Trainer Registry Pattern.

Provides:
- BaseTrainer: Abstract base class for all trainers
- TrainerRegistry: Registry for mapping model types to trainers
- Concrete trainers: DiffusionTrainer, VAETrainer, BaselineTrainer
"""

from tsgen.training.base import BaseTrainer
from tsgen.training.registry import TrainerRegistry

# Import trainers to trigger registration
from tsgen.training.diffusion_trainer import DiffusionTrainer
from tsgen.training.vae_trainer import VAETrainer
from tsgen.training.baseline_trainer import BaselineTrainer

__all__ = [
    # Base infrastructure
    'BaseTrainer',
    'TrainerRegistry',

    # Concrete trainers
    'DiffusionTrainer',
    'VAETrainer',
    'BaselineTrainer',
]
