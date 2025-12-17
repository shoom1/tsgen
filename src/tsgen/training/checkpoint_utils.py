"""Utilities for checkpoint management and resumption."""

import os
import re
from pathlib import Path
from typing import Optional, List, Tuple


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    Find the latest checkpoint file in a directory.

    Args:
        checkpoint_dir: Directory containing checkpoint files

    Returns:
        Path to the latest checkpoint file, or None if no checkpoints found
    """
    if not os.path.exists(checkpoint_dir):
        return None

    # Look for checkpoint files matching pattern: checkpoint_epoch_*.pt
    checkpoint_pattern = re.compile(r'checkpoint_epoch_(\d+)\.pt')
    checkpoints = []

    for filename in os.listdir(checkpoint_dir):
        match = checkpoint_pattern.match(filename)
        if match:
            epoch = int(match.group(1))
            filepath = os.path.join(checkpoint_dir, filename)
            checkpoints.append((epoch, filepath))

    if not checkpoints:
        return None

    # Return checkpoint with highest epoch number
    checkpoints.sort(key=lambda x: x[0], reverse=True)
    return checkpoints[0][1]


def list_checkpoints(checkpoint_dir: str) -> List[Tuple[int, str]]:
    """
    List all checkpoints in a directory.

    Args:
        checkpoint_dir: Directory containing checkpoint files

    Returns:
        List of (epoch, filepath) tuples, sorted by epoch descending
    """
    if not os.path.exists(checkpoint_dir):
        return []

    checkpoint_pattern = re.compile(r'checkpoint_epoch_(\d+)\.pt')
    checkpoints = []

    for filename in os.listdir(checkpoint_dir):
        match = checkpoint_pattern.match(filename)
        if match:
            epoch = int(match.group(1))
            filepath = os.path.join(checkpoint_dir, filename)
            checkpoints.append((epoch, filepath))

    checkpoints.sort(key=lambda x: x[0], reverse=True)
    return checkpoints


def get_checkpoint_path(experiment_dir: str, epoch: Optional[int] = None) -> Optional[str]:
    """
    Get checkpoint path for a specific epoch or latest.

    Args:
        experiment_dir: Experiment directory
        epoch: Specific epoch number, or None for latest

    Returns:
        Path to checkpoint file, or None if not found
    """
    checkpoint_dir = os.path.join(experiment_dir, 'artifacts', 'checkpoints')

    if epoch is None:
        # Get latest checkpoint
        return find_latest_checkpoint(checkpoint_dir)
    else:
        # Get specific epoch checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
        return checkpoint_path if os.path.exists(checkpoint_path) else None


def extract_epoch_from_checkpoint(checkpoint_path: str) -> Optional[int]:
    """
    Extract epoch number from checkpoint filename.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Epoch number, or None if pattern doesn't match
    """
    filename = os.path.basename(checkpoint_path)
    match = re.match(r'checkpoint_epoch_(\d+)\.pt', filename)
    return int(match.group(1)) if match else None
