"""
Experiment tracking utilities.

Provides different tracking backends:
- MLflow: Full-featured experiment tracking
- File: JSON Lines metrics logging with human-readable logs
- Console: Simple console logging
- NoOp: Disabled tracking (for testing)
"""

from tsgen.tracking.base import ExperimentTracker, NoOpTracker, ConsoleTracker, FileTracker, ArtifactType
from tsgen.tracking.factory import create_tracker


def __getattr__(name):
    if name == "MLFlowTracker":
        from tsgen.tracking.mlflow_tracker import MLFlowTracker
        return MLFlowTracker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Base classes
    "ExperimentTracker",
    "NoOpTracker",
    "ConsoleTracker",
    "FileTracker",
    # Implementations
    "MLFlowTracker",
    # Factory
    "create_tracker",
    # Types
    "ArtifactType",
]
