from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Literal

# Artifact types for organized storage
ArtifactType = Literal['model', 'plot', 'checkpoint', 'data', 'other']

class ExperimentTracker(ABC):
    """
    Abstract base class for experiment tracking.

    Provides interface for logging parameters, metrics, and artifacts.
    Artifacts are organized by type ('model', 'plot', 'checkpoint', 'data', 'other').
    """

    @abstractmethod
    def start_run(self, run_name: Optional[str] = None):
        pass

    @abstractmethod
    def log_params(self, params: Dict[str, Any]):
        pass

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        pass

    @abstractmethod
    def log_artifact(
        self,
        local_path: str,
        artifact_type: ArtifactType = 'other',
        artifact_name: Optional[str] = None
    ):
        """
        Log an artifact with type-based organization.

        Args:
            local_path: Path to the artifact file (can be temporary)
            artifact_type: Type of artifact ('model', 'plot', 'checkpoint', 'data', 'other')
            artifact_name: Optional custom name (defaults to basename of local_path)
        """
        pass

    @abstractmethod
    def end_run(self):
        pass

    # Helper methods (concrete implementations)
    def get_artifact_subdir(self, artifact_type: ArtifactType) -> str:
        """
        Get subdirectory for artifact type.

        Returns:
            Subdirectory name (e.g., 'models', 'plots', 'checkpoints')
        """
        mapping = {
            'model': 'models',
            'plot': 'plots',
            'checkpoint': 'checkpoints',
            'data': 'data',
            'other': ''
        }
        return mapping.get(artifact_type, '')

    def get_artifact_path(
        self,
        artifact_name: str,
        artifact_type: ArtifactType
    ) -> Optional[str]:
        """
        Get path to previously logged artifact (for loading in evaluation).

        Returns None if tracker doesn't support artifact retrieval.
        Subclasses can override to provide artifact loading.

        Args:
            artifact_name: Name of the artifact file
            artifact_type: Type of artifact

        Returns:
            Path to artifact file, or None if not available
        """
        return None

class NoOpTracker(ExperimentTracker):
    """
    Does nothing. Useful for testing or when tracking is disabled.
    """
    def start_run(self, run_name: Optional[str] = None):
        pass
    def log_params(self, params: Dict[str, Any]):
        pass
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        pass
    def log_artifact(
        self,
        local_path: str,
        artifact_type: ArtifactType = 'other',
        artifact_name: Optional[str] = None
    ):
        pass
    def end_run(self):
        pass

class ConsoleTracker(ExperimentTracker):
    """
    Prints tracking information to the console.
    """
    def start_run(self, run_name: Optional[str] = None):
        print(f"\n[Tracker] Starting run: {run_name}")

    def log_params(self, params: Dict[str, Any]):
        print(f"[Tracker] Params: {params}")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        step_str = f" | Step: {step}" if step is not None else ""
        print(f"[Tracker] Metrics{step_str}: {metrics}")

    def log_artifact(
        self,
        local_path: str,
        artifact_type: ArtifactType = 'other',
        artifact_name: Optional[str] = None
    ):
        artifact_name_str = artifact_name if artifact_name else "auto"
        print(f"[Tracker] Saving artifact: {local_path} (type={artifact_type}, name={artifact_name_str})")

    def end_run(self):
        print("[Tracker] Run ended.")

class FileTracker(ExperimentTracker):
    """
    Writes tracking information to log files.
    Ideal for long-running experiments where you want logs saved to file
    while keeping tqdm progress bars clean in the console.

    Metrics are written to metrics.jsonl in JSON Lines format for easy parsing.
    Other logs (params, artifacts) are written to training.log in human-readable format.

    When used with ExperimentManager, logs are saved to the experiment folder.
    """
    def __init__(self, log_file: str = "training.log", experiment_dir: Optional[str] = None):
        """
        Initialize FileTracker.

        Args:
            log_file: Log filename or path for general logs
            experiment_dir: Optional experiment directory. If provided, files are saved relative to this dir.
        """
        import os

        if experiment_dir:
            # Files are in the experiment directory
            self.log_file = os.path.join(experiment_dir, log_file)
            self.metrics_file = os.path.join(experiment_dir, "metrics.jsonl")
            self.experiment_dir = experiment_dir
        else:
            # Standalone log files
            self.log_file = log_file
            # Put metrics.jsonl in same directory as log_file
            log_dir = os.path.dirname(log_file) if os.path.dirname(log_file) else "."
            self.metrics_file = os.path.join(log_dir, "metrics.jsonl")
            self.experiment_dir = None

        # Create directories
        os.makedirs(os.path.dirname(self.log_file) if os.path.dirname(self.log_file) else ".", exist_ok=True)
        os.makedirs(os.path.dirname(self.metrics_file) if os.path.dirname(self.metrics_file) else ".", exist_ok=True)

        # Create/clear the log file
        with open(self.log_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("Experiment Log\n")
            f.write("="*80 + "\n\n")

        # Create/clear the metrics file
        with open(self.metrics_file, 'w') as f:
            pass  # Empty file, will be written in JSON Lines format

    def _write(self, message: str):
        """Write a message to the log file."""
        with open(self.log_file, 'a') as f:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}] {message}\n")

    def start_run(self, run_name: Optional[str] = None):
        self._write(f"Starting run: {run_name}")

    def log_params(self, params: Dict[str, Any]):
        self._write(f"Params: {params}")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        import json
        import numpy as np

        # Write to human-readable log
        step_str = f" | Step: {step}" if step is not None else ""
        self._write(f"Metrics{step_str}: {metrics}")

        # Write to metrics.jsonl in JSON Lines format
        # Convert numpy types to Python native types for JSON serialization
        metric_entry = {}
        for key, value in metrics.items():
            if isinstance(value, (np.floating, np.float32, np.float64)):
                metric_entry[key] = float(value)
            elif isinstance(value, (np.integer, np.int32, np.int64)):
                metric_entry[key] = int(value)
            elif isinstance(value, np.ndarray):
                metric_entry[key] = value.tolist()
            else:
                metric_entry[key] = value

        if step is not None:
            metric_entry['step'] = int(step) if isinstance(step, (np.integer, np.int32, np.int64)) else step

        with open(self.metrics_file, 'a') as f:
            json.dump(metric_entry, f)
            f.write('\n')

    def log_artifact(
        self,
        local_path: str,
        artifact_type: ArtifactType = 'other',
        artifact_name: Optional[str] = None
    ):
        """
        Log artifact with type-based organization.

        Organizes artifacts into typed subdirectories:
            artifacts/models/       # artifact_type='model'
            artifacts/plots/        # artifact_type='plot'
            artifacts/checkpoints/  # artifact_type='checkpoint'
            artifacts/data/         # artifact_type='data'
            artifacts/              # artifact_type='other' (no subdir)

        Args:
            local_path: Path to the artifact file (can be temporary)
            artifact_type: Type of artifact ('model', 'plot', 'checkpoint', 'data', 'other')
            artifact_name: Optional custom name (defaults to basename of local_path)
        """
        import shutil
        import os

        # Determine artifact name
        if artifact_name is None:
            artifact_name = os.path.basename(local_path)

        self._write(f"Logging artifact: {artifact_name} (type={artifact_type}, source={local_path})")

        # Determine base artifacts directory
        if self.experiment_dir:
            artifacts_base = os.path.join(self.experiment_dir, "artifacts")
        else:
            artifacts_base = "artifacts"

        # Get type-specific subdirectory
        subdir = self.get_artifact_subdir(artifact_type)

        # Build destination path
        if subdir:
            dest_dir = os.path.join(artifacts_base, subdir)
            dest_path = os.path.join(dest_dir, artifact_name)
        else:
            dest_path = os.path.join(artifacts_base, artifact_name)

        # Create subdirectory if needed
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        # Copy file if it exists
        if os.path.exists(local_path):
            shutil.copy2(local_path, dest_path)
            self._write(f"Artifact saved to: {dest_path}")
        else:
            self._write(f"Warning: Artifact file not found: {local_path}")

    def get_artifact_path(
        self,
        artifact_name: str,
        artifact_type: ArtifactType
    ) -> Optional[str]:
        """
        Get path to previously logged artifact (for loading in evaluation).

        Args:
            artifact_name: Name of the artifact file
            artifact_type: Type of artifact

        Returns:
            Path to artifact file if it exists, None otherwise
        """
        import os

        # Determine base artifacts directory
        if self.experiment_dir:
            artifacts_base = os.path.join(self.experiment_dir, "artifacts")
        else:
            artifacts_base = "artifacts"

        # Get type-specific subdirectory
        subdir = self.get_artifact_subdir(artifact_type)

        # Build path
        if subdir:
            artifact_path = os.path.join(artifacts_base, subdir, artifact_name)
        else:
            artifact_path = os.path.join(artifacts_base, artifact_name)

        # Return path if file exists
        if os.path.exists(artifact_path):
            return artifact_path
        return None

    def end_run(self):
        self._write("Run ended.")
        self._write("="*80 + "\n")
