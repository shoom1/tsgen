import mlflow
from typing import Any, Dict, Optional
from .base import ExperimentTracker, ArtifactType

class MLFlowTracker(ExperimentTracker):
    """
    Experiment tracker using MLflow with typed artifact organization.

    Provides production-ready experiment tracking with:
    - Cloud storage support (S3, Azure, GCS)
    - Web UI for browsing runs and artifacts
    - Typed artifact paths for organization
    """
    def __init__(
        self,
        experiment_name: str = "Default",
        tracking_uri: Optional[str] = None,
        artifact_location: Optional[str] = None
    ):
        """
        Initialize MLflow tracker.

        Args:
            experiment_name: MLflow experiment name
            tracking_uri: Optional tracking URI (defaults to ./mlruns)
            artifact_location: Optional base artifact location (local path or cloud URI)
        """
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        # Set or create experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(
                experiment_name,
                artifact_location=artifact_location
            )
        else:
            experiment_id = experiment.experiment_id

        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name
        self.experiment_id = experiment_id

    def start_run(self, run_name: Optional[str] = None):
        mlflow.start_run(run_name=run_name)

    def log_params(self, params: Dict[str, Any]):
        mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(
        self,
        local_path: str,
        artifact_type: ArtifactType = 'other',
        artifact_name: Optional[str] = None
    ):
        """
        Log artifact to MLflow with type-based organization.

        MLflow structure:
            mlruns/
            └── experiment_id/
                └── run_id/
                    └── artifacts/
                        ├── models/
                        ├── plots/
                        ├── checkpoints/
                        └── data/

        Args:
            local_path: Path to the artifact file
            artifact_type: Type of artifact ('model', 'plot', 'checkpoint', 'data', 'other')
            artifact_name: Optional custom name (not used, MLflow uses original filename)
        """
        # Get type-specific subdirectory
        subdir = self.get_artifact_subdir(artifact_type)

        # Log to MLflow with artifact_path parameter
        # This creates the typed subdirectory structure in MLflow
        mlflow.log_artifact(local_path, artifact_path=subdir if subdir else None)

    def end_run(self):
        mlflow.end_run()
