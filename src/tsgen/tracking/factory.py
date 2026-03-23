"""Factory for creating experiment trackers."""

from tsgen.tracking.base import ConsoleTracker, NoOpTracker, FileTracker
from tsgen.tracking.mlflow_tracker import MLFlowTracker


def create_tracker(config, experiment_dir=None):
    """
    Create tracker based on config.

    Args:
        config: ExperimentConfig or configuration dictionary
        experiment_dir: Optional experiment directory for file-based tracking

    Returns:
        ExperimentTracker instance (defaults to ConsoleTracker if not specified)
    """
    # Support both ExperimentConfig (attribute access) and dict
    is_dict = isinstance(config, dict)

    def _get(key, default=None):
        if is_dict:
            return config.get(key, default)
        return getattr(config, key, default)

    # 1. Look for 'tracker' at root
    tracker_conf = _get('tracker', {})

    output_type = 'console'  # DEFAULT
    output_dir = None

    if isinstance(tracker_conf, dict):
        output_type = tracker_conf.get('output_type', 'console').lower()
        output_dir = tracker_conf.get('output_dir')
    elif hasattr(tracker_conf, 'output_type'):
        # TrackerConfig pydantic model
        output_type = tracker_conf.output_type.lower()
        output_dir = tracker_conf.output_dir
    elif tracker_conf is not None and tracker_conf != {}:
        output_type = str(tracker_conf).lower()

    # Determine experiment name
    exp_conf = _get('experiment', {})
    if isinstance(exp_conf, dict):
        exp_name = exp_conf.get('name', _get('experiment_name', 'Default_Experiment'))
    elif hasattr(exp_conf, 'name'):
        exp_name = exp_conf.name or _get('experiment_name', 'Default_Experiment')
    else:
        exp_name = _get('experiment_name', 'Default_Experiment')

    if output_type == 'mlflow':
        return MLFlowTracker(
            experiment_name=exp_name,
            tracking_uri=_get('mlflow_tracking_uri'),
            artifact_location=_get('mlflow_artifact_location')
        )
    elif output_type == 'console':
        return ConsoleTracker()
    elif output_type == 'file':
        # Priority 1: Use experiment_dir parameter
        if experiment_dir:
            return FileTracker(log_file="training.log", experiment_dir=experiment_dir)

        # Priority 2: Use output_dir from tracker config
        if output_dir:
            return FileTracker(log_file="training.log", experiment_dir=output_dir)

        # Priority 3: Use experiment_dir from experiment section
        if isinstance(exp_conf, dict):
            exp_root_dir = exp_conf.get('experiment_dir')
        elif hasattr(exp_conf, 'experiment_dir'):
            exp_root_dir = exp_conf.experiment_dir
        else:
            exp_root_dir = None
        if exp_root_dir:
             return FileTracker(log_file="training.log", experiment_dir=exp_root_dir)

        # Priority 4: Legacy 'output_dir' in root
        root_output_dir = _get('output_dir')
        if root_output_dir:
             return FileTracker(log_file="training.log", experiment_dir=root_output_dir)

        # Fallback: Standalone log file
        log_file = _get('log_file', f"logs/{exp_name}.log")
        return FileTracker(log_file=log_file)

    elif output_type == 'noop':
        return NoOpTracker()
    else:
        print(f"Warning: Unknown tracker type '{output_type}'. Defaulting to Console.")
        return ConsoleTracker()
