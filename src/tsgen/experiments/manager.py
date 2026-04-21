"""
Experiment management utilities for structured experiment tracking.

Each experiment gets a dedicated folder with:
- Unique 4-digit experiment number
- Descriptive name (2-3 lowercase words with underscores)
- README.md with description and setup
- Multiple runs with different configurations
- Each run has its own subdirectory with outputs

Folder structure:
    experiments/
    ├── 0001_timevae_all_stocks/
    │   ├── README.md
    │   ├── config_baseline.yaml
    │   ├── config_low_freebits.yaml
    │   ├── run_baseline/
    │   │   ├── metrics.jsonl
    │   │   ├── model_final.pt
    │   │   ├── processor.pkl
    │   │   ├── plots/
    │   │   └── checkpoints/
    │   └── run_low_freebits/
    │       ├── metrics.jsonl
    │       ├── model_final.pt
    │       └── ...
"""

import os
import yaml
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional


class ExperimentManager:
    """
    Manages experiment folder creation and organization.

    Folder structure:
        experiments/
        ├── 0001_multivariate_baseline/
        │   ├── README.md
        │   ├── config.yaml
        │   ├── training.log
        │   ├── results.md
        │   ├── plots/
        │   └── artifacts/
        ├── 0002_timevae_496_features/
        │   └── ...
    """

    def __init__(self, base_dir: str = "experiments"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)

    def get_next_experiment_number(self) -> int:
        """Get the next available experiment number."""
        existing = [d.name for d in self.base_dir.iterdir() if d.is_dir()]
        if not existing:
            return 1

        # Extract numbers from folder names like "0001_name" -> 1
        numbers = []
        for dirname in existing:
            try:
                num_str = dirname.split('_')[0]
                numbers.append(int(num_str))
            except (ValueError, IndexError):
                continue

        return max(numbers) + 1 if numbers else 1

    def create_experiment(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None,
        description: str = "",
        experiment_number: Optional[int] = None
    ) -> Path:
        """
        Create a new experiment folder with all necessary structure.

        Args:
            name: Short experiment name (2-3 words with underscores)
            config: Optional experiment configuration dictionary (for single-model experiments)
            description: Detailed experiment description
            experiment_number: Optional experiment number (auto-assigned if None)

        Returns:
            Path to created experiment folder
        """
        # Get experiment number
        if experiment_number is None:
            experiment_number = self.get_next_experiment_number()

        # Create folder name: 0001_experiment_name
        folder_name = f"{experiment_number:04d}_{name}"
        exp_dir = self.base_dir / folder_name

        # Create directory structure
        exp_dir.mkdir(exist_ok=True)
        (exp_dir / "plots").mkdir(exist_ok=True)
        (exp_dir / "artifacts").mkdir(exist_ok=True)

        # Save config if provided (for single-model experiments)
        if config:
            config_path = exp_dir / "config.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        # Create README.md
        readme_path = exp_dir / "README.md"
        self._create_readme(readme_path, experiment_number, name, config, description)

        # Create results.md template
        results_path = exp_dir / "results.md"
        self._create_results_template(results_path, experiment_number, name)

        print(f"Created experiment folder: {exp_dir}")
        return exp_dir

    def add_model_config(self, experiment_path: Path, model_name: str, config: Dict[str, Any]):
        """
        Add a model configuration to an existing experiment.

        This allows multiple models to be compared within a single experiment.
        Each model gets its own config file, logs, and artifact subdirectories.

        Args:
            experiment_path: Path to experiment folder
            model_name: Short name for the model (e.g., 'baseline', 'timevae')
            config: Model configuration dictionary
        """
        # Save config to config_{model_name}.yaml
        config_path = experiment_path / f"config_{model_name}.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        # Create model-specific subdirectories
        (experiment_path / "plots" / model_name).mkdir(parents=True, exist_ok=True)
        (experiment_path / "artifacts" / model_name).mkdir(parents=True, exist_ok=True)

        print(f"Added {model_name} model config to experiment: {experiment_path}")

    def create_run(
        self,
        experiment_path: Path,
        run_name: str,
        config: Optional[Dict[str, Any]] = None,
        auto_increment: bool = True
    ) -> Path:
        """
        Create a run directory within an experiment for a specific configuration.

        This allows multiple parameter configurations to be tested within the same
        experiment without overwriting previous results.

        Args:
            experiment_path: Path to experiment folder
            run_name: Name for this run (e.g., 'baseline', 'low_freebits', 'run_001')
            config: Optional configuration dictionary to save
            auto_increment: If True and run exists, auto-increment (run_001 -> run_002)

        Returns:
            Path to created run directory

        Example:
            manager = ExperimentManager()
            exp_path = Path('experiments/0001_timevae_all_stocks')
            run_path = manager.create_run(exp_path, 'baseline', config)
            # Creates: experiments/0001_timevae_all_stocks/run_baseline/
        """
        # Handle auto-incrementing if run already exists
        if auto_increment:
            base_run_name = run_name
            counter = 1
            run_dir = experiment_path / f"run_{run_name}"

            while run_dir.exists():
                run_name = f"{base_run_name}_{counter:03d}"
                run_dir = experiment_path / f"run_{run_name}"
                counter += 1
        else:
            run_dir = experiment_path / f"run_{run_name}"

        # Create run directory structure
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "plots").mkdir(exist_ok=True)
        (run_dir / "checkpoints").mkdir(exist_ok=True)

        # Save config to the run directory if provided
        if config:
            # Save to both experiment root (for reference) and run directory
            exp_config_path = experiment_path / f"config_{run_name}.yaml"
            run_config_path = run_dir / "config.yaml"

            # Update config to output to this run directory
            config_copy = config.copy()
            config_copy['output_dir'] = str(run_dir)
            config_copy['run_name'] = run_name

            with open(exp_config_path, 'w') as f:
                yaml.dump(config_copy, f, default_flow_style=False, sort_keys=False)
            with open(run_config_path, 'w') as f:
                yaml.dump(config_copy, f, default_flow_style=False, sort_keys=False)

        # Create a run info file
        run_info_path = run_dir / "run_info.txt"
        with open(run_info_path, 'w') as f:
            f.write(f"Run: {run_name}\n")
            f.write(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Experiment: {experiment_path.name}\n")
            if config:
                f.write(f"\nKey Configuration:\n")
                important_keys = ['vae_beta', 'vae_free_bits', 'vae_annealing_epochs',
                                'learning_rate', 'epochs', 'batch_size']
                for key in important_keys:
                    if key in config:
                        f.write(f"  {key}: {config[key]}\n")

        print(f"Created run: {run_dir}")
        return run_dir

    def get_next_run_number(self, experiment_path: Path) -> int:
        """Get the next available run number in an experiment."""
        run_dirs = [d for d in experiment_path.iterdir()
                   if d.is_dir() and d.name.startswith('run_')]

        if not run_dirs:
            return 1

        # Extract numbers from run_001, run_002, etc.
        numbers = []
        for run_dir in run_dirs:
            try:
                # Extract number from run_NAME_001 or run_001
                parts = run_dir.name.split('_')
                # Try last part first (run_baseline_001 -> 001)
                if parts[-1].isdigit():
                    numbers.append(int(parts[-1]))
                # Try second part (run_001 -> 001)
                elif len(parts) > 1 and parts[1].isdigit():
                    numbers.append(int(parts[1]))
            except (ValueError, IndexError):
                continue

        return max(numbers) + 1 if numbers else 1

    def _create_readme(
        self,
        path: Path,
        exp_num: int,
        name: str,
        config: Optional[Dict[str, Any]],
        description: str
    ):
        """Create README.md for the experiment."""
        # For multi-model experiments, config is None
        if config is None:
            readme_content = f"""# Experiment {exp_num:04d}: {name.replace('_', ' ').title()}

**Created**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview

{description if description else 'Model comparison experiment. Add model configurations using `add_model_config()` or run with `--model-name` parameter.'}

## Models

_Add models to this experiment using:_
```bash
python main.py --experiment-number {exp_num} --model-name <model_name> --config <config_path> --mode train_eval
```

Models will be listed here automatically as they are added.

## Files

- `config_<model>.yaml` - Configuration for each model
- `training_<model>.log` - Training logs per model
- `results.md` - Comparison results and analysis
- `plots/<model>/` - Visualizations per model
- `artifacts/<model>/` - Model checkpoints per model

## Status

- [ ] All models trained
- [ ] Results compared
- [ ] Analysis documented

## Notes

_Add any observations, issues, or insights here during the experiment._

"""
        else:
            # Single-model experiment README
            model_type = config.get('model_type', 'unknown')
            data_info = f"{len(config.get('tickers', []))} tickers" if 'tickers' in config else "N/A"
            date_range = f"{config.get('start_date', 'N/A')} to {config.get('end_date', 'N/A')}"

            readme_content = f"""# Experiment {exp_num:04d}: {name.replace('_', ' ').title()}

**Created**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview

{description if description else 'No description provided.'}

## Configuration

- **Model**: {model_type}
- **Data**: {data_info}
- **Date Range**: {date_range}
- **Sequence Length**: {config.get('sequence_length', 'N/A')}
- **Batch Size**: {config.get('batch_size', 'N/A')}
- **Epochs**: {config.get('epochs', 'N/A')}

## Model-Specific Parameters

"""

            # Add model-specific parameters
            model_params = {
                'timevae': ['hidden_dim', 'latent_dim', 'encoder_type', 'vae_beta', 'vae_use_annealing'],
                'multivariate_gaussian': ['full_covariance'],
                'bootstrap': [],
                'ccc_garch': ['p', 'q', 'distribution'],
                'unet': ['hidden_channels', 'timesteps'],
                'transformer': ['d_model', 'nhead', 'num_layers', 'timesteps'],
                'dit': ['dim', 'depth', 'heads', 'mlp_ratio', 'timesteps'],
                'diffwave': ['residual_channels', 'num_blocks', 'dilation_cycle_length', 'timesteps'],
                'mamba': ['dim', 'depth', 'd_state', 'timesteps'],
            }

            params_to_show = model_params.get(model_type, [])
            if params_to_show:
                for param in params_to_show:
                    if param in config:
                        readme_content += f"- **{param}**: {config[param]}\n"
            else:
                readme_content += "_No additional model parameters_\n"

            readme_content += f"""
## Files

- `config.yaml` - Complete experiment configuration
- `training.log` - Training logs and metrics
- `results.md` - Results summary and analysis
- `plots/` - Generated visualizations
- `artifacts/` - Model checkpoints and saved objects

## Running This Experiment

```bash
conda activate tsgen
python main.py --config {path.parent / 'config.yaml'} --mode train_eval
```

## Status

- [ ] Training started
- [ ] Training completed
- [ ] Evaluation completed
- [ ] Results analyzed
- [ ] Documented

## Notes

_Add any observations, issues, or insights here during the experiment._

"""

        with open(path, 'w') as f:
            f.write(readme_content)

    def _create_results_template(self, path: Path, exp_num: int, name: str):
        """Create results.md template."""
        results_content = f"""# Results: Experiment {exp_num:04d}

**Experiment**: {name.replace('_', ' ').title()}
**Date Completed**: _Not yet completed_

## Summary

_Brief summary of results (2-3 sentences)_

## Training Metrics

### Final Training Loss
- **Total Loss**:
- **Reconstruction Loss**:
- **KL Divergence** (if VAE):

### Training Time
- **Total Duration**:
- **Time per Epoch**:

## Evaluation Metrics

### Correlation Structure
- **Frobenius Norm**:
- **Max Correlation Difference**:
- **Mean Correlation Difference**:
- **Eigenvalue MSE**:

### Stylized Facts
- **Kurtosis**:
  - Real:
  - Synthetic:
- **Skewness**:
  - Real:
  - Synthetic:
- **Volatility Clustering (ACF)**:

### Distribution Tests
- **Kolmogorov-Smirnov**:
- **Cramér-von Mises**:
- **Anderson-Darling**:

### Model Quality
- **Discriminator Accuracy**: (target: 0.5)
- **TSTR MSE**:

## Visualizations

_List and describe key plots generated_

- `plots/correlation_matrix_real.png` -
- `plots/correlation_matrix_synthetic.png` -
- `plots/eigenvalue_spectrum.png` -
- `plots/sample_paths.png` -

## Key Findings

1.
2.
3.

## Issues Encountered

_List any problems, errors, or unexpected behavior_

## Conclusions

_What did we learn from this experiment?_

## Next Steps

_Suggested follow-up experiments or improvements_

---

**Analysis Completed**: _Date_
**Reviewed By**: _Name_
"""

        with open(path, 'w') as f:
            f.write(results_content)

    def get_experiment_path(self, experiment_id: str) -> Optional[Path]:
        """
        Get path to experiment folder by number or name.

        Args:
            experiment_id: Either "0001" or "0001_name" or "name"

        Returns:
            Path to experiment folder or None if not found
        """
        # Try exact match first
        exp_dir = self.base_dir / experiment_id
        if exp_dir.exists():
            return exp_dir

        # Try finding by number
        if experiment_id.isdigit():
            for exp_dir in self.base_dir.iterdir():
                if exp_dir.is_dir() and exp_dir.name.startswith(experiment_id.zfill(4)):
                    return exp_dir

        # Try finding by name
        for exp_dir in self.base_dir.iterdir():
            if exp_dir.is_dir() and experiment_id in exp_dir.name:
                return exp_dir

        return None

    def list_experiments(self) -> list:
        """List all experiments with their basic info."""
        experiments = []
        for exp_dir in sorted(self.base_dir.iterdir()):
            if not exp_dir.is_dir():
                continue

            # Parse folder name
            parts = exp_dir.name.split('_', 1)
            if len(parts) != 2:
                continue

            exp_num = parts[0]
            exp_name = parts[1]

            # Read config
            config_path = exp_dir / "config.yaml"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                model_type = config.get('model_type', 'unknown')
            else:
                model_type = 'unknown'

            # Check completion status
            results_path = exp_dir / "results.md"
            completed = False
            if results_path.exists():
                with open(results_path, 'r') as f:
                    content = f.read()
                    completed = "Date Completed" in content and "_Not yet completed_" not in content

            experiments.append({
                'number': exp_num,
                'name': exp_name,
                'path': exp_dir,
                'model': model_type,
                'completed': completed
            })

        return experiments


def create_experiment_from_config(config_path: str, description: str = "") -> Path:
    """
    Convenience function to create experiment from config file.

    Args:
        config_path: Path to YAML config file
        description: Experiment description

    Returns:
        Path to created experiment folder
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Generate experiment name from config
    exp_name = config.get('experiment_name', 'unnamed_experiment')
    # Convert to lowercase with underscores, limit to 2-3 words
    exp_name = exp_name.lower().replace('-', '_').replace(' ', '_')
    parts = exp_name.split('_')
    if len(parts) > 3:
        exp_name = '_'.join(parts[:3])

    manager = ExperimentManager()
    return manager.create_experiment(exp_name, config, description)
