# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2024-12-12

### Changed - Breaking

- **Artifact Management**: Refactored to Tracker-Managed Storage Pattern
  - `get_artifact_path()` and `get_checkpoint_dir()` utilities removed from training module
  - Training/evaluation now save to temporary files; trackers manage final storage
  - Artifacts organized in typed subdirectories: `artifacts/models/`, `artifacts/plots/`, `artifacts/checkpoints/`, `artifacts/data/`
  - `training/utils.py` module deleted (deprecated path utilities removed)

### Added

- **Enhanced Tracker API**:
  - `log_artifact()` now accepts `artifact_type` parameter (`'model'`, `'plot'`, `'checkpoint'`, `'data'`, `'other'`)
  - `get_artifact_path()` method for retrieving logged artifacts (FileTracker implementation)
  - Automatic typed subdirectory organization
- **MLflow Configuration**:
  - Support for `mlflow_tracking_uri` config parameter
  - Support for `mlflow_artifact_location` config parameter
  - Improved MLflow defaults for production use
- **Deprecation Warnings**:
  - `output_dir` config key now triggers DeprecationWarning with migration guidance

### Improved

- **Code Quality**:
  - Eliminated artifact file duplication (files now in ONE location only)
  - Removed 58+ lines of deprecated path utility code
  - Simplified training code with tempfile pattern
  - Cleaner separation: training produces, tracker manages storage
- **Organization**:
  - Typed artifact subdirectories for better organization
  - Easy to find artifacts by type
  - Matches MLflow best practices
- **Extensibility**:
  - Easy to add new artifact types
  - Simple to implement new tracker backends

### Migration Guide

**For Config Users:**

Old (deprecated but still works):
```yaml
tracker: file
output_dir: 'experiments/0001_name'
```

New (recommended):
```yaml
tracker: file
experiment_dir: 'experiments/0001_name'
# Or use --experiment-number CLI flag
```

**For MLflow Users:**

```yaml
tracker: mlflow
experiment_name: 'my_experiment'
mlflow_tracking_uri: 'http://localhost:5000'  # Optional
mlflow_artifact_location: 's3://my-bucket/mlartifacts'  # Optional
```

**Artifact Locations Changed:**

- Models: `artifacts/models/model_final.pt` (was: `artifacts/model_name/model_final.pt`)
- Plots: `artifacts/plots/stylized_facts.png` (was: `plots/model_name/stylized_facts.png`)
- Checkpoints: `artifacts/checkpoints/model_epoch_10.pt` (was: `artifacts/model_name/checkpoints/...`)
- Data: `artifacts/data/processor.pkl` (was: `artifacts/processor.pkl`)

**For Internal API Users:**

If you imported path utilities (unlikely):
```python
# Old (removed):
from tsgen.training.utils import get_artifact_path, get_checkpoint_dir

# New (use tracker):
tracker.log_artifact(temp_path, artifact_type='model')
model_path = tracker.get_artifact_path('model_final.pt', artifact_type='model')
```

### Testing

- All 240+ tests passing
- 6 new integration tests for artifact organization
- New deprecation warning test
- End-to-end tests verify no duplication

### Benefits

1. **No Duplication**: Files exist in ONE location (tracker-managed)
2. **Better Organization**: Typed subdirectories make artifacts easy to find
3. **Simpler Code**: Training code has no path logic
4. **MLflow Ready**: Production experiment tracking with cloud storage support
5. **Backward Compatible**: Old configs work with deprecation warnings

## [0.2.0] - 2024-12-12

### Changed - Breaking

- **Training Architecture**: Refactored to Trainer Registry Pattern
  - `train_vae()` function removed - use `VAETrainer` class instead
  - Training dispatch now uses `TrainerRegistry` instead of if/elif conditionals
  - Path utilities moved to `tsgen.training.utils`

### Added

- **Trainer Classes** implementing Strategy Pattern:
  - `BaseTrainer`: Abstract base class for all trainers
  - `DiffusionTrainer`: For UNet and Transformer models
  - `VAETrainer`: For TimeVAE model
  - `BaselineTrainer`: For GBM, Bootstrap, and Multivariate LogNormal
- **Trainer Registry**:
  - `TrainerRegistry`: Maps model types to trainer classes
  - Decorator-based registration: `@TrainerRegistry.register('unet')`
  - Factory method: `get_trainer()` creates appropriate trainer instance
- **Shared Utilities**:
  - `tsgen.training.utils.get_checkpoint_dir()` - unified checkpoint path management
  - `tsgen.training.utils.get_artifact_path()` - unified artifact path management

### Improved

- **Code Quality**:
  - Eliminated 120+ lines of duplicate code
  - Reduced train.py from 309 lines to 108 lines (65% reduction)
  - Cleaner separation of concerns with Strategy Pattern
  - Easier to add new model types (just create class + register)
  - Better testability with isolated trainer classes
- **Architecture**:
  - Strategy Pattern: Each trainer encapsulates a training algorithm
  - Registry Pattern: Declarative model type → trainer mapping
  - Factory Pattern: Centralized trainer instantiation

### Migration Guide

**For internal API users (if you imported `train_vae` directly):**

Before (v0.1.0):
```python
from tsgen.training.vae_trainer import train_vae
model = train_vae(model, dataloader, config, tracker, device)
```

After (v0.2.0):
```python
from tsgen.training import VAETrainer
trainer = VAETrainer(model, config, tracker, device)
model = trainer.train(dataloader)
```

**For CLI users:**
No changes needed - `train_model()` API is fully backward compatible.

### Testing

- All 223 tests passing (27 VAE tests updated, 8 new registry tests added)
- Test coverage maintained at same level
- No regression in training functionality

## [0.1.0] - 2024-12-12

### Added

#### Models
- UNet1D diffusion model for time series generation
- Diffusion Transformer architecture
- TimeVAE model for variational time series generation
- Classical baselines:
  - Geometric Brownian Motion (GBM)
  - Bootstrap resampling
  - Multivariate LogNormal model (with correlation structure)

#### Data Pipeline
- Composable data pipeline functions:
  - `load_prices()`: Load data from findata database
  - `clean_data()`: Handle missing values with multiple strategies
  - `split_temporal()`: Chronological train/test splitting
  - `process_prices()`: LogReturnProcessor for price transformations
  - `create_windows()`: Sliding window generation
  - `create_dataloader()`: PyTorch DataLoader creation
- LogReturnProcessor: Converts prices → log-returns → z-scores
- Temporal train/test splitting to prevent data leakage

#### Evaluation
- Stylized facts metrics:
  - Kurtosis and skewness
  - Volatility clustering (ACF of squared returns)
  - Autocorrelation analysis
- Discriminator accuracy (LSTM-based real vs. fake classifier)
- Distribution tests:
  - Kolmogorov-Smirnov test
  - Cramér-von Mises test
  - Anderson-Darling test
- TSTR (Train on Synthetic, Test on Real) evaluation
- Correlation structure metrics:
  - Correlation matrix comparison (Frobenius norm)
  - Eigenvalue spectrum comparison
  - Cross-asset correlation preservation

#### CLI Commands
- `tsgen`: Main training and evaluation command
- `tsgen-experiments`: Experiment management (list, info, create, open)
- `tsgen-backtest`: Rolling window backtesting

#### Experiment Tracking
- File-based tracker (JSON logs)
- Console tracker (stdout)
- MLflow tracker (optional)
- NoOp tracker (for testing)

#### Other
- Comprehensive test suite (18 test files, 100+ tests)
- Experiment folder structure with configs
- Diffusion utilities (forward/reverse process, DDPM/DDIM sampling)
- Model checkpointing and loading
- Reproducible random seeds

### Dependencies
- Python 3.12+
- PyTorch 2.0+
- findata package (separate repository for database management)

### Notes
- Initial release for research and experimentation
- Requires findata package for historical market data access
- All experiment data is gitignored by default
