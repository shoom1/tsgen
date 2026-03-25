# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-03-24

### Changed - Breaking

- **Composable Evaluation Pipeline**: Replaced monolithic `evaluate_model()` with pluggable `EvaluationPipeline` + `MetricEvaluator` classes
  - Old: Single 300-line function with all metrics hardcoded inline (discriminator, stylized facts, correlation, distribution tests, TSTR, plotting all interleaved)
  - New: `EvaluationPipeline` orchestrates independent `MetricEvaluator` subclasses (`StylizedFactsEvaluator`, `CorrelationEvaluator`, `DistributionTestEvaluator`, `DiscriminatorEvaluator`, `TSTREvaluator`)
  - Evaluators are configurable per-experiment via `EvaluationPipeline.from_config()` or manually composed
  - `EvaluationResult` container with `summary()` and `generate_plots()` for clean output
  - New modules: `tsgen/evaluation/pipeline.py`, `tsgen/evaluation/evaluators.py`

- **Typed Configuration System**: Replaced raw dict configs with Pydantic-validated `ExperimentConfig`
  - All config access through typed accessors: `config.get_model_config()`, `config.get_training_config()`, `config.get_data_config()`, `config.get_evaluation_config()`
  - Per-model configs: `UNetConfig`, `TransformerConfig`, `MambaConfig`, `TimeVAEConfig`, `BaselineModelConfig`
  - Per-paradigm training configs: `DiffusionTrainingConfig`, `VAETrainingConfig`, `BaselineTrainingConfig`
  - Config validation on load catches typos and invalid values with clear error messages

- **Removed deprecated flat config fields**: Old flat keys (`epochs`, `learning_rate`, `timesteps`, etc.) at root level no longer supported — must use nested sections (`training:`, `model:`, `data:`)

- **Removed old config classes**: Legacy `models/factory.py`, config shims, and `factory.py` compatibility layer removed

### Added

- **Model Base Class Hierarchy**: `GenerativeModel` → `DiffusionModel`, `VAEModel`, `StatisticalModel` with standardized `generate()` and `from_config()` interfaces
- **ModelRegistry**: Decorator-based registration (`@ModelRegistry.register('unet')`) — replaces `create_model()` factory function

### Improved

- **Code Quality**:
  - Deduplicated `from_config()` across models with config caching
  - Optimized `create_windows()` with stride views
  - Extracted shared helpers, removed dead code
  - Fixed config mutation bugs in backtest and evaluation
  - Fixed `.gitignore` to properly track `src/tsgen/data/` package

## [0.3.2] - 2025-12-16

### Added

#### MambaDiffusion Model
- **New State Space Model architecture**: MambaDiffusion based on Selective SSMs (Mamba/S4)
  - Pure PyTorch implementation for CPU/GPU/Mac compatibility (no CUDA kernels required)
  - Sequential SSM recurrence for broad hardware support
  - RMSNorm layers for improved training stability
  - Support for class-conditional generation via label embeddings
  - Competitive with Transformer while being more efficient
  - Files: `src/tsgen/models/mamba.py`
  - Tests: `tests/test_mamba.py` (7 comprehensive tests covering all components)
  - Factory integration: Use `model_type: 'mamba'` in configs

#### Checkpoint Resumption System
- **Full checkpoint support** for training resumption with complete state preservation
  - Saves: model weights, optimizer state (learning rate, momentum, etc.), epoch number, step count
  - Three convenient resumption modes:
    - `--resume-latest`: Auto-resume from latest checkpoint
    - `--resume-from-checkpoint <path>`: Resume from specific checkpoint file
    - `--list-checkpoints`: List all available checkpoints and exit
  - New checkpoint utilities module: `src/tsgen/training/checkpoint_utils.py`
    - `find_latest_checkpoint()`: Find most recent checkpoint
    - `list_checkpoints()`: List all checkpoints sorted by epoch
    - `get_checkpoint_path()`: Get checkpoint for specific epoch or latest
    - `extract_epoch_from_checkpoint()`: Parse epoch from filename
  - Enhanced BaseTrainer with `save_checkpoint()` and `load_checkpoint()` methods
  - Checkpoints saved every 10 epochs with naming: `checkpoint_epoch_N.pt`
  - Supports extending training (e.g., 50→100 epochs) without retraining
  - Tests: `tests/test_checkpoint_resumption.py` (10 comprehensive tests)

#### Tracker Factory Pattern
- **Centralized tracker creation** via factory pattern for better organization
  - New module: `src/tsgen/tracking/factory.py`
  - Function: `create_tracker(config, experiment_dir=None)`
  - Default behavior: Returns `ConsoleTracker` when no tracker specified (no config needed)
  - Supports all tracker types: MLflow, Console, File, NoOp
  - Backward compatible with legacy config formats
  - Cleaner separation: tracker creation logic isolated from main CLI

### Changed

#### Config Structure Improvements
- **Grouped configuration structure** for better organization and clarity
  - Top-level sections: `experiment`, `data`, `model`, `training`, `diffusion`, `evaluation`, `tracker`
  - Backward compatible with flat config structure via fallback pattern `config.get('section', config)`
  - Improved readability and maintainability
  - Example structure:
    ```yaml
    experiment:
      name: "0006_mamba_default"
      experiment_number: 6
      seed: 42

    tracker:
      output_type: "file"

    data:
      tickers: [...]
      sequence_length: 64
      batch_size: 64

    training:
      epochs: 50
      learning_rate: 0.0002

    diffusion:
      time_steps: 1000
      beta_schedule: "linear"

    evaluation:
      num_samples: 1000
    ```
  - All trainers and evaluation code updated to handle grouped configs:
    - `src/tsgen/training/diffusion_trainer.py` - handles `training` and `diffusion` sections
    - `src/tsgen/training/vae_trainer.py` - handles `training` section
    - `src/tsgen/evaluate.py` - handles `data`, `diffusion`, `evaluation` sections
    - `src/tsgen/train.py` - handles `data` section
  - Config keys support both new and old names:
    - `diffusion.time_steps` (new) and `timesteps` (old)
    - Graceful degradation ensures no breaking changes

#### Checkpoint File Naming
- **New checkpoint naming convention**: `checkpoint_epoch_N.pt` (was `model_epoch_N.pt`)
  - Old format: Model weights only (`state_dict`)
  - New format: Complete training state (model + optimizer + metadata)
  - Backward compatible: Old checkpoints still work for evaluation (weights-only loading)
  - Migration: New training runs automatically create new-format checkpoints

#### Code Organization
- Moved tracker creation logic from `cli/main.py` to dedicated `tracking/factory.py`
- Consolidated checkpoint utilities into `training/checkpoint_utils.py`
- Updated all imports to use factory pattern
- Cleaner separation of concerns throughout codebase

### Fixed
- **evaluate.py config parsing**: Fixed KeyError when accessing tickers with grouped config
  - Now properly reads from `config['data']['tickers']` with fallback to root
  - Handles missing tickers by recovering from processor metadata
- **Config parsing in trainers**: Support both `time_steps` (new) and `timesteps` (old)
  - Ensures backward compatibility while encouraging new format
- **Test suite updates**: Updated all test files to use new naming conventions
  - `MultivariateGBM` instead of `GBMGenerativeModel`
  - `create_tracker` instead of `get_tracker`
  - Fixed import paths in `test_cli.py`, `test_checkpoints.py`, `test_training_registry.py`

### Tests
- **Added 17 new tests** (294 total tests passing):
  - 7 tests for MambaDiffusion model:
    - RMSNorm layer functionality
    - MambaBlock forward pass
    - SSM recurrence correctness
    - Factory model creation
    - Output shape verification
    - Class-conditional generation
    - Reproducibility with fixed seeds
  - 10 tests for checkpoint resumption:
    - Save and load full checkpoints
    - Model weight preservation
    - Optimizer state preservation
    - Find latest checkpoint
    - List all checkpoints
    - Get checkpoint by epoch
    - Extract epoch from filename
    - Extra state preservation (step count, custom metrics)
- **Updated existing tests**:
  - Fixed test imports for renamed classes
  - Updated CLI tests for factory pattern
  - Fixed baseline model references
  - All MLflow tracker mocking fixed
- **Test Status**: 284 tests passing, 5 skipped, 2 pre-existing failures (unrelated)

### Documentation
- **Checkpoint Resumption Guide**: Complete usage documentation with examples
- **CLI Help Text**: Updated for new checkpoint arguments
- **Config Examples**: Updated `experiments/0006_mamba_default/config.yaml` with grouped structure
- **Code Comments**: Enhanced docstrings for checkpoint and factory modules

### Migration Guide

#### Config Structure (Optional - Fully Backward Compatible)

```yaml
# Old format (still works):
epochs: 50
learning_rate: 0.0002
timesteps: 1000

# New format (recommended):
training:
  epochs: 50
  learning_rate: 0.0002

diffusion:
  time_steps: 1000
```

**Note**: No changes required for existing configs. The new format is recommended for clarity but not mandatory.

#### Resuming from Old Checkpoints

Old checkpoints (`model_epoch_*.pt`) contain only weights:
- ✅ Can be loaded for **evaluation** (works normally)
- ⚠️ For **training resumption**, optimizer state is lost
- **Recommended**: Train a few more epochs to create new-format checkpoints

#### Tracker Configuration

```yaml
# Old format (still works):
tracker: "console"

# New format (recommended):
tracker:
  output_type: "console"
```

#### Using MambaDiffusion

```yaml
model_type: mamba
model:
  name: "MambaDiffusion"
  params:
    dim: 128
    depth: 4
    d_state: 16
    d_conv: 4
    expand: 2
    num_classes: 0  # For unconditional generation
```

### Benefits

1. **MambaDiffusion Model**:
   - More efficient than Transformer for long sequences
   - Works on any hardware (CPU/GPU/Mac) without specialized CUDA kernels
   - Competitive generation quality with faster training

2. **Checkpoint Resumption**:
   - ✅ Continue interrupted training without losing progress
   - ✅ Experiment with different epoch counts efficiently
   - ✅ Extend training (50→100 epochs) without starting over
   - ✅ Preserve optimizer momentum for better convergence
   - ✅ Save compute time and resources

3. **Grouped Config Structure**:
   - 📖 Better organization and readability
   - 🔧 Easier to maintain and update
   - 🔄 Backward compatible with existing configs
   - 🎯 Clear separation of concerns (data, training, model)

4. **Tracker Factory**:
   - 🏭 Centralized creation logic
   - 🎮 Default console tracker (no config needed)
   - 🧹 Cleaner main CLI code
   - 🔌 Easy to add new tracker types

## [0.3.1] - 2025-12-13

### Changed - Breaking

- **Pipeline Configuration Format**: Simplified and renamed
  - Config key renamed: `pipeline.steps` → `DataPipeline`
  - Direct list format instead of nested structure
  - **Old format (no longer supported)**:
    ```yaml
    pipeline:
      steps:
        - load_prices: {}
        - clean_data: {}
    ```
  - **New format (required)**:
    ```yaml
    DataPipeline:
      - load_prices: {}
      - clean_data: {}
    ```

- **Removed Legacy Manual Pipeline**:
  - All configs must now include `DataPipeline` section
  - No backward compatibility with configs lacking pipeline configuration
  - Eliminates code duplication from hardcoded pipeline logic

### Added

- **YAML-Configurable Data Pipeline**: Functional composition pattern for data preprocessing
  - `DataPipeline` class: Compose pipeline steps from YAML configuration
  - `PipelineStep` wrapper: Adds metadata and validation to pipeline functions
  - `PIPELINE_REGISTRY`: Maps step names to functions with type/parameter specs
  - Inspired by `torchvision.transforms.Compose` but for data pipelines

- **Pipeline Steps** with metadata and type checking:
  - `load_prices`: Load data from findata database
  - `clean_data`: Handle NaN values with configurable strategies
  - `split_temporal`: Chronological train/test splitting
  - `process_prices`: Convert prices → log-returns → standardized
  - `create_windows`: Sliding window generation
  - `create_dataloader`: PyTorch DataLoader creation

- **Type Safety**:
  - Pipeline validates step output/input type compatibility at build time
  - Catches mismatched types (DataFrame → ndarray → DataLoader) before execution
  - Clear error messages with step context

- **Parameter Validation**:
  - Required vs optional parameter checking
  - Runtime parameter override of config parameters
  - Function signature inspection for automatic parameter filtering

- **Documentation**:
  - `notes/PIPELINE_GUIDE.md`: Comprehensive pipeline configuration guide
  - Example configs: `configs/example_pipeline_*.yaml`
  - 37 new tests covering pipeline functionality

### Changed

- **Unified Baseline Model**: Merged `GBMGenerativeModel` and `MultivariateLogNormalModel` into single `MultivariateGBM` class
  - New `full_covariance` parameter (default=`True`) controls correlation modeling
  - `full_covariance=True`: Captures cross-asset correlations via Cholesky decomposition (previous `multivariate_lognormal` behavior)
  - `full_covariance=False`: Independent per-feature sampling (previous `gbm` behavior)
  - Backward compatible: Old model_type strings still work (`'gbm'`, `'multivariate_lognormal'`)
  - New recommended model_type: `'multivariate_gbm'` with configurable `full_covariance`

- **train.py**: Simplified to use only DataPipeline
  - Removed legacy manual pipeline code
  - Single, unified data pipeline approach
  - Cleaner, more maintainable codebase

- **evaluate.py**: Simplified evaluation pipeline
  - Direct use of processor.transform() and create_windows()
  - Maintains `df_real` for plotting compatibility

### Improved

- **Code Quality**:
  - Eliminated pipeline code duplication in train.py and evaluate.py
  - Reduced repetitive parameter extraction from config
  - Centralized pipeline logic in reusable classes
  - Cleaner separation of concerns

- **Configurability**:
  - Per-experiment pipeline customization via YAML
  - Easy to add/remove/reorder pipeline steps
  - No code changes needed for pipeline experiments

- **Testability**:
  - Individual pipeline steps remain independently testable
  - Integration tests verify end-to-end pipeline flow

### Testing

- All 270+ existing tests updated and passing ✅
- 37 pipeline tests updated:
  - `test_pipeline_builder.py`: 29 tests (PipelineStep, DataPipeline, registry, type validation)
  - `test_train_with_pipeline.py`: 3 tests (training with DataPipeline)
  - `test_evaluate_with_pipeline.py`: 1 test (evaluation with DataPipeline)

### Benefits

1. **No Code Duplication**: Pipeline defined once in config, used everywhere
2. **Simplified Configuration**: Direct list format, no nested structure
3. **Type Safe**: Validates step composition at build time
4. **Composable**: Functions remain independent, just wrapped
5. **Extensible**: Easy to add new pipeline steps to registry
6. **Cleaner Codebase**: Removed legacy manual pipeline code

### Migration Guide

**⚠️ Breaking Change**: All configs must be updated to use `DataPipeline` format.

**Update your configs**:

```yaml
# Before (v0.3.0):
pipeline:
  steps:
    - load_prices:
        column: 'adj_close'
    - clean_data:
        strategy: 'ffill_drop'
    - create_windows:
        sequence_length: 64

# After (v0.3.1):
DataPipeline:
  - load_prices:
      column: 'adj_close'
  - clean_data:
      strategy: 'ffill_drop'
  - create_windows:
      sequence_length: 64
```

**Changes required**:
1. Rename `pipeline:` to `DataPipeline:`
2. Remove `steps:` level - list steps directly under `DataPipeline`

**See Also:**
- `notes/PIPELINE_GUIDE.md` - Complete pipeline configuration guide
- `configs/example_pipeline_*.yaml` - Example configurations

---

## [0.3.0] - 2025-12-12

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

## [0.2.0] - 2025-12-12

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

## [0.1.0] - 2025-12-12

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
