# tsgen

> Synthetic Financial Time Series Generation using Diffusion Models

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A research framework for generating realistic synthetic financial time series using Denoising Diffusion Probabilistic Models (DDPM). The project benchmarks deep learning architectures (UNet, Transformer) against classical baselines (Multivariate GBM, Bootstrap) for capturing "Stylized Facts" of financial returns.

## Purpose

- Generate realistic synthetic time series for Monte Carlo simulations (e.g., Counterparty Credit Risk modeling)
- Create training data augmentation for financial ML models
- Research and benchmark generative models for financial time series
- Preserve key statistical properties: fat tails, volatility clustering, cross-asset correlations

## Features

- **Deep Learning Models**: UNet1D, Diffusion Transformer, TimeVAE
- **Classical Baselines**: Geometric Brownian Motion (GBM), Bootstrap resampling, Multivariate LogNormal
- **Comprehensive Evaluation**:
  - Stylized facts (kurtosis, skewness, volatility clustering)
  - Discriminator accuracy (distinguishability test)
  - Distribution tests (KS, Cramér-von Mises, Anderson-Darling)
  - TSTR (Train on Synthetic, Test on Real)
  - Correlation structure preservation
- **Composable Data Pipeline**: Flexible functions for loading, cleaning, processing time series
- **Experiment Management**: Track and organize multiple experiments with configs and results
- **Database Integration**: Uses `findata` project for historical market data management

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/yourusername/tsgen.git
cd tsgen
pip install -e .

# 2. Install findata dependency (for historical data access)
cd ../findata  # or clone from https://github.com/yourusername/findata
pip install -e .
cd ../tsgen

# 3. Run a simple experiment
tsgen --config experiments/example/config.yaml --mode train_eval
```

## Installation

### Prerequisites

- **Python**: 3.12 or higher
- **findata package**: Required for loading historical market data
  - This is a separate package that manages the time series database
  - Install from: `https://github.com/yourusername/findata`
  - tsgen has read-only access to the database managed by findata

### Install tsgen

**Standard installation:**
```bash
# Clone repository
git clone https://github.com/yourusername/tsgen.git
cd tsgen

# Install package
pip install -e .

# With development dependencies (pytest, black, etc.)
pip install -e ".[dev]"
```

**Using Conda (recommended):**
```bash
# Create and activate environment
conda env create -f environment.yml
conda activate tsgen

# Install tsgen
pip install -e .
```

### Install findata dependency

```bash
# Clone findata repository (sibling to tsgen)
cd ..
git clone https://github.com/yourusername/findata.git
cd findata

# Install findata
pip install -e .

# Return to tsgen
cd ../tsgen
```

### Verify Installation

```bash
# Check that CLI commands are available
tsgen --help
tsgen-experiments --help
tsgen-backtest --help

# Test imports
python -c "from tsgen import train_model, evaluate_model"
python -c "from tsgen.models import create_model, UNet1D"
python -c "from tsgen.data.pipeline import load_prices"
```

## Usage

### Command Line Interface

**Train and evaluate a model:**
```bash
# Train and evaluate with experiment config
tsgen --config experiments/my_experiment/config.yaml --mode train_eval

# Train only
tsgen --config experiments/my_experiment/config.yaml --mode train

# Evaluate only (requires existing model artifacts)
tsgen --config experiments/my_experiment/config.yaml --mode eval
```

**Experiment management:**
```bash
# List all experiments
tsgen-experiments list

# Show experiment details
tsgen-experiments info <experiment_number>

# Create new experiment
tsgen-experiments create my_experiment --model unet --description "Test UNet on 3 stocks"

# Open experiment directory
tsgen-experiments open <experiment_number>
```

**Run backtest:**
```bash
# Run rolling window backtest experiment
tsgen-backtest --config experiments/my_backtest/config.yaml
```

### Python API

```python
from tsgen import train_model, evaluate_model
from tsgen.models import create_model, UNet1D, DiffusionTransformer
from tsgen.data.pipeline import load_prices, clean_data, process_prices
from tsgen.data.processor import LogReturnProcessor
from tsgen.tracking.base import ConsoleTracker

# Load and process data using pipeline
df = load_prices(['AAPL', 'MSFT'], '2020-01-01', '2024-01-01')
df_clean = clean_data(df)

processor = LogReturnProcessor()
data_scaled = process_prices(df_clean, processor, fit=True)

# Create model from config
config = {
    'model_type': 'unet',
    'sequence_length': 256,
    'timesteps': 500,
    'base_channels': 32,
    'tickers': ['AAPL', 'MSFT'],
    'start_date': '2020-01-01',
    'end_date': '2024-01-01',
    # ... other config parameters
}

model = create_model(config)

# Train model
tracker = ConsoleTracker()
train_model(config, tracker)

# Evaluate model
evaluate_model(config, tracker)
```

## Configuration

Experiments are configured via YAML files in `experiments/XXXX_name/config.yaml`. Key parameters:

```yaml
# Experiment identification
experiment_name: 'my_experiment'
experiment_number: 1
description: 'Experiment description'
output_dir: 'experiments/0001_my_experiment'

# Model configuration
model_type: unet  # Options: unet, transformer, gbm, bootstrap, multivariate_lognormal, timevae
base_channels: 128
sequence_length: 64
timesteps: 500
learning_rate: 1e-3

# Data configuration
tickers: [AAPL, MSFT, GOOG]
start_date: '2020-01-01'
end_date: '2024-01-01'
column: adj_close  # Options: open, high, low, close, adj_close, volume
train_test_split: 0.8  # Optional: temporal split ratio

# Training configuration
batch_size: 32
epochs: 100

# Experiment tracking
tracker: file  # Options: file, console, mlflow, noop
```

## Project Structure

```
tsgen/
├── src/
│   └── tsgen/              # Main package
│       ├── cli/            # CLI entry points
│       │   ├── main.py     # Main training/evaluation CLI
│       │   └── experiments.py  # Experiment management CLI
│       ├── models/         # Model architectures
│       │   ├── unet.py     # UNet1D diffusion model
│       │   ├── transformer.py  # Diffusion Transformer
│       │   ├── timevae.py  # TimeVAE model
│       │   ├── baselines.py    # GBM, Bootstrap, Multivariate LogNormal
│       │   ├── diffusion.py    # Diffusion utilities (forward/reverse process)
│       │   └── factory.py  # Model factory
│       ├── data/           # Data pipeline (loading, cleaning, processing)
│       │   ├── pipeline.py # Composable pipeline functions
│       │   └── processor.py # LogReturnProcessor for price transformations
│       ├── analysis/       # Evaluation metrics and tests
│       │   ├── metrics.py  # Stylized facts metrics
│       │   ├── discriminator.py  # Discriminator model
│       │   ├── distribution_tests.py  # Statistical tests
│       │   ├── tstr.py     # Train on Synthetic, Test on Real
│       │   └── correlation_metrics.py  # Correlation structure analysis
│       ├── tracking/       # Experiment tracking (MLflow, console, file)
│       ├── experiments/    # Experiment management and backtesting
│       ├── training/       # Training utilities
│       └── utils/          # General utilities
├── tests/                  # Test suite (18 test files)
├── experiments/            # Experiment folders (created during training)
│   └── .gitkeep            # Folder structure preserved, content gitignored
└── pyproject.toml          # Package configuration
```

## Development

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=tsgen --cov-report=html

# Run specific test file
pytest tests/test_models.py -v
```

### Code Formatting

```bash
# Format code with black
black src/ tests/

# Check with flake8
flake8 src/ tests/
```

## Tech Stack

- **Python**: 3.12+
- **Deep Learning**: PyTorch 2.0+
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Visualization**: Matplotlib
- **Experiment Tracking**: MLflow (optional)
- **Database**: findata package (sibling project)

## Related Projects

- **findata**: Historical market data management (sibling repository)
  - Manages the time series database
  - Must be installed before using tsgen
  - tsgen has READ-ONLY access to the database
  
## License

MIT License - See LICENSE file for details

