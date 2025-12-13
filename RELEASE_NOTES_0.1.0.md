# Release Notes: tsgen v0.1.0

**Release Date:** 2024-12-12

## Overview

This is the initial public release of `tsgen`, a research framework for generating realistic synthetic financial time series using Denoising Diffusion Probabilistic Models (DDPM).

## What's Included

### Models
- **UNet1D**: Diffusion model adapted for 1D time series
- **Diffusion Transformer**: Transformer-based diffusion architecture
- **TimeVAE**: Variational autoencoder for time series
- **Classical Baselines**:
  - Geometric Brownian Motion (GBM)
  - Bootstrap resampling
  - Multivariate LogNormal (with correlation structure)

### Data Pipeline
- Composable pipeline functions for maximum flexibility
- Direct integration with `findata` package for database access
- LogReturnProcessor: prices → log-returns → z-scores
- Temporal train/test splitting to prevent data leakage
- Multiple NaN handling strategies

### Evaluation Framework
- **Stylized Facts**: kurtosis, skewness, volatility clustering (ACF)
- **Discriminator Accuracy**: LSTM-based real vs. fake classifier
- **Distribution Tests**: KS, Cramér-von Mises, Anderson-Darling
- **TSTR**: Train on Synthetic, Test on Real
- **Correlation Metrics**: Frobenius norm, eigenvalue spectrum, cross-asset correlations

### CLI Tools
- `tsgen`: Main training and evaluation command
- `tsgen-experiments`: Experiment management (list, create, info, open)
- `tsgen-backtest`: Rolling window backtesting

### Experiment Tracking
- File-based tracker (JSON logs)
- Console tracker
- MLflow tracker (optional)
- NoOp tracker (for testing)

## Test Coverage

- **Total Tests**: 215 passing, 5 skipped
- **Test Files**: 18 comprehensive test suites
- **Coverage Areas**:
  - Models and architectures
  - Data pipeline functions
  - Evaluation metrics
  - Diffusion utilities
  - CLI commands
  - Experiment management
  - Integration tests

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/tsgen.git
cd tsgen

# Install package
pip install -e .

# Install findata dependency (required)
cd ../findata
pip install -e .
```

## Quick Start

```bash
# Create experiment directory with config
mkdir -p experiments/0001_my_first_experiment
# Add your config.yaml to the directory

# Train and evaluate
tsgen --config experiments/0001_my_first_experiment/config.yaml --mode train_eval
```

## Known Limitations

1. **Requires findata package**: Must install separately for database access
2. **CPU-focused**: While GPU support exists, initial release is optimized for CPU
3. **Limited data sources**: Currently only supports database source (managed by findata)
4. **DDIM sampling**: DDIM tests skipped (feature not fully tested yet)

## Breaking Changes

N/A - This is the first release

## Dependencies

- Python 3.12+
- PyTorch 2.0+
- NumPy <2.0.0 (compatibility)
- Pandas, Scikit-learn, SciPy, Statsmodels
- findata package (separate repository)

## Documentation

- **README.md**: Installation and usage guide
- **CHANGELOG.md**: Detailed changelog
- **LICENSE**: MIT License
- **Project Structure**: See README.md for detailed structure

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## Support

- **Issues**: https://github.com/yourusername/tsgen/issues
- **Wiki**: https://github.com/yourusername/tsgen/wiki

## Next Steps

For users starting with tsgen:
1. Install both tsgen and findata packages
2. Set up findata database with historical data
3. Create your first experiment configuration
4. Run training and evaluation
5. Analyze results in the experiment directory

## Acknowledgments

Thanks to all contributors and the research community for feedback during development.
