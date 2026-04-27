# tsgen

> Synthetic financial time-series generation and evaluation for research workflows.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-0.4.1-blue.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`tsgen` benchmarks generative models for multi-asset financial return series.
It includes diffusion models, TimeVAE, Mamba, DiT, DiffWave, and classical
baselines, with evaluation focused on whether generated samples preserve
stylized facts, cross-asset dependence, tail-risk behavior, and out-of-sample
distinguishability.

The framework is research-oriented: evaluation methodology is treated as part
of the product, not as a throwaway script.

## Current Status

Version `0.4.1` includes the April 2026 methodology refresh:

- chronological held-out evaluation windows when configs declare a split;
- discriminator train/test split with held-out accuracy and AUC;
- per-feature distribution tests on reduced-overlap windows;
- explicit evaluator failure metrics in aggregate summaries;
- fail-fast behavior when requested tickers are missing from `finbase`;
- stabilized Mamba scan discretization;
- Colab benchmark workflow that records the exact git SHA and aggregates runs.

Recent full benchmark runs completed successfully with `Eval Failures = 0`,
but the held-out discriminator still separates real and synthetic samples
almost perfectly. Treat that as a research finding: the current models should
not be described as producing indistinguishable synthetic market data.

## Models

Neural models:

- `unet` - 1D U-Net diffusion model
- `transformer` - diffusion transformer
- `mamba` - Mamba-style diffusion model
- `diffwave` - DiffWave-style 1D diffusion model
- `dit` - DiT-style 1D diffusion transformer
- `timevae` - variational autoencoder baseline

Classical baselines:

- `multivariate_gaussian` - full-covariance Gaussian baseline on returns
- `bootstrap` - stationary/block bootstrap baseline
- `ccc_garch` - constant conditional correlation GARCH baseline

## Evaluation

The standard evaluation pipeline reports:

- stylized facts: kurtosis, skewness, autocorrelation, volatility clustering;
- correlation structure: matrix norm, eigenvalue fit, rolling-correlation stability;
- distribution tests: per-feature KS, Cramer-von Mises, Anderson-Darling aggregates;
- discriminator distinguishability: held-out accuracy and AUC;
- TSTR: train on synthetic, test on real;
- tail risk: VaR and expected shortfall differences;
- evaluator health: `evaluation_failed_count` and per-evaluator failure flags.

Use `scripts/aggregate_results.py` to build comparison tables:

```bash
conda run -n tsgen python scripts/aggregate_results.py experiments/
```

This writes:

```text
experiments/summary.csv
experiments/summary.md
```

## Data Access

Market data is loaded through `finbase.DataClient`.

`tsgen` is a read-only consumer of the market database. Load and manage market
data in `finbase`; use `tsgen` for generation, evaluation, and analysis.

The database path is usually discovered from `~/.finbaserc`. A typical config:

```yaml
database:
  path: /path/to/timeseries.db
```

Install `finbase` before running data-backed experiments:

```bash
pip install finbase
```

For local editable development, install the sibling project if applicable:

```bash
cd ../finbase
pip install -e .
cd ../tsgen
```

## Installation

Python 3.12+ is required.

```bash
git clone https://github.com/shoom1/tsgen.git
cd tsgen
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
```

Recommended local environment:

```bash
conda env create -f environment.yml
conda activate tsgen
pip install -e .
```

Verify the install:

```bash
tsgen --help
tsgen-experiments --help
tsgen-backtest --help
python -c "from tsgen import train_model, evaluate_model; print('ok')"
```

## Standard Benchmark Workflow

Generate the standardized 10-run benchmark configs:

```bash
conda run -n tsgen python scripts/generate_standard_configs.py
```

Run a quick end-to-end smoke test:

```bash
conda run -n tsgen python scripts/smoke_test_experiments.py
conda run -n tsgen python scripts/aggregate_results.py smoke_test/
```

Run a full experiment:

```bash
conda run -n tsgen tsgen --config experiments/0004_transformer_all_stocks/config.yaml --mode train_eval
```

Run the full benchmark on Colab:

1. Upload the `finbase` SQLite database to `My Drive/tsgen/timeseries.db`.
2. Open [notebooks/colab_train.ipynb](notebooks/colab_train.ipynb) in Colab.
3. Select a GPU runtime.
4. Run cells top-to-bottom.
5. Confirm `Eval Failures = 0` in the aggregate summary before interpreting metrics.

See [notebooks/README.md](notebooks/README.md) for the Colab workflow details.

## Configuration

Experiments are YAML-driven. Current standard configs use the nested schema:

```yaml
experiment_name: transformer_all_stocks
output_dir: experiments/0004_transformer_all_stocks
model_type: transformer

data:
  column: adj_close
  tickers: [AAPL, MSFT, GOOGL]
  start_date: "2005-01-01"
  end_date: "2024-12-31"
  sequence_length: 64
  train_test_split: 0.8

training:
  epochs: 200
  batch_size: 32
  learning_rate: 0.001
  timesteps: 500

evaluation:
  num_samples: 500
  discriminator_epochs: 20
  tstr_epochs: 10

data_pipeline:
  - load_prices:
      column: adj_close
  - clean_data:
      strategy: mask
  - split_temporal:
      train_ratio: 0.8
  - process_prices:
      fit: true
  - create_windows:
      sequence_length: 64
      stride: 1
  - create_dataloader:
      batch_size: 32
      shuffle: true
```

## CLI

Train and evaluate:

```bash
tsgen --config path/to/config.yaml --mode train_eval
```

Train only:

```bash
tsgen --config path/to/config.yaml --mode train
```

Evaluate saved artifacts:

```bash
tsgen --config path/to/config.yaml --mode eval
```

Override config fields:

```bash
tsgen --config path/to/config.yaml --mode train_eval \
  --override training.epochs=3 \
  --override evaluation.num_samples=50 \
  --override output_dir=smoke_test/debug_run
```

## Project Layout

```text
tsgen/
├── configs/                 # Small example configs
├── notebooks/               # Colab benchmark notebook
├── scripts/                 # Config generation, smoke tests, aggregation
├── src/tsgen/
│   ├── analysis/            # Metrics and statistical tests
│   ├── cli/                 # CLI entry points
│   ├── config/              # Pydantic config schema
│   ├── data/                # finbase loading, cleaning, processing, windows
│   ├── evaluation/          # Evaluation pipeline and evaluators
│   ├── experiments/         # Experiment management and backtesting
│   ├── models/              # Model registry and architectures
│   ├── tracking/            # File, console, MLflow tracking
│   └── training/            # Trainer registry and training loops
├── tests/                   # Unit and regression tests
└── pyproject.toml
```

`experiments/`, `smoke_test/`, generated artifacts, and local agent notes are
ignored by git. Standard experiment configs are generated by
`scripts/generate_standard_configs.py`.

## Development

Run tests in the project conda environment:

```bash
conda run -n tsgen pytest tests/
```

Focused methodology checks:

```bash
conda run -n tsgen pytest \
  tests/test_load_prices_missing.py \
  tests/test_evaluate_real_data.py \
  tests/test_evaluation_pipeline_behavior.py \
  tests/test_distribution_tests.py \
  tests/test_mamba_parallel_scan.py -q
```

Before relying on benchmark outputs:

```bash
conda run -n tsgen python scripts/aggregate_results.py experiments/
```

Confirm:

- all expected runs are present;
- `Eval Failures` is zero;
- the run manifest records the expected git SHA;
- discriminator accuracy and AUC are interpreted as held-out metrics.

## License

MIT License. See [LICENSE](LICENSE) for details.
