#!/usr/bin/env python3
"""One-shot helper that writes the standardized `config.yaml` for every
experiment in `experiments/000X/` according to the agreed research protocol.

Run from the repo root:

    conda run -n tsgen python scripts/generate_standard_configs.py

The generated configs share a common ticker list, date range, window length,
temporal split, cleaning strategy, scaling, and evaluation settings. Per-model
identity (architecture hyperparameters and training budget) differs.

Safe to re-run: overwrites `config.yaml` in each experiment folder. Does not
touch `config_*.yaml` (the TimeVAE ablation sub-configs in 0001) or
documentation (`README.md`, `MULTI_RUN_GUIDE.md`).
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent


# ---------------------------------------------------------------------------
# Shared protocol
# ---------------------------------------------------------------------------

TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'JPM',
    'AMGN', 'AXP', 'BA', 'A', 'ABBV', 'ABNB', 'ABT', 'ACGL', 'ACN', 'ADBE',
    'ADI', 'ADM', 'ADP', 'ADSK', 'AEE', 'AEP', 'AES', 'AFL', 'AIG', 'AIZ',
    'AJG', 'AKAM', 'ALB', 'ALGN', 'ALL', 'ALLE', 'AMAT', 'AMCR', 'AMD', 'AME',
    'AMP', 'AMT', 'ANET', 'AON', 'AOS', 'APA', 'APD', 'APH', 'APO', 'APP',
    'APTV', 'ARE', 'ATO', 'AVB', 'AVGO', 'AVY', 'AWK', 'AXON', 'AZO', 'BAC',
    'BALL', 'BAX', 'BBY', 'BDX', 'BEN', 'BG', 'BIIB', 'BK', 'BKNG', 'BKR',
    'BLDR', 'BLK', 'BMY', 'BR', 'BRO', 'BSX', 'BX', 'BXP', 'C', 'CAG',
    'CAH', 'CARR', 'CAT', 'CB', 'CBOE', 'CBRE', 'CCI', 'CCL', 'CDNS', 'CDW',
    'CEG', 'CF', 'CFG', 'CHD', 'CHRW', 'CHTR', 'CI', 'CINF', 'CL', 'CLX',
]

START_DATE = '2005-01-01'
END_DATE = '2024-12-31'
SEQUENCE_LENGTH = 64
TRAIN_TEST_SPLIT = 0.8

# Per-paradigm training budgets (shared scale only; per-model arch
# hyperparameters are left to each model spec below).
DIFFUSION_EPOCHS = 200
VAE_EPOCHS = 100
BASELINE_EPOCHS = 1
DIFFUSION_TIMESTEPS = 500


# ---------------------------------------------------------------------------
# YAML rendering helpers
# ---------------------------------------------------------------------------

def _format_tickers() -> str:
    """Render the ticker list indented as a YAML array-of-strings."""
    lines = []
    chunk = 10
    for i in range(0, len(TICKERS), chunk):
        row = ", ".join(f"'{t}'" for t in TICKERS[i:i + chunk])
        lines.append(f"    {row},")
    # Drop trailing comma on last line
    lines[-1] = lines[-1].rstrip(',')
    body = "\n".join(lines)
    return f"[\n{body}\n  ]"


TICKERS_YAML = _format_tickers()


def _data_block() -> str:
    """Data section common to all experiments."""
    return dedent(f"""\
        data:
          column: 'adj_close'
          tickers: {TICKERS_YAML}
          start_date: '{START_DATE}'
          end_date: '{END_DATE}'
          sequence_length: {SEQUENCE_LENGTH}
          train_test_split: {TRAIN_TEST_SPLIT}""")


def _data_block_no_scaling() -> str:
    """Data section for CCC-GARCH (raw log-returns, no z-scoring)."""
    return dedent(f"""\
        data:
          column: 'adj_close'
          tickers: {TICKERS_YAML}
          start_date: '{START_DATE}'
          end_date: '{END_DATE}'
          sequence_length: {SEQUENCE_LENGTH}
          train_test_split: {TRAIN_TEST_SPLIT}
          scaling: 'none'""")


def _evaluation_block() -> str:
    return dedent("""\
        evaluation:
          num_samples: 500
          discriminator_epochs: 20
          stylized_facts_lags: 20
          var_alpha: 0.05
          tstr_epochs: 10""")


def _pipeline_block(batch_size: int, shuffle: bool = True) -> str:
    shuffle_str = 'true' if shuffle else 'false'
    return dedent(f"""\
        data_pipeline:
          - load_prices:
              column: 'adj_close'
          - clean_data:
              strategy: 'mask'
          - split_temporal:
              train_ratio: {TRAIN_TEST_SPLIT}
          - process_prices:
              fit: true
          - create_windows:
              sequence_length: {SEQUENCE_LENGTH}
              stride: 1
          - create_dataloader:
              batch_size: {batch_size}
              shuffle: {shuffle_str}""")


def _diffusion_training_block(batch_size: int = 32) -> str:
    return dedent(f"""\
        training:
          batch_size: {batch_size}
          epochs: {DIFFUSION_EPOCHS}
          learning_rate: 0.001
          timesteps: {DIFFUSION_TIMESTEPS}
          sampling_method: 'ddpm'
          validation_interval: 20
          num_validation_samples: 100""")


def _vae_training_block(batch_size: int = 32) -> str:
    return dedent(f"""\
        training:
          batch_size: {batch_size}
          epochs: {VAE_EPOCHS}
          learning_rate: 0.001
          beta: 0.5
          use_annealing: true
          annealing_epochs: 50
          use_free_bits: true
          free_bits: 0.5
          teacher_forcing_ratio: 0.5""")


def _baseline_training_block(batch_size: int = 128) -> str:
    return dedent(f"""\
        training:
          batch_size: {batch_size}
          epochs: {BASELINE_EPOCHS}""")


def _assemble(
    *,
    experiment_name: str,
    experiment_number: int,
    description: str,
    output_dir: str,
    model_type: str,
    model_block: str,
    training_block: str,
    data_block: str,
    pipeline_block: str,
    header_comment: str = "",
) -> str:
    header = header_comment.strip()
    if header:
        header = "\n".join(f"# {line}" if line else "#" for line in header.splitlines()) + "\n\n"
    return (
        header
        + f"experiment_name: '{experiment_name}'\n"
        + f"experiment_number: {experiment_number}\n"
        + f"description: '{description}'\n"
        + f"output_dir: '{output_dir}'\n"
        + "\n"
        + f"model_type: '{model_type}'\n"
        + "\n"
        + model_block.strip() + "\n"
        + "\n"
        + data_block.strip() + "\n"
        + "\n"
        + training_block.strip() + "\n"
        + "\n"
        + _evaluation_block().strip() + "\n"
        + "\n"
        + "tracker: 'file'\n"
        + "\n"
        + pipeline_block.strip() + "\n"
    )


# ---------------------------------------------------------------------------
# Per-experiment spec
# ---------------------------------------------------------------------------

CONFIGS = [
    {
        "path": "0001_timevae_all_stocks/config.yaml",
        "kwargs": dict(
            experiment_name="timevae_all_stocks",
            experiment_number=1,
            description="TimeVAE with LSTM encoder/decoder, free-bits + beta annealing. Baseline config; ablations live in config_*.yaml.",
            output_dir="experiments/0001_timevae_all_stocks",
            model_type="timevae",
            model_block=dedent("""\
                model:
                  hidden_dim: 64
                  latent_dim: 16
                  encoder_type: 'lstm'
                  num_layers: 2"""),
            training_block=_vae_training_block(),
            data_block=_data_block(),
            pipeline_block=_pipeline_block(batch_size=32),
            header_comment="TimeVAE - 100 stocks (standardized protocol)\nAblation sub-configs in config_*.yaml.",
        ),
    },
    {
        "path": "0002_unet_all_stocks/config.yaml",
        "kwargs": dict(
            experiment_name="unet_all_stocks",
            experiment_number=2,
            description="UNet 1D diffusion (pooling baseline). Paired with 0008_diffwave for pooling-vs-dilation ablation.",
            output_dir="experiments/0002_unet_all_stocks",
            model_type="unet",
            model_block=dedent("""\
                model:
                  base_channels: 128"""),
            training_block=_diffusion_training_block(),
            data_block=_data_block(),
            pipeline_block=_pipeline_block(batch_size=32),
            header_comment="UNet1D (MaxPool downsampling) - 100 stocks",
        ),
    },
    {
        "path": "0003_unet_data_fix/config.yaml",
        "kwargs": dict(
            experiment_name="unet_all_stocks_data_fix",
            experiment_number=3,
            description="UNet 1D diffusion - retained for reproducibility of original train/test-split work. Protocol is identical to 0002 under the standardized protocol; kept as a historical marker.",
            output_dir="experiments/0003_unet_data_fix",
            model_type="unet",
            model_block=dedent("""\
                model:
                  base_channels: 128"""),
            training_block=_diffusion_training_block(),
            data_block=_data_block(),
            pipeline_block=_pipeline_block(batch_size=32),
            header_comment="UNet1D (data-fix lineage) - 100 stocks\nHistorically introduced the temporal train/test split. Under the\nstandardized protocol this is now equivalent to 0002; both kept.",
        ),
    },
    {
        "path": "0004_transformer_all_stocks/config.yaml",
        "kwargs": dict(
            experiment_name="transformer_all_stocks",
            experiment_number=4,
            description="DiffusionTransformer (additive conditioning baseline). Paired with 0009_dit for conditioning-scheme ablation (additive vs adaLN-Zero).",
            output_dir="experiments/0004_transformer_all_stocks",
            model_type="transformer",
            model_block=dedent("""\
                model:
                  dim: 128
                  depth: 6
                  heads: 8
                  mlp_dim: 256
                  dropout: 0.1
                  num_classes: 0"""),
            training_block=_diffusion_training_block(),
            data_block=_data_block(),
            pipeline_block=_pipeline_block(batch_size=32),
            header_comment="DiffusionTransformer (additive conditioning) - 100 stocks",
        ),
    },
    {
        "path": "0005_multivariate_gaussian/config.yaml",
        "kwargs": dict(
            experiment_name="multivariate_gaussian_baseline",
            experiment_number=5,
            description="Multivariate Gaussian baseline on scaled log-returns (full covariance). Captures first + second moments only; floor benchmark.",
            output_dir="experiments/0005_multivariate_gaussian",
            model_type="multivariate_gaussian",
            model_block=dedent("""\
                model:
                  full_covariance: true"""),
            training_block=_baseline_training_block(),
            data_block=_data_block(),
            pipeline_block=_pipeline_block(batch_size=128),
            header_comment="Multivariate Gaussian - 100 stocks\nFits Gaussian (mean + full covariance) to z-scored log-returns.",
        ),
    },
    {
        "path": "0006_mamba_default/config.yaml",
        "kwargs": dict(
            experiment_name="mamba_all_stocks",
            experiment_number=6,
            description="MambaDiffusion (selective SSM) via pure-PyTorch Heinsen parallel scan. Linear-in-L scaling; compare with Transformer and DiffWave.",
            output_dir="experiments/0006_mamba_default",
            model_type="mamba",
            model_block=dedent("""\
                model:
                  dim: 128
                  depth: 4
                  d_state: 16
                  d_conv: 4
                  expand: 2
                  num_classes: 0"""),
            training_block=_diffusion_training_block(),
            data_block=_data_block(),
            pipeline_block=_pipeline_block(batch_size=32),
            header_comment="MambaDiffusion (selective SSM) - 100 stocks",
        ),
    },
    {
        "path": "0007_ccc_garch/config.yaml",
        "kwargs": dict(
            experiment_name="ccc_garch_baseline",
            experiment_number=7,
            description="CCC-GARCH(1,1) with Student-t innovations. Captures volatility clustering (univariate GARCH per ticker) + linear cross-correlation (constant R).",
            output_dir="experiments/0007_ccc_garch",
            model_type="ccc_garch",
            model_block=dedent("""\
                model:
                  p: 1
                  q: 1
                  distribution: 't'"""),
            training_block=_baseline_training_block(),
            data_block=_data_block_no_scaling(),
            # GARCH reconstructs chronological series from windows; no shuffle.
            pipeline_block=_pipeline_block(batch_size=128, shuffle=False)
                + dedent("""\

                    # IMPORTANT: CCC-GARCH requires raw log-returns (scaling='none'
                    # above) and ordered windows (shuffle: false). Under these two
                    # settings, the fit path reconstructs the flat chronological
                    # series from stride-1 windows before fitting univariate GARCH
                    # models per ticker.""").rstrip(),
            header_comment="CCC-GARCH(1,1) Student-t - 100 stocks\nFits raw log-returns. scaling='none' + shuffle=false required.",
        ),
    },
    {
        "path": "0008_diffwave/config.yaml",
        "kwargs": dict(
            experiment_name="diffwave_all_stocks",
            experiment_number=8,
            description="DiffWave-style dilated-conv diffusion. Paired with 0002_unet_all_stocks for pooling-vs-dilation ablation.",
            output_dir="experiments/0008_diffwave",
            model_type="diffwave",
            model_block=dedent("""\
                model:
                  residual_channels: 64
                  num_blocks: 10
                  dilation_cycle_length: 5
                  kernel_size: 3
                  num_classes: 0"""),
            training_block=_diffusion_training_block(),
            data_block=_data_block(),
            pipeline_block=_pipeline_block(batch_size=32),
            header_comment="DiffWave1D (dilated convs, no pooling) - 100 stocks",
        ),
    },
    {
        "path": "0009_dit/config.yaml",
        "kwargs": dict(
            experiment_name="dit_all_stocks",
            experiment_number=9,
            description="DiT-style transformer with adaLN-Zero conditioning and sinusoidal positional encoding. Paired with 0004_transformer_all_stocks for conditioning-scheme ablation.",
            output_dir="experiments/0009_dit",
            model_type="dit",
            model_block=dedent("""\
                model:
                  dim: 128
                  depth: 4
                  heads: 4
                  mlp_ratio: 4.0
                  dropout: 0.0
                  num_classes: 0"""),
            training_block=_diffusion_training_block(),
            data_block=_data_block(),
            pipeline_block=_pipeline_block(batch_size=32),
            header_comment="DiT1D (adaLN-Zero) - 100 stocks",
        ),
    },
    {
        "path": "0010_bootstrap/config.yaml",
        "kwargs": dict(
            experiment_name="bootstrap_baseline",
            experiment_number=10,
            description="Stationary block bootstrap (Politis-Romano). Expected block length 10. Non-parametric baseline that preserves the empirical marginal and short-range dependence.",
            output_dir="experiments/0010_bootstrap",
            model_type="bootstrap",
            model_block=dedent("""\
                model:
                  block_p: 0.1"""),
            training_block=_baseline_training_block(),
            data_block=_data_block(),
            # Bootstrap reconstructs chronological series from windows; no shuffle.
            pipeline_block=_pipeline_block(batch_size=128, shuffle=False)
                + dedent("""\

                    # IMPORTANT: The stationary block bootstrap reconstructs the
                    # flat chronological series from stride-1 windows; shuffle
                    # must be false.""").rstrip(),
            header_comment="Stationary Block Bootstrap - 100 stocks\nNon-parametric resampling of geometrically-sized blocks with\ncircular wrap. Target expected block length = 10 trading days.",
        ),
    },
]


# ---------------------------------------------------------------------------

def main() -> None:
    root = Path(__file__).resolve().parent.parent / "experiments"
    for spec in CONFIGS:
        target = root / spec["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(_assemble(**spec["kwargs"]))
        print(f"wrote {target.relative_to(root.parent)}")


if __name__ == "__main__":
    main()
