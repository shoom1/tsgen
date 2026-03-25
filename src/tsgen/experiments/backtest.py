"""
Backtesting module for comparing generative models.

Trains multiple model types on historical data and generates
synthetic paths for visual comparison against held-out test data.
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta

from tsgen.data.pipeline import load_prices, clean_data
from tsgen.models.registry import ModelRegistry
from tsgen.models.base_model import StatisticalModel
from tsgen.tracking.base import ConsoleTracker, FileTracker
from tsgen.train import train_model
from tsgen.config.schema import ExperimentConfig, DataConfig

# Import models package to trigger ModelRegistry registration
import tsgen.models


DEFAULT_MODELS = ['gbm', 'bootstrap', 'unet', 'transformer']
MODEL_COLORS = {
    'gbm': 'blue',
    'bootstrap': 'green',
    'unet': 'orange',
    'transformer': 'purple',
    'mamba': 'red',
    'timevae': 'brown',
}


def get_backtest_output_dir(config):
    """Get output directory for backtest results."""
    output_dir = getattr(config, 'output_dir', None) or "backtest_results"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def _train_single_model(model_type, base_config, train_end, output_dir):
    """
    Train a single model and save artifacts via tracker.

    Returns:
        dict with 'model_path' and 'processor_path', or None if training failed.
    """
    model_path = os.path.join(output_dir, f"{model_type}_model.pt")
    processor_path = os.path.join(output_dir, f"{model_type}_processor.pkl")

    if os.path.exists(model_path):
        print(f"Model {model_type} already trained. Skipping.")
        return {'model_path': model_path, 'processor_path': processor_path}

    new_data = base_config.data.model_copy(update={'end_date': train_end.strftime('%Y-%m-%d')})
    curr_config = base_config.model_copy(update={'model_type': model_type, 'data': new_data})

    tracker = FileTracker(log_file=f"backtest_{model_type}.log", experiment_dir=output_dir)
    try:
        train_model(curr_config, tracker)
    except (ValueError, RuntimeError, KeyError) as e:
        print(f"Training failed for {model_type}: {type(e).__name__}: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error training {model_type}: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Copy artifacts from tracker to backtest output dir
    tracker_model = tracker.get_artifact_path('model')
    tracker_data = tracker.get_artifact_path('data')

    if tracker_model:
        # Find model file in tracker's artifact dir
        for fname in os.listdir(tracker_model):
            if fname.endswith('.pt'):
                src = os.path.join(tracker_model, fname)
                os.replace(src, model_path)
                break

    if tracker_data:
        for fname in os.listdir(tracker_data):
            if fname.endswith('.pkl'):
                src = os.path.join(tracker_data, fname)
                os.replace(src, processor_path)
                break

    if os.path.exists(model_path):
        return {'model_path': model_path, 'processor_path': processor_path}

    print(f"Warning: No artifacts found for {model_type}")
    return None


def _generate_paths(model_type, paths, base_config, n_tickers, n_paths, seq_len, device):
    """
    Load a trained model and generate synthetic paths.

    Returns:
        np.ndarray of shape (n_paths, seq_len, n_tickers) log-returns, or None.
    """
    import joblib

    loaded_obj = torch.load(paths['model_path'], map_location=device, weights_only=False)
    processor = joblib.load(paths['processor_path'])

    model_config = base_config.model_copy(update={'model_type': model_type})

    if isinstance(loaded_obj, dict):
        model = ModelRegistry.create(model_config, features=n_tickers).to(device)
        model.load_state_dict(loaded_obj)
    else:
        model = loaded_obj

    gen_seqs = model.generate(n_samples=n_paths, seq_len=seq_len, device=device)
    gen_seqs_np = gen_seqs.cpu().numpy()

    initial_prices = np.ones(n_tickers) * 100
    gen_prices = processor.inverse_transform(gen_seqs_np, initial_prices)
    return np.diff(np.log(gen_prices), axis=1)


def _plot_backtest_results(tickers, generated_data, real_prices, seq_len,
                           n_paths, train_end, output_dir):
    """Generate per-ticker comparison plots."""
    for i, ticker in enumerate(tickers):
        plt.figure(figsize=(12, 6))

        real_path = real_prices[ticker].values
        if len(real_path) > 0:
            norm_real = real_path / real_path[0] * 100
            plt.plot(norm_real, label='Real (Test Data)', color='black',
                     linewidth=2, zorder=10)

        for model_type, gen_returns in generated_data.items():
            asset_returns = gen_returns[:, :, i]
            cumulative = np.cumsum(asset_returns, axis=1)
            cumulative = np.hstack([np.zeros((n_paths, 1)), cumulative])
            generated_prices = 100 * np.exp(cumulative[:, :-1])

            p05 = np.percentile(generated_prices, 5, axis=0)
            p50 = np.percentile(generated_prices, 50, axis=0)
            p95 = np.percentile(generated_prices, 95, axis=0)

            color = MODEL_COLORS.get(model_type, 'gray')
            plt.plot(p50, label=f'{model_type} Median', color=color, linestyle='--')
            plt.fill_between(range(len(p05)), p05, p95, color=color, alpha=0.15,
                             label=f'{model_type} 90% CI')

        plt.title(f"{ticker}: 1-Year Forecast Validation (Train end: {train_end.date()})")
        plt.xlabel("Trading Days")
        plt.ylabel("Normalized Price")
        plt.legend()
        plt.grid(True, alpha=0.3)

        save_path = os.path.join(output_dir, f"plot_{ticker}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved plot to {save_path}")


def run_backtest_experiment(config=None):
    """
    Run backtest experiment comparing multiple model types.

    Args:
        config: Optional ExperimentConfig. If None, uses default config.
    """
    if config is None:
        config = ExperimentConfig(
            experiment_name="Backtest_Exp",
            model_type='unet',
            data=DataConfig(
                tickers=['AAPL', 'MSFT', 'GOOG'],
                start_date='2015-01-01',
                end_date='2024-01-01',
                sequence_length=256,
            ),
            training={
                'epochs': 50, 'batch_size': 32, 'learning_rate': 1e-3,
                'timesteps': 500,
            },
            model={'base_channels': 32, 'dim': 32, 'depth': 2, 'heads': 4},
            tracker='console',
        )

    seq_len = config.data.sequence_length
    n_paths = getattr(config, 'backtest_n_paths', 1000)
    models_to_test = getattr(config, 'backtest_models', None) or DEFAULT_MODELS

    output_dir = get_backtest_output_dir(config)
    print(f"Backtest outputs will be saved to: {output_dir}")

    # Define split
    full_end = pd.to_datetime(config.data.end_date)
    test_start = full_end - timedelta(days=365)
    train_end = test_start

    print(f"--- Experiment Setup ---")
    print(f"Train Period: {config.data.start_date} -> {train_end.strftime('%Y-%m-%d')}")
    print(f"Test Period:  {test_start.strftime('%Y-%m-%d')} -> {config.data.end_date}")

    # Data preparation
    tickers = config.data.tickers
    df_full = load_prices(tickers, config.data.start_date, config.data.end_date,
                          column=config.data.column)
    df_full = clean_data(df_full, strategy='ffill_drop')

    df_test = df_full[df_full.index >= test_start]
    if len(df_test) < seq_len:
        print(f"Warning: Test set ({len(df_test)}) < sequence_length ({seq_len})")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_tickers = len(tickers)

    # Training loop
    results = {}
    for model_type in models_to_test:
        print(f"\n>>> Processing Model: {model_type}")
        result = _train_single_model(model_type, config, train_end, output_dir)
        if result:
            results[model_type] = result

    # Generation loop
    print("\n>>> Starting Generation...")
    generated_data = {}
    for model_type, paths in results.items():
        print(f"Generating paths for {model_type}...")
        try:
            gen_returns = _generate_paths(
                model_type, paths, config, n_tickers, n_paths, seq_len, device)
            generated_data[model_type] = gen_returns
        except Exception as e:
            print(f"Generation failed for {model_type}: {type(e).__name__}: {e}")

    # Plotting
    print("\n>>> Creating Plots...")
    real_prices = df_test.iloc[:seq_len]
    _plot_backtest_results(tickers, generated_data, real_prices, seq_len,
                           n_paths, train_end, output_dir)


def main():
    """Entry point for backtest CLI command."""
    run_backtest_experiment()


if __name__ == "__main__":
    main()
