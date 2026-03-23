import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
import joblib

# Import internal modules
from tsgen.data.factory import create_datasource
from tsgen.models.factory import create_model
from tsgen.models.diffusion import DiffusionUtils
from tsgen.tracking.base import ConsoleTracker
from tsgen.train import train_model
from tsgen.config.schema import ExperimentConfig


def get_backtest_output_dir(config):
    """
    Get output directory for backtest results.

    Args:
        config: Configuration dictionary

    Returns:
        Path to output directory
    """
    if 'output_dir' in config:
        output_dir = config['output_dir']
    else:
        output_dir = "backtest_results"

    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def run_backtest_experiment():
    # --- Configuration ---
    # Hardcoded for this specific experiment logic
    SEQ_LEN = 256 
    N_PATHS = 1000
    
    base_config = {
        'experiment_name': "Backtest_Exp",
        'tracker': 'console',
        'data_source_type': 'database',  # Use database source (managed by findata project)
        'tickers': ['AAPL', 'MSFT', 'GOOG'],
        'start_date': '2015-01-01',
        'end_date': '2024-01-01',
        'sequence_length': SEQ_LEN,
        'batch_size': 32,
        'epochs': 50,
        'timesteps': 500,
        'learning_rate': 1e-3,
        'base_channels': 32,
        'dim': 32,
        'depth': 2,
        'heads': 4,
        'mlp_dim': 64,
        'dropout': 0.0
        # 'output_dir': 'experiments/0001_experiment/run_backtest'  # Can be set for experiment mode
    }

    # Get output directory for results
    output_dir = get_backtest_output_dir(base_config)
    print(f"Backtest outputs will be saved to: {output_dir}")

    # Define Split
    full_end = pd.to_datetime(base_config['end_date'])
    test_start = full_end - timedelta(days=365)
    train_end = test_start
    
    print(f"--- Experiment Setup ---")
    print(f"Train Period: {base_config['start_date']} -> {train_end.strftime('%Y-%m-%d')}")
    print(f"Test Period:  {test_start.strftime('%Y-%m-%d')} -> {base_config['end_date']}")
    
    # --- Data Preparation ---
    ds = create_datasource(base_config)
    df_full = ds.get_data()
    
    df_test = df_full[df_full.index >= test_start]
    if len(df_test) < SEQ_LEN:
        print(f"Warning: Test set length ({len(df_test)}) is smaller than sequence length ({SEQ_LEN}). Adjusting...")
    
    train_config = base_config.copy()
    train_config['end_date'] = train_end.strftime('%Y-%m-%d')
    
    models_to_test = ['gbm', 'bootstrap', 'unet', 'transformer']
    results = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Training Loop ---
    for model_type in models_to_test:
        print(f"\n>>> Processing Model: {model_type}")

        curr_config = train_config.copy()
        curr_config['model_type'] = model_type

        # Check if model already exists to skip training
        model_path = os.path.join(output_dir, f"{model_type}_model.pt")
        processor_path = os.path.join(output_dir, f"{model_type}_processor.pkl")

        if os.path.exists(model_path):
            print(f"Model {model_type} already trained. Skipping.")
            results[model_type] = {
                'model_path': model_path,
                'processor_path': processor_path
            }
            continue

        # Run Training
        tracker = ConsoleTracker()
        try:
            train_model(ExperimentConfig(**curr_config), tracker)
        except (ValueError, RuntimeError, KeyError) as e:
            print(f"Training failed for {model_type}: {type(e).__name__}: {e}")
            continue
        except Exception as e:
            print(f"Unexpected error training {model_type}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            continue

        # Rename Artifacts
        model_path = os.path.join(output_dir, f"{model_type}_model.pt")
        processor_path = os.path.join(output_dir, f"{model_type}_processor.pkl")

        if os.path.exists("diffusion_model_final.pt"):
            os.rename("diffusion_model_final.pt", model_path)
        if os.path.exists("processor.pkl"):
            os.rename("processor.pkl", processor_path)

        results[model_type] = {'model_path': model_path, 'processor_path': processor_path}

    # --- Generation Loop (All Tickers Jointly) ---
    print("\n>>> Starting Generation (Jointly)...")
    
    # Store generated returns: {model_type: np.array(N_PATHS, SEQ_LEN, N_TICKERS)}
    generated_data = {}
    tickers = base_config['tickers']
    n_tickers = len(tickers)

    for model_type, paths in results.items():
        print(f"Generating paths for {model_type}...")
        try:
            loaded_obj = torch.load(paths['model_path'], map_location=device)
            processor = joblib.load(paths['processor_path'])
            
            # Re-instantiate or use directly
            if isinstance(loaded_obj, dict) and not model_type in ['gbm', 'bootstrap']: 
                model = create_model({**train_config, 'model_type': model_type}).to(device)
                model.load_state_dict(loaded_obj)
            else:
                model = loaded_obj
            
            # Generate
            if hasattr(model, 'sample'):
                gen_seqs = model.sample(N_PATHS, SEQ_LEN).to(device)
            else:
                diff_utils = DiffusionUtils(T=train_config['timesteps'], device=device)
                gen_seqs = diff_utils.sample(model, image_size=(SEQ_LEN, n_tickers), batch_size=N_PATHS)
            
            # Inverse Transform using DataProcessor
            gen_seqs_np = gen_seqs.cpu().numpy()  # (N, Seq, Feat)

            # Use processor's inverse_transform with initial prices
            initial_prices = np.ones(n_tickers) * 100  # Normalized to 100
            gen_prices = processor.inverse_transform(gen_seqs_np, initial_prices)  # (N, Seq+1, Feat)

            # Convert back to returns for consistency
            gen_returns = np.diff(np.log(gen_prices), axis=1)  # (N, Seq, Feat)

            generated_data[model_type] = gen_returns

        except FileNotFoundError as e:
            print(f"Model artifacts not found for {model_type}: {e}")
        except (ValueError, RuntimeError) as e:
            print(f"Generation failed for {model_type}: {type(e).__name__}: {e}")
        except Exception as e:
            print(f"Unexpected error generating for {model_type}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

    # --- Plotting Loop ---
    print("\n>>> Creating Plots...")
    print(f"Test Data Points: {len(df_test)}")
    real_prices = df_test.iloc[:SEQ_LEN]
    
    for i, ticker in enumerate(tickers):
        plt.figure(figsize=(12, 6))
        
        # 1. Plot Real Data
        real_path = real_prices[ticker].values
        if len(real_path) > 0:
            norm_real = real_path / real_path[0] * 100
            plt.plot(norm_real, label='Real (Test Data)', color='black', linewidth=2, zorder=10)
        
        # 2. Plot Models
        for model_type, gen_returns in generated_data.items():
            # Specific asset returns
            asset_returns = gen_returns[:, :, i] # (N_Paths, Seq)
            
            # Convert to Price Paths (Normalized start=100)
            cumulative = np.cumsum(asset_returns, axis=1)
            cumulative = np.hstack([np.zeros((N_PATHS, 1)), cumulative])
            generated_prices = 100 * np.exp(cumulative[:, :-1]) # Match length to SEQ_LEN
            
            # Calculate Percentiles
            p05 = np.percentile(generated_prices, 5, axis=0)
            p50 = np.percentile(generated_prices, 50, axis=0)
            p95 = np.percentile(generated_prices, 95, axis=0)
            
            color = {'gbm': 'blue', 'bootstrap': 'green', 'unet': 'orange', 'transformer': 'purple'}.get(model_type, 'gray')
            
            plt.plot(p50, label=f'{model_type} Median', color=color, linestyle='--')
            plt.fill_between(range(len(p05)), p05, p95, color=color, alpha=0.15, label=f'{model_type} 90% CI')

        plt.title(f"{ticker}: 1-Year Forecast Validation (Train end: {train_end.date()})")
        plt.xlabel("Trading Days")
        plt.ylabel("Normalized Price")
        plt.legend()
        plt.grid(True, alpha=0.3)

        save_path = os.path.join(output_dir, f"plot_{ticker}.png")
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
        plt.close()


def main():
    """Entry point for backtest CLI command."""
    run_backtest_experiment()


if __name__ == "__main__":
    main()