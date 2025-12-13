import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import tempfile
from tsgen.models.factory import create_model
from tsgen.models.diffusion import DiffusionUtils
from tsgen.data.pipeline import load_prices, clean_data, process_prices, create_windows
from tsgen.data.processor import DataProcessor
from tsgen.tracking.base import ExperimentTracker
from tsgen.analysis.metrics import (
    calculate_stylized_facts,
    plot_stylized_facts,
    compute_correlation_structure_metrics,
    plot_correlation_structure
)
from tsgen.analysis.tstr import train_and_evaluate_tstr
from tsgen.analysis.distribution_tests import run_all_distribution_tests


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])
        return self.sigmoid(out)

def train_discriminator(real_data, fake_data, device, epochs=20):
    real_labels = torch.ones(len(real_data), 1)
    fake_labels = torch.zeros(len(fake_data), 1)
    X = torch.cat([torch.FloatTensor(real_data), torch.FloatTensor(fake_data)])
    y = torch.cat([real_labels, fake_labels])
    
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = Discriminator(input_dim=real_data.shape[2]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()
    
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            
    # Eval
    model.eval()
    with torch.no_grad():
        all_preds = model(X.to(device))
        predicted_labels = (all_preds > 0.5).float().cpu()
        acc = (predicted_labels == y).float().mean().item()
        
    return acc

def evaluate_model(config, tracker: ExperimentTracker):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Evaluating...")
    
    features = len(config['tickers']) 
    
    # Model Factory
    model = create_model(config).to(device)
    
    try:
        # Get artifact paths via tracker (with fallback for non-file trackers)
        model_path = tracker.get_artifact_path("model_final.pt", artifact_type='model')
        processor_path = tracker.get_artifact_path("processor.pkl", artifact_type='data')

        # Fallback for trackers that don't implement get_artifact_path
        if model_path is None:
            # Try common default locations
            if 'output_dir' in config:
                model_path = os.path.join(config['output_dir'], "model_final.pt")
                processor_path = os.path.join(config['output_dir'], "processor.pkl")
            else:
                model_path = "model_final.pt"
                processor_path = "processor.pkl"

        if hasattr(model, 'fit'):
            # Load full model for baselines
            model = torch.load(model_path, map_location=device)
        else:
            # Load state dict for diffusion models
            model.load_state_dict(torch.load(model_path, map_location=device))

        # Load Processor
        processor = DataProcessor.load(processor_path)

    except FileNotFoundError as e:
        print(f"Artifacts not found: {e}. Please run training first.")
        return
    except (RuntimeError, KeyError) as e:
        print(f"Error loading model: {type(e).__name__}: {e}")
        print("Model architecture may have changed. Try retraining.")
        return

    # 1. Generate Data
    num_samples = 500
    print(f"Generating {num_samples} synthetic samples for analysis...")
    
    # Determine sampling method
    sampling_method = config.get('sampling_method', 'ddpm').lower()
    ddim_steps = config.get('ddim_steps', 50)
    
    # Conditional Generation Setup (for inference)
    num_classes = config.get('num_classes', 0)
    y_sampling = None
    if num_classes > 0:
        # For evaluation, we can generate samples for a specific class or random classes
        # Here, we generate random classes for each sample
        y_sampling = torch.randint(0, num_classes, (num_samples,), device=device).long()
        print(f"Generating samples conditioned on {num_classes} classes.")

    if hasattr(model, 'sample'):
        # Baseline Model Generation
        gen_seqs = model.sample(num_samples, config['sequence_length']).to(device)
    else:
        # Diffusion Model Generation
        diff_utils = DiffusionUtils(T=config['timesteps'], device=device)
        if sampling_method == 'ddim':
            print(f"Using DDIM sampling with {ddim_steps} steps.")
            gen_seqs = diff_utils.ddim_sample(model, image_size=(config['sequence_length'], features), batch_size=num_samples, num_inference_steps=ddim_steps, y=y_sampling)
        else:
            print("Using DDPM sampling.")
            gen_seqs = diff_utils.sample(model, image_size=(config['sequence_length'], features), batch_size=num_samples, y=y_sampling)
        
    gen_seqs_np = gen_seqs.cpu().numpy() # (N, Seq, Feat) - Scaled returns
    
    # 2. Real Data Preparation - Use pipeline
    df = load_prices(
        config['tickers'],
        config['start_date'],
        config['end_date'],
        column=config.get('column', 'adj_close'),
        db_path=config.get('db_path')
    )

    df_real = clean_data(df, strategy='ffill_drop')

    # Use processor to transform
    real_data_scaled = processor.transform(df_real)

    # Create windows using pipeline
    real_seqs_scaled = create_windows(real_data_scaled, sequence_length=config['sequence_length'])
    
    # Ensure we compare same amount of data if possible
    limit = min(len(real_seqs_scaled), num_samples)
    real_sample = real_seqs_scaled[:limit]
    fake_sample = gen_seqs_np[:limit]
    
    # --- Advanced Metrics ---
    print("Calculating Stylized Facts...")
    sf_metrics = calculate_stylized_facts(real_sample, fake_sample)
    
    # Log Metrics
    tracker.log_metrics({
        "kurtosis_diff_mean": np.mean(sf_metrics['kurtosis_diff']),
        "skew_diff_mean": np.mean(sf_metrics['skew_diff']),
        "acf_ret_diff_mse": sf_metrics['acf_ret_diff'],
        "acf_sq_ret_diff_mse": sf_metrics['acf_sq_ret_diff'],
        "corr_matrix_norm_diff": sf_metrics['corr_matrix_diff_norm'],
        "var_diff_mean": np.mean(sf_metrics['var_diff']),
        "es_diff_mean": np.mean(sf_metrics['es_diff'])
    })

    # --- Correlation Structure Analysis ---
    print("Analyzing Correlation Structure...")
    corr_metrics = compute_correlation_structure_metrics(real_sample, fake_sample)

    # Log correlation metrics
    correlation_log_metrics = {
        "corr_frobenius_norm": corr_metrics['corr_frobenius_norm'],
        "corr_max_diff": corr_metrics['corr_max_diff'],
        "corr_mean_diff": corr_metrics['corr_mean_diff'],
        "eigenvalue_mse": corr_metrics['eigenvalue_mse'],
        "eigenvalue_max_diff": corr_metrics['eigenvalue_max_diff'],
        "explained_var_ratio_diff": corr_metrics['explained_var_ratio_diff']
    }

    # Add rolling correlation metrics if available
    if not np.isnan(corr_metrics.get('rolling_corr_stability', np.nan)):
        correlation_log_metrics['rolling_corr_stability'] = corr_metrics['rolling_corr_stability']
        correlation_log_metrics['rolling_corr_std_diff'] = corr_metrics['rolling_corr_std_diff']

    tracker.log_metrics(correlation_log_metrics)

    print(f"Correlation Structure Metrics:")
    print(f"  Frobenius Norm: {corr_metrics['corr_frobenius_norm']:.4f}")
    print(f"  Max Difference: {corr_metrics['corr_max_diff']:.4f}")
    print(f"  Mean Difference: {corr_metrics['corr_mean_diff']:.4f}")
    print(f"  Eigenvalue MSE: {corr_metrics['eigenvalue_mse']:.4f}")
    if not np.isnan(corr_metrics.get('rolling_corr_stability', np.nan)):
        print(f"  Rolling Correlation Stability: {corr_metrics['rolling_corr_stability']:.4f}")

    # --- Distribution Tests ---
    print("Running Distribution Tests (KS, CvM, AD)...")
    dist_results = run_all_distribution_tests(real_sample, fake_sample)

    dist_metrics = {}
    for test_name, res in dist_results.items():
        short_name = test_name.replace(" ", "").replace("-", "")
        if 'statistic' in res:
            dist_metrics[f"dist_{short_name}_stat"] = res['statistic']
        if 'p_value' in res:
            dist_metrics[f"dist_{short_name}_p"] = res['p_value']

    tracker.log_metrics(dist_metrics)
    print(f"Distribution Test Results: {dist_metrics}")

    # TSTR
    print("Calculating TSTR Score (Train Synthetic, Test Real)...")
    tstr_mse = train_and_evaluate_tstr(fake_sample, real_sample, epochs=10, device=device)
    print(f"TSTR MSE: {tstr_mse:.6f}")
    tracker.log_metrics({"tstr_mse": tstr_mse})

    # --- Generate and save all plots using tempfile ---
    with tempfile.TemporaryDirectory() as tmpdir:
        # Plot correlation structure
        corr_plot_path = os.path.join(tmpdir, "correlation_structure.png")
        plot_correlation_structure(corr_metrics, config['tickers'], save_path=corr_plot_path)
        tracker.log_artifact(corr_plot_path, artifact_type='plot')

        # Plot Stylized Facts
        sf_plot_path = os.path.join(tmpdir, "stylized_facts.png")
        plot_stylized_facts(sf_metrics, config['tickers'], save_path=sf_plot_path)
        tracker.log_artifact(sf_plot_path, artifact_type='plot')

        # --- Visualizations (Price Paths) ---
        # Inverse transform a subset using the processor
        subset_size = 5
        gen_subset = gen_seqs_np[:subset_size]  # (5, Seq, Feat)

        # Use processor inverse_transform
        # Initial price 100 for all assets
        initial_prices = np.ones(features) * 100
        # prices: (5, Seq+1, Feat)
        generated_prices_subset = processor.inverse_transform(gen_subset, initial_prices)

        fig, axes = plt.subplots(features, 1, figsize=(12, 6 * features))
        if features == 1:
            axes = [axes]

        for i, ticker in enumerate(config['tickers']):
            ax = axes[i]
            real_price_section = df_real[ticker].iloc[-config['sequence_length']-1:].values
            if len(real_price_section) > 0:
                norm_real = real_price_section / real_price_section[0] * 100
                ax.plot(norm_real, label='Real', color='black')

                for j in range(subset_size):
                    # Plot generated path. Note generated_prices_subset has length Seq+1
                    # generated_prices_subset[j, :, i]
                    ax.plot(generated_prices_subset[j, :, i], alpha=0.6, linestyle='--')
                ax.set_title(f"{ticker}")

        plot_path = os.path.join(tmpdir, "synthetic_comparison.png")
        plt.tight_layout()
        plt.savefig(plot_path)
        tracker.log_artifact(plot_path, artifact_type='plot')
        print(f"Saved synthetic comparison plot")

    # Discriminator
    score = train_discriminator(real_sample, fake_sample, device)
    print(f"Discriminator Accuracy: {score:.4f}")
    tracker.log_metrics({"discriminator_accuracy": score})

    # Return metrics for testing/validation
    return {"discriminator_accuracy": score}
