"""
Evaluation module for synthetic time series quality assessment.

Uses composable EvaluationPipeline with modular MetricEvaluator classes
for comprehensive analysis of generated time series.

Example:
    from tsgen import evaluate_model
    from tsgen.evaluation import EvaluationPipeline, StylizedFactsEvaluator

    # Use default evaluators
    result = evaluate_model(config, tracker)

    # Or create custom pipeline
    pipeline = EvaluationPipeline([
        StylizedFactsEvaluator(lags=30),
    ], tracker=tracker)
    result = evaluate_model(config, tracker, pipeline=pipeline)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import tempfile
from tsgen.models.registry import ModelRegistry
from tsgen.models.base_model import StatisticalModel
from tsgen.data.pipeline import load_prices, clean_data, create_windows
from tsgen.data.processor import DataProcessor
from tsgen.tracking.base import ExperimentTracker
from tsgen.evaluation import EvaluationPipeline, EvaluationResult
from tsgen.config.schema import ExperimentConfig


def evaluate_model(
    config: ExperimentConfig,
    tracker: ExperimentTracker,
    pipeline: EvaluationPipeline = None
) -> EvaluationResult:
    """
    Evaluate model using composable EvaluationPipeline.

    Loads a trained model and processor from tracker artifacts, generates
    synthetic samples, loads real data for comparison, and runs all
    configured evaluators (stylized facts, correlation, distribution tests,
    discriminator, TSTR).

    Args:
        config: ExperimentConfig instance
        tracker: Experiment tracker for logging
        pipeline: Optional custom EvaluationPipeline (uses default if None)

    Returns:
        EvaluationResult with metrics and plotting support

    Example:
        from tsgen.evaluation import StylizedFactsEvaluator, DiscriminatorEvaluator

        # Use default pipeline
        result = evaluate_model(config, tracker)

        # Or create custom pipeline
        pipeline = EvaluationPipeline([
            StylizedFactsEvaluator(lags=30),
            DiscriminatorEvaluator(epochs=30),
        ], tracker=tracker)
        result = evaluate_model(config, tracker, pipeline=pipeline)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Evaluating...")

    # Resolve config sections
    data_conf = config.get_data_config()
    tickers = data_conf.tickers

    # Model Factory
    model = ModelRegistry.create(config).to(device)

    try:
        # Get artifact paths via tracker
        model_path = tracker.get_artifact_path("model_final.pt", artifact_type='model')
        processor_path = tracker.get_artifact_path("processor.pkl", artifact_type='data')

        # Fallback for trackers that don't implement get_artifact_path
        if model_path is None:
            output_dir = getattr(config, 'output_dir', None)
            if output_dir:
                model_path = os.path.join(output_dir, "model_final.pt")
                processor_path = os.path.join(output_dir, "processor.pkl")
            else:
                model_path = "model_final.pt"
                processor_path = "processor.pkl"

        # isinstance check for model loading only (serialization concern:
        # StatisticalModel saves full object, DiffusionModel saves state_dict)
        if isinstance(model, StatisticalModel):
            model = torch.load(model_path, map_location=device)
        else:
            model.load_state_dict(torch.load(model_path, map_location=device))

        processor = DataProcessor.load(processor_path)

    except FileNotFoundError as e:
        print(f"Artifacts not found: {e}. Please run training first.")
        raise
    except (RuntimeError, KeyError) as e:
        print(f"Error loading model: {type(e).__name__}: {e}")
        print("Model architecture may have changed. Try retraining.")
        raise

    # Resolve diffusion config
    diff_conf = config.get_diffusion_config()
    timesteps = diff_conf.time_steps

    # Handle tickers recovery from processor
    if not tickers:
        if hasattr(processor, 'feature_names_in_'):
            tickers = list(processor.feature_names_in_)
        elif hasattr(processor, 'n_features_'):
            tickers = [f"Asset_{i}" for i in range(processor.n_features_)]

    features = len(tickers)

    # Generate synthetic samples
    eval_conf = config.get_evaluation_config()
    num_samples = eval_conf.num_samples
    print(f"Generating {num_samples} synthetic samples...")

    # Conditional Generation Setup (for class-conditioned models)
    model_params = config.get_model_params_config()
    num_classes = model_params.num_classes
    y_sampling = None
    if num_classes > 0:
        y_sampling = torch.randint(0, num_classes, (num_samples,), device=device).long()
        print(f"Generating samples conditioned on {num_classes} classes.")

    # Use unified generate() interface for all model types
    gen_seqs = model.generate(
        n_samples=num_samples,
        seq_len=data_conf.sequence_length,
        device=device,
        y=y_sampling
    )

    gen_seqs_np = gen_seqs.cpu().numpy()

    # Load and prepare real data
    df = load_prices(
        tickers,
        data_conf.start_date,
        data_conf.end_date,
        column=data_conf.column,
        db_path=data_conf.db_path
    )
    df_real = clean_data(df, strategy='ffill_drop')
    real_data_scaled = processor.transform(df_real)
    real_seqs_scaled = create_windows(
        real_data_scaled,
        sequence_length=data_conf.sequence_length
    )

    # Ensure we compare same amount of data
    limit = min(len(real_seqs_scaled), num_samples)
    real_sample = real_seqs_scaled[:limit]
    fake_sample = gen_seqs_np[:limit]

    # Create pipeline if not provided
    if pipeline is None:
        pipeline = EvaluationPipeline.from_config(config, tracker=tracker)

    # Run evaluation
    metrics = pipeline.run(
        real_sample,
        fake_sample,
        device=device,
        tickers=tickers
    )

    # Create result object
    result = EvaluationResult(
        metrics=metrics,
        real_data=real_sample,
        synthetic_data=fake_sample,
        tickers=tickers
    )

    # Generate and save plots
    with tempfile.TemporaryDirectory() as tmpdir:
        result.generate_plots(tmpdir, tracker=tracker)

        # Also generate price path comparison plot
        _generate_price_comparison_plot(
            gen_seqs_np, processor, features, tickers,
            df_real, data_conf, tmpdir, tracker
        )

    print(f"\nEvaluation complete.")
    print(result.summary())

    return result


def _generate_price_comparison_plot(
    gen_seqs_np, processor, features, tickers,
    df_real, data_conf, tmpdir, tracker
):
    """Generate price path comparison plot (helper function)."""
    subset_size = 5
    gen_subset = gen_seqs_np[:subset_size]
    initial_prices = np.ones(features) * 100
    generated_prices_subset = processor.inverse_transform(gen_subset, initial_prices)

    fig, axes = plt.subplots(features, 1, figsize=(12, 6 * features))
    if features == 1:
        axes = [axes]

    seq_len = data_conf.sequence_length

    for i, ticker in enumerate(tickers):
        ax = axes[i]
        real_price_section = df_real[ticker].iloc[-seq_len-1:].values
        if len(real_price_section) > 0:
            norm_real = real_price_section / real_price_section[0] * 100
            ax.plot(norm_real, label='Real', color='black')

            for j in range(subset_size):
                ax.plot(generated_prices_subset[j, :, i], alpha=0.6, linestyle='--')
            ax.set_title(f"{ticker}")

    plot_path = os.path.join(tmpdir, "synthetic_comparison.png")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    tracker.log_artifact(plot_path, artifact_type='plot')
