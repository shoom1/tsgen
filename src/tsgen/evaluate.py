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
from tsgen.data.pipeline import load_prices, clean_data, create_windows, split_temporal
from tsgen.data.processor import DataProcessor
from tsgen.tracking.base import ExperimentTracker
from tsgen.evaluation import EvaluationPipeline, EvaluationResult
from tsgen.config.schema import ExperimentConfig, TRAINING_CONFIG_MAP, BaselineTrainingConfig


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

    # Determine if this is a statistical (baseline) model before constructing
    is_baseline = TRAINING_CONFIG_MAP.get(config.model_type) is BaselineTrainingConfig

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

        # StatisticalModel saves full object; others save state_dict
        if is_baseline:
            model = torch.load(model_path, map_location=device, weights_only=False)
        else:
            model = ModelRegistry.create(config).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))

        processor = DataProcessor.load(processor_path)

    except FileNotFoundError as e:
        print(f"Artifacts not found: {e}. Please run training first.")
        raise
    except (RuntimeError, KeyError) as e:
        print(f"Error loading model: {type(e).__name__}: {e}")
        print("Model architecture may have changed. Try retraining.")
        raise

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
    model_conf = config.get_model_config()
    num_classes = getattr(model_conf, 'num_classes', 0)
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

    # Load and prepare real held-out data when the config defines a split.
    real_seqs_scaled, df_real, real_eval_meta = _prepare_real_evaluation_data(
        config, processor, tickers
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

    evaluation_meta_metrics = {
        "evaluation_heldout": float(real_eval_meta["heldout"]),
        "evaluation_real_windows": float(len(real_seqs_scaled)),
        "evaluation_real_rows": float(len(df_real)),
    }
    metrics.update(evaluation_meta_metrics)
    if tracker:
        tracker.log_metrics(evaluation_meta_metrics)

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


def _get_pipeline_step_params(config: ExperimentConfig, step_name: str):
    """Return params for the first matching data_pipeline step, if present."""
    for step in config.data_pipeline or []:
        if isinstance(step, dict) and step_name in step:
            return step.get(step_name) or {}
    return None


def _get_evaluation_train_ratio(config: ExperimentConfig):
    """Resolve the chronological split ratio used for held-out evaluation."""
    split_params = _get_pipeline_step_params(config, "split_temporal")
    if split_params is not None:
        return split_params.get("train_ratio", 0.8)
    return config.get_data_config().train_test_split


def _get_clean_data_params(config: ExperimentConfig):
    """Use the configured cleaning step for evaluation, defaulting to ffill/drop."""
    return _get_pipeline_step_params(config, "clean_data") or {"strategy": "ffill_drop"}


def _prepare_real_evaluation_data(config: ExperimentConfig, processor, tickers):
    """Load and transform real data for evaluation.

    If the config declares a chronological split, only the held-out test
    portion is used. For mask-cleaned universes, evaluation keeps rows where
    every requested feature is valid so zero-filled pre-IPO placeholders do
    not enter quality metrics.
    """
    data_conf = config.get_data_config()
    df = load_prices(
        tickers,
        data_conf.start_date,
        data_conf.end_date,
        column=data_conf.column,
        db_path=data_conf.db_path,
    )

    clean_params = _get_clean_data_params(config)
    cleaned = clean_data(df, **clean_params)
    eval_mask = None
    if isinstance(cleaned, tuple):
        df_clean, mask = cleaned
    else:
        df_clean, mask = cleaned, None

    train_ratio = _get_evaluation_train_ratio(config)
    heldout = train_ratio is not None
    if heldout:
        train_df, eval_df = split_temporal(df_clean, train_ratio=train_ratio)
        if mask is not None:
            eval_mask = mask.iloc[len(train_df):]
    else:
        eval_df = df_clean
        eval_mask = mask

    if eval_mask is not None:
        all_valid_rows = eval_mask.astype(bool).all(axis=1)
        eval_df = eval_df.loc[all_valid_rows]

        if eval_df.empty:
            raise ValueError(
                "Held-out evaluation data has no rows where every requested "
                "ticker is valid. Use a later evaluation window, fewer tickers, "
                "or a mask-aware evaluator."
            )

    real_data_scaled = processor.transform(eval_df)
    real_seqs_scaled = create_windows(
        real_data_scaled,
        sequence_length=data_conf.sequence_length,
    )
    return real_seqs_scaled, eval_df, {"heldout": heldout}


def _render_comparison_page(
    feature_idx, tickers, df_real, gen_prices, num_synthetic,
    seq_len, page_num, total_pages, panels_per_page,
):
    """Render one page of the comparison plot and return the Figure."""
    n_panels = len(feature_idx)
    ncols = min(3, max(1, n_panels))
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 3.2 * nrows),
        sharex=True, squeeze=False,
    )
    axes = axes.flatten()

    for panel_i, feature_i in enumerate(feature_idx):
        ax = axes[panel_i]
        ticker = tickers[feature_i]
        if ticker not in df_real.columns:
            ax.set_visible(False)
            continue
        real_section = df_real[ticker].iloc[-seq_len - 1:].values
        if len(real_section) == 0 or real_section[0] == 0:
            ax.set_visible(False)
            continue
        norm_real = real_section / real_section[0] * 100
        ax.plot(norm_real, color='black', linewidth=2.0, label='Real', zorder=3)
        syn_paths = gen_prices[:, :, feature_i]
        for j in range(num_synthetic):
            ax.plot(
                syn_paths[j],
                color='tab:blue', alpha=0.45, linestyle='--', linewidth=1.1,
                label='Synthetic' if j == 0 else None,
            )
        ax.set_title(ticker, fontsize=10)
        ax.grid(alpha=0.3)

        # Clamp y-axis around the real trajectory. Never-trained tickers
        # (e.g. post-2020 IPOs with no pre-IPO data in the training window)
        # can produce 1e+50 synthetic prices that otherwise collapse the
        # real line to a flat 0 on a shared axis. Synthetic values outside
        # the clamp clip off-screen, which is what you want — they're junk.
        real_lo, real_hi = float(np.min(norm_real)), float(np.max(norm_real))
        real_span = max(real_hi - real_lo, 5.0)
        ax.set_ylim(real_lo - 2.0 * real_span, real_hi + 2.0 * real_span)

    for k in range(n_panels, len(axes)):
        axes[k].set_visible(False)

    fig.supxlabel('Trading day within window', fontsize=11)
    fig.supylabel('Price (normalized to 100 at start)', fontsize=11)
    suptitle = f'Real vs {num_synthetic} synthetic price paths'
    if total_pages > 1:
        suptitle += f' — page {page_num}/{total_pages}'
    fig.suptitle(suptitle, fontsize=13)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, loc='upper left', fontsize=9, frameon=True)

    fig.tight_layout()
    return fig


def _generate_price_comparison_plot(
    gen_seqs_np, processor, features, tickers,
    df_real, data_conf, tmpdir, tracker,
    panels_per_page: int = 9,
    num_synthetic: int = 5,
):
    """Real-vs-synthetic price path comparison.

    Writes two artifacts:

    - ``synthetic_comparison.png`` — first ``panels_per_page`` tickers only.
      Quick preview, embeddable in notes/docs.
    - ``synthetic_comparison.pdf`` — all tickers, paginated (``panels_per_page``
      tickers per page, 3 columns). Prior single-page versions clipped 90% of
      a 100-ticker universe; this surfaces every series without producing an
      unrenderable 600-inch PNG.
    """
    from matplotlib.backends.backend_pdf import PdfPages

    num_synthetic = min(num_synthetic, len(gen_seqs_np))
    gen_subset = gen_seqs_np[:num_synthetic]
    initial_prices = np.ones(features) * 100
    gen_prices = processor.inverse_transform(gen_subset, initial_prices)
    seq_len = data_conf.sequence_length

    all_idx = np.arange(features)
    pages = [all_idx[i:i + panels_per_page]
             for i in range(0, features, panels_per_page)]
    total_pages = len(pages)

    # Page 1 → PNG (quick preview)
    png_path = os.path.join(tmpdir, "synthetic_comparison.png")
    fig1 = _render_comparison_page(
        pages[0], tickers, df_real, gen_prices, num_synthetic,
        seq_len, page_num=1, total_pages=total_pages,
        panels_per_page=panels_per_page,
    )
    fig1.savefig(png_path, dpi=120, bbox_inches='tight')
    plt.close(fig1)
    tracker.log_artifact(png_path, artifact_type='plot')

    # All pages → PDF (only worth producing if there's more than one page)
    if total_pages > 1:
        pdf_path = os.path.join(tmpdir, "synthetic_comparison.pdf")
        with PdfPages(pdf_path) as pdf:
            for page_num, feature_idx in enumerate(pages, start=1):
                fig = _render_comparison_page(
                    feature_idx, tickers, df_real, gen_prices, num_synthetic,
                    seq_len, page_num=page_num, total_pages=total_pages,
                    panels_per_page=panels_per_page,
                )
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
        tracker.log_artifact(pdf_path, artifact_type='plot')
