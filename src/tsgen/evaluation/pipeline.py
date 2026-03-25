"""
Evaluation pipeline for orchestrating multiple evaluators.

Provides a composable pipeline that runs multiple evaluators,
logs metrics, and generates visualization artifacts.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import os
import tempfile

from tsgen.tracking.base import ExperimentTracker
from tsgen.evaluation.evaluators import MetricEvaluator, StylizedFactsEvaluator, CorrelationEvaluator, create_default_evaluators
from tsgen.analysis.metrics import plot_stylized_facts, plot_correlation_structure


class EvaluationPipeline:
    """
    Orchestrates evaluation of synthetic time series quality.

    Runs multiple evaluators in sequence, logs metrics to tracker,
    and optionally generates visualization artifacts.

    Example:
        pipeline = EvaluationPipeline(
            evaluators=[
                StylizedFactsEvaluator(),
                DiscriminatorEvaluator(),
            ],
            tracker=tracker
        )
        results = pipeline.run(real_data, synthetic_data)
    """

    def __init__(
        self,
        evaluators: List[MetricEvaluator],
        tracker: Optional[ExperimentTracker] = None,
        verbose: bool = True
    ):
        """
        Initialize evaluation pipeline.

        Args:
            evaluators: List of MetricEvaluator instances to run
            tracker: Optional experiment tracker for logging
            verbose: Whether to print progress messages
        """
        self.evaluators = evaluators
        self.tracker = tracker
        self.verbose = verbose

    def run(
        self,
        real_data: np.ndarray,
        synthetic_data: np.ndarray,
        device: str = 'cpu',
        tickers: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run all evaluators and aggregate metrics.

        Args:
            real_data: Real data array (N, Seq_Len, Features)
            synthetic_data: Synthetic data array (N, Seq_Len, Features)
            device: Device for GPU-based evaluators
            tickers: List of ticker symbols for plotting
            **kwargs: Additional parameters passed to evaluators

        Returns:
            Dictionary of all metric names to values.
            If any evaluators failed, includes '_failed_evaluators' key
            mapping evaluator names to error messages.
        """
        all_metrics = {}
        failed = {}
        raw_results = {}

        for evaluator in self.evaluators:
            if self.verbose:
                print(f"Running {evaluator.name} evaluator...")

            try:
                metrics = evaluator.evaluate(
                    real_data,
                    synthetic_data,
                    device=device,
                    tickers=tickers,
                    **kwargs
                )
                all_metrics.update(metrics)

                # Collect raw results from evaluators that cache them
                if hasattr(evaluator, 'last_raw_results'):
                    raw_results[evaluator.name] = evaluator.last_raw_results

                # Log metrics if tracker available
                if self.tracker:
                    self.tracker.log_metrics(metrics)

                if self.verbose:
                    self._print_metrics(evaluator.name, metrics)

            except Exception as e:
                failed[evaluator.name] = str(e)
                if self.verbose:
                    print(f"  WARNING: {evaluator.name} failed: {e}")

        if failed:
            all_metrics['_failed_evaluators'] = failed
        all_metrics['_raw_results'] = raw_results

        return all_metrics

    def _print_metrics(self, evaluator_name: str, metrics: Dict[str, float]):
        """Print metrics in readable format."""
        print(f"  {evaluator_name} metrics:")
        for name, value in metrics.items():
            if isinstance(value, float):
                print(f"    {name}: {value:.6f}")
            else:
                print(f"    {name}: {value}")

    @classmethod
    def from_config(
        cls,
        config,
        tracker: Optional[ExperimentTracker] = None
    ) -> 'EvaluationPipeline':
        """
        Create pipeline from ExperimentConfig or configuration dictionary.

        Args:
            config: ExperimentConfig or configuration dictionary
            tracker: Optional experiment tracker

        Returns:
            Configured EvaluationPipeline instance
        """
        evaluators = create_default_evaluators(config)
        return cls(evaluators=evaluators, tracker=tracker)


class EvaluationResult:
    """
    Container for evaluation results with visualization support.

    Stores metrics and provides methods for generating plots
    and saving artifacts.
    """

    def __init__(
        self,
        metrics: Dict[str, float],
        real_data: np.ndarray,
        synthetic_data: np.ndarray,
        tickers: List[str]
    ):
        """
        Initialize evaluation result.

        Args:
            metrics: Dictionary of metric values
            real_data: Real data used in evaluation
            synthetic_data: Synthetic data used in evaluation
            tickers: List of ticker symbols
        """
        self.metrics = metrics
        self.real_data = real_data
        self.synthetic_data = synthetic_data
        self.tickers = tickers

        # Seed caches from evaluator raw results (avoids recomputation in generate_plots)
        raw = metrics.get('_raw_results', {})
        self._stylized_facts = raw.get('stylized_facts')
        self._correlation_metrics = raw.get('correlation_structure')

    def generate_plots(
        self,
        output_dir: str,
        tracker: Optional[ExperimentTracker] = None
    ):
        """
        Generate visualization plots and save as artifacts.

        Args:
            output_dir: Directory to save plots
            tracker: Optional tracker to log artifacts
        """
        from tsgen.analysis.metrics import (
            calculate_stylized_facts,
            compute_correlation_structure_metrics,
        )

        os.makedirs(output_dir, exist_ok=True)

        # Calculate metrics for plotting if not cached
        if self._stylized_facts is None:
            self._stylized_facts = calculate_stylized_facts(
                self.real_data, self.synthetic_data
            )

        if self._correlation_metrics is None:
            self._correlation_metrics = compute_correlation_structure_metrics(
                self.real_data, self.synthetic_data
            )

        # Generate stylized facts plot
        sf_plot_path = os.path.join(output_dir, "stylized_facts.png")
        plot_stylized_facts(self._stylized_facts, self.tickers, save_path=sf_plot_path)
        if tracker:
            tracker.log_artifact(sf_plot_path, artifact_type='plot')

        # Generate correlation structure plot
        corr_plot_path = os.path.join(output_dir, "correlation_structure.png")
        plot_correlation_structure(
            self._correlation_metrics, self.tickers, save_path=corr_plot_path
        )
        if tracker:
            tracker.log_artifact(corr_plot_path, artifact_type='plot')

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'metrics': self.metrics,
            'tickers': self.tickers,
            'real_data_shape': self.real_data.shape,
            'synthetic_data_shape': self.synthetic_data.shape,
        }

    def summary(self) -> str:
        """Generate human-readable summary of results."""
        lines = ["Evaluation Summary", "=" * 40]

        # Key metrics
        key_metrics = [
            ('discriminator_accuracy', 'Discriminator Accuracy (target: 0.5)'),
            ('tstr_mse', 'TSTR MSE'),
            ('kurtosis_diff_mean', 'Kurtosis Diff'),
            ('corr_frobenius_norm', 'Correlation Frobenius Norm'),
        ]

        for metric_key, label in key_metrics:
            if metric_key in self.metrics:
                value = self.metrics[metric_key]
                lines.append(f"{label}: {value:.6f}")

        return "\n".join(lines)
