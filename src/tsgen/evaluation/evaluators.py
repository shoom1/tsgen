"""
Composable evaluator classes for synthetic time series quality assessment.

Each evaluator implements a specific set of metrics and can be combined
in an EvaluationPipeline for comprehensive analysis.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn

from tsgen.analysis.metrics import (
    calculate_stylized_facts,
    compute_correlation_structure_metrics,
)
from tsgen.analysis.distribution_tests import run_all_distribution_tests
from tsgen.analysis.tstr import train_and_evaluate_tstr


class MetricEvaluator(ABC):
    """
    Abstract base class for evaluation metrics.

    Each evaluator computes a specific set of metrics comparing real
    and synthetic data. Evaluators can be composed into an EvaluationPipeline.

    Subclasses must implement:
        - name: Evaluator identifier
        - evaluate(): Compute metrics from real and synthetic data
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return evaluator name for logging and identification."""
        pass

    @abstractmethod
    def evaluate(
        self,
        real_data: np.ndarray,
        synthetic_data: np.ndarray,
        **kwargs
    ) -> Dict[str, float]:
        """
        Evaluate synthetic data quality against real data.

        Args:
            real_data: Real data array (N, Seq_Len, Features)
            synthetic_data: Synthetic data array (N, Seq_Len, Features)
            **kwargs: Additional parameters (e.g., tickers, device)

        Returns:
            Dictionary of metric names to values
        """
        pass


class StylizedFactsEvaluator(MetricEvaluator):
    """
    Evaluate financial stylized facts.

    Computes metrics for:
    - Fat tails (kurtosis)
    - Skewness
    - Volatility clustering (ACF of squared returns)
    - Value-at-Risk (VaR)
    - Expected Shortfall (ES)
    - Correlation matrix similarity
    """

    def __init__(self, lags: int = 20, alpha: float = 0.05):
        """
        Initialize stylized facts evaluator.

        Args:
            lags: Number of lags for ACF computation
            alpha: Confidence level for VaR/ES (default: 5%)
        """
        self.lags = lags
        self.alpha = alpha

    @property
    def name(self) -> str:
        return "stylized_facts"

    def evaluate(
        self,
        real_data: np.ndarray,
        synthetic_data: np.ndarray,
        **kwargs
    ) -> Dict[str, float]:
        """Compute stylized facts metrics."""
        sf_metrics = calculate_stylized_facts(real_data, synthetic_data)

        return {
            "kurtosis_diff_mean": float(np.mean(sf_metrics['kurtosis_diff'])),
            "skew_diff_mean": float(np.mean(sf_metrics['skew_diff'])),
            "acf_ret_diff_mse": float(sf_metrics['acf_ret_diff']),
            "acf_sq_ret_diff_mse": float(sf_metrics['acf_sq_ret_diff']),
            "corr_matrix_norm_diff": float(sf_metrics['corr_matrix_diff_norm']),
            "var_diff_mean": float(np.mean(sf_metrics['var_diff'])),
            "es_diff_mean": float(np.mean(sf_metrics['es_diff'])),
        }


class CorrelationEvaluator(MetricEvaluator):
    """
    Evaluate correlation structure preservation.

    Computes metrics for:
    - Correlation matrix Frobenius norm difference
    - Eigenvalue spectrum comparison
    - Rolling correlation stability
    """

    def __init__(self, window: int = 20):
        """
        Initialize correlation evaluator.

        Args:
            window: Window size for rolling correlation
        """
        self.window = window

    @property
    def name(self) -> str:
        return "correlation_structure"

    def evaluate(
        self,
        real_data: np.ndarray,
        synthetic_data: np.ndarray,
        **kwargs
    ) -> Dict[str, float]:
        """Compute correlation structure metrics."""
        corr_metrics = compute_correlation_structure_metrics(real_data, synthetic_data)

        results = {
            "corr_frobenius_norm": float(corr_metrics['corr_frobenius_norm']),
            "corr_max_diff": float(corr_metrics['corr_max_diff']),
            "corr_mean_diff": float(corr_metrics['corr_mean_diff']),
            "eigenvalue_mse": float(corr_metrics['eigenvalue_mse']),
            "eigenvalue_max_diff": float(corr_metrics['eigenvalue_max_diff']),
            "explained_var_ratio_diff": float(corr_metrics['explained_var_ratio_diff']),
        }

        # Add rolling correlation metrics if available
        if not np.isnan(corr_metrics.get('rolling_corr_stability', np.nan)):
            results['rolling_corr_stability'] = float(corr_metrics['rolling_corr_stability'])
            results['rolling_corr_std_diff'] = float(corr_metrics['rolling_corr_std_diff'])

        return results


class DistributionTestEvaluator(MetricEvaluator):
    """
    Evaluate distributional similarity using statistical tests.

    Computes:
    - Kolmogorov-Smirnov test
    - Cramér-von Mises test
    - Anderson-Darling test
    """

    @property
    def name(self) -> str:
        return "distribution_tests"

    def evaluate(
        self,
        real_data: np.ndarray,
        synthetic_data: np.ndarray,
        **kwargs
    ) -> Dict[str, float]:
        """Run distribution tests."""
        dist_results = run_all_distribution_tests(real_data, synthetic_data)

        metrics = {}
        for test_name, res in dist_results.items():
            short_name = test_name.replace(" ", "").replace("-", "")
            if 'statistic' in res:
                metrics[f"dist_{short_name}_stat"] = float(res['statistic'])
            if 'p_value' in res:
                metrics[f"dist_{short_name}_p"] = float(res['p_value'])

        return metrics


class DiscriminatorEvaluator(MetricEvaluator):
    """
    Evaluate synthetic data quality using discriminator accuracy.

    Trains an LSTM discriminator to classify real vs synthetic data.
    Target accuracy is 0.5 (can't distinguish = perfect generation).
    """

    def __init__(self, epochs: int = 20, hidden_dim: int = 64):
        """
        Initialize discriminator evaluator.

        Args:
            epochs: Training epochs for discriminator
            hidden_dim: LSTM hidden dimension
        """
        self.epochs = epochs
        self.hidden_dim = hidden_dim

    @property
    def name(self) -> str:
        return "discriminator"

    def evaluate(
        self,
        real_data: np.ndarray,
        synthetic_data: np.ndarray,
        device: str = 'cpu',
        **kwargs
    ) -> Dict[str, float]:
        """Train discriminator and compute accuracy."""
        accuracy = self._train_discriminator(real_data, synthetic_data, device)
        return {"discriminator_accuracy": float(accuracy)}

    def _train_discriminator(
        self,
        real_data: np.ndarray,
        fake_data: np.ndarray,
        device: str
    ) -> float:
        """Train LSTM discriminator and return accuracy."""
        real_labels = torch.ones(len(real_data), 1)
        fake_labels = torch.zeros(len(fake_data), 1)
        X = torch.cat([torch.FloatTensor(real_data), torch.FloatTensor(fake_data)])
        y = torch.cat([real_labels, fake_labels])

        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

        model = _Discriminator(input_dim=real_data.shape[2], hidden_dim=self.hidden_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.BCELoss()

        for _ in range(self.epochs):
            model.train()
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                preds = model(batch_x)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            all_preds = model(X.to(device))
            predicted_labels = (all_preds > 0.5).float().cpu()
            acc = (predicted_labels == y).float().mean().item()

        return acc


class _Discriminator(nn.Module):
    """LSTM discriminator for real vs synthetic classification."""

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])
        return self.sigmoid(out)


class TSTREvaluator(MetricEvaluator):
    """
    Train on Synthetic, Test on Real (TSTR) evaluation.

    Trains a model on synthetic data and evaluates on real data.
    Lower MSE indicates better synthetic data quality.
    """

    def __init__(self, epochs: int = 10):
        """
        Initialize TSTR evaluator.

        Args:
            epochs: Training epochs for TSTR model
        """
        self.epochs = epochs

    @property
    def name(self) -> str:
        return "tstr"

    def evaluate(
        self,
        real_data: np.ndarray,
        synthetic_data: np.ndarray,
        device: str = 'cpu',
        **kwargs
    ) -> Dict[str, float]:
        """Compute TSTR MSE."""
        tstr_mse = train_and_evaluate_tstr(
            synthetic_data, real_data, epochs=self.epochs, device=device
        )
        return {"tstr_mse": float(tstr_mse)}


class CompositeEvaluator(MetricEvaluator):
    """
    Compose multiple evaluators into a single evaluator.

    Runs all child evaluators and aggregates their metrics.
    """

    def __init__(self, evaluators: list):
        """
        Initialize composite evaluator.

        Args:
            evaluators: List of MetricEvaluator instances
        """
        self.evaluators = evaluators

    @property
    def name(self) -> str:
        return "composite"

    def evaluate(
        self,
        real_data: np.ndarray,
        synthetic_data: np.ndarray,
        **kwargs
    ) -> Dict[str, float]:
        """Run all evaluators and aggregate metrics."""
        all_metrics = {}
        for evaluator in self.evaluators:
            try:
                metrics = evaluator.evaluate(real_data, synthetic_data, **kwargs)
                all_metrics.update(metrics)
            except Exception as e:
                print(f"Warning: {evaluator.name} failed: {e}")
        return all_metrics


def create_default_evaluators(config: Optional[Dict[str, Any]] = None) -> list:
    """
    Create default set of evaluators based on config.

    Args:
        config: Optional configuration dictionary

    Returns:
        List of MetricEvaluator instances
    """
    config = config or {}
    eval_conf = config.get('evaluation', {})

    return [
        StylizedFactsEvaluator(
            lags=eval_conf.get('stylized_facts_lags', 20),
            alpha=eval_conf.get('var_alpha', 0.05)
        ),
        CorrelationEvaluator(window=20),
        DistributionTestEvaluator(),
        DiscriminatorEvaluator(
            epochs=eval_conf.get('discriminator_epochs', 20),
            hidden_dim=eval_conf.get('discriminator_hidden_dim', 64)
        ),
        TSTREvaluator(epochs=eval_conf.get('tstr_epochs', 10)),
    ]
