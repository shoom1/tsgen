"""
Evaluation module for synthetic time series quality assessment.

This module provides composable evaluators and an evaluation pipeline
for comprehensive analysis of generated time series.

Example usage:
    from tsgen.evaluation import EvaluationPipeline, StylizedFactsEvaluator

    # Create custom pipeline
    pipeline = EvaluationPipeline([
        StylizedFactsEvaluator(),
        DiscriminatorEvaluator(),
    ], tracker=tracker)

    # Run evaluation
    results = pipeline.run(real_data, synthetic_data, device='cuda')

    # Or use default evaluators from config
    pipeline = EvaluationPipeline.from_config(config, tracker=tracker)
    results = pipeline.run(real_data, synthetic_data)
"""

from tsgen.evaluation.evaluators import (
    MetricEvaluator,
    StylizedFactsEvaluator,
    CorrelationEvaluator,
    DistributionTestEvaluator,
    DiscriminatorEvaluator,
    TSTREvaluator,
    create_default_evaluators,
)
from tsgen.evaluation.pipeline import (
    EvaluationPipeline,
    EvaluationResult,
)

__all__ = [
    # Base class
    'MetricEvaluator',
    # Evaluators
    'StylizedFactsEvaluator',
    'CorrelationEvaluator',
    'DistributionTestEvaluator',
    'DiscriminatorEvaluator',
    'TSTREvaluator',
    # Factory
    'create_default_evaluators',
    # Pipeline
    'EvaluationPipeline',
    'EvaluationResult',
]
