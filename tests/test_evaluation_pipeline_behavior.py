import numpy as np
import pytest

from tsgen.evaluation.evaluators import DiscriminatorEvaluator, MetricEvaluator
from tsgen.evaluation.pipeline import EvaluationPipeline


def test_discriminator_reports_heldout_scores():
    rng = np.random.default_rng(0)
    real = rng.normal(0.0, 1.0, size=(20, 8, 2))
    synthetic = rng.normal(1.0, 1.0, size=(20, 8, 2))

    metrics = DiscriminatorEvaluator(
        epochs=2,
        hidden_dim=8,
        random_state=123,
    ).evaluate(real, synthetic, device="cpu")

    assert 0.0 <= metrics["discriminator_accuracy"] <= 1.0
    assert 0.0 <= metrics["discriminator_train_accuracy"] <= 1.0
    assert 0.0 <= metrics["discriminator_auc"] <= 1.0


class _FailingEvaluator(MetricEvaluator):
    @property
    def name(self):
        return "failing"

    def evaluate(self, real_data, synthetic_data, **kwargs):
        raise RuntimeError("intentional failure")


def test_pipeline_marks_failed_evaluators():
    data = np.zeros((4, 5, 1))
    pipeline = EvaluationPipeline([_FailingEvaluator()], verbose=False)

    metrics = pipeline.run(data, data)

    assert metrics["evaluation_failed_count"] == 1.0
    assert metrics["evaluation_failing_failed"] == 1.0
    assert metrics["_failed_evaluators"] == {"failing": "intentional failure"}

