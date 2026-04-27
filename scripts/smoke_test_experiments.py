#!/usr/bin/env python3
"""Smoke-test every experiment config at a small training budget.

Runs each experiment via the installed ``tsgen`` CLI with ``--override``
arguments that slash the training cost (epochs, sample counts, DDIM sampling
with 5 inference steps) and route artifacts to ``smoke_test/<name>/``. The
goal is not to produce meaningful models — only to verify every config wires
through the pipeline end-to-end on the current codebase.

Run from repo root:

    conda run -n tsgen python scripts/smoke_test_experiments.py

Prints a per-experiment PASS/FAIL summary with wall-clock timings. Full logs
for each run land in ``smoke_test/<name>/training_*.log``.
"""

from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


# Per-experiment override sets. Keys are paradigm buckets; values apply to
# every config in that bucket. Individual experiments may also have
# experiment-specific overrides.

DIFFUSION_OVERRIDES = [
    "training.epochs=3",
    "training.validation_interval=0",
    "training.sampling_method=ddim",
    "training.num_inference_steps=5",
]

VAE_OVERRIDES = [
    "training.epochs=3",
    "training.annealing_epochs=1",
]

BASELINE_OVERRIDES = [
    # epochs already 1; nothing to reduce
]

SHARED_OVERRIDES = [
    "evaluation.num_samples=50",
    "evaluation.discriminator_epochs=3",
    "evaluation.tstr_epochs=2",
]


@dataclass
class ExperimentSpec:
    name: str
    config_path: str
    paradigm: str  # 'diffusion' | 'vae' | 'baseline'


EXPERIMENTS = [
    ExperimentSpec("0001_timevae", "experiments/0001_timevae_all_stocks/config.yaml", "vae"),
    ExperimentSpec("0002_unet", "experiments/0002_unet_all_stocks/config.yaml", "diffusion"),
    ExperimentSpec("0003_unet_fix", "experiments/0003_unet_data_fix/config.yaml", "diffusion"),
    ExperimentSpec("0004_transformer", "experiments/0004_transformer_all_stocks/config.yaml", "diffusion"),
    ExperimentSpec("0005_multivariate_gaussian", "experiments/0005_multivariate_gaussian/config.yaml", "baseline"),
    ExperimentSpec("0006_mamba", "experiments/0006_mamba_default/config.yaml", "diffusion"),
    ExperimentSpec("0007_ccc_garch", "experiments/0007_ccc_garch/config.yaml", "baseline"),
    ExperimentSpec("0008_diffwave", "experiments/0008_diffwave/config.yaml", "diffusion"),
    ExperimentSpec("0009_dit", "experiments/0009_dit/config.yaml", "diffusion"),
    ExperimentSpec("0010_bootstrap", "experiments/0010_bootstrap/config.yaml", "baseline"),
]


def overrides_for(paradigm: str) -> list[str]:
    base = list(SHARED_OVERRIDES)
    if paradigm == "diffusion":
        return base + DIFFUSION_OVERRIDES
    if paradigm == "vae":
        return base + VAE_OVERRIDES
    if paradigm == "baseline":
        return base + BASELINE_OVERRIDES
    raise ValueError(f"Unknown paradigm {paradigm!r}")


def run_one(exp: ExperimentSpec) -> tuple[bool, float, str]:
    """Run a single smoke test. Returns (ok, seconds, tail_of_output)."""
    out_dir = f"smoke_test/{exp.name}"
    Path(ROOT / out_dir).mkdir(parents=True, exist_ok=True)

    overrides = overrides_for(exp.paradigm)
    overrides.append(f"output_dir={out_dir}")

    cmd = ["tsgen", "--config", exp.config_path, "--mode", "train_eval"]
    for ov in overrides:
        cmd += ["--override", ov]

    log_path = ROOT / out_dir / "smoke.log"
    t0 = time.perf_counter()
    with open(log_path, "w") as lf:
        result = subprocess.run(
            cmd, cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT,
            text=True,
        )
    elapsed = time.perf_counter() - t0
    ok = result.returncode == 0

    # Grab the last bit of the log for the summary
    tail = ""
    if log_path.exists():
        tail_lines = log_path.read_text().splitlines()[-6:]
        tail = "\n    ".join(tail_lines)

    return ok, elapsed, tail


def main() -> int:
    results = []
    overall_t0 = time.perf_counter()

    for exp in EXPERIMENTS:
        print(f"[{exp.name}] running... ", end="", flush=True)
        ok, seconds, tail = run_one(exp)
        flag = "PASS" if ok else "FAIL"
        print(f"{flag} ({seconds:.1f}s)")
        if not ok:
            print(f"    tail: {tail}")
        results.append((exp, ok, seconds))

    overall_elapsed = time.perf_counter() - overall_t0

    print()
    print("=" * 70)
    print(f"Smoke-test summary ({overall_elapsed:.1f}s total)")
    print("=" * 70)
    for exp, ok, seconds in results:
        flag = "PASS" if ok else "FAIL"
        print(f"  {flag}  {exp.name:<30s} {seconds:>6.1f}s   log: smoke_test/{exp.name}/smoke.log")

    n_fail = sum(1 for _, ok, _ in results if not ok)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
