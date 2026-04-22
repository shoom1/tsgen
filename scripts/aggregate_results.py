#!/usr/bin/env python3
"""Aggregate evaluation metrics across experiment runs into a comparison table.

For each experiment directory containing a ``metrics.jsonl`` (FileTracker
output), extract the final values of the evaluation metrics and render a
side-by-side table. Works on any tree of run folders — pass one or more
root paths on the command line.

Usage:
    # Local runs (baselines)
    scripts/aggregate_results.py experiments/

    # Colab runs on Drive (after syncing down from Drive)
    scripts/aggregate_results.py /path/to/drive/tsgen/runs/

    # Combined: mix local + drive
    scripts/aggregate_results.py experiments/ /path/to/drive/runs/

Outputs:
  - A comparison table printed to stdout (Markdown).
  - <first_root>/summary.csv with all metrics as columns.
  - <first_root>/summary.md with the rendered table.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Metric selection + ordering for the printed table.
# Order = display order; keep the "quality summary" first, then the
# stylized-fact and distribution metrics.
# ---------------------------------------------------------------------------

HEADLINE_METRICS = [
    # (key, pretty label, best_direction) — 'close_to_half', 'lower', 'higher'
    ('discriminator_accuracy', 'Disc. Acc. (best 0.5)', 'close_to_half'),
    ('tstr_mse', 'TSTR MSE', 'lower'),
    ('kurtosis_diff_mean', 'Kurt Δ', 'lower'),
    ('skew_diff_mean', 'Skew Δ', 'lower'),
    ('acf_sq_ret_diff_mse', 'ACF(r²) MSE', 'lower'),
    ('corr_frobenius_norm', 'Corr Frob', 'lower'),
    ('eigenvalue_mse', 'Eig MSE', 'lower'),
    ('var_diff_mean', 'VaR Δ', 'lower'),
    ('es_diff_mean', 'ES Δ', 'lower'),
    ('dist_KolmogorovSmirnov_stat', 'KS stat', 'lower'),
]

# Keys that identify training-time / per-step entries we should skip
STEP_KEYS = {'step', 'batch_loss', 'epoch_loss', 'epoch', 'batch_recon_loss',
             'batch_kl_loss', 'beta', 'mean_kl_per_dim', 'mu_std', 'collapsed',
             'active_dimensions', 'val_kurtosis_diff_mean', 'val_skew_diff_mean',
             'val_acf_ret_diff_mse', 'val_acf_sq_ret_diff_mse', 'val_es_diff_mean'}


def _is_step_entry(obj: dict) -> bool:
    """True if the entry is a per-step / per-epoch training log, not eval."""
    keys = set(obj.keys())
    return bool(keys & STEP_KEYS) or 'epoch' in keys


def load_experiment_metrics(exp_dir: Path) -> dict[str, Any] | None:
    """Return a flat dict of the latest evaluation metrics, or None if missing."""
    metrics_path = exp_dir / 'metrics.jsonl'
    if not metrics_path.exists():
        return None
    latest: dict[str, Any] = {}
    with metrics_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if _is_step_entry(obj):
                continue
            # Overwrite any existing keys — we want the last occurrence
            latest.update({k: v for k, v in obj.items() if k not in STEP_KEYS})
    return latest or None


def infer_model_type(exp_dir: Path) -> str | None:
    """Find model_type from the nearest config.yaml (check both run dir and source experiments/)."""
    # 1) config in the run dir (unusual but possible)
    for candidate in [exp_dir / 'config.yaml']:
        if candidate.exists():
            try:
                return yaml.safe_load(candidate.read_text()).get('model_type')
            except Exception:
                pass
    # 2) lookup under experiments/<name>/
    experiments_dir = Path(__file__).resolve().parent.parent / 'experiments'
    name = exp_dir.name
    # Exact match first, then loose prefix match
    for candidate in sorted(experiments_dir.glob('*')):
        if candidate.name.startswith(name.split('_')[0]):  # e.g. "0001"
            cfg = candidate / 'config.yaml'
            if cfg.exists():
                try:
                    return yaml.safe_load(cfg.read_text()).get('model_type')
                except Exception:
                    pass
    return None


def discover_experiments(roots: list[Path]) -> list[Path]:
    """Yield experiment dirs under each root (one level deep) that have metrics.jsonl."""
    seen: set[Path] = set()
    out: list[Path] = []
    for root in roots:
        if not root.exists():
            print(f"warning: {root} does not exist; skipping", file=sys.stderr)
            continue
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            if child.name.startswith('.'):
                continue
            if (child / 'metrics.jsonl').exists():
                if child.resolve() not in seen:
                    seen.add(child.resolve())
                    out.append(child)
    return out


def format_table(df: pd.DataFrame) -> str:
    """Render the DataFrame as an aligned Markdown table."""
    # pandas' to_markdown requires tabulate; fall back to manual if unavailable.
    try:
        return df.to_markdown(floatfmt='.4f')
    except ImportError:
        cols = ['experiment', 'model'] + [label for _, label, _ in HEADLINE_METRICS]
        lines = ['| ' + ' | '.join(cols) + ' |',
                 '|' + '|'.join(['---'] * len(cols)) + '|']
        for _, row in df.iterrows():
            cells = []
            for c in cols:
                v = row.get(c, '')
                if isinstance(v, float):
                    cells.append(f'{v:.4f}')
                else:
                    cells.append(str(v))
            lines.append('| ' + ' | '.join(cells) + ' |')
        return '\n'.join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'roots', nargs='+', type=Path,
        help='Directories containing experiment run subfolders.',
    )
    parser.add_argument(
        '-o', '--output-prefix', type=Path, default=None,
        help='Path prefix for summary.csv / summary.md (default: first root).',
    )
    args = parser.parse_args()

    experiments = discover_experiments(args.roots)
    if not experiments:
        print('No experiment directories with metrics.jsonl found.', file=sys.stderr)
        return 1

    rows = []
    for exp_dir in experiments:
        metrics = load_experiment_metrics(exp_dir)
        if metrics is None:
            continue
        model = infer_model_type(exp_dir) or '(unknown)'
        row = {'experiment': exp_dir.name, 'model': model}
        for key, label, _direction in HEADLINE_METRICS:
            row[label] = metrics.get(key)
        rows.append(row)

    if not rows:
        print('No usable evaluation metrics found.', file=sys.stderr)
        return 1

    df = pd.DataFrame(rows)
    # Sort by experiment number (leading digits in name)
    df['_sort_key'] = df['experiment'].str.extract(r'^(\d+)')[0].astype(int, errors='ignore')
    df = df.sort_values('_sort_key').drop(columns='_sort_key').reset_index(drop=True)

    table_md = format_table(df)
    print(table_md)

    # Persist outputs
    out_prefix = args.output_prefix if args.output_prefix else args.roots[0]
    csv_path = Path(str(out_prefix).rstrip('/') + '/summary.csv') if out_prefix.is_dir() else Path(str(out_prefix) + '.csv')
    md_path = Path(str(out_prefix).rstrip('/') + '/summary.md') if out_prefix.is_dir() else Path(str(out_prefix) + '.md')
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    md_path.write_text(table_md + '\n')
    print(f'\nWrote {csv_path} and {md_path}', file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
