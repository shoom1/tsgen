# Colab Training Notebook

`colab_train.ipynb` runs the diffusion and VAE experiments on a Colab GPU. Baselines (`multivariate_gaussian`, `bootstrap`, `ccc_garch`) are fast enough to run locally — they're excluded by default.

## One-time setup

### 1. Upload the finbase database to Google Drive

The experiments read price data via `finbase.DataClient`, which auto-detects a SQLite DB file via `~/.finbaserc`. Ship the DB file to Drive once:

1. Locate it locally: the path is in your `~/.finbaserc` under `database.path`. On the current machine it lives at `~/.finbase/timeseries.db` (~345 MB).
2. In the Drive web UI create a folder `My Drive/tsgen/`.
3. Upload `timeseries.db` into that folder. The upload takes a few minutes depending on connection speed.

Target Drive layout after this step:
```
My Drive/tsgen/
  └── timeseries.db
```

The notebook creates `My Drive/tsgen/runs/` automatically on first use.

### 2. Open the notebook in Colab

- **Option A (recommended):** In GitHub, navigate to `notebooks/colab_train.ipynb` and click the **Open in Colab** badge (or use the Colab URL scheme `https://colab.research.google.com/github/shoom1/tsgen/blob/develop/notebooks/colab_train.ipynb`).
- **Option B:** Download the file and upload it to Colab manually (File → Upload notebook).

### 3. Request a GPU

In Colab: **Runtime → Change runtime type → T4 GPU**. Free tier gives you a T4 with roughly 12 hours of GPU time per day.

## Running

The notebook has 8 cells you run top-to-bottom:

1. **Runtime check** — confirms a GPU is allocated.
2. **Mount Drive** — asks for permission; `~/tsgen` must already exist on Drive.
3. **Copy DB to Colab local disk** — one-time ~10s operation. SQLite over Drive-mount is painfully slow; local copy makes queries fast.
4. **Install dependencies** — `pip install finbase arch ...` plus cloning/installing `tsgen` from `github.com/shoom1/tsgen` at the `develop` branch.
5. **Generate experiment configs** — runs `scripts/generate_standard_configs.py` to materialize `experiments/00*/config.yaml` (not shipped in the repo since they're personal research artifacts).
6. **Run experiments** — the main training cell. Each experiment saves to `My Drive/tsgen/runs/<exp_name>/`, so work survives runtime disconnects.
7. **Inspect artifacts** — quick file listing of what ended up on Drive.
8. **(Optional) smoke test** — re-run every config at a tiny budget to verify end-to-end wiring. Useful after pulling new code.

## Resuming across sessions

Free-tier T4 sessions time out after ~12h. To resume:

1. Open the notebook again; all prior cells can be re-executed safely (the DB-copy step is no-op if the local copy is up to date).
2. In cell 6 (`EXPERIMENTS = [...]`), comment out the ones you've already completed.
3. Run cell 6 again. New artifacts go to Drive.

## Notes

- **Which branch to clone.** The notebook pulls `develop` by default. To switch (e.g., to `main` after a merge), edit the `BRANCH = 'develop'` line in cell 4.
- **Updating code mid-session.** The install is editable (`pip install -e`), and each run pulls the latest `develop`. If you push changes mid-session, re-run cell 4's second sub-cell and the new code is active.
- **Device override.** The run loop passes `--override device=cuda` so models train on GPU. The config files default to `device: cpu` since experiments also run locally. No edit required.
- **Interrupting a training run.** Interrupting a cell kills the subprocess; partial artifacts remain on Drive. Re-running relaunches from scratch (no checkpoint resume wired into the notebook — use `--resume-from-checkpoint` manually if needed).
