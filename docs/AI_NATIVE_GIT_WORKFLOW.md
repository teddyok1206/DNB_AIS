# AI-Native Git Workflow

This repository should track the reasoning, source code, configs, and lightweight manifests needed to reproduce a deep-learning run. It should not track generated geospatial data, checkpoints, raw AIS, NetCDF, GeoTIFFs, or notebook output bulk.

## Repository Contract

Track:

- Python source in `sub_module/` and small project scripts.
- Notebook generator code and curated notebooks when useful.
- Design notes in `docs/` and important Markdown files.
- Experiment configs in `configs/`.
- Small split manifests or curated metrics only when intentionally promoted from `outputs/`.

Do not track:

- `outputs/`, `artifacts/`, checkpoints, tensor caches, model exports.
- GeoTIFF, NetCDF, SQLite/DB, FITS, shapefile, QGIS, and raw AIS data.
- Generated PNG/PDF/ZIP files unless moved into `docs/` intentionally.

## Experiment Reproducibility

Every training run should produce a run directory like:

```text
artifacts/dnb_density_unet/run_YYYYMMDD_HHMMSS/
  config.json
  checkpoint_last.pt
  checkpoint_best.pt
  training_history.csv
  validation_metrics.json
  run_manifest.json
```

`run_manifest.json` must include:

```json
{
  "git_commit": "<commit sha>",
  "git_dirty": true,
  "config_path": "configs/dnb_density_unet_base.json",
  "config_sha256": "<hash>",
  "scene_split_manifest": "<path>",
  "model_class": "MaskedDensityUNet",
  "target_builder": "sum_preserving_pixel_kernel",
  "checkpoint_last": "checkpoint_last.pt",
  "checkpoint_best": "checkpoint_best.pt"
}
```

If `git_dirty` is true, also store a patch file in the run directory:

```sh
git diff > run_git_dirty.patch
```

This lets an AI agent reconstruct what code produced a checkpoint even before the change was committed.

## Commit Policy

Use small, scoped commits:

```text
feat(detector): add PH candidate detector
feat(target): add sum-preserving pixel kernel GT
feat(model): add masked density U-Net
exp(config): add baseline DNB density config
docs(method): document PH-masked U-Net design
```

Before committing:

```sh
./scripts/git_ai_status.sh
python -m py_compile sub_module/*.py
python scripts/strip_notebook_outputs.py --check "[3]_DNB_AIS - (STEP 3)/[C]_metadata_analyzer.ipynb"
```

Do not commit `.pt`, `.tif`, `.nc`, `.db`, `.npy`, `.npz`, or generated `outputs/` files.

## Branch Policy

Keep `main` usable. For substantial experiments, use short topic branches:

```text
feat/density-unet
feat/ph-detector
exp/loss-sweep
```

Do not use branches to store data. Store artifacts under `artifacts/` and record metadata.

## AI Agent Rules

- Check `git status --short --untracked-files=no` before edits.
- Never revert user changes unless explicitly requested.
- Prefer adding new files or narrow patches over broad rewrites.
- Keep checkpoint and data paths out of git.
- When creating a run, write config and manifest first, then checkpoint.
- If a run depends on dirty code, save `git diff` into the run directory.

## Quick Commands

```sh
./scripts/git_ai_status.sh

git add .gitignore .gitattributes docs configs scripts sub_module

git commit -m "chore: set up AI-native git workflow"
```
