# DNB_AIS

Research code for VIIRS DNB nighttime-light ship-density estimation using AIS-derived supervision.

The active deep-learning direction is PH-assisted U-Net density heatmap prediction:

```text
DNB GeoTIFF brightness + sea mask + PH hierarchy features
-> multi-channel crop tensor
-> U-Net density model
-> per-pixel expected ship-count density heatmap
```

The GAT path is retired from active development. Historical GAT notes and logs are kept only for reference.

## Current Status

- Active model family: `CountSpatialDensityUNet`.
- Active config: `configs/dnb_density_unet_count_spatial.json`.
- Active target: sum-preserving Gaussian density map from AIS/bbox ground truth.
- Active partitioning: PH anchors first, fallback grid second, so valid sea pixels are covered.
- Active PH mode: H0 components from `cripser`, with hierarchical child PH splitting for oversized anchors.
- Preferred device: Apple MPS on the M2 Max MacBook Pro.
- Runtime outputs, checkpoints, GeoTIFFs, NetCDFs, DB files, NumPy arrays, and bulk bbox outputs are intentionally not tracked by git.

## Repository Layout

```text
configs/                       Model, loss, target, detector, and experiment configs.
sub_module/                    Active reusable Python modules and runnable entrypoints.
scripts/                       Repo maintenance and experiment helper scripts.
docs/                          Design notes, workflow docs, and curated experiment reports.
docs/experiments/              Lightweight experiment summaries promoted from runtime outputs.
outputs/                       Ignored runtime outputs; only outputs/README.md is tracked.
artifacts/                     Ignored long-lived artifact area; only artifacts/README.md is tracked.
_Readings/                     Classified research readings and notes.
_Meetings/                     Meeting notes.
_archive/                      Archived or retired project material.
[3]_DNB_AIS - (STEP 3)/        Active operational workspace for A/D/E preprocessing jobs.
```

`[3]_DNB_AIS - (STEP 3)` is still kept in place because current operational scripts and some pipeline defaults use exact paths there.

Important STEP3 scripts:

```text
[A]_dnb2geotif_v2_modified_4326_metadata.py   DNB L1/NetCDF to GeoTIFF conversion/compositing.
[D]_ship_class_SQL_fast.py                    Optimized AIS interpolation and DB update workflow.
[E]_bounding_box.py                           Bbox/GeoJSON rebuild workflow.
```

## Data And Artifact Policy

Keep code, configs, docs, scripts, and small curated manifests in git.

Do not commit heavy/generated data:

```text
*.pt *.pth *.ckpt *.tif *.tiff *.nc *.db *.npy *.npz
raw AIS files
bulk bbox/GeoJSON output folders
bulk outputs/ run directories
```

Runtime artifacts should live under `outputs/` or `artifacts/` and include metadata when possible:

```text
git_commit
git_dirty
config path and config hash
scene split manifest
target generation config
model config
checkpoint paths
```

If a checkpoint is created from dirty code, save the diff in the run directory as `run_git_dirty.patch`.

## Environment

The current local Python environment is expected to be the conda environment used by the DNB/AIS project:

```bash
/Users/jungtaeuk/anaconda3/envs/DNB_AIS/bin/python
```

For commands below, either activate that environment or set:

```bash
export DNB_AIS_PYTHON=/Users/jungtaeuk/anaconda3/envs/DNB_AIS/bin/python
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_ENABLE_MPS_FALLBACK=0
export PYTHONUNBUFFERED=1
```

## Quick Checks

Run lightweight source checks before committing code changes:

```bash
./scripts/git_ai_status.sh
/Users/jungtaeuk/anaconda3/envs/DNB_AIS/bin/python -m py_compile sub_module/*.py
git diff --check
```

Notebook output hygiene check:

```bash
./scripts/strip_notebook_outputs.py --check "[3]_DNB_AIS - (STEP 3)/[C]_metadata_analyzer.ipynb"
```

## Split Smoke Training

The current standard pilot training script is:

```bash
bash scripts/run_density_count_spatial_scaled_patchmix.sh
```

Useful overrides:

```bash
RUN_TAG=count_spatial_manual_$(date +%Y%m%d_%H%M%S) \
EPOCHS=20 \
BATCH_SIZE=2 \
MAX_SCENES_PER_SPLIT=30 \
MAX_PATCHES_PER_SCENE=64 \
MAX_PH_PATCHES_PER_SCENE=48 \
MAX_FALLBACK_PATCHES_PER_SCENE=16 \
bash scripts/run_density_count_spatial_scaled_patchmix.sh
```

Typical run output path:

```text
outputs/dnb_density/runs/<run_tag>/
```

Monitor a run:

```bash
tail -f outputs/dnb_density/runs/<run_tag>/run.log
```

## Enhanced Preview Rendering

After a training run, render qualitative prediction previews:

```bash
PYTHONPATH=. /Users/jungtaeuk/anaconda3/envs/DNB_AIS/bin/python \
  -m sub_module.render_density_enhanced_previews \
  --run-dir outputs/dnb_density/runs/<run_tag> \
  --split test \
  --device mps
```

Preview panels are designed to compare brightness, PH structure, target density, prediction density, explained density overlap, and absolute error.

## Core Method Notes

The density model predicts a continuous non-negative density map, not integer count classes.

```text
model(input) -> density_pred[h, w] >= 0
ship_count(region) = sum(density_pred over region)
```

The true ship count is integer-valued, but training remains continuous and differentiable. Count is recovered by integrating or summing the density map over a valid region.

PH is used as a structural prior and partitioning mechanism, not as a hard censor for supervision. Ground-truth ships inside the crop can still contribute to the target even if strict PH masks miss them.

## Key Docs

- `docs/ACTIVE_WORKSPACE_MAP.md`: current directory and workspace policy.
- `docs/DNB_DENSITY_MODEL_SCAFFOLD.md`: active U-Net density model design.
- `docs/PH_HIERARCHY_UNET_HYBRID_DESIGN.md`: PH hierarchy and U-Net integration design.
- `docs/DNB_DENSITY_OUTPUT_WORKSPACE_20260602.md`: output workspace organization.
- `docs/experiments/density_count_spatial_good_baseline_20260602.md`: current good qualitative baseline record.

## Git Workflow

This repository follows the project policy in `AGENTS.md`:

- commit meaningful code, config, doc, and experiment scaffold progress;
- stage only focused related files;
- never stage heavy generated artifacts;
- run lightweight checks before commits;
- do not push unless explicitly requested.
