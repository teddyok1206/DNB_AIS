# DNB_AIS

Research code for VIIRS DNB nighttime-light ship-density estimation using AIS-derived supervision.

The active deep-learning direction is PH-assisted U-Net occupancy/spatial heatmap prediction:

```text
DNB GeoTIFF brightness + sea mask + PH hierarchy features
-> multi-channel crop tensor
-> U-Net occupancy/spatial model
-> per-pixel ship-presence evidence heatmap
```

The GAT executable code path has been removed from active source. Historical GAT notes and logs remain only in archive/log documents for design-history reference.

## Current Status

- Active model family: `OccupancySpatialUNet`.
- Active baseline config: `configs/dnb_density_unet_occupancy_spatial.json`.
- Active target: O/X patch occupancy plus positive-patch spatial distribution from AIS/bbox ground truth.
- Active partitioning: PH anchors first, fallback grid second, so valid sea pixels are covered.
- Active PH mode: H0 components from `cripser`, with hierarchical child PH splitting for oversized anchors.
- Deferred target: direct ship-count regression. See `docs/count_reintroduction/COUNT_HEAD_REINTRODUCTION_PLAN.md`.
- Preferred device: Apple MPS on the M2 Max MacBook Pro.
- Runtime outputs, checkpoints, GeoTIFFs, NetCDFs, DB files, NumPy arrays, and bulk bbox outputs are intentionally not tracked by git.
- Retired count/inverse/fast-density configs and scripts live under `_archive/legacy_density_configs_20260608/`.

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
bash scripts/run_density_occupancy_spatial_patchmix.sh
```

Useful overrides:

```bash
RUN_TAG=occupancy_spatial_manual_$(date +%Y%m%d_%H%M%S) \
EPOCHS=20 \
BATCH_SIZE=2 \
MAX_SCENES_PER_SPLIT=30 \
MAX_PATCHES_PER_SCENE=64 \
MAX_PH_PATCHES_PER_SCENE=48 \
MAX_FALLBACK_PATCHES_PER_SCENE=16 \
POSITIVE_PATCHES_PER_SCENE=24 \
NEGATIVE_PATCHES_PER_SCENE=24 \
SELECTION_SEED=20260608 \
bash scripts/run_density_occupancy_spatial_patchmix.sh
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
  --checkpoint-kind best_val_occupancy_f1 \
  --device mps
```

Preview panels are designed to compare brightness, PH structure, target density, prediction density, explained density overlap, and absolute error.

Full-scene merged prediction:

```bash
PYTHONPATH=. /Users/jungtaeuk/anaconda3/envs/DNB_AIS/bin/python \
  -m sub_module.render_density_full_scene_predictions \
  --run-dir outputs/dnb_density/runs/<run_tag> \
  --split test \
  --checkpoint-kind best_val_occupancy_f1 \
  --limit-scenes 3 \
  --device mps
```

Heuristic baseline evaluation:

```bash
PYTHONPATH=. /Users/jungtaeuk/anaconda3/envs/DNB_AIS/bin/python \
  -m sub_module.evaluate_density_baselines \
  --scene-split-csv outputs/dnb_density/splits/density_smoke_split_10_3_2/scene_split.csv \
  --config configs/dnb_density_unet_occupancy_spatial.json \
  --calibration-split train \
  --eval-split test
```

## Core Method Notes

The active model predicts continuous occupancy evidence, not integer count classes.

```text
model(input) -> P(ship exists in patch), p(pixel | ship exists)
occupancy_heatmap[h, w] = P(ship exists in patch) * p(pixel | ship exists)
```

The patch-level prediction sum is a probability of ship presence. Direct count regression is deferred until O/X and localization are stable, then count can be reintroduced as a conditional positive-patch head.

PH is used as a structural prior and partitioning mechanism, not as a hard censor for supervision. Ground-truth ships inside the crop can still contribute to the target even if strict PH masks miss them.

## Key Docs

- `docs/ACTIVE_WORKSPACE_MAP.md`: current directory and workspace policy.
- `docs/DNB_DENSITY_MODEL_SCAFFOLD.md`: active U-Net density model design.
- `docs/PH_HIERARCHY_UNET_HYBRID_DESIGN.md`: PH hierarchy and U-Net integration design.
- `docs/DNB_DENSITY_OUTPUT_WORKSPACE_20260602.md`: output workspace organization.
- `docs/DENSITY_PIPELINE_REVIEW_20260608.md`: full pipeline review, quantitative metrics, and performance plan.
- `docs/count_reintroduction/COUNT_HEAD_REINTRODUCTION_PLAN.md`: deferred count-head design and reintroduction criteria.
- `docs/experiments/density_count_spatial_good_baseline_20260602.md`: historical count-spatial qualitative baseline record.

## Git Workflow

This repository follows the project policy in `AGENTS.md`:

- commit meaningful code, config, doc, and experiment scaffold progress;
- stage only focused related files;
- never stage heavy generated artifacts;
- run lightweight checks before commits;
- do not push unless explicitly requested.
