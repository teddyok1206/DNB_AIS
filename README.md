# DNB_AIS

Research code for VIIRS DNB nighttime ship-presence mapping with AIS-derived supervision.

The active deep-learning path is now deliberately narrow:

```text
DNB GeoTIFF brightness + sea mask + minimal PH persistence/seed features
-> recursive PH exact-cover patches
-> PixelBinaryOccupancyUNet
-> per-pixel ship-presence probability
```

Older count-regression, occupancy/spatial-softmax, brightness-threshold baseline, and preview/render variants are archived under `_archive/retired_density_complexity_20260609/` so they can be recovered without remaining in the active path.

## Current Status

- Active model: `PixelBinaryOccupancyUNet`.
- Active loss: `pixel_binary_occupancy_loss`.
- Active config: `configs/dnb_density_unet_pixel_binary_recursive_ph_hardtarget_20260609.json`.
- Active target: hard pixel occupancy, `target_pixel = 1[raw_count > 0]`, masked by the valid partition owner sea mask.
- Active input channels: `brightness`, `ph_persistence_map`, `ph_seed_map`.
- Active partitioning: PH anchors first, recursive PH subdivision for oversized anchors, fallback grid for exact sea-pixel coverage.
- Primary metrics: `pixel_f1`, `pixel_iou`, `pixel_precision`, `pixel_recall`, `pixel_brier`.
- Secondary diagnostics only: patch-level `occupancy_*`, `spatial_overlap_mean_positive`, and mass-ratio metrics.

## Repository Layout

```text
configs/                       Active experiment config.
sub_module/                    Active Python modules and runnable entrypoints.
scripts/                       Repo maintenance and active experiment helper scripts.
docs/                          Active design notes, workflow docs, and current experiment summaries.
docs/experiments/              Curated current experiment reports.
outputs/                       Ignored runtime outputs; only lightweight metadata should be promoted.
artifacts/                     Ignored long-lived artifact area.
_Readings/                     Classified research readings and notes.
_Meetings/                     Meeting notes.
_archive/                      Archived or retired project material.
[3]_DNB_AIS - (STEP 3)/        Operational data/preprocessing workspace.
```

`[3]_DNB_AIS - (STEP 3)` stays in place because preprocessing and ground-truth resolution still use exact local paths there.

## Data And Artifact Policy

Keep code, configs, docs, scripts, and small curated manifests in git.

Do not commit heavy/generated data:

```text
*.pt *.pth *.ckpt *.tif *.tiff *.nc *.db *.npy *.npz
raw AIS files
bulk bbox/GeoJSON output folders
bulk outputs/ run directories
```

Runtime artifacts should live under `outputs/` or `artifacts/` and record config path/hash, scene split, target config, model config, checkpoint paths, and git state.

## Environment

```bash
export DNB_AIS_PYTHON=/Users/jungtaeuk/anaconda3/envs/DNB_AIS/bin/python
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_ENABLE_MPS_FALLBACK=0
export PYTHONUNBUFFERED=1
```

## Quick Checks

```bash
./scripts/git_ai_status.sh
/Users/jungtaeuk/anaconda3/envs/DNB_AIS/bin/python -m py_compile sub_module/*.py
git diff --check
```

## Active Training Command

Foreground run:

```bash
bash scripts/run_density_pixel_binary_recursive_ph.sh
```

Useful overrides:

```bash
RUN_TAG=pixel_binary_manual_$(date +%Y%m%d_%H%M%S) \
EPOCHS=18 \
BATCH_SIZE=8 \
MAX_PATCHES_PER_SCENE=64 \
POSITIVE_PATCHES_PER_SCENE=32 \
NEGATIVE_PATCHES_PER_SCENE=32 \
bash scripts/run_density_pixel_binary_recursive_ph.sh
```

Typical output path:

```text
outputs/dnb_density/runs/<run_tag>/
```

Monitor a run:

```bash
tail -f outputs/dnb_density/runs/<run_tag>/run.log
```

Evaluate the active checkpoint:

```bash
PYTHONPATH=. /Users/jungtaeuk/anaconda3/envs/DNB_AIS/bin/python \
  -m sub_module.evaluate_density_checkpoint \
  --run-dir outputs/dnb_density/runs/<run_tag> \
  --checkpoint best_val_pixel_f1 \
  --split test \
  --calibration-split val \
  --device mps
```

## Method Notes

The active model predicts independent per-pixel probabilities:

```text
P_pixel = sigmoid(pixel_logits)
Y_pixel = 1[raw_count > 0]
```

This replaces the retired spatial-softmax interpretation:

```text
retired: P(patch positive) * softmax(pixel | positive patch)
active: independent valid-pixel ship-presence probability
```

The patch-level O/X head remains only as a small auxiliary regularizer. It is not the main reporting target.

GT smoothing is not used as the supervised label. It remains acceptable for visualization because DNB light is physically diffuse, but AIS identity labels should not be spread into neighboring pixels for the primary loss.

## Key Docs

- `docs/experiments/density_pixel_binary_recursive_ph_hardtarget_20260609.md`: current experiment rationale and metric interpretation.
- `docs/PH_HIERARCHY_UNET_HYBRID_DESIGN.md`: PH hierarchy and U-Net integration design.
- `docs/FINAL_REPORT_PREP_IEEE.md`: final-report structure aligned to the active pipeline.
- `_archive/retired_density_complexity_20260609/README.md`: list of retired configs/scripts/docs/utilities and restore pattern.

## Git Workflow

This repository follows `AGENTS.md`: commit focused meaningful code/config/doc progress, do not stage heavy generated artifacts, run lightweight checks before commit, and do not push unless explicitly requested.
