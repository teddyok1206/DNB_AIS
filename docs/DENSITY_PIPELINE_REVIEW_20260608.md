# Density Pipeline Review And Performance Plan - 2026-06-08

## Executive Conclusion

The active PH-assisted CountSpatial U-Net pipeline can produce meaningful quantitative results, but the final claim must be framed as density-map/count estimation, not individual ship detection.

Current evidence from smoke/pilot runs is promising but not final:

```text
best current pilot: count_spatial_patchmix64_20260602_194800
train patches: 1280
val patches: 192
test patches: 128
test target sum: 5762.0
test pred/target ratio: 0.720
test loss_mean: 1.220
```

Interpretation:

```text
spatial structure is learnable.
count calibration is still under-counting.
existing smoke runs are too small for a final claim.
full 1200-scene training should materially improve stability if labels and split are clean.
```

## Current Active Pipeline

```text
[A] JPSS-2 / VIIRS DNB NetCDF -> arctan-encoded GeoTIFF + metadata
[D] AIS dynamic DB -> per-scene IMG_DT-matched/interpolated ship positions
[E] bbox/GeoJSON generation from DB rows where IMG_DT exists and ITPL >= 0
KR EEZ + 12 nm sea mask
full-scene H0 PH anchor extraction with factor-4 max downsampling
hierarchical child PH for oversized anchors
exact-cover sea partitioning: PH anchors first, fallback grid for all remaining valid sea pixels
sum-preserving Gaussian density target
CountSpatialDensityUNet: density = count_head(x) * softmax(spatial_logits over valid owner pixels)
patch-level and full-scene merged inference
```

## Data Readiness Gate

Before final 250/60/55 day split training, verify:

```text
GeoTIFF index count >= expected full-scene count
Bboxes_completed.txt count matches completed bbox/GeoJSON count
all scene_split rows have existing tif_path and geojson_path
DB rows used by generated GeoJSON satisfy IMG_DT IS NOT NULL and ITPL >= 0
same day never crosses train/val/test
```

Current local warning:

```text
/Volumes/SAMSUNG/JPSS-2_VIIRS/GeoTIFF currently reports 0 .tif files in this session.
The external volume path may be unmounted or the GeoTIFF folder path may differ.
```

## Quantitative Metrics To Report

Patch-level metrics:

```text
patch_count_mae
patch_count_rmse
patch_count_smape
patch_count_bias_mean
pred_target_ratio
count_ratio_abs_log_error
```

Density/spatial metrics:

```text
target_explained = sum(min(pred, target)) / sum(target)
pred_matched = sum(min(pred, target)) / sum(pred)
spatial_overlap_mean_positive = mean(sum(min(pred/sum(pred), target/sum(target))) over positive patches)
```

Scene-level metrics:

```text
scene_count_mae
scene_count_rmse
scene_pred_target_ratio
scene_target_explained
scene_pred_matched
scene_spatial_overlap
```

Baselines that must be compared:

```text
all-zero density baseline
constant-per-valid-ocean-pixel baseline
brightness-proportional density baseline
PH-soft-attention-proportional density baseline
trained CountSpatial U-Net
lifetime-weighted CountSpatial U-Net pilot
```

A result is meaningful if the trained model beats brightness/PH heuristic baselines on held-out days in both:

```text
count error: lower MAE/RMSE/SMAPE
spatial quality: higher normalized spatial overlap and target_explained
```

## Fixes Already Implemented In This Review

```text
GAT executable source files removed from active source tree.
run_batches now reports count MAE/RMSE/SMAPE, count ratio error, target_explained, pred_matched, spatial overlap.
training now saves checkpoint_best_val_loss.pt.
training now saves checkpoint_best_val_count_ratio.pt.
full-scene and enhanced preview renderers can choose last/best_val_loss/best_val_count_ratio checkpoints.
run output now stores config_snapshot.json and config_hash for reproducibility.
lifetime pilot config/script added: use lifetime as input/loss confidence prior, not merge weight.
```

## Highest-Leverage Performance Improvements

### 1. Use Best Checkpoint Selection

Do not report only `checkpoint_last.pt`.

Preferred evaluation order:

```text
checkpoint_best_val_loss.pt
checkpoint_best_val_count_ratio.pt
checkpoint_last.pt only as fallback
```

Rationale:

```text
MPS overnight runs can drift after the best validation epoch.
count calibration and spatial loss may peak at different epochs.
```

### 2. Increase Training Coverage Gradually

Current smoke caps discard many candidate patches:

```text
count_spatial_patchmix64_20260602_194800 dropped_by_cap_total = 16659
selected_patches_total = 1600
```

Plan:

```text
pilot: MAX_SCENES_PER_SPLIT=30, MAX_PATCHES_PER_SCENE=64
medium: MAX_SCENES_PER_SPLIT=120, MAX_PATCHES_PER_SCENE=96
full: day split 250/60/55, MAX_PATCHES_PER_SCENE=96 or 128 if memory allows
```

Expected impact:

```text
more fallback/background examples improve false-positive control.
more high-density PH examples improve count-head calibration.
```

### 3. Keep Exact-Cover Merge, Do Not Lifetime-Weight Merge

Use lifetime as confidence prior only:

```text
input channel: anchor_lifetime_map
loss sample weight: normalized log1p(lifetime)
```

Do not use lifetime-weighted full-scene overlap averaging by default because the active partitioner is exact-cover and owner-mask based.

### 4. Add Baseline Heuristic Evaluation

Implemented utility:

```text
sub_module.evaluate_density_baselines
```

For each split/scene, compute train-calibrated density maps from:

```text
valid_uniform
brightness_weighted
ph_soft_attention_weighted
brightness_times_ph_attention
```

Then evaluate with the same metrics as model predictions. This gives defensible final-report evidence that the neural model is not just reproducing brightness/PH heuristics.

Command:

```sh
PYTHONPATH=. /Users/jungtaeuk/anaconda3/envs/DNB_AIS/bin/python \
  -m sub_module.evaluate_density_baselines \
  --scene-split-csv outputs/dnb_density/splits/density_smoke_split_10_3_2/scene_split.csv \
  --config configs/dnb_density_unet_count_spatial.json \
  --calibration-split train \
  --eval-split test \
  --limit-calibration-scenes 10 \
  --limit-eval-scenes 3
```

### 5. Count Calibration Head

Current best pilot still under-counts:

```text
test pred/target ratio ~= 0.72
```

Implementation options:

```text
add global patch statistics to count head: brightness sum/mean/max/std, PH area, lifetime, valid area
calibrate count post-hoc on validation: pred_count_calibrated = a * pred_count + b or scale-only a * pred_count
save calibration parameters in run_summary.json
```

Preferred first step:

```text
scale-only validation calibration
```

Reason:

```text
simple, hard to overfit, directly addresses systematic under-counting.
```

### 6. Cloud / Bright Outlier Handling

Do not hard-zero cloud pixels initially. Add diagnostic channels or masks instead:

```text
scene_brightness_p99 channel or bright_outlier_mask
robust z-score brightness channel
cloud-like component flag based on area + brightness distribution
```

Rationale:

```text
arctan-preserved imagery keeps cloud/bright artifacts visible.
hard clipping may remove real vessel signals.
model should first see artifact indicators and learn suppression.
```

### 7. Split Quality

Current split is random day-level. For full training, use day-level stratification by:

```text
scene count per day
target ship count per day
PH anchor count per day
cloud/brightness summary if available
```

This avoids train/val/test imbalance where validation has unusually dense or sparse fishing days.

## Concrete Run Plan

### Step 0. Data Audit

```sh
PYTHONPATH=. /Users/jungtaeuk/anaconda3/envs/DNB_AIS/bin/python \
  -m sub_module.build_density_scene_split \
  --tif-root /Volumes/SAMSUNG/JPSS-2_VIIRS \
  --train-days 10 --val-days 3 --test-days 2
```

Accept only if `split_summary.json` reports:

```text
tif_index_count >= completed_count or at least no unexpected missing_tif_count
usable_day_count >= 15 for smoke, >= 365 for final
```

### Step 1. Lifetime Pilot Comparison

```sh
bash scripts/run_density_count_spatial_scaled_patchmix.sh
bash scripts/run_density_count_spatial_lifetime_patchmix.sh
```

Compare:

```text
best_val_loss checkpoint
test patch_count_mae / patch_count_rmse / pred_target_ratio / spatial_overlap_mean_positive
full-scene prediction metrics for representative test scenes
```

### Step 2. Medium Run

```sh
RUN_TAG=count_spatial_medium_$(date +%Y%m%d_%H%M%S) \
MAX_SCENES_PER_SPLIT=120 \
MAX_PATCHES_PER_SCENE=96 \
MAX_PH_PATCHES_PER_SCENE=72 \
MAX_FALLBACK_PATCHES_PER_SCENE=24 \
EPOCHS=30 \
bash scripts/run_density_count_spatial_lifetime_patchmix.sh
```

### Step 3. Full Run

```sh
SPLIT_DIR=outputs/dnb_density/splits/density_final_250_60_55 \
TRAIN_DAYS=250 VAL_DAYS=60 TEST_DAYS=55 \
MAX_SCENES_PER_SPLIT=0 \
MAX_PATCHES_PER_SCENE=96 \
MAX_PH_PATCHES_PER_SCENE=72 \
MAX_FALLBACK_PATCHES_PER_SCENE=24 \
EPOCHS=40 \
bash scripts/run_density_count_spatial_lifetime_patchmix.sh
```

### Step 4. Full-Scene Report Figures

```sh
PYTHONPATH=. /Users/jungtaeuk/anaconda3/envs/DNB_AIS/bin/python \
  -m sub_module.render_density_full_scene_predictions \
  --run-dir outputs/dnb_density/runs/<run_tag> \
  --split test \
  --checkpoint-kind best_val_loss \
  --limit-scenes 3 \
  --device mps
```

## Final Judgment

The current pipeline is scientifically defensible if final held-out-day metrics show:

```text
trained model beats heuristic baselines on count MAE/RMSE
trained model beats heuristic baselines on normalized spatial overlap
test pred/target ratio is close enough to 1 after validation calibration
full-scene figures show density in AIS target regions, not merely brightness blobs
```

If the model only improves spatial overlap but count remains biased, report the output as a relative vessel-density probability heatmap plus calibrated expected count, not as raw integer count per pixel.
