# Final Report Prep - IEEE-Style EESRL Report

This document keeps the final-report plan aligned with the active DNB/AIS density pipeline.

## Important Pivot

The midterm report was written around a DRUID + GAT direction. That path is historical only.

The active final-report method is now:

```text
PH-assisted recursive exact-cover patching + PixelBinaryOccupancyUNet
```

The main label is hard pixel ship presence from AIS-derived raw count pixels:

```text
Y_pixel = 1[raw_count > 0]
```

Patch-level O/X, smoothed density maps, brightness baselines, and count regression are secondary or archived material unless explicitly revived.

## Active Method Summary

```text
problem: VIIRS DNB pixels are coarse enough that vessel lights can overlap and bloom.
data: 2025 JPSS-2/VIIRS DNB GeoTIFF + AIS-derived GeoJSON point/bbox supervision.
preprocessing: arctan-encoded DNB brightness, KR EEZ + 12 nm sea mask, day-level split.
structure prior: H0 persistent-homology anchors from encoded brightness.
partitioning: PH anchors first, recursive PH rerun for oversized anchors, fallback tiles for exact valid-sea coverage.
model: PixelBinaryOccupancyUNet.
output: independent per-pixel ship-presence probability.
training target: hard valid-pixel O/X from raw_count > 0.
```

Current input channels from `configs/dnb_density_unet_pixel_binary_recursive_ph_hardtarget_20260609.json`:

```text
0 brightness
1 ph_persistence_map
2 ph_seed_map
```

Parent/child PH masks, soft attention, and anchor lifetime are retained only as proposal-building metadata when needed. They are not U-Net input channels.

## Core Equations

```text
encoded = (2 / pi) * arctan(radiance / 1e-9)
Y_pixel = 1[raw_count_pixel > 0] * valid_owner_mask
P_pixel = sigmoid(pixel_logits) * valid_owner_mask
O_target = 1[sum(Y_pixel) > 0]
P_patch = sigmoid(occupancy_logit)
```

Main loss:

```text
L = lambda_pixel * BCEWithLogits(pixel_logits, Y_pixel)
  + lambda_patch * BCEWithLogits(occupancy_logit, O_target)
  + lambda_dice * Dice(pixel_logits, Y_pixel)   # currently disabled
```

Current config:

```text
lambda_pixel = 0.9
lambda_patch = 0.1
lambda_dice = 0.0
pixel_pos_weight = 256.0
```

GT smoothing note:

```text
DNB brightness is physically diffuse, so smoothed GT remains acceptable for visualization panels.
It is not the supervised label. AIS-derived identity should not be spread into neighboring pixels for the main O/X objective.
```

## Proposed Final Report Outline

### 1. Abstract

Include:

```text
VIIRS DNB low-resolution nighttime vessel-light overlap
AIS-derived supervision
PH-assisted recursive exact-cover patch construction
PixelBinaryOccupancyUNet per-pixel ship-presence mapping
final train/validation/test pixel metrics
```

### 2. Introduction

Emphasize:

```text
AIS-only monitoring has blind spots.
VIIRS DNB observes nighttime lights over large sea areas.
Direct integer count is poorly posed at current pixel/patch scale.
The final task is therefore ship-presence probability mapping, with count reintroduction deferred.
```

### 3. Data

Subsections:

```text
3.1 VIIRS DNB GeoTIFF scenes
3.2 AIS interpolation and GeoJSON-derived supervision
3.3 KR EEZ + 12 nm sea mask
3.4 Day-level train/validation/test split
```

Tables:

```text
Table I: split by days and scenes
Table II: selected patch counts by split
Table III: valid-pixel positive/negative distribution by split
```

### 4. Method

Subsections:

```text
4.1 DNB radiance preprocessing and arctan encoding
4.2 AIS-to-image supervision rasterization
4.3 KR sea masking
4.4 PH-assisted exact-cover partitioning
4.5 Recursive PH subdivision for oversized anchors
4.6 PH feature-channel construction
4.7 PixelBinaryOccupancyUNet architecture
4.8 Hard pixel O/X loss and auxiliary patch O/X head
4.9 Patch inference and full-scene merge
4.10 Deferred count head extension
```

### 5. Experiments

Report:

```text
hardware: Apple Silicon Mac, MPS backend
config: configs/dnb_density_unet_pixel_binary_recursive_ph_hardtarget_20260609.json
split: outputs/dnb_density/splits/ox_spatial_25pct_63_15_14_20260609_011421/scene_split.csv
checkpoint policy: last, best validation loss, best validation pixel F1
threshold policy: fixed 0.5 plus validation-calibrated threshold for pixel F1
```

Primary metrics:

```text
pixel_precision
pixel_recall
pixel_f1
pixel_iou
pixel_brier
```

Secondary diagnostics:

```text
occupancy_f1: patch auxiliary behavior only
spatial_overlap_mean_positive: retired softmax-style localization diagnostic only
pred_target_ratio: probability-mass vs positive-pixel-count diagnostic, not ship-count error
```

### 6. Results

Required figures:

```text
Figure 1: active pipeline diagram
Figure 2: VIIRS DNB scene + KR sea mask
Figure 3: PH parent / recursive child / fallback partition visualization
Figure 4: patch input channels with PH features
Figure 5: target pixel O/X, predicted probability, thresholded prediction, error map
Figure 6: train/validation loss and pixel F1 curves
Figure 7: representative success/failure test patches or scenes
```

Required tables:

```text
Table IV: final checkpoint metrics on train/val/test
Table V: fixed 0.5 threshold vs validation-calibrated threshold on test
```

If a baseline is included, label it clearly as archived/secondary:

```text
rule-based brightness threshold baseline
```

Do not present the retired occupancy/spatial-softmax model as the active method.

### 7. Discussion

Discuss:

```text
why hard pixel O/X is better posed than direct count at current resolution
why GT smoothing is visualization-only, not label spreading
why PH is a partition/input prior rather than a label censor
how recursive PH addresses oversized patches
limitations from AIS supervision, DNB blooming, cloud/bright artifacts, and sparse positives
what would justify reintroducing count prediction later
```

### 8. Conclusion

State:

```text
The completed pipeline maps DNB and PH-derived features to per-pixel vessel-presence probability.
It is evaluated as pixel O/X detection and localization, not as final integer vessel counting.
```

## Citation File

Use `docs/FINAL_REPORT_CITATIONS_IEEE.md`. Update that file before changing references.

## Next Report-Readiness Tasks

```text
1. finish the active pixel-binary recursive PH experiment
2. freeze best_val_pixel_f1 test metrics with fixed and calibrated thresholds
3. generate qualitative target/probability/error panels
4. prepare final metric tables
5. draft the final LaTeX report from this outline
```
