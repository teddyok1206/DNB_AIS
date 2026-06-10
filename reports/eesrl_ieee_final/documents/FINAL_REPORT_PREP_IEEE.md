# Final Report Prep - IEEE-Style EESRL Report

This document keeps the final-report plan aligned with the active DNB/AIS probability-map pipeline.

## Important Pivot

The midterm DRUID + GAT direction is historical only.

The active final-report method is now:

```text
PH-assisted recursive exact-cover patching + PixelProbabilityUNet ship-presence probability map
```

The project goal is not integer vessel counting. The goal is to show that high values in the learned probability map correspond to ship presence more reliably than high values in raw VIIRS DNB brightness.

## Active Method Summary

```text
problem: VIIRS DNB pixels are coarse enough that vessel lights can overlap and bloom.
data: 2025 JPSS-2/VIIRS DNB GeoTIFF + AIS-derived GeoJSON point/bbox supervision.
preprocessing: arctan-encoded DNB brightness, KR EEZ + 12 nm sea mask, day-level split.
structure prior: H0 persistent-homology anchors from encoded brightness.
partitioning: PH anchors first, recursive PH rerun for oversized anchors, fallback tiles for exact valid-sea coverage.
model: PixelProbabilityUNet.
output: independent per-pixel ship-presence probability.
training target: Gaussian proximity probability field seeded by raw_count > 0.
evaluation baseline: raw DNB brightness ranked against the same AIS presence target.
```

Current input channels from `configs/dnb_density_unet_probability_field_recursive_ph_20260610.json`:

```text
0 brightness
1 ph_persistence_map
2 ph_seed_map
```

Parent/child PH masks are computed only transiently during partition construction. Soft attention, anchor lifetime maps, count-density heads, and patch O/X auxiliary objectives are removed from the active path.

## Core Equations

```text
encoded = (2 / pi) * arctan(radiance / 1e-9)
Y_seed = 1[raw_count_pixel > 0] * valid_owner_mask
Y_field = exp(-0.5 * distance_to_nearest(Y_seed)^2 / sigma_pixels^2) * valid_owner_mask
P_pixel = sigmoid(pixel_logits) * valid_owner_mask
```

Main loss:

```text
L = mean_valid((1 + alpha * Y_field) * SmoothL1(P_pixel, Y_field; beta))
```

Current active config:

```text
model.name = PixelProbabilityUNet
loss.name = weighted_smooth_l1_probability_loss
pixel_weight = 1.0
occupancy_weight = 0.0
dice_weight = 0.0
sigma_pixels = 4.0
radius_pixels = 12
probability_target_threshold = 0.25
field_weight_strength = 8.0
smooth_l1_beta = 0.1
```

Target note:

```text
AIS labels remain exact seeds.
The supervised target is a proximity probability field because DNB pixels are coarse and AIS-to-DNB alignment is brittle.
This is not count-mass smoothing and should not be interpreted as integer vessel density.
```

## Evaluation Policy

Primary question:

```text
Does model probability rank AIS-positive pixels above AIS-negative pixels better than raw DNB brightness does?
```

Primary metrics:

```text
presence_probability.model_probability.average_precision
presence_probability.brightness_baseline.average_precision
presence_probability.model_vs_brightness_lift.average_precision_ratio
presence_probability.model_probability.precision_at_top.top_1pct.precision
presence_probability.model_probability.brier
presence_probability.model_calibrated_threshold.f1
```

Radius-tolerant sensitivity analysis:

```text
radius_presence.by_sigma.*.model_probability.average_precision
radius_presence.by_sigma.*.brightness_baseline.average_precision
radius_presence.by_sigma.*.model_vs_brightness_lift.average_precision_ratio
```

Retired metrics and objectives:

```text
ship-count regression
pred_target_ratio
soft_target_explained
soft_pred_matched
spatial_overlap_mean_positive
patch-level occupancy headline metrics
sum-preserving density/count mass
```

Brightness is a score baseline, not a calibrated probability. Report Brier/reliability bins for the model probability map, not for raw brightness unless a separate calibration model is explicitly fitted.

## Proposed Final Report Outline

### 1. Abstract

Include:

```text
VIIRS DNB low-resolution nighttime vessel-light overlap
AIS-derived supervision
PH-assisted recursive exact-cover patch construction
PixelProbabilityUNet per-pixel ship-presence probability mapping
model probability vs raw brightness ranking results
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
4.7 PixelProbabilityUNet architecture
4.8 Radius probability-field loss
4.9 Presence-ranking evaluation against raw brightness baseline
4.10 Patch inference and full-scene merge
4.11 Deferred count prediction extension
```

### 5. Experiments

Report:

```text
hardware: Apple Silicon Mac, MPS backend
config: configs/dnb_density_unet_probability_field_recursive_ph_20260610.json
split: outputs/dnb_density/splits/ox_spatial_25pct_63_15_14_20260609_011421/scene_split.csv
checkpoint policy: best validation field F1 plus final presence-ranking evaluation
threshold policy: fixed 0.5 plus validation-calibrated threshold for field F1
```

Current draft:

```text
docs/FINAL_REPORT_DRAFT_IEEE.md
```

### 6. Results

Required figures:

```text
Figure 1: active pipeline diagram
Figure 2: VIIRS DNB scene + KR sea mask
Figure 3: PH parent / recursive child / fallback partition visualization
Figure 4: patch input channels with PH features
Figure 5: target probability field, predicted probability field, thresholded field, error map
Figure 6: model probability vs raw brightness precision-recall curves
Figure 7: reliability diagram for model probability
Figure 8: representative success/failure test patches or scenes
```

Required tables:

```text
Table IV: final checkpoint presence-ranking metrics on train/val/test
Table V: raw brightness baseline vs model probability AP and Top-1% precision
Table VI: fixed 0.5 threshold vs validation-calibrated threshold on test
Table VII: radius_presence sigma sweep
```

### 7. Discussion

Discuss:

```text
why probability fields are better posed than exact AIS-pixel hits at current resolution
why this target is proximity probability, not count-density smoothing
why raw brightness is the correct baseline for the project claim
why PH is a partition/input prior rather than a label censor
how recursive PH addresses oversized patches
limitations from AIS supervision, DNB blooming, cloud/bright artifacts, and sparse positives
what evidence would justify reintroducing count prediction later
```

## Citation Maintenance

When adding final-report citations, update `docs/FINAL_REPORT_CITATIONS_IEEE.md` first. Keep references tied to the active DNB/AIS probability-map pipeline in the core list; retired GAT/SAR/radar/count-density paths should stay historical unless explicitly discussed.
