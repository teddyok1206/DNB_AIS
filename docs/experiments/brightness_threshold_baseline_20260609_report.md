# Brightness Threshold Baseline Comparison - 2026-06-09

## Purpose

Compare the active PH-assisted OccupancySpatial U-Net against a simple rule-based brightness threshold baseline on the same held-out day split and the same patch-selection policy.

This baseline is intentionally simple:

```text
pred_positive_patch = any(encoded_brightness >= threshold over valid owner pixels)
pred_spatial_map = normalize(binary_threshold_mask over valid owner pixels)
```

The purpose is not to reproduce a historical legacy method, but to provide a defensible non-learning baseline for the final report.

## Runs

Model run:

```text
outputs/dnb_density/runs/ox_spatial_25pct_63_15_14_20260609_011421
```

Baseline run:

```text
outputs/dnb_density/baselines/brightness_threshold_ox_spatial_25pct_valtest_20260609_0242
```

Shared scene split:

```text
outputs/dnb_density/splits/ox_spatial_25pct_63_15_14_20260609_011421/scene_split.csv
```

Baseline threshold candidates:

```text
0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.90, 0.95
```

The high thresholds `0.85, 0.90, 0.95` are retained for interpretability, but the sweep includes lower thresholds because current arctan-encoded GeoTIFF brightness often peaks below `0.85`.

## Patch Counts

| split | kept patches | positive patches | negative patches |
|---|---:|---:|---:|
| validation | 1,222 | 598 | 624 |
| test | 1,017 | 489 | 528 |

Excluded scenes matched the active pipeline policy:

```text
validation: ph_anchor_count_zero=13, raster-overlap exception=1
test: ph_anchor_count_zero=8, raster-overlap exception=1
```

## Validation Threshold Selection

The selected threshold is `0.35`, chosen by highest validation `occupancy_f1`.

| threshold | val precision | val recall | val F1 | val Brier | val pred/target |
|---:|---:|---:|---:|---:|---:|
| 0.35 | 0.4842 | 0.7709 | 0.5948 | 0.5139 | 1.5920 |
| 0.45 | 0.4988 | 0.7074 | 0.5851 | 0.4910 | 1.4181 |
| 0.55 | 0.5243 | 0.6672 | 0.5872 | 0.4591 | 1.2726 |
| 0.65 | 0.5445 | 0.6037 | 0.5726 | 0.4411 | 1.1087 |
| 0.75 | 0.5607 | 0.5251 | 0.5423 | 0.4337 | 0.9365 |
| 0.85 | 0.5852 | 0.4766 | 0.5253 | 0.4214 | 0.8144 |
| 0.90 | 0.6023 | 0.4431 | 0.5106 | 0.4157 | 0.7358 |
| 0.95 | 0.6387 | 0.3311 | 0.4361 | 0.4190 | 0.5184 |

## Test Comparison

The table compares the validation-selected brightness threshold baseline against the final epoch-12 active model.

| method | threshold | precision | recall | F1 | accuracy | Brier | TP | FP | FN | TN | pred/target | spatial overlap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| brightness threshold | 0.35 | 0.4656 | 0.7342 | 0.5698 | 0.4671 | 0.5329 | 359 | 412 | 130 | 116 | 1.5767 | 0.0507 |
| PH-assisted OccupancySpatial U-Net | n/a | 0.6298 | 0.7342 | 0.6780 | 0.6647 | 0.2178 | 359 | 211 | 130 | 317 | 1.0097 | 0.0884 |

## Interpretation

The active model preserves the same test recall as the selected brightness baseline while cutting false positives from `412` to `211`.

This is the key result:

```text
same TP and FN, much lower FP
```

Quantitatively:

```text
F1 improvement: +0.1082 absolute, +19.0% relative
precision improvement: +0.1642 absolute
accuracy improvement: +0.1976 absolute
Brier reduction: 59.1%
false-positive reduction: 48.8%
```

The brightness baseline can recover many true positives only by over-detecting bright structures. The U-Net uses PH context and learned spatial evidence to suppress many of those false positives without sacrificing recall on this test split.

Spatial overlap is still modest for both methods, but the model improves it from `0.0507` to `0.0884`. This supports the current direction: first stabilize occupancy, then improve localization/peak sharpness.

## Caveat

The model numbers above are from the final epoch-12 checkpoint. The best-validation-F1 checkpoint is epoch 8 and should be re-evaluated on test before final reporting.
