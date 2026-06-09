# Probability Field Recursive PH Pipeline

Date: 2026-06-10

## Decision

The active target is now a pure per-pixel ship-presence probability map.

```text
source_pixel = 1[raw_count > 0]
target_probability = exp(-0.5 * distance_to_nearest_source_pixel^2 / sigma_pixels^2)
P_pixel = sigmoid(pixel_logits)
```

The Gaussian field is only a tolerance/proximity label around exact AIS seed pixels. It is not count-density smoothing, it is not mass preserving, and it is not a ship-count target.

## What Was Removed

The active pipeline no longer reports or optimizes:

```text
patch O/X auxiliary objective
ship-count regression
sum-preserving density/count mass
pred_target_ratio
soft_target_explained
soft_pred_matched
spatial_overlap_mean_positive
patch occupancy metrics as headline results
```

Scene/patch target sums can still appear in build logs as dataset sanity checks, but they are not model-quality metrics.

## Active Config

```text
configs/dnb_density_unet_probability_field_recursive_ph_20260610.json
```

Important settings:

```text
model.name = PixelProbabilityUNet
training.loss.name = radius_probability_loss
training.loss.target_mode = radius_probability
radius_probability_sigma_pixels = 4.0
radius_probability_radius_pixels = 12
probability_target_threshold = 0.25
pixel_weight = 1.0
occupancy_weight = 0.0
dice_weight = 0.0
```

## Inputs

Active input channels remain fixed:

```text
brightness
ph_persistence_map
ph_seed_map
```

PH remains useful in two places:

```text
1. recursive PH-guided patch/partition construction
2. physically interpretable PH persistence and seed feature channels
```

Broad parent/child masks, soft attention, anchor lifetime maps, and patch O/X heads are not active model inputs/objectives.

## Evaluation

The central question is:

```text
Are bright pixels in the learned probability map more likely to indicate AIS ship presence than bright pixels in the raw DNB image?
```

The checkpoint evaluator now compares model probability and raw brightness on the same valid pixels and the same binary presence target.

Primary readout:

```text
presence_probability.model_probability.average_precision
presence_probability.brightness_baseline.average_precision
presence_probability.model_vs_brightness_lift.average_precision_ratio
presence_probability.model_probability.precision_at_top.top_1pct.precision
presence_probability.model_probability.brier
presence_probability.model_calibrated_threshold.f1
```

Radius-tolerant sweep:

```text
radius_presence.by_sigma.*.model_probability.average_precision
radius_presence.by_sigma.*.brightness_baseline.average_precision
radius_presence.by_sigma.*.model_vs_brightness_lift.average_precision_ratio
```

Brightness is treated as a ranking baseline, not a calibrated probability, so Brier/calibration are reported for the model probability map only.

## Operational Notes

The default runner still uses:

```text
scripts/run_density_pixel_binary_recursive_ph.sh
```

For quick feasibility checks, run without patch cache:

```text
--patch-cache-mode off
```

For proper comparison, evaluate a checkpoint with:

```text
python -m sub_module.evaluate_density_checkpoint \
  --run-dir <run_dir> \
  --checkpoint best_val_pixel_f1 \
  --split test \
  --calibration-split val \
  --radius-sigmas 1,2,4,8
```

The resulting JSON schema is version 3 and is centered on `presence_probability` and `radius_presence`.
