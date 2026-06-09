# Probability Field Pixel Pos Weight Probe

Date: 2026-06-10

## Status

This probe was run before the metric cleanup. Its numeric table is retained as historical context for choosing `pixel_pos_weight`, but the old mass/soft-overlap columns are no longer active success criteria.

## Fixed Settings At Probe Time

```text
radius_probability_sigma_pixels = 4.0
radius_probability_radius_pixels = 12
probability_target_threshold = 0.25
input_channels = brightness, ph_persistence_map, ph_seed_map
max_scenes_per_split = 5
max_patches_per_scene = 32
positive_patches_per_scene = 16
negative_patches_per_scene = 16
```

The probe cache was:

```text
outputs/dnb_density/patch_caches/probability_field_posweight_probe_5scene_20260610_024330
```

Probe configs remain available:

```text
configs/dnb_density_unet_probability_field_recursive_ph_posw6_probe_20260610.json
configs/dnb_density_unet_probability_field_recursive_ph_posw4_probe_20260610.json
configs/dnb_density_unet_probability_field_recursive_ph_posw2_probe_20260610.json
```

## Historical Results

| pixel_pos_weight | old fixed field F1 | old calibrated field F1 | old calibrated threshold | old sigma4 AP | old sigma8 AP | old pixel Brier |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 0.2232 | 0.2128 | 0.45 | 0.1313 | 0.2478 | 0.1078 |
| 6 | 0.1813 | 0.2047 | 0.31 | 0.1331 | 0.2628 | 0.0896 |
| 4 | 0.1623 | 0.2021 | 0.22 | 0.1274 | 0.2560 | 0.0742 |
| 2 | 0.0749 | 0.2059 | 0.14 | 0.1208 | 0.2502 | 0.0581 |

## Interpretation

Lowering `pixel_pos_weight` made the probability field less broad and improved Brier, but too-low weights reduced useful confidence at fixed threshold.

The best next candidates are still:

```text
pixel_pos_weight = 8
pixel_pos_weight = 6
```

However, selection should now use the cleaned presence-ranking evaluator:

```text
model AP vs brightness AP
model Top-1% precision vs brightness Top-1% precision
model Brier and reliability bins
validation-calibrated threshold F1
radius_presence AP lift over brightness
```

Do not use `pred_target_ratio`, `soft_target_explained`, or `soft_pred_matched` as active decision metrics.

## New Presence-Ranking Smoke Eval

After the metric cleanup, the old `posw6` checkpoint was re-evaluated with schema v3:

```text
run: outputs/dnb_density/runs/probability_field_posweight_probe_20260610_024330/posw6
checkpoint: best_val_pixel_f1
split: test
output: evaluations/presence_schema_v3_smoke_eval.json
```

Main presence target (`pixel target >= 0.25`):

| score | AP | Top-1% precision | Top-5% precision | Brier |
| --- | ---: | ---: | ---: | ---: |
| model probability | 0.1326 | 0.2799 | 0.2200 | 0.0896 |
| raw brightness baseline | 0.1985 | 0.6449 | 0.2494 | n/a |

Radius presence `sigma=8`:

| score | AP | Top-1% precision | Top-5% precision |
| --- | ---: | ---: | ---: |
| model probability | 0.2547 | 0.4780 | 0.4316 |
| raw brightness baseline | 0.2922 | 0.8590 | 0.4538 |

Interpretation: the old probability-field model is smoother and somewhat useful, but it does not yet beat raw DNB brightness at the actual project claim. Future experiments should be selected by AP/Top-k lift over brightness, not by old fixed-threshold field F1 alone.
