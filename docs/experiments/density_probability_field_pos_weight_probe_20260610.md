# Probability Field Pixel Pos Weight Probe

Date: 2026-06-10

## Purpose

The previous probability-field probe used `pixel_pos_weight = 8.0`. It produced the best fixed-threshold field F1, but the predicted probability mass was too broad:

```text
pred_target_ratio ~= 6.48
pixel_pred_mean ~= 0.292
probability_target_mean ~= 0.048
```

This probe isolates `pixel_pos_weight` while keeping the radius-probability target fixed.

## Fixed Settings

```text
radius_probability_sigma_pixels = 4.0
radius_probability_radius_pixels = 12
probability_target_threshold = 0.25
pixel_weight = 0.9
occupancy_weight = 0.1
input_channels = brightness, ph_persistence_map, ph_seed_map
max_scenes_per_split = 5
max_patches_per_scene = 32
positive_patches_per_scene = 16
negative_patches_per_scene = 16
```

The `pos_weight=4`, `2`, and `6` probes reused the same patch cache:

```text
outputs/dnb_density/patch_caches/probability_field_posweight_probe_5scene_20260610_024330
```

The prior `pos_weight=8` no-cache probe used the same selected scene/patch target sums, so it is comparable.

## Probe Configs

```text
configs/dnb_density_unet_probability_field_recursive_ph_posw6_probe_20260610.json
configs/dnb_density_unet_probability_field_recursive_ph_posw4_probe_20260610.json
configs/dnb_density_unet_probability_field_recursive_ph_posw2_probe_20260610.json
```

## Results

Best checkpoint selection: `best_val_pixel_f1`.

| pixel_pos_weight | pred_target_ratio | pixel_pred_mean | pixel Brier | fixed field F1 | calibrated field F1 | calibrated threshold | sigma4 AP | sigma4 soft Brier | sigma8 AP | sigma8 calibrated F1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 6.480 | 0.2923 | 0.1078 | 0.2232 | 0.2128 | 0.45 | 0.1313 | 0.0848 | 0.2478 | 0.2666 |
| 6 | 5.588 | 0.2418 | 0.0896 | 0.1813 | 0.2047 | 0.31 | 0.1331 | 0.0651 | 0.2628 | 0.2789 |
| 4 | 4.405 | 0.1935 | 0.0742 | 0.1623 | 0.2021 | 0.22 | 0.1274 | 0.0484 | 0.2560 | 0.2781 |
| 2 | 2.615 | 0.1142 | 0.0581 | 0.0749 | 0.2059 | 0.14 | 0.1208 | 0.0298 | 0.2502 | 0.2709 |

## Interpretation

Lowering `pixel_pos_weight` does reduce over-broad probability mass. The monotonic pattern is clear:

```text
8 -> 6 -> 4 -> 2
pred_target_ratio: 6.48 -> 5.59 -> 4.41 -> 2.62
pixel Brier:       0.108 -> 0.090 -> 0.074 -> 0.058
```

However, lower weights also reduce peak confidence. At fixed threshold `0.5`, field recall drops too much:

```text
fixed field F1: 8=0.223, 6=0.181, 4=0.162, 2=0.075
```

`pixel_pos_weight=2` is too conservative. It improves Brier and mass ratio, but most useful predictions fall below the fixed threshold and require a very low calibrated threshold (`0.14`).

`pixel_pos_weight=4` is also conservative for fixed-threshold detection, though it is useful as a calibration sanity check.

`pixel_pos_weight=6` is the best compromise in this probe:

```text
sigma4 AP: 0.1331, slightly above pos_weight=8
sigma8 AP: 0.2628, above pos_weight=8
pixel Brier: 0.0896, improved from 0.1078
fixed field F1: 0.1813, lower than pos_weight=8 but not collapsed
```

## Decision

Do not switch all defaults blindly to `2` or `4`.

Use `pixel_pos_weight=6` as the next candidate for a larger run if the goal is probability-field quality and radius-ranking behavior. Keep `pixel_pos_weight=8` as the fixed-threshold F1 baseline.

The next larger run should compare only:

```text
pixel_pos_weight = 8
pixel_pos_weight = 6
```

Use the same split and the same patch-cache policy, then select by a combined readout:

```text
fixed field F1
sigma4 AP
sigma8 AP
pixel Brier
pred_target_ratio
```
