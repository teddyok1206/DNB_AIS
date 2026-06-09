# Probability Field Recursive PH Run

Date: 2026-06-10

## Decision

The active target is now a radius-tolerant ship-proximity probability field. Exact AIS pixel hit supervision is intentionally abandoned.

```text
source_pixel = 1[raw_count > 0]
target_probability = exp(-0.5 * distance_to_nearest_source_pixel^2 / sigma_pixels^2)
```

The field is truncated to `radius_pixels`, clipped to `[0, 1]`, and masked by the valid owner sea mask. This is not count-density smoothing: AIS points are exact seeds, and the model learns proximity probability rather than conserved ship mass.

## Rationale

The hard pixel run showed that patch-level O/X was learnable, but exact pixel localization was not a useful primary objective at the current DNB resolution and AIS alignment quality. A no-cache probe on 2026-06-10 produced:

```text
patch O/X calibrated F1: 0.625
exact pixel F1: 0.0
radius sigma=8 AP: 0.180
radius sigma=8 calibrated F1: 0.223
```

This means the model can learn coarse target presence and broad proximity, but exact pixel supervision is too brittle.

## Active Config

```text
configs/dnb_density_unet_probability_field_recursive_ph_20260610.json
```

Important loss settings:

```text
training.loss.name = radius_probability_occupancy_loss
training.loss.target_mode = radius_probability
radius_probability_sigma_pixels = 4.0
radius_probability_radius_pixels = 12
probability_target_threshold = 0.25
pixel_pos_weight = 8.0
pixel_weight = 0.9
occupancy_weight = 0.1
```

## Model And Inputs

The model remains `PixelBinaryOccupancyUNet` and still emits independent per-pixel sigmoid logits plus an auxiliary patch O/X head.

Active input channels remain fixed:

```text
brightness
ph_persistence_map
ph_seed_map
```

PH remains central for recursive proposal/patch construction and for the two physically interpretable PH feature channels. Broad parent/child masks, soft attention, and lifetime maps remain removed from U-Net inputs.

## Metrics

`pixel_*` metrics now mean thresholded probability-field metrics, not exact AIS-pixel hit metrics. The target threshold is recorded as `pixel_target_threshold`.

Primary readout:

```text
pixel_f1
pixel_iou
pixel_precision
pixel_recall
pixel_brier
radius_probability.by_sigma.*.eval_soft.average_precision
radius_probability.by_sigma.*.eval_soft.soft_target_explained
radius_probability.by_sigma.*.eval_soft.soft_pred_matched
```

Secondary readout:

```text
occupancy_f1: patch-level auxiliary behavior
pred_target_ratio: probability-mass calibration diagnostic, not count error
spatial_overlap_mean_positive: normalized map overlap diagnostic
```

## Operational Notes

The default runner now points to the probability-field config:

```text
scripts/run_density_pixel_binary_recursive_ph.sh
```

For quick feasibility checks, run without patch cache:

```text
--patch-cache-mode off
```

The checkpoint evaluator can now evaluate no-cache runs by rebuilding the required split patches in memory from `command.txt`; it does not write patch cache files in that path.
