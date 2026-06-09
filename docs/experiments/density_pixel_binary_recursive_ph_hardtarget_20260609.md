# Pixel Binary Recursive PH Hard-Target Run

Date: 2026-06-09

## Decision

The active direction is no longer patch-level O/X plus spatial softmax. The main supervised target is hard pixel occupancy:

```text
target_pixel = 1 if raw_count > 0 else 0
```

The target is masked by the partition owner sea mask. Gaussian GT smoothing remains available only for legacy preview and compatibility paths; it is not the main label for this run.

## Pipeline Changes

1. Recursive PH partitioning

Large PH anchors are recursively reprocessed by local PH until they are below the configured target size or the max depth is reached. If a child PH region remains too large, it is not emitted as a training patch when recursive mode is enabled. Its descendants or fallback grid patches cover that area instead.

2. Smaller exact-cover patches

The fallback grid is reduced to 64 px with 16 px halo. PH anchor padding is reduced to 12 px, and recursive child padding is 6 px. This makes patch-level O/X less dominant and gives pixel supervision a more meaningful local context.

3. Hard pixel target

The collate path now carries `raw_count`, and `pixel_binary_occupancy_loss` builds the label from `raw_count > 0`. The smoothed target is not consumed by the loss.

4. Independent pixel probability

`PixelBinaryOccupancyUNet` emits `pixel_logits`. The prediction is `sigmoid(pixel_logits)`, not a spatial softmax. There is no forced per-patch probability mass allocation.

5. Auxiliary options are separated

Patch-level O/X remains as a small auxiliary regularizer. Dice and smoothed-density paths are explicit optional knobs, disabled in the first run.

## Primary Metrics

Use these as the main readout:

- `pixel_f1`
- `pixel_iou`
- `pixel_precision`
- `pixel_recall`
- `pixel_brier`

Checkpoint selection should prefer `best_val_pixel_f1` for this family.

## Secondary Metrics

Use these only for diagnostics:

- `occupancy_f1`: patch-level auxiliary behavior.
- `spatial_overlap_mean_positive`: legacy overlap between normalized predicted and target maps. It is not the objective for this run because the new pixel head is independent sigmoid, not a per-patch probability distribution.
- `pred_target_ratio`: now compares predicted positive probability mass against positive pixel count, not vessel count.

## Config

Config path:

```text
configs/dnb_density_unet_pixel_binary_recursive_ph_hardtarget_20260609.json
```

Important separation:

- Main target: `primary_target.definition`
- Main loss: `training.loss.name = pixel_binary_occupancy_loss`
- Recursive PH: `partitioning.hierarchical_ph.recursive`
- Auxiliary-only knobs: `auxiliary_options`

## First Experiment Protocol

Use the fixed 25% day split:

```text
outputs/dnb_density/splits/ox_spatial_25pct_63_15_14_20260609_011421/scene_split.csv
```

Recommended first non-smoke settings:

```text
epochs: 18
batch_size: 8
max_patches_per_scene: 64
positive_patches_per_scene: 32
negative_patches_per_scene: 32
max_patch_height: 160
max_patch_width: 160
checkpoint: best_val_pixel_f1
```

If the first run collapses to low recall, the next controlled change should be `pixel_pos_weight` or enabling the disabled Dice auxiliary. Do not reintroduce smoothed labels as the main target unless the hard-label premise is intentionally abandoned.
