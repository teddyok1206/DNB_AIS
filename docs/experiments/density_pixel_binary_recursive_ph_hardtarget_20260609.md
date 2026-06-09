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

5. Minimal physically interpretable input channels

The active U-Net input is fixed to three channels:

```text
brightness
ph_persistence_map
ph_seed_map
```

Parent PH masks and child PH union masks are not model inputs and are not stored in patch/batch memory after target construction. Soft attention maps and anchor lifetime maps are removed from the active path. These features are too easy for the network to use as broad proposal shortcuts and are less physically direct than brightness, persistence, and seed evidence.

6. Auxiliary options are separated

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
- Main input channels: `patching.input_channels = [brightness, ph_persistence_map, ph_seed_map]`
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

## First Experiment Result

Run directory:

```text
outputs/dnb_density/runs/pixel_binary_recursive_ph_hardtarget_e18_20260609_194707
```

Run metadata:

- Git commit: `43594083cd54c866337bc3a83000ee6fdfc8dd22`
- Git dirty: `false`
- Device: `mps`
- Epochs: `18`
- Checkpoint evaluated on test: `best_val_pixel_f1`
- Config hash: `f509ae8e5a5fbe85893797367a76d2e1cfecb4f12592458ed95db123178d3277`
- Train patches: `7106`
- Val patches: `1688`
- Test patches: `1354`

Best validation checkpoints:

- Best val loss: epoch `7`, `0.5660`
- Best val occupancy F1: epoch `17`, `0.7260`
- Best val pixel F1: epoch `14`, `0.0631`
- Best val pixel IoU: epoch `14`, `0.0326`
- Best val pixel precision: epoch `14`, `0.0363`
- Best val pixel recall: epoch `11`, `0.3072`

Final epoch 18 validation:

- Occupancy F1: `0.7083`
- Pixel F1: `0.0573`
- Pixel IoU: `0.0295`
- Pixel precision: `0.0322`
- Pixel recall: `0.2577`
- Pixel Brier: `0.0173`

Test evaluation for `best_val_pixel_f1` at threshold `0.5`:

- Occupancy F1: `0.6616`
- Occupancy precision: `0.6600`
- Occupancy recall: `0.6631`
- Pixel F1: `0.0514`
- Pixel IoU: `0.0264`
- Pixel precision: `0.0284`
- Pixel recall: `0.2707`
- Pixel Brier: `0.0203`
- Pixel positives: target `2379`, predicted `22700`, TP `644`, FP `22056`, FN `1735`

Test evaluation with pixel threshold calibrated on validation:

- Threshold: `0.925`
- Pixel F1: `0.1048`
- Pixel IoU: `0.0553`
- Pixel precision: `0.0848`
- Pixel recall: `0.1370`
- Pixel positives: target `2379`, predicted `3845`, TP `326`, FP `3519`, FN `2053`

Interpretation:

- The hard pixel target pipeline runs end-to-end and produces a cleaner scientific question than the previous smoothed-density target.
- Patch-level context is learnable; occupancy F1 reaches about `0.72` on validation and improves to `0.7213` on test after occupancy-threshold calibration.
- Pixel localization remains the bottleneck. At threshold `0.5`, the model overpredicts positive pixels by roughly `9.5x` on test.
- Pixel-threshold calibration roughly doubles pixel F1 from `0.0514` to `0.1048`, but it does so by raising precision and sacrificing recall.
- Pixel accuracy is not meaningful because positive pixels are about `0.1%` of valid pixels.
- Legacy mass and overlap diagnostics are not primary metrics for this family; they mostly expose probability-mass calibration problems.

Next controlled changes:

- Add calibrated pixel-threshold reporting as a standard output for this experiment family.
- Inspect qualitative FP/FN previews before changing the model; current previews show broad positive probability in some valid fallback regions.
- Run a small controlled loss sweep before adding architecture complexity: `pixel_pos_weight` around `128`, `256`, and `512`, and optionally one Dice/Tversky auxiliary run.
- Consider a separately reported tolerance-band pixel metric if exact AIS-to-DNB pixel alignment is too strict, but keep hard labels as the main target unless that premise is intentionally changed.

## Input Channel Simplification

After the first result, the active input policy was simplified permanently:

```text
keep:
  brightness
  ph_persistence_map
  ph_seed_map

remove as U-Net inputs:
  parent_ph_mask
  child_ph_union_mask
  ph_soft_attention
  anchor_lifetime_map
```

Rationale:

- `brightness` is the actual DNB observation.
- `ph_persistence_map` captures physically meaningful topological strength.
- `ph_seed_map` gives sparse local PH anchor evidence.
- Broad binary PH masks and soft attention can encourage false-positive blobs over proposal regions.
- Anchor lifetime is mostly patch metadata and does not provide pixel-local physical evidence.

PH remains central to recursive proposal generation and exact-cover patch construction. It is no longer exposed to the U-Net as broad binary masks or soft-attention fields.

Implementation note: parent/child PH masks are now transient arrays only. New patch caches store the compact patch schema; older schema caches are compacted immediately after load.

## Radius-Tolerant Post-Evaluation

The active training target remains hard pixel occupancy:

```text
target_pixel = 1[raw_count > 0]
prediction = sigmoid(pixel_logits)
```

For diagnostics only, checkpoint evaluation now also runs a Gaussian radius-tolerant probability sweep. This does not change the loss or checkpoint selection.

Evaluation transform:

```text
pred_radius = masked_gaussian_smooth(sigmoid(pixel_logits), sigma)
target_radius = exp(-0.5 * distance_to_nearest_positive_pixel^2 / sigma^2)
target_binary_for_radius_f1 = target_radius >= 0.25
```

Default sweep:

```text
sigma_pixels = 1, 2, 4, 8
radius_pixels = ceil(3 * sigma_pixels)
```

Interpretation:

- Low hard pixel F1 but better radius F1 means the model is spatially near the AIS label but not exact-pixel aligned.
- Low hard pixel F1 and low radius F1 means the model is not detecting the target.
- High radius recall with poor precision means the model is producing broad false-positive probability fields.
- The primary report should still keep hard pixel metrics separate from radius-tolerant metrics.
