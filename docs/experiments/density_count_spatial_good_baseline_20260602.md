# Density Count-Spatial Good Baseline - 2026-06-02

## Baseline Purpose

This file locks the first qualitatively good DNB density-map baseline before further model/loss changes.

The important observation is not count calibration yet. The important observation is that the model started to produce a spatial density pattern that closely follows the AIS-derived target density on held-out test patches.

This is the rollback point if later experiments damage the spatial allocation behavior.

## Git State

Good code baseline before this note:

```text
f4556b7 feat(density): add count-spatial U-Net pilot
```

Recommended rollback tag after this note is committed:

```text
density-count-spatial-good-20260602
```

The run was produced after the count-spatial code/config had been implemented. The original run metadata records a dirty pre-commit state, but the relevant code/config was later committed in `f4556b7`.

## Model State

Model:

```text
CountSpatialDensityUNet
```

Prediction factorization:

```text
pred_density = predicted_count * masked_spatial_probability
```

Loss:

```text
CountSpatialDensityLoss
```

Configured objective:

```text
loss = count_weight * log_count_mse
     + spatial_weight * spatial_KL
     + density_weight * weak_density_huber
     + background_weight * weak_background
```

Current config:

```text
../configs/dnb_density_unet_count_spatial.json
```

Main code files:

```text
../sub_module/dnb_density_models.py
../sub_module/dnb_density_losses.py
../sub_module/run_density_split_smoke_train.py
```

Input channels:

```text
brightness
parent_ph_mask
child_ph_union_mask
ph_seed_map
ph_persistence_map
ph_soft_attention
```

## Pilot Run

Output directory:

```text
outputs/density_pilot_20260602_unet_mps_count_spatial_v2
```

Run summary:

```text
outputs/density_pilot_20260602_unet_mps_count_spatial_v2/run_summary.json
```

Scene metrics:

```text
outputs/density_pilot_20260602_unet_mps_count_spatial_v2/scene_metrics.csv
```

Inference previews:

```text
outputs/density_pilot_20260602_unet_mps_count_spatial_v2/inference_previews/
```

Run settings:

```text
device=mps
seed=20260529
epochs=6
batch_size=2
max_scenes_per_split=6
max_patches_per_scene=16
max_patch_height=512
max_patch_width=512
skip_ph_anchor_zero=true
```

Split size actually used:

```text
train: 6 scenes, 96 selected patches
val:   3 kept scenes, 48 selected patches
test:  2 scenes, 32 selected patches
```

Test scenes:

```text
A2025009_1706_021
A2025125_1730_021
```

## Quantitative Snapshot

Count-spatial v2 aggregate test result:

```text
pred_sum=3008.14
target_sum=3947.00
pred/target=0.7621
```

This still undercounts globally after 6 epochs, but the spatial allocation is the important retained behavior.

## Key Qualitative Preview

Preview selected by visual inspection:

```text
outputs/density_pilot_20260602_unet_mps_count_spatial_v2/inference_previews/inference_preview_007_ph_anchor_113.png
```

Patch identity:

```text
split=test
scene=A2025009_1706_021
partition_kind=ph_anchor
partition_id=113
target_sum=90.00
pred_sum=155.74
```

Interpretation:

```text
The prediction overcounts this patch, but the spatial density pattern aligns surprisingly well with target density.
For the research objective, this is valuable because the spatial map can be normalized and interpreted as a ship-location probability surface.
```

## Leakage Check Snapshot

The preview was not from the train split. It came from the test split.

Input feature to target relationship for preview 007:

```text
brightness        corr=0.513, top200_hit=0.925
parent_ph_mask    corr=0.483, top200_hit=0.840
child_union_mask  corr=0.483, top200_hit=0.840
persistence_map   corr=0.489, top200_hit=0.760
soft_attention    corr=0.353, top200_hit=0.670
seed_map          corr=0.010, top200_hit=0.380
```

Current conclusion:

```text
No direct target-density input leakage was found in the model input tensor.
However, smoke-test patch selection is GT-biased because selected patches are prioritized by raw_count_sum.
This does not invalidate the preview as a held-out patch result, but it means full-scene/random-patch evaluation is still required.
```

## Preserve This Behavior

Do not lose these properties in later changes:

```text
1. Keep the count/spatial factorization available.
2. Keep MPS-compatible training/inference.
3. Keep the six-channel brightness+PH input path available.
4. Keep the count-spatial config path available.
5. Keep preview generation available for side-by-side brightness, soft_attention, target, prediction, error, and valid mask.
```

Count calibration should be improved later, but not by destroying the clean spatial allocation behavior.

## Rollback

To inspect the locked baseline without overwriting the current working branch:

```text
git switch -c inspect-density-count-spatial-good density-count-spatial-good-20260602
```

To compare later code against this baseline:

```text
git diff density-count-spatial-good-20260602..HEAD -- \
  sub_module/dnb_density_models.py \
  sub_module/dnb_density_losses.py \
  sub_module/run_density_split_smoke_train.py \
  configs/dnb_density_unet_count_spatial.json
```

Runtime artifacts are intentionally not committed. If outputs are deleted, the code/config state can be restored from git, but preview PNGs must be regenerated.
