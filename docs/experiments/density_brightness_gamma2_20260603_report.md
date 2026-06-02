# Density Brightness Gamma2 Pilot - 2026-06-03

## Purpose

This pilot tests a simpler alternative to dual radiance branches:

```text
Use one image radiance channel only.
Keep PH channels unchanged.
Replace encoded brightness with encoded_brightness ** 2.
```

The motivation is to emphasize bright pixels while preserving the original count-spatial U-Net structure.

## Transform

Baseline image channel:

```text
brightness = encoded_brightness
```

Pilot image channel:

```text
brightness_gamma_2 = encoded_brightness ** 2
```

No patchwise normalization is applied.

## Config And Script

Config:

```text
configs/dnb_density_unet_brightness_gamma2.json
```

Script:

```text
scripts/run_density_brightness_gamma2_patchmix.sh
```

Code/config commit:

```text
d605a66 config(density): add brightness gamma2 pilot
```

## Model Structure

Model:

```text
CountSpatialDensityUNet
```

Input channels:

```text
0 brightness_gamma_2
1 parent_ph_mask
2 child_union_mask
3 ph_seed_map
4 ph_persistence_map
5 ph_soft_attention
```

The count and spatial heads both receive the same transformed brightness plus PH context through the shared encoder.

PH extraction is still based on the original encoded brightness scene, not on the gamma2 channel.

## Run

Run directory:

```text
outputs/dnb_density/runs/brightness_gamma2_pilot_20260603_015702
```

Run summary:

```text
outputs/dnb_density/runs/brightness_gamma2_pilot_20260603_015702/run_summary.json
```

Enhanced previews:

```text
outputs/dnb_density/runs/brightness_gamma2_pilot_20260603_015702/enhanced_previews/
```

Settings:

```text
device: mps
epochs: 6
batch_size: 2
max_scenes_per_split: 6
max_patches_per_scene: 16
max_ph_patches_per_scene: 16
max_fallback_patches_per_scene: 0
max_patch_height: 512
max_patch_width: 512
```

Actual split after PH-anchor filtering:

```text
train: 5 kept scenes, 80 selected patches, target_sum=4957.0
val:   3 kept scenes, 48 selected patches, target_sum=3768.0
test:  2 kept scenes, 32 selected patches, target_sum=3630.0
```

Run git metadata:

```text
git_commit: d605a66
git_dirty: false
```

## Result

Final test:

```text
pred_sum=1806.23
target_sum=3630.00
pred/target=0.498
```

Final test component means:

```text
loss_total=1.0336
loss_count=0.6719
loss_spatial=1.6248
loss_density=0.000578
pred_count_mean=56.44
target_count_mean=113.44
```

Enhanced preview metrics over 12 test previews:

```text
target_explained=0.309
pred_matched=0.462
spatial_overlap=0.395
pred_sum_mean=64.72
target_sum_mean=137.42
```

## Comparison

Same split/patch-cap comparison:

```text
raw inverse only count branch:
  test pred/target=0.263

raw inverse + PH count branch, softplus:
  test pred/target=0.412

brightness_gamma_2 + PH, shared baseline model:
  test pred/target=0.498
```

The gamma2 single-radiance model improves over the raw inverse experiments, but it remains below the earlier encoded-brightness count-spatial baseline recorded in:

```text
docs/experiments/density_count_spatial_good_baseline_20260602.md
```

That baseline reported:

```text
test pred/target=3008.14/3947.00=0.762
```

The baseline run used a similar but not identical test target total, so treat this as directional rather than a strict controlled comparison.

## Interpretation

The gamma2 transform is stable and improves count relative to raw inverse variants.

However, it still under-counts:

```text
test pred/target=0.498
```

Preview-level counts are also compressed:

```text
pred_sum range:   56.55 to 72.74
target_sum range: 54.00 to 455.00
```

This suggests `encoded ** 2` may over-suppress mid-bright structure and still does not give the count head enough dynamic range for dense patches.

## Next Step

The single-radiance direction is cleaner than dual raw inverse branches.

Recommended next ablations:

```text
1. brightness_gamma_1p5 + PH
2. brightness + bright_residual channel is not allowed if we want one radiance channel, so avoid for now
3. train for more epochs only after the transform choice is better
```

The immediate next controlled test should be:

```text
brightness_gamma_1p5 = encoded_brightness ** 1.5
```

This keeps bright emphasis but should preserve more mid-bright signal than gamma2.
