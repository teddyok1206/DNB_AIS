# Density Stabilized Inverse P99 Pilot - 2026-06-03

## Purpose

This pilot tests the mathematically direct inverse of the STEP3 arctan compression, while stabilizing the tail.

Original STEP3 encoding:

```text
encoded = (2 / pi) * arctan(radiance / 1e-9)
```

Exact inverse ratio:

```text
ratio = radiance / 1e-9 = tan((pi / 2) * encoded)
```

Pilot channel:

```text
stabilized_inverse = clip(ratio / train_p99, 0, 1)
```

PH extraction remains based on the original encoded brightness scene.

## Cap Estimation

Cap source:

```text
first 6 train scenes from outputs/dnb_density/splits/density_smoke_split_10_3_2/scene_split.csv
```

Sampling method:

```text
positive finite pixels, deterministic stride to about 1M samples per scene
```

Sample count:

```text
6027309
```

Estimated percentiles for:

```text
ratio = tan((pi / 2) * encoded)
```

Global sampled values:

```text
p95  = 1.225919645353719
p99  = 5.478349152183705
p995 = 11.092256298298269
p999 = 37.79361212829353
max  = 5225.394961102604
```

Chosen cap:

```text
train_p99 = 5.478349152183705
```

## Config And Script

Config:

```text
configs/dnb_density_unet_stabilized_inverse_p99.json
```

Script:

```text
scripts/run_density_stabilized_inverse_p99_patchmix.sh
```

Code/config commit:

```text
a6ee5ce config(density): add stabilized inverse p99 pilot
```

## Model Structure

Model:

```text
CountSpatialDensityUNet
```

Input channels:

```text
0 inverse_ratio_p99_5p478349
1 parent_ph_mask
2 child_union_mask
3 ph_seed_map
4 ph_persistence_map
5 ph_soft_attention
```

The count and spatial heads share the same stabilized inverse radiance channel plus PH context.

## Run

Run directory:

```text
outputs/dnb_density/runs/stabilized_inverse_p99_pilot_20260603_021320
```

Run summary:

```text
outputs/dnb_density/runs/stabilized_inverse_p99_pilot_20260603_021320/run_summary.json
```

Enhanced previews:

```text
outputs/dnb_density/runs/stabilized_inverse_p99_pilot_20260603_021320/enhanced_previews/
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
git_commit: a6ee5ce
git_dirty: false
```

## Result

Final test:

```text
pred_sum=1788.50
target_sum=3630.00
pred/target=0.493
```

Final test component means:

```text
loss_total=1.0220
loss_count=0.6774
loss_spatial=1.5936
loss_density=0.000578
pred_count_mean=55.89
target_count_mean=113.44
```

Enhanced preview metrics over 12 test previews:

```text
target_explained=0.298
pred_matched=0.446
spatial_overlap=0.387
pred_sum_mean=63.42
target_sum_mean=137.42
```

## Comparison

Same split/patch-cap comparison:

```text
raw inverse only count branch:
  pred/target=0.263

raw inverse + PH count branch, softplus:
  pred/target=0.412

brightness_gamma_2 + PH:
  pred/target=0.498

stabilized inverse p99 + PH:
  pred/target=0.493
```

The stabilized inverse p99 channel performs almost the same as gamma2. It is stable and mathematically more interpretable, but it does not recover the stronger count calibration expected from inverse radiance.

## Interpretation

The result suggests that the problem is no longer numerical explosion, but count dynamic-range compression after stabilization.

Preview-level counts remain compressed:

```text
pred_sum range:   56.40 to 70.16
target_sum range: 54.00 to 455.00
```

The p99 cap makes the input safe but may clip or saturate exactly the tail information needed to distinguish dense ship groups.

## Next Step

The inverse family should not be discarded yet, but p99 may be too aggressive.

Potential next ablations:

```text
1. stabilized inverse p99.5
2. stabilized inverse p99.9
3. two-channel input: encoded brightness + stabilized inverse
```

If preserving one radiance channel is required, the next mathematically consistent test is:

```text
stabilized_inverse_p99p5 = clip(tan((pi/2)*encoded) / 11.092256298298269, 0, 1)
```

This should retain more high-brightness dynamic range than p99 while still preventing the extreme tail from dominating.
