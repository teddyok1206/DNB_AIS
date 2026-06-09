# Density Dual Inverse+PH Count Softplus Pilot - 2026-06-03

## Purpose

This pilot tests whether the previous inverse-radiance-plus-PH count branch failed mainly because of exponential count decoding.

Previous result:

```text
count branch = raw inverse_radiance + PH context
count_activation = exp
test pred/target = 7.06
```

This pilot keeps the same count input but changes:

```text
count_activation: exp -> softplus
```

No other intended methodology change was made.

## Config And Script

Config:

```text
configs/dnb_density_unet_dual_inverse_phcount_softplus.json
```

Script:

```text
scripts/run_density_dual_inverse_phcount_softplus_patchmix.sh
```

Code/config commit:

```text
58e1292 config(density): add softplus inverse PH count pilot
```

## Model Structure

Input channels:

```text
0 brightness
1 inverse_radiance
2 parent_ph_mask
3 child_union_mask
4 ph_seed_map
5 ph_persistence_map
6 ph_soft_attention
```

Branch inputs:

```text
spatial branch = [0, 2, 3, 4, 5, 6]
count branch   = [1, 2, 3, 4, 5, 6]
```

PH is still extracted from the encoded brightness scene, not from raw inverse radiance.

## Run

Run directory:

```text
outputs/dnb_density/runs/dual_inverse_phcount_softplus_pilot_20260603_014130
```

Run summary:

```text
outputs/dnb_density/runs/dual_inverse_phcount_softplus_pilot_20260603_014130/run_summary.json
```

Enhanced previews:

```text
outputs/dnb_density/runs/dual_inverse_phcount_softplus_pilot_20260603_014130/enhanced_previews/
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
git_commit: 58e1292
git_dirty: false
```

## Result

Final test:

```text
pred_sum=1497.34
target_sum=3630.00
pred/target=0.412
```

Final test component means:

```text
loss_total=1.2478
loss_count=1.1575
loss_spatial=1.6153
loss_density=0.000601
pred_count_mean=46.79
target_count_mean=113.44
```

Enhanced preview metrics over 12 test previews:

```text
target_explained=0.167
pred_matched=0.399
spatial_overlap=0.327
pred_sum_mean=45.81
target_sum_mean=137.42
```

## Comparison

Same split/patch-cap comparison:

```text
inverse only count branch:
  commit=2a071fd
  test pred/target=0.263

inverse + PH count branch, exp activation:
  commit=936a7ba
  test pred/target=7.062

inverse + PH count branch, softplus activation:
  commit=58e1292
  test pred/target=0.412
```

The softplus activation fixed the catastrophic over-counting caused by `exp`, but it did not restore the stronger count calibration seen in the earlier encoded-brightness baseline.

## Interpretation

The experiment supports two points:

```text
1. PH context is necessary for the count branch.
2. raw inverse_radiance + PH context is numerically unstable with exp count activation.
```

However, simply replacing `exp` with `softplus` makes the model conservative and under-counting:

```text
test pred/target=0.412
```

This suggests the raw inverse-radiance signal is not yet in a good learnable scale.

## Next Step

The inverse-radiance idea is still plausible, but the raw unscaled value is probably not the right representation.

Recommended next ablation:

```text
Keep the same dual branch and PH context.
Use a scaled inverse-radiance channel before the count branch.
Do not change PH extraction.
```

Candidate scaling options:

```text
1. inverse_radiance / 1e-9
2. inverse_radiance / scene or train percentile
3. log10 inverse_radiance with a documented floor
```

The least invasive next test is probably:

```text
count branch = inverse_radiance / 1e-9 + PH
count_activation = softplus
```

This keeps the no-log direction while avoiding the 1e-10 to 1e-8 numerical scale.
