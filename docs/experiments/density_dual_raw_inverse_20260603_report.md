# Density Dual Raw-Inverse Radiance Pilot - 2026-06-03

## Purpose

This pilot tests the hypothesis that arctan-compressed DNB brightness makes ship-count regression unnecessarily hard.

The STEP3 GeoTIFF brightness encoding is:

```text
encoded = (2 / pi) * arctan(radiance / 1e-9)
```

The pilot adds the exact inverse radiance channel:

```text
inverse_radiance = 1e-9 * tan((pi / 2) * encoded)
```

Per the experiment request, this first ablation intentionally does not apply:

```text
clip
maximum/floor
log transform
robust normalization
```

## Implemented Structure

The active baseline factorization is preserved:

```text
pred_density = predicted_count * predicted_spatial_probability
```

New model:

```text
DualRadianceCountSpatialUNet
```

Branch structure:

```text
spatial branch input:
  encoded brightness + PH hierarchy channels

count branch input:
  raw inverse_radiance

output:
  spatial_logits
  count
  count_raw
```

The count branch also receives raw inverse-radiance summary statistics:

```text
sum
mean
max
std
```

These are computed directly from raw inverse radiance without clipping, flooring, log transform, or normalization.

## Code And Config

Code commit:

```text
2a071fd feat(density): add raw inverse radiance count branch
```

Main files:

```text
configs/dnb_density_unet_dual_raw_inverse.json
scripts/run_density_dual_raw_inverse_patchmix.sh
sub_module/dnb_density_common.py
sub_module/dnb_density_models.py
sub_module/run_density_split_smoke_train.py
sub_module/render_density_enhanced_previews.py
```

## Smoke Test

Run:

```text
outputs/dnb_density/runs/dual_raw_inverse_smoke_20260603_011019
```

Settings:

```text
train/val/test scenes: 1/1/1
epochs: 1
batch_size: 1
max_patches_per_scene: 2
device: mps
```

Result:

```text
MPS forward/backward/checkpoint/preview completed.
No immediate nan/inf failure occurred on the sampled scenes.
```

This smoke run is not a performance benchmark.

## Comparison Pilot

Run:

```text
outputs/dnb_density/runs/dual_raw_inverse_pilot_20260603_011326
```

Run summary:

```text
outputs/dnb_density/runs/dual_raw_inverse_pilot_20260603_011326/run_summary.json
```

Enhanced previews:

```text
outputs/dnb_density/runs/dual_raw_inverse_pilot_20260603_011326/enhanced_previews/
```

Enhanced preview metrics:

```text
outputs/dnb_density/runs/dual_raw_inverse_pilot_20260603_011326/enhanced_previews/enhanced_preview_metrics.csv
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

Git metadata recorded by the run:

```text
git_commit: 2a071fd61bbc3944375a444120fde12b9b498cf5
git_dirty: false
```

## Quantitative Result

Final test:

```text
pred_sum=953.96
target_sum=3630.00
pred/target=0.263
```

Final test component means:

```text
loss_total=1.4261
loss_count=1.5406
loss_spatial=1.6284
loss_density=0.000529
pred_count_mean=29.81
target_count_mean=113.44
```

Enhanced preview metric means over the first 12 test previews:

```text
target_explained=0.145
pred_matched=0.462
spatial_overlap=0.355
pred_sum_mean=29.50
target_sum_mean=137.42
```

## Interpretation

The raw inverse radiance channel did not improve count calibration in this direct form.

The most important symptom is:

```text
preview pred_sum is nearly flat around 28-30 even when target_sum ranges from 54 to 455
```

That means the count branch did not learn a useful monotonic patch-count mapping from the unnormalized raw inverse-radiance input during this pilot.

The hypothesis that arctan compression damages count information is still plausible, but this experiment shows that feeding raw inverse radiance without scaling is not enough.

Likely causes:

```text
1. Raw inverse radiance values are very small, usually around 1e-10 to 1e-8.
2. The count branch receives radiance at a numerically awkward scale.
3. The count head uses exp activation, so small count_raw shifts can dominate optimization.
4. No log or robust normalization means the model sees a difficult dynamic range directly.
5. PH-biased patch selection and only 6 epochs remain too small for a definitive conclusion.
```

## Next Decision

Do not discard the inverse-radiance idea.

Recommended next ablation:

```text
Keep the same dual-branch model.
Replace raw inverse_radiance with normalized inverse-radiance features.
Try log10 inverse radiance or train-set robust z-score / percentile scaling.
Keep encoded brightness for spatial PH behavior.
Compare against the locked count-spatial baseline using the same split and patch caps.
```

This pilot is useful as a negative result: the problem is not solved by simply inverting arctan and passing the raw radiance map into the count branch.
