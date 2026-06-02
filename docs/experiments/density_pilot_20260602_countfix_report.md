# Density Count-Fix Pilot Report - 2026-06-02

## Purpose

The previous U-Net pilot verified that the training pipeline runs on MPS, but the model collapsed toward near-zero density:

```text
old test pred_sum / target_sum = 18.60 / 3947.00 = 0.004711
```

This pilot changed the loss so that predicting almost zero density is no longer a cheap solution.

## Code Changes

Files changed:

```text
../sub_module/dnb_density_losses.py
../sub_module/run_density_split_smoke_train.py
../configs/dnb_density_unet_countfix.json
```

Loss additions:

```text
log-count MSE
foreground-positive density loss
positive-target local count loss
empty-window false-positive loss
robust masked reductions that avoid 0 * NaN contamination
```

Runner additions:

```text
grad_clip_norm from training config
non-finite loss guard
grad_clip_norm recorded in run_summary.json
```

Count-fix config:

```text
count_loss = log_mse
lr = 1e-4
grad_clip_norm = 1.0
pixel_weight = 0.10
foreground_weight = 0.20
count_weight = 0.30
batch_count_weight = 0.12
local_count_weight = 0.15
empty_count_weight = 0.02
background_weight = 0.01
positive_local_count_only = true
local_count_windows = [32, 64, 128]
```

## Numerical Result

Same split and same scene cap were used:

```text
scene split: outputs/density_pilot_20260602_10_3_2/scene_split.csv
train input scenes: 6
val input scenes: 6
test input scenes: 2
patches: train 96, val 48, test 32
```

Old loss:

```text
split  loss_mean  pred_sum  target_sum  pred/target
train  0.051538   76.60     4174.00     0.018353
val    0.068738   36.67     3877.00     0.009459
test   0.081886   18.60     3947.00     0.004711
```

Count-fix v3:

```text
split  loss_mean  pred_sum  target_sum  pred/target
train  0.817691   5861.13   4174.00     1.404199
val    0.424027   3413.77   3877.00     0.880518
test   0.602606   4076.18   3947.00     1.032730
```

Conclusion:

```text
Count-collapse is fixed for this pilot.
```

The old loss made near-zero prediction look numerically good. The new log-count/foreground/local-count loss keeps strong pressure on the integral count.

## Qualitative Result

Representative preview:

```text
outputs/density_pilot_20260602_unet_mps_countfix_v3/inference_previews/inference_preview_000_ph_anchor_25.png
pred_sum = 571.01
target_sum = 656.00
```

This is a large improvement over the old run:

```text
old pred_sum = 0.91
old target_sum = 656.00
```

However, spatial localization is still weak. The prediction often spreads along PH soft-attention/coastline-like structure instead of concentrating on the GT density peaks.

Current interpretation:

```text
The loss now preserves count, but the model can satisfy count by distributing mass over broad PH/brightness structures.
```

So the next problem is no longer count-collapse. The next problem is spatial localization.

## Failed Intermediate

The first count-fix attempt used stronger weights and no robust invalid-pixel masking. It produced non-finite values.

The final v3 run succeeded after:

```text
lower lr
gradient clipping
weaker local/count weights
mask-safe loss reductions
non-finite loss guard
```

## Next Step

The next loss/model ablation should target localization:

```text
1. Reduce PH soft-attention dominance in loss weighting.
2. Try removing or weakening ph_soft_attention as an input channel.
3. Increase foreground/peak-local density supervision relative to broad count conservation.
4. Add a small high-target weighted pixel term so GT peaks are not washed into PH contours.
5. Compare current arctan-preserved input against cloud/outlier-suppressed variants.
```

Acceptance criteria for the next run:

```text
pred_sum / target_sum remains between roughly 0.5 and 1.5
pred density visually overlaps target density peaks better than v3
abs error decreases around GT clusters, not only in global count
```

## Artifacts

```text
outputs/density_pilot_20260602_unet_mps_countfix_v3/run_summary.json
outputs/density_pilot_20260602_unet_mps_countfix_v3/scene_metrics.csv
outputs/density_pilot_20260602_unet_mps_countfix_v3/inference_previews/
../configs/dnb_density_unet_countfix.json
```
