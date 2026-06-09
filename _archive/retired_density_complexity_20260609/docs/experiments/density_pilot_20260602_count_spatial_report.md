# Density Count-Spatial Pilot Report - 2026-06-02

## Purpose

The previous count-fix loss solved count collapse but became over-complex:

```text
pixel + foreground + count + batch_count + local_count + empty_count + background
```

This pilot replaces that patched loss structure with a simpler factorized task:

```text
pred_density = predicted_count * predicted_spatial_probability
```

The model now solves two explicit subtasks:

```text
1. Count head: predict total ship count in the patch.
2. Spatial head: predict where that count should be distributed inside the valid mask.
```

## Code Changes

Files changed:

```text
../sub_module/dnb_density_models.py
../sub_module/dnb_density_losses.py
../sub_module/run_density_split_smoke_train.py
../configs/dnb_density_unet_count_spatial.json
```

New model:

```text
CountSpatialDensityUNet
```

Output:

```text
spatial_logits: B x 1 x H x W
count:          B x 1
count_raw:      B x 1
```

New loss:

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

Configurable coefficients:

```text
count_weight
spatial_weight
density_weight
background_weight
spatial_temperature
count_loss
spatial_loss
density_loss
learning rate
weight decay
grad_clip_norm
count_activation
```

## Pilot Runs

Same split and patch caps were used as the previous pilots:

```text
scene split: outputs/density_pilot_20260602_10_3_2/scene_split.csv
train scenes used: 6
val scenes used:   3 kept from 6 input
test scenes used:  2
```

### Count-Spatial v1

Count head used:

```text
count_activation = softplus
```

Result:

```text
test pred_sum / target_sum = 259.15 / 3947.00 = 0.0657
```

Interpretation:

```text
The structure worked, but softplus count prediction learned too slowly from near-zero initial counts.
```

### Count-Spatial v2

Count head changed to:

```text
count_activation = exp
```

This makes the scalar count prediction naturally optimize in log-count space.

Result:

```text
test pred_sum / target_sum = 3008.14 / 3947.00 = 0.7621
```

Final epoch:

```text
train pred_sum / target_sum = 1845.73 / 4174.00 = 0.4422
val   pred_sum / target_sum = 1906.09 / 3877.00 = 0.4916
test  pred_sum / target_sum = 3008.14 / 3947.00 = 0.7621
```

## Comparison

Previous patched count-fix loss:

```text
test pred_sum / target_sum = 4076.18 / 3947.00 = 1.0327
```

New factorized count-spatial loss:

```text
test pred_sum / target_sum = 3008.14 / 3947.00 = 0.7621
```

The patched loss still gives better count calibration after 6 epochs. The new factorized structure is cleaner, but the count head needs more training and likely count-head tuning.

## Qualitative Interpretation

Representative preview:

```text
outputs/density_pilot_20260602_unet_mps_count_spatial_v2/inference_previews/inference_preview_000_ph_anchor_25.png
pred_sum = 110.36
target_sum = 656.00
```

Spatial bias is still visible:

```text
predicted density follows PH/shoreline-like structure more than GT peaks.
```

So the simplified structure did not immediately solve localization. It did, however, separate the failure modes:

```text
count underprediction is now a count-head problem.
spatial misallocation is now a spatial-head problem.
```

This is the main design improvement.

## Next Step

Do not add more patch terms to the loss yet. The next ablations should stay within the clean factorized design:

```text
1. Train count-spatial for more epochs.
2. Increase count_weight or use a short count-head warmup.
3. Test spatial_temperature < 1.0 to sharpen spatial allocation.
4. Try removing or weakening PH soft-attention input if spatial predictions keep following PH boundaries.
5. Add explicit channel-selection config only if PH input bias remains.
```

Acceptance criteria:

```text
count ratio: 0.5 to 1.5
spatial preview: density peaks overlap GT peaks better than count-fix v3
loss design: keep count/spatial/density terms only, avoid returning to many patched local/background terms
```

## Artifacts

```text
../configs/dnb_density_unet_count_spatial.json
outputs/density_pilot_20260602_unet_mps_count_spatial/run_summary.json
outputs/density_pilot_20260602_unet_mps_count_spatial_v2/run_summary.json
outputs/density_pilot_20260602_unet_mps_count_spatial_v2/inference_previews/
```
