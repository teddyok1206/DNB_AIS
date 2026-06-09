# Density Encoded Baseline Scale20 Pilot - 2026-06-03

## Purpose

This pilot returns to the strongest current input hypothesis: use the original STEP3 encoded brightness channel as the radiance input, while keeping the PH feature channels.

Recent inverse-radiance and gamma pilots showed that mathematically reversing the arctan compression is plausible, but the actual short-run results were weaker than the encoded baseline. This run scales the encoded baseline up to a larger patch budget before making the next modeling decision.

## Config And Script

Config:

```text
configs/dnb_density_unet_count_spatial.json
```

Script:

```text
scripts/run_density_count_spatial_scaled_patchmix.sh
```

Launch settings used for this run:

```text
EPOCHS=20
MAX_SCENES_PER_SPLIT=20
MAX_PATCHES_PER_SCENE=64
MAX_PH_PATCHES_PER_SCENE=48
MAX_FALLBACK_PATCHES_PER_SCENE=16
PREVIEW_PATCHES=24
BATCH_SIZE=2
NUM_WORKERS=0
```

## Model Structure

Model:

```text
CountSpatialDensityUNet
```

Input channels:

```text
0 brightness
1 parent_ph_mask
2 child_ph_union_mask
3 ph_seed_map
4 ph_persistence_map
5 ph_soft_attention
```

The count and spatial heads share the same encoded brightness plus PH context.

## Run

Run directory:

```text
outputs/dnb_density/runs/encoded_baseline_scale20_20260603_024204
```

Run summary:

```text
outputs/dnb_density/runs/encoded_baseline_scale20_20260603_024204/run_summary.json
```

Enhanced previews:

```text
outputs/dnb_density/runs/encoded_baseline_scale20_20260603_024204/enhanced_previews/
```

Device:

```text
mps
```

Run git metadata:

```text
git_commit: f662c52dbfba116d32f5b32c5450bb1d8db5c6a7
git_dirty: false
```

Actual split after PH-anchor filtering:

```text
train: input=20, kept=15, patches=960, target_sum=27857.0
val:   input=4,  kept=3,  patches=192, target_sum=7475.0
test:  input=2,  kept=2,  patches=128, target_sum=5396.0
```

Skipped train scenes due to zero PH anchors:

```text
A2025033_1800_021
A2025060_1754_021
A2025069_1636_021
A2025108_1612_021
A2025108_1754_021
```

Skipped val scene due to zero PH anchors:

```text
A2025090_1642_021
```

## Training Dynamics

Train loss decreased steadily:

```text
epoch 1:  loss_total=1.7775, pred/target=0.458
epoch 10: loss_total=1.3571, pred/target=0.613
epoch 20: loss_total=1.0101, pred/target=0.720
```

Validation was noisier because only 3 kept scenes and 192 patches were used. The best count-calibrated validation moment was epoch 19:

```text
epoch 19 val:
pred_sum=7816.78
target_sum=7475.00
pred/target=1.046
loss_total=1.1450
loss_spatial=1.7375
```

Final epoch validation undercounted again:

```text
epoch 20 val:
pred_sum=4412.49
target_sum=7475.00
pred/target=0.590
loss_total=1.1212
loss_spatial=1.7394
```

This means validation loss and count calibration are not perfectly aligned under the current loss coefficients.

## Test Result

Final test:

```text
pred_sum=2789.82
target_sum=5396.00
pred/target=0.517
```

Final test component means:

```text
loss_total=1.0373
loss_count=0.5593
loss_spatial=1.7457
loss_density=0.000276
pred_count_mean=21.80
target_count_mean=42.16
```

Enhanced preview metrics over 24 test previews:

```text
pred_sum=793.71
target_sum=2084.00
pred/target=0.381
target_explained_mean=0.243
pred_matched_mean=0.497
spatial_overlap_mean=0.403
```

Best preview examples by normalized spatial overlap:

```text
enhanced_preview_003_ph_anchor_610.png:
  pred_sum=28.6, target_sum=108.0, target_explained=0.240, pred_matched=0.907, spatial_overlap=0.703

enhanced_preview_019_ph_anchor_781.png:
  pred_sum=19.5, target_sum=30.0, target_explained=0.525, pred_matched=0.808, spatial_overlap=0.702

enhanced_preview_011_ph_anchor_381.png:
  pred_sum=38.0, target_sum=54.0, target_explained=0.549, pred_matched=0.780, spatial_overlap=0.670
```

Worst preview examples by target explained mass:

```text
enhanced_preview_002_ph_anchor_539.png:
  pred_sum=11.1, target_sum=143.0, target_explained=0.057, pred_matched=0.742, spatial_overlap=0.443

enhanced_preview_013_ph_anchor_518.png:
  pred_sum=9.3, target_sum=46.0, target_explained=0.061, pred_matched=0.301, spatial_overlap=0.177

enhanced_preview_021_ph_anchor_204.png:
  pred_sum=26.9, target_sum=27.0, target_explained=0.079, pred_matched=0.079, spatial_overlap=0.079
```

## Interpretation

The larger encoded baseline still undercounts on held-out test patches, but the visual failure mode is informative.

The good examples place density at the right target locations, but with weak peak intensity. The bad examples also often show nonzero prediction around the correct region, but the predicted density is too faint to explain the target mass.

Primary failure mode:

```text
spatial localization is partially learned, count/density amplitude calibration is weak.
```

This differs from a pure detection failure. The model is not simply hallucinating density in unrelated regions; it often finds the right structures but assigns too little mass.

## Comparison With Recent Input Pilots

Recent same-family pilots:

```text
raw inverse only count branch:
  pred/target=0.263

raw inverse + PH count branch, softplus:
  pred/target=0.412

brightness_gamma_2 + PH:
  pred/target=0.498

stabilized inverse p99 + PH:
  pred/target=0.493

encoded baseline scale20:
  pred/target=0.517 on final test
  val epoch 19 pred/target=1.046
```

The encoded baseline remains the safest mainline input. The inverse-radiance idea should not be deleted as a physical hypothesis, but it is not the next best engineering move based on current evidence.

## Next Modeling Implication

Do not keep adding radiance transforms immediately.

The next useful change is loss/checkpoint selection:

```text
1. track validation count ratio explicitly
2. save best checkpoints by both val loss and count-calibrated validation score
3. evaluate test with the selected best checkpoint, not only the last checkpoint
4. consider increasing count loss weight or adding a count-ratio calibration term
```

This is more directly connected to the observed failure than changing the input channel again.

