# Density Pilot Report - 2026-06-02

## Run

Command target:

```text
sub_module.run_density_split_smoke_train
```

Config:

```text
../configs/dnb_density_unet_main.json
```

Output directory:

```text
outputs/density_pilot_20260602_unet_mps
```

Split manifest:

```text
outputs/density_pilot_20260602_10_3_2/scene_split.csv
```

Execution settings:

```text
device=mps
epochs=6
batch_size=2
max_scenes_per_split=6
max_patches_per_scene=16
max_patch_height=512
max_patch_width=512
skip_ph_anchor_zero=true
```

## Data Used

The split builder selected:

```text
train: 10 days, 28 scenes in manifest
val:   3 days, 7 scenes in manifest
test:  2 days, 2 scenes in manifest
```

The train runner used a scene cap:

```text
train: 6 input scenes, 6 kept scenes, 96 selected patches
val:   6 input scenes, 3 kept scenes, 48 selected patches
test:  2 input scenes, 2 kept scenes, 32 selected patches
```

Skipped validation scenes:

```text
A2025025_1700_021: ph_anchor_count_zero
A2025055_1606_021: ph_anchor_count_zero
A2025055_1748_021: ph_anchor_count_zero
```

## Main Metrics

Epoch-level count summary:

```text
epoch  split  loss_mean  pred_sum  target_sum  pred/target
1      train  9.200108   36866.75  4174.00     8.832475
1      val    0.067089   98.27     3877.00     0.025346
2      train  0.052833   141.04    4174.00     0.033790
2      val    0.067877   59.07     3877.00     0.015237
3      train  0.053742   128.79    4174.00     0.030856
3      val    0.067867   57.97     3877.00     0.014951
4      train  0.052591   94.19     4174.00     0.022567
4      val    0.068699   37.29     3877.00     0.009619
5      train  0.052140   76.12     4174.00     0.018236
5      val    0.068609   38.74     3877.00     0.009992
6      train  0.051538   76.60     4174.00     0.018353
6      val    0.068738   36.67     3877.00     0.009459
test   test   0.081886   18.60     3947.00     0.004711
```

Representative preview:

```text
outputs/density_pilot_20260602_unet_mps/inference_previews/inference_preview_000_ph_anchor_25.png
pred_sum=0.91
target_sum=656.00
```

## Interpretation

The pipeline executed correctly on MPS and produced split manifests, quicklook PNGs, training metrics, scene metrics, and inference previews.

The model did not learn a useful density predictor in this pilot. It collapsed toward near-zero density after the first epoch. The strongest evidence is the final test count ratio:

```text
pred_sum / target_sum = 0.004711
```

This is not just poor localization. It is count collapse.

The likely immediate cause is the current structured loss configuration. The relative Huber count term with `count_huber_delta=0.25` saturates for large relative under-count errors. Once predictions become very small, the loss stays numerically modest instead of strongly forcing integral count recovery.

## Next Loss Fix

The next pilot should keep the same data split and model, but change the count terms before increasing dataset size.

Recommended next ablation:

```text
count_loss = relative_mse
count_weight >= 0.40
batch_count_weight >= 0.15
local_count_weight >= 0.25
pixel_weight <= 0.20
background_weight <= 0.02
```

Alternative:

```text
keep relative_huber
increase count_huber_delta from 0.25 to 1.0 or 2.0
increase count and local-count weights
```

The success criterion for the next run is not only lower loss. It must also satisfy:

```text
train pred_sum / target_sum: not collapsed, ideally 0.5 to 1.5 during pilot
val pred_sum / target_sum: not collapsed, ideally 0.3 to 2.0 during pilot
preview PNG: pred density mass appears near target density regions
```

## Cloud And Outlier Follow-Up

The current GeoTIFF synthesis preserves more high-radiance signal because the original top-1% clipping/outlier removal was replaced by an arctan-style transform. This better preserves raw DNB information, but it also leaves cloud-scattered radiance and broad glow structures visible.

This issue should be treated as a separate input-normalization/cloud-mask ablation, not as a reason to discard the current density pipeline.

Future experiments should compare at least these input variants:

```text
1. current arctan-preserving GeoTIFF
2. legacy top-1% clipped or zeroed GeoTIFF behavior
3. arctan GeoTIFF plus cloud/outlier auxiliary channel
4. arctan GeoTIFF plus robust high-radiance suppression mask
```

Important interpretation rule:

```text
If cloud regions would have been zeroed in the legacy synthesis, poor prediction in those regions should not be treated as a ship-density modeling failure until cloud/outlier handling is explicitly ablated.
```

## Artifacts

```text
outputs/density_pilot_20260602_10_3_2/split_summary.json
outputs/density_pilot_20260602_10_3_2/visuals/contact_sheet_train.png
outputs/density_pilot_20260602_10_3_2/visuals/contact_sheet_val.png
outputs/density_pilot_20260602_10_3_2/visuals/contact_sheet_test.png
outputs/density_pilot_20260602_unet_mps/run_summary.json
outputs/density_pilot_20260602_unet_mps/scene_metrics.csv
outputs/density_pilot_20260602_unet_mps/inference_previews/
```
