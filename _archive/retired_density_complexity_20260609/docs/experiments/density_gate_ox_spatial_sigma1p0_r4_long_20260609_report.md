# DNB Density Gated O/X + Spatial U-Net Long Run

Date: 2026-06-09

## Decision

Use `OccupancySpatialUNet` as an internal O/X gate over a masked spatial distribution:

- model output: occupancy logit plus spatial logit
- final density: `sigmoid(occupancy_logit) * masked_spatial_softmax(spatial_logit)`
- no count regression head
- target kernel: `sigma_pixels=1.0`, `radius_pixels=4`
- loss weights: `occupancy_weight=0.85`, `spatial_weight=0.15`
- learning rate: `3e-5`
- epochs: `72`

The run improves the O/X metric over the previous 24-epoch O/X+spatial run, but it overfits strongly after the best validation checkpoints. Report validation-selected checkpoint metrics, not the final epoch.

## Canonical Run

Run directory:

`outputs/dnb_density/runs/gate_ox_spatial_sigma1p0_r4_e72_lr3e5_20260609_153802`

Log:

`outputs/dnb_density/runs/gate_ox_spatial_sigma1p0_r4_e72_lr3e5_20260609_153802/run.log`

Config:

`configs/dnb_density_unet_occupancy_spatial_gate_sigma1p0_r4_e72_lr3e5.json`

Patch cache:

`/Volumes/SAMSUNG/dnb_density_training_patch_cache/ox_spatial_25pct_48p_sigma1p0_r4_20260609`

Training git commit:

`5c73637cde36a904a43b63a461fcb261b0e17df1`

Config hash:

`db9f64dd9b2ff4df31dbdc6d8d986bce282d6d4f7eaa9636601d0fbe7781d287`

## Split Size

| split | scenes | selected patches | selected target sum |
| --- | ---: | ---: | ---: |
| train | 157 | 5334 | 16772.0001 |
| val | 40 | 1222 | 3605.0000 |
| test | 31 | 1017 | 2840.0000 |

## Validation Checkpoints

| checkpoint | selected epoch | validation selection metric |
| --- | ---: | ---: |
| `best_val_loss` | 12 | loss `1.132523` |
| `best_val_occupancy_mass_ratio` | 9 | ratio log error `0.000386`, pred/target `0.999614` |
| `best_val_occupancy_f1` | 51 | F1 `0.706997` |
| `best_val_spatial_overlap` | 23 | spatial overlap `0.091079` |

## Test Metrics

| checkpoint | epoch | fixed F1 | calibrated F1 | threshold | fixed spatial | pred/target | Brier | precision | recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `best_val_occupancy_f1` | 51 | 0.7117 | 0.7111 | 0.465 | 0.0726 | 1.1430 | 0.2361 | 0.6118 | 0.8507 |
| `best_val_spatial_overlap` | 23 | 0.6848 | 0.7027 | 0.235 | 0.0728 | 0.9741 | 0.2218 | 0.6326 | 0.7464 |
| `best_val_loss` | 12 | 0.6990 | 0.7001 | 0.395 | 0.0584 | 1.0933 | 0.2212 | 0.6069 | 0.8241 |
| `best_val_occupancy_mass_ratio` | 9 | 0.6781 | 0.6872 | 0.295 | 0.0601 | 1.0521 | 0.2200 | 0.6078 | 0.7669 |
| final epoch 72 | 72 | 0.6066 | n/a | 0.500 | 0.0781 | 0.8604 | 0.2916 | 0.6300 | 0.5849 |

## Enhanced Previews

Generated preview directories:

- `outputs/dnb_density/runs/gate_ox_spatial_sigma1p0_r4_e72_lr3e5_20260609_153802/enhanced_previews_best_val_occupancy_f1`
- `outputs/dnb_density/runs/gate_ox_spatial_sigma1p0_r4_e72_lr3e5_20260609_153802/enhanced_previews_best_val_spatial_overlap`

The preview renderer now supports cached runs by loading patches from `run_summary.patch_cache.dir` when `filtered_scene_split.csv` is empty. This is required for read-only cache runs where scene build records are not regenerated.

## Interpretation

- `best_val_occupancy_f1` is the strongest O/X checkpoint: fixed-threshold test F1 `0.7117`, calibrated test F1 `0.7111`.
- `best_val_spatial_overlap` has the strongest test spatial overlap among evaluated validation-selected checkpoints: `0.0728`, but its fixed-threshold F1 is lower.
- The final epoch has better-looking train metrics but worse test F1 and Brier, so epoch 72 should be treated as overfit diagnostic output only.
- Compared with the previous 24-epoch sigma1.0/r4 O/X+spatial run, this long run modestly improves best O/X F1 and spatial overlap, but does not solve spatial localization; spatial overlap remains low on test.

## Active Recommendation

For the report table, keep two rows:

- `best_val_occupancy_f1` for the main O/X-gated density result
- `best_val_spatial_overlap` as the spatial-localization-selected diagnostic

For the next run, prefer explicit early stopping or shorter training around the validation-selected window rather than extending epochs further without regularization or data expansion.
