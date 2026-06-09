# DNB Density O/X + Spatial U-Net: Sigma 1.0 / Radius 4

Date: 2026-06-09

## Decision

Use the `OccupancySpatialUNet` path with:

- target kernel: `sigma_pixels=1.0`, `radius_pixels=4`
- loss weights: `occupancy_weight=0.9`, `spatial_weight=0.1`
- learning rate: `5e-5`
- epochs: `24`
- split/cache: fixed 25% day split and retargeted sigma 1.0/r4 patch cache

This is the current main candidate for the O/X-first density pipeline. The final epoch is not the selection target; use validation-selected checkpoints.

## Canonical Run

Run directory:

`outputs/dnb_density/runs/ox_spatial_ow09_sw01_sigma1p0_r4_e24_lr5e5_spatialckpt_20260609_140325`

Log:

`outputs/dnb_density/runs/ox_spatial_ow09_sw01_sigma1p0_r4_e24_lr5e5_spatialckpt_20260609_140325/run.log`

Config:

`configs/dnb_density_unet_occupancy_spatial_ow09_sw01_sigma1p0_r4_e24_lr5e5.json`

Patch cache:

`/Volumes/SAMSUNG/dnb_density_training_patch_cache/ox_spatial_25pct_48p_sigma1p0_r4_20260609`

## Split Size

| split | scenes | selected patches | selected target sum |
| --- | ---: | ---: | ---: |
| train | 157 | 5334 | 16772.0001 |
| val | 40 | 1222 | 3605.0000 |
| test | 31 | 1017 | 2840.0000 |

## Validation Checkpoints

| checkpoint | selected epoch | validation selection metric |
| --- | ---: | ---: |
| `best_val_loss` | 12 | loss `0.963331` |
| `best_val_occupancy_mass_ratio` | 1 | ratio log error `0.001862` |
| `best_val_occupancy_f1` | 21 | F1 `0.704748` |
| `best_val_spatial_overlap` | 14 | spatial overlap `0.089199` |

## Test Metrics

| model/checkpoint | threshold | F1 | precision | recall | Brier | spatial overlap | pred/target ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline `sigma=1.5/r5`, final | 0.50 | 0.677998 | 0.629825 | 0.734151 | 0.217809 | 0.088442 | 1.009666 |
| `sigma=1.0/r4`, final epoch 24 | 0.50 | 0.679476 | 0.592988 | 0.795501 | 0.235036 | 0.063847 | 1.108189 |
| `sigma=1.0/r4`, `best_val_loss` | 0.50 | 0.701139 | 0.613497 | 0.817996 | 0.221330 | 0.062107 | 1.092057 |
| `sigma=1.0/r4`, `best_val_occupancy_f1` | 0.50 | 0.696104 | 0.603604 | 0.822086 | 0.224298 | 0.070042 | 1.095536 |
| `sigma=1.0/r4`, `best_val_occupancy_f1` + val threshold | 0.41 | 0.700248 | 0.587258 | 0.867076 | 0.224298 | n/a | n/a |
| `sigma=1.0/r4`, `best_val_spatial_overlap` | 0.50 | 0.703259 | 0.605613 | 0.838446 | 0.222547 | 0.067805 | 1.113996 |
| `sigma=1.0/r4`, `best_val_spatial_overlap` + val threshold | 0.275 | 0.695720 | 0.561558 | 0.914110 | 0.222547 | n/a | n/a |
| `sigma=1.0/r4`, `best_val_occupancy_mass_ratio` | 0.50 | 0.646602 | 0.615527 | 0.680982 | 0.225155 | 0.050607 | 1.037708 |

## Interpretation

- The final epoch is not reliable for model selection. It underperforms validation-selected checkpoints.
- The strongest fixed-threshold O/X checkpoint is `best_val_spatial_overlap`, with test F1 `0.703259`.
- The strongest calibrated-threshold result is `best_val_occupancy_f1` with threshold `0.41`, test F1 `0.700248`.
- `best_val_spatial_overlap` unexpectedly has the best fixed-threshold test F1, but its test spatial overlap remains modest.
- The `sigma=1.0/r4` target is sharper than the baseline `sigma=1.5/r5`, so spatial overlap is harder to compare directly across those two target kernels.
- The current active protocol should report both fixed threshold `0.5` and validation-calibrated threshold, but avoid choosing a threshold on test.

## Active Recommendation

For the next full-data O/X-first run:

- keep `sigma_pixels=1.0`, `radius_pixels=4`
- keep `occupancy_weight=0.9`, `spatial_weight=0.1`
- save and evaluate `best_val_loss`, `best_val_occupancy_f1`, and `best_val_spatial_overlap`
- treat final epoch metrics as diagnostic only
- use validation-selected checkpoint metrics for report tables
