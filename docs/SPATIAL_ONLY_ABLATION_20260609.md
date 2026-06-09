# Spatial-Only Ablation, 2026-06-09

## Context

The patch-level O/X head was removed after clarifying that the desired O/X concept, if used later, should be pixel-level rather than patch-level. This ablation keeps the PH-assisted U-Net input stack and trains only a positive-patch spatial density distribution.

## Run

- Run directory: `outputs/dnb_density/runs/spatial_only_sigma1p0_r4_e24_lr5e5_20260609_145942`
- Config: `configs/dnb_density_unet_spatial_only_sigma1p0_r4_e24_lr5e5.json`
- Patch cache: `/Volumes/SAMSUNG/dnb_density_training_patch_cache/ox_spatial_25pct_48p_sigma1p0_r4_20260609`
- Split: fixed 25% day split, cached patches
- Model: `SpatialOnlyUNet`
- Loss: `spatial_only_loss`, KL over normalized positive-patch target density
- Target: Gaussian density, `sigma_pixels=1.0`, `radius_pixels=4`
- Device: MPS

The run was stopped after epoch 14 because validation spatial overlap peaked early and then degraded while train spatial overlap continued increasing.

## Validation Behavior

| Checkpoint | Epoch | Val loss | Val spatial overlap |
|---|---:|---:|---:|
| `best_val_loss` | 4 | 4.0215 | 0.0877 |
| `best_val_spatial_overlap` | 4 | 4.0215 | 0.0877 |
| `best_val_count_ratio` | 10 | 4.1100 | 0.0851 |

Observed trend:

- Train spatial overlap increased from about `0.0620` at epoch 1 to `0.2571` by epoch 14.
- Validation spatial overlap peaked at epoch 4 and did not recover afterward.
- This indicates overfitting under the current spatial-only KL setup.

## Test Results

| Checkpoint | Test loss | Test spatial overlap | Note |
|---|---:|---:|---|
| `best_val_loss` | 4.1869 | 0.0628 | Same epoch as best val spatial |
| `best_val_spatial_overlap` | 4.1869 | 0.0628 | Main spatial checkpoint |
| `best_val_count_ratio` | 4.2121 | 0.0667 | Best test spatial among evaluated spatial-only checkpoints |

Reference from the previous O/X+spatial run on the same cached setup:

- `best_val_spatial_overlap` test spatial was about `0.0678`.
- `best_val_occupancy_f1` test spatial was about `0.0700`.

Therefore, the current spatial-only ablation does not improve spatial localization over the previous O/X+spatial setup.

## Qualitative Finding

Preview directory:

- `outputs/dnb_density/runs/spatial_only_sigma1p0_r4_e24_lr5e5_20260609_145942/inference_previews_best_val_count_ratio`

Key observation:

- Negative patches still receive a normalized predicted spatial distribution with `pred_sum=1.00` because the spatial-only model has no mechanism to express “no ship here.”
- Positive patches often produce multiple diffuse peaks rather than concentrating sharply on the target Gaussian peak.

This is expected from the current formulation: the model is trained only to distribute probability within positive patches. It is not trained to suppress empty patches or estimate mass.

## Conclusion

Spatial-only is a useful ablation but not a deployable inference model by itself.

Recommended active direction:

1. Keep spatial distribution learning, because it isolates the location problem.
2. Do not revive patch-level O/X as the main objective.
3. If O/X is added later, implement it as pixel-level foreground/background supervision, not a patch scalar.
4. For the next spatial experiment, use stronger regularization or a smaller/less diffuse target/patch setup before increasing epochs.
5. Treat `pred_target_ratio` from spatial-only runs as non-informative, because predictions are normalized over positive patches by construction.

## Utility Patch

`sub_module/evaluate_density_checkpoint.py` was updated so interrupted runs without `run_summary.json` can still be evaluated when `config_snapshot.json`, a checkpoint, and `--patch-cache-dir` are available.
