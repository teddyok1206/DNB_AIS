# Count Head Reintroduction Plan

Date: 2026-06-08

## Current Active Phase

The active model path intentionally removes direct count regression.

Current output:

```text
input patch -> OccupancySpatialUNet -> P(ship exists in patch), p(pixel | ship exists)
occupancy_heatmap = P(ship exists in patch) * p(pixel | ship exists)
```

Interpretation:

- The patch-level sum of the prediction is an occupancy probability, not expected ship count.
- A positive target patch is normalized to sum to 1 for spatial supervision.
- An empty target patch is all zero.
- Evaluation should prioritize O/X metrics and spatial overlap before count calibration.

## Why Count Is Deferred

Direct count regression was under-constrained in the current data regime:

- The DNB arctan encoding compresses high brightness differences, making 1 ship vs many ships hard to separate from brightness alone.
- PH partition patches can include very different physical areas and cluster complexities.
- The model was learning plausible spatial shape before learning reliable integral count.

The simpler task is:

```text
ship exists? -> where is the ship-like evidence?
```

Only after this is stable should the count head be reintroduced.

## Reintroduction Gate

Add count only when the O/X model satisfies, on validation/test splits:

```text
occupancy_f1 is stable and high
occupancy_recall is not sacrificing too many positives
spatial_overlap_mean_positive is visually and quantitatively acceptable
full-scene occupancy maps are not dominated by false-positive fallback tiles
```

Exact numeric thresholds should be set after a 10/3/2-day pilot and a larger date-level validation run.

## Future Model Form

Recommended future output:

```text
P = sigmoid(occupancy_logit)
S[h,w] = masked_softmax(spatial_logits)[h,w]
C_pos = positive scalar estimate of E[count | ship exists]
density[h,w] = P * C_pos * S[h,w]
```

This preserves the successful O/X decomposition while adding count as a conditional positive-patch estimate.

## Future Loss Form

Recommended loss:

```text
L = w_occ * BCEWithLogits(P, y_exists)
  + w_spatial * KL(target_spatial || S) on positive patches only
  + w_count * log_count_loss(C_pos, target_count) on positive patches only
  + optional w_cal * full-scene or batch count calibration after enough data
```

Important constraints:

- Do not train count on negative patches except through the occupancy BCE.
- Keep count loss conditional on positives, otherwise empty patches dominate the scalar head.
- Keep `density = P * C_pos * S` for inference and full-scene merge only after count is reintroduced.
- Keep O/X metrics even after count returns, because count can improve while detection recall collapses.

## Implementation Notes

The active implementation already keeps reusable components:

- `OccupancySpatialUNet`: active O/X + spatial model.
- `CountSpatialDensityUNet`: legacy count-spatial model kept for checkpoint compatibility and future reference.
- `OccupancySpatialLoss`: active count-free loss.
- `CountSpatialDensityLoss`: legacy/future count loss reference.

A future conditional-count model should not replace the active O/X path blindly. Add it as a new class and config, for example:

```text
ConditionalCountSpatialUNet
conditional_count_spatial_loss
configs/dnb_density_unet_conditional_count_spatial.json
scripts/run_density_conditional_count_spatial_patchmix.sh
```

## Active Files To Compare Against

- `configs/dnb_density_unet_occupancy_spatial.json`
- `scripts/run_density_occupancy_spatial_patchmix.sh`
- `sub_module/dnb_density_models.py`
- `sub_module/dnb_density_losses.py`
- `sub_module/run_density_split_smoke_train.py`

## Archived Legacy References

The retired count, inverse-radiance, and fast-dilated configs/scripts were moved out of the active `configs/` and `scripts/` directories:

- `_archive/legacy_density_configs_20260608/configs/`
- `_archive/legacy_density_configs_20260608/scripts/`

Use them only as historical references when designing the future conditional count head.
