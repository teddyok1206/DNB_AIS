# Retired Density Complexity Archive - 2026-06-09

This folder preserves density-pipeline pieces removed from the active source tree when the project narrowed to the hard pixel-occupancy pipeline.

## Active Replacement

The active path is:

```text
configs/dnb_density_unet_pixel_binary_recursive_ph_hardtarget_20260609.json
sub_module/run_density_split_smoke_train.py
sub_module/evaluate_density_checkpoint.py
sub_module/dnb_density_models.py        # PixelBinaryOccupancyUNet only
sub_module/dnb_density_losses.py        # PixelBinaryOccupancyLoss only
scripts/run_density_pixel_binary_recursive_ph.sh
```

The active target is `raw_count > 0` at each valid owner pixel. Gaussian density smoothing is retained only for preview/legacy diagnostics and is not the supervised label.

## Archived Categories

- `configs/`: occupancy/spatial, occupancy-only, spatial-only, and related long-run configs.
- `docs/`: retired design notes, older experiment reports, and count-head reintroduction material.
- `scripts/`: brightness threshold, retargeting, and old occupancy/spatial run helpers.
- `sub_module/`: retired analysis/render/baseline utilities.
- `sub_module/dnb_density_models_legacy_full.py`: pre-cleanup model zoo source.
- `sub_module/dnb_density_losses_legacy_full.py`: pre-cleanup loss zoo source.

## Restore Pattern

Use `git mv` from this archive back into the original path if a retired path becomes active again. Example:

```sh
git mv _archive/retired_density_complexity_20260609/configs/<file>.json configs/<file>.json
```

For model/loss variants, copy from the `*_legacy_full.py` files into active source deliberately instead of bulk-restoring the full zoo. The point of this archive is to make old work recoverable without letting old branches remain active by accident.
