# Legacy Density Configs And Scripts

Archived on 2026-06-08 when the active path was narrowed to PH-assisted `OccupancySpatialUNet`.

These files are retained for history, checkpoint compatibility, and future count-head reintroduction reference only. They should not be used for active training without an explicit decision to revive the retired count/inverse/dilated experiments.

Contents:

- `configs/`: retired count, inverse-radiance, base, and fast-dilated configs.
- `scripts/`: retired count/inverse experiment launcher scripts.
- `sub_module/`: retired smoke-test entrypoints that predate the active occupancy/spatial split runner.
