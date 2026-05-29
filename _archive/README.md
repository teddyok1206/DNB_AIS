# DNB_AIS Archive Policy

This directory stores lightweight archive metadata only. Heavy legacy archives are not tracked by git.

## Policy

- Keep active pipeline code in `configs/`, `sub_module/`, `scripts/`, and `[3]_DNB_AIS - (STEP 3)/` until path aliases are introduced.
- Keep inventory, checksums, and restore notes in `_archive/manifests/`.
- Store heavy legacy archives outside git. Default external target: `/Volumes/SAMSUNG/DNB_AIS_archive/legacy_steps`.
- Do not delete legacy source folders until archive checksums and restore listings have been verified.
