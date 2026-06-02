# Legacy Placeholder Relocation - 20260530_011500

- repo_root: `/Users/jungtaeuk/Desktop/SATGEO/[DNB_AIS]`
- moved_at: `2026-05-30 01:15 KST`
- source_folders_removed_from_top_level:
  - `[1]_DNB_AIS - (STEP 1)`
  - `[2]_DNB_AIS - (STEP 2)`
- placeholder_target: `_archive/placeholders/legacy_steps/`
- move_method: same-volume `mv` directory rename; no copy/checksum/tar was used
- reason: keep dataless/zero-allocated placeholder paths without leaving legacy step folders in active root

## Validation

- remaining_files_under_placeholder_target: 369
- files_matching_archive_exclude_keep_list: 369
- extra_files_not_in_keep_list: 0
- missing_files_from_keep_list: 0
- dataless_zero_block_files: 352
- nonzero_block_keep_list_files: 17

## Archive References

- archive_manifest: `_archive/manifests/legacy_archive_20260530_003528.md`
- external_archive_dir: `/Volumes/SAMSUNG/DNB_AIS_archive/legacy_steps`
- step1_excluded_zero_allocated_payloads: `/Volumes/SAMSUNG/DNB_AIS_archive/legacy_steps/step1_excluded_zero_allocated_payloads_20260530_003528.tsv`
- step2_excluded_zero_allocated_payloads: `/Volumes/SAMSUNG/DNB_AIS_archive/legacy_steps/step2_excluded_zero_allocated_payloads_20260530_003528.tsv`

## Git Hygiene

`_archive/placeholders/` was added to `.git/info/exclude` as a local-only ignore rule because the preserved placeholder files are runtime/data artifacts and should not enter git.
