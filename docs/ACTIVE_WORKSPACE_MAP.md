# Active Workspace Map

This repository currently has one active execution workspace and two legacy step workspaces.
Keep this map updated when the canonical path layout changes.

## Current Canonical Areas

- `configs/`: model, target, loss, and experiment configuration.
- `sub_module/`: reusable Python modules and runnable pipeline entrypoints.
- `scripts/`: repo maintenance and long-running operational helpers.
- `docs/`: design decisions, methodology notes, workflow records, and curated experiment interpretation.
- `docs/experiments/`: curated lightweight experiment reports promoted out of the active STEP3 runtime folder.
- `reports/eesrl_ieee_final/`: active IEEE/EESRL final-report workspace with draft documents, templates, and curated figure assets.
- `outputs/`: runtime outputs and smoke-test diagnostics; ignored by git except `outputs/README.md`.
- `artifacts/`: long-lived model/checkpoint metadata; heavy files are not tracked.
- `[3]_DNB_AIS - (STEP 3)/`: active DNB/AIS operational workspace for current data preparation and bbox generation.

## Active Step 3 Workspace

`[3]_DNB_AIS - (STEP 3)` is intentionally kept in place for now. Multiple scripts still resolve this path directly, including density split tooling and STEP3 operational jobs.

Important active items:

- `[A]_dnb2geotif_v2_modified_4326_metadata.py`: GeoTIFF conversion workflow.
- `[D]_ship_class_SQL_fast.py`: optimized AIS interpolation/database workflow.
- `[E]_bounding_box.py`: bbox rebuild workflow.
- `metadata_JPSS-2.csv`: current JPSS-2 metadata table used by split and bbox tooling.
- `bboxes_JPSS-2/`: current bbox/GeoJSON outputs used for density targets.
- `outputs/`: local runtime output for split, PH, partition, and smoke-train diagnostics.

Lightweight operational source files in this folder are tracked because current
jobs still run them by exact path. Heavy runtime products, raw AIS, GeoTIFFs,
bbox bulk outputs, DB files, and generated analysis tables stay untracked.

Do not rename this folder until path aliases or config-driven paths have replaced direct `STEP3` constants in the active code.

## Legacy Step Workspaces

`[1]_DNB_AIS - (STEP 1)` and `[2]_DNB_AIS - (STEP 2)` are legacy snapshots from earlier project phases. They should be preserved for reproducibility, but they are not the canonical location for new deep-learning pipeline work.

Archive policy:

- Use `scripts/archive_legacy_steps.sh` after active `[A]`, `[D]`, and `[E]` jobs finish.
- Default archive target is `/Volumes/SAMSUNG/DNB_AIS_archive/legacy_steps` because local repo disk space is currently insufficient for repo-local 90GB+ archives.
- Keep archive manifests in `_archive/manifests/`.
- Do not delete source legacy folders until archive checksums and listing heads are verified.

## Documentation Policy

- Put durable design docs in `docs/`.
- Put final-report drafting materials, IEEE templates, and curated report figures in `reports/eesrl_ieee_final/`.
- Put raw or chronological agent logs in `codex_logs/`.
- Keep one-off notebook outputs and generated figures in `outputs/`, not in git.
- If a generated result becomes important, promote only a small curated summary or figure to `docs/`.

## Git Policy

- Track code, configs, docs, scripts, and lightweight manifests.
- Do not track checkpoints, TIFF/NetCDF/DB files, bbox bulk outputs, raw AIS, ZIP/PDF archives, or run directories.
- Stage only focused changes. Leave unrelated modified notebooks or active workspace files untouched unless explicitly part of a requested change.
