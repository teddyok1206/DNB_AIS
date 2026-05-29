# DNB_AIS Inventory - 20260529_213033

## Summary

- repo_root: `/Users/jungtaeuk/Desktop/SATGEO/[DNB_AIS]`
- generated_at: `2026-05-29T12:30:33Z`
- active_workspace: `[3]_DNB_AIS - (STEP 3)`
- archive_policy: legacy step folders are preserved; no source deletion in this cleanup pass
- archive_execution: deferred because active [A]/[D]/[E] jobs are running and internal disk has insufficient free space for repo-local archives
- external_archive_target: `/Volumes/SAMSUNG/DNB_AIS_archive/legacy_steps`

## Active DNB Jobs Observed

```text
  PID     ELAPSED  %CPU %MEM COMMAND
39875       00:00   2.5  0.0 /bin/sh -lc stamp="$(date +%Y%m%d_%H%M%S)"\012inv="_archive/manifests/dnb_ais_inventory_${stamp}.md"\012latest="_archive/manifests/dnb_ais_inventory_latest.md"\012{\012  echo "# DNB_AIS Inventory - ${stamp}"\012  echo\012  echo "## Summary"\012  echo\012  echo "- repo_root: \`$(pwd)\`"\012  echo "- generated_at: \`$(date -u +%Y-%m-%dT%H:%M:%SZ)\`"\012  echo "- active_workspace: \`[3]_DNB_AIS - (STEP 3)\`"\012  echo "- archive_policy: legacy step folders are preserved; no source deletion in this cleanup pass"\012  echo "- archive_execution: deferred because active [A]/[D]/[E] jobs are running and internal disk has insufficient free space for repo-local archives"\012  echo "- external_archive_target: \`/Volumes/SAMSUNG/DNB_AIS_archive/legacy_steps\`"\012  echo\012  echo "## Active DNB Jobs Observed"\012  echo\012  echo '```text'\012  ps -ax -o pid,etime,pcpu,pmem,command | rg '\[A\]_dnb2geotif|\[D\]_ship_class|\[E\]_bounding_box|PID' || true\012  echo '```'\012  echo\012  echo "## Disk Snapshot"\012  echo\012  echo '```text'\012  df -h . /Volumes/SAMSUNG 2>/dev/null || df -h .\012  echo '```'\012  echo\012  echo "## Top-Level Size Snapshot"\012  echo\012  echo '```text'\012  du -sh . ./* 2>/dev/null | sort -h\012  echo '```'\012  echo\012  echo "## Step Folder Size Snapshot"\012  echo\012  echo '```text'\012  for d in "[1]_DNB_AIS - (STEP 1)" "[2]_DNB_AIS - (STEP 2)" "[3]_DNB_AIS - (STEP 3)"; do\012    echo "### $d"\012    find "$d" -maxdepth 1 -mindepth 1 -print0 2>/dev/null | xargs -0 du -sh 2>/dev/null | sort -h\012    echo\012  done\012  echo '```'\012  echo\012  echo "## File Count by Step Subdirectory"\012  echo\012  echo '```text'\012  find "[1]_DNB_AIS - (STEP 1)" "[2]_DNB_AIS - (STEP 2)" "[3]_DNB_AIS - (STEP 3)" -maxdepth 2 -type f 2>/dev/null | awk 'BEGIN{FS="/"} {print $1"/"$2}' | sort | uniq -c | sort -nr\012  echo '```'\012  echo\012  echo "## Git Status Snapshot"\012  echo\012  echo '```text'\012  git status --short\012  echo '```'\012  echo\012  echo "## Classification"\012  echo\012  echo "- Active code/config/docs: \`sub_module/\`, \`configs/\`, \`docs/\`, \`scripts/\`."\012  echo "- Active step workspace: \`[3]_DNB_AIS - (STEP 3)\`; keep path stable for current scripts."\012  echo "- Legacy archive candidates: \`[1]_DNB_AIS - (STEP 1)\`, \`[2]_DNB_AIS - (STEP 2)\`."\012  echo "- Generated runtime outputs: \`outputs/\`, \`[3]_DNB_AIS - (STEP 3)/outputs/\`; keep out of git except README/curated summaries."\012  echo "- Heavy raw/geospatial data: \`AIS/\`, \`bboxes_*\`, TIFF/DB/ZIP/PDF artifacts; keep out of git."\012  echo\012  echo "## Follow-Up Command"\012  echo\012  echo '```sh'\012  echo './scripts/archive_legacy_steps.sh'\012  echo '```'\012} > "$inv"\012cp "$inv" "$latest"\012printf '%s\n' "$inv"\012
39884       00:00   0.0  0.0 rg \[A\]_dnb2geotif|\[D\]_ship_class|\[E\]_bounding_box|PID
18721    02:59:30   0.0  0.0 sudo python [D]_ship_class_SQL_fast.py
18749    02:59:20  93.1  2.5 python [D]_ship_class_SQL_fast.py
18711    02:59:33  77.2 19.2 python [A]_dnb2geotif_v2_modified_4326_metadata.py
39148       04:48  85.9  0.7 python -u ./[E]_bounding_box.py --REBUILD_BBOXES=1
```

## Disk Snapshot

```text
Filesystem      Size    Used   Avail Capacity iused ifree %iused  Mounted on
/dev/disk3s5   926Gi   878Gi   9.2Gi    99%    3.1M   96M    3%   /System/Volumes/Data
/dev/disk5s1   1.8Ti   1.1Ti   740Gi    61%    9.5k  4.3G    0%   /Volumes/SAMSUNG
```

## Top-Level Size Snapshot

```text
4.0K	./AGENTS.md
4.0K	./artifacts
 12K	./_archive
 12K	./configs
 12K	./scripts
 32K	./docs
 52K	./codex_stamp.md
 88K	./DNB_GAT_v1.ipynb
116K	./codex_logs
684K	./2105.14491v3.pdf
1.2M	./sub_module
 25M	./_Meetings
116M	./outputs
237M	./_Readings
7.1G	./[2]_DNB_AIS - (STEP 2)
 18G	./[3]_DNB_AIS - (STEP 3)
 89G	./[1]_DNB_AIS - (STEP 1)
115G	.
```

## Step Folder Size Snapshot

```text
### [1]_DNB_AIS - (STEP 1)
  0B	[1]_DNB_AIS - (STEP 1)/A_dnb2geotif_v2_modified_4326.py
  0B	[1]_DNB_AIS - (STEP 1)/dnb_02.tar.gz
  0B	[1]_DNB_AIS - (STEP 1)/dnb_03.tar.gz
4.0K	[1]_DNB_AIS - (STEP 1)/OnlyIn.txt
8.0K	[1]_DNB_AIS - (STEP 1)/B_metadata_generator.py
 12K	[1]_DNB_AIS - (STEP 1)/.DS_Store
 12K	[1]_DNB_AIS - (STEP 1)/C_polygon_handler.py
 12K	[1]_DNB_AIS - (STEP 1)/eez_12nm
 24K	[1]_DNB_AIS - (STEP 1)/IMG_min:max
540K	[1]_DNB_AIS - (STEP 1)/metadata.csv
716K	[1]_DNB_AIS - (STEP 1)/eez
1.3M	[1]_DNB_AIS - (STEP 1)/D_metadata_analyzer.ipynb
2.4M	[1]_DNB_AIS - (STEP 1)/2025년_S-NPP_DNB영상_관측통계.pdf
3.8M	[1]_DNB_AIS - (STEP 1)/layers
 83M	[1]_DNB_AIS - (STEP 1)/2025년_S-NPP_DNB영상_관측통계.pptx
463M	[1]_DNB_AIS - (STEP 1)/references
 89G	[1]_DNB_AIS - (STEP 1)/results_4326_OVERLAP_O

### [2]_DNB_AIS - (STEP 2)
4.0K	[2]_DNB_AIS - (STEP 2)/IMG_min:max
4.0K	[2]_DNB_AIS - (STEP 2)/[F]_pixel_overlap.py
4.0K	[2]_DNB_AIS - (STEP 2)/[sub2]_crosscheck.py
4.0K	[2]_DNB_AIS - (STEP 2)/[sub3]_fill_tif_name.py
4.0K	[2]_DNB_AIS - (STEP 2)/[sub4]_refresh.py
4.0K	[2]_DNB_AIS - (STEP 2)/percentile.m
8.0K	[2]_DNB_AIS - (STEP 2)/[sub1]_preprocessing.ipynb
 12K	[2]_DNB_AIS - (STEP 2)/[B]_polygon_handler.py
 12K	[2]_DNB_AIS - (STEP 2)/[E]_bounding_box.py
 16K	[2]_DNB_AIS - (STEP 2)/OnlyIn.txt
 16K	[2]_DNB_AIS - (STEP 2)/[A]_dnb2geotif_v2_modified_4326_metadata.py
 20K	[2]_DNB_AIS - (STEP 2)/.DS_Store
 24K	[2]_DNB_AIS - (STEP 2)/[D]_ship_class_SQL.py
 28K	[2]_DNB_AIS - (STEP 2)/cross-check
 44K	[2]_DNB_AIS - (STEP 2)/qgz_archieve
 92K	[2]_DNB_AIS - (STEP 2)/[G]_pixel_analyzer.ipynb
120K	[2]_DNB_AIS - (STEP 2)/eez_12nm
532K	[2]_DNB_AIS - (STEP 2)/metadata_JPSS-2.csv
536K	[2]_DNB_AIS - (STEP 2)/metadata_JPSS-1.csv
540K	[2]_DNB_AIS - (STEP 2)/metadata_S-NPP.csv
708K	[2]_DNB_AIS - (STEP 2)/eez
1.3M	[2]_DNB_AIS - (STEP 2)/[C]_metadata_analyzer.ipynb
4.8M	[2]_DNB_AIS - (STEP 2)/layers_JPSS-2
402M	[2]_DNB_AIS - (STEP 2)/pixel_overlap.csv
417M	[2]_DNB_AIS - (STEP 2)/pixel_overlap-allITPL.csv
440M	[2]_DNB_AIS - (STEP 2)/bboxes_JPSS-2
1.1G	[2]_DNB_AIS - (STEP 2)/toTW.zip
4.8G	[2]_DNB_AIS - (STEP 2)/AIS

### [3]_DNB_AIS - (STEP 3)
  0B	[3]_DNB_AIS - (STEP 3)/RESULTS_TEST_5_A2025001_1754_021.tif
  0B	[3]_DNB_AIS - (STEP 3)/RESULTS_TEST_5_A2025001_1754_021_KR.tif
4.0K	[3]_DNB_AIS - (STEP 3)/PH_masked_density_unet_design.md
4.0K	[3]_DNB_AIS - (STEP 3)/[F]_pixel_overlap.py
4.0K	[3]_DNB_AIS - (STEP 3)/[sub2]_crosscheck.py
4.0K	[3]_DNB_AIS - (STEP 3)/[sub3]_fill_tif_name.py
4.0K	[3]_DNB_AIS - (STEP 3)/[sub4]_refresh.py
4.0K	[3]_DNB_AIS - (STEP 3)/percentile_mk1.m
4.0K	[3]_DNB_AIS - (STEP 3)/percentile_mk2.m
8.0K	[3]_DNB_AIS - (STEP 3)/[sub1]_preprocessing.ipynb
 12K	[3]_DNB_AIS - (STEP 3)/[B]_polygon_handler.py
 16K	[3]_DNB_AIS - (STEP 3)/OnlyIn.txt
 16K	[3]_DNB_AIS - (STEP 3)/[A]_dnb2geotif_v2_modified_4326_metadata.py
 16K	[3]_DNB_AIS - (STEP 3)/[E]_bounding_box.py
 24K	[3]_DNB_AIS - (STEP 3)/.DS_Store
 24K	[3]_DNB_AIS - (STEP 3)/__pycache__
 32K	[3]_DNB_AIS - (STEP 3)/[D]_ship_class_SQL.py
 32K	[3]_DNB_AIS - (STEP 3)/percentiles.mat
 40K	[3]_DNB_AIS - (STEP 3)/RESULTS_TEST_5_A2025001_1754_021_batch_1.tif
 40K	[3]_DNB_AIS - (STEP 3)/[D]_ship_class_SQL_fast.py
 48K	[3]_DNB_AIS - (STEP 3)/QGIS_prj_files
 48K	[3]_DNB_AIS - (STEP 3)/cross-check
 92K	[3]_DNB_AIS - (STEP 3)/[G]_pixel_analyzer.ipynb
 96K	[3]_DNB_AIS - (STEP 3)/ship_heatmap_dl_trend_summary.pdf
120K	[3]_DNB_AIS - (STEP 3)/eez_12nm
184K	[3]_DNB_AIS - (STEP 3)/[H]_pixel_analyzer_v2.ipynb
336K	[3]_DNB_AIS - (STEP 3)/ships_radiance
532K	[3]_DNB_AIS - (STEP 3)/metadata_JPSS-2.csv
536K	[3]_DNB_AIS - (STEP 3)/metadata_JPSS-1.csv
540K	[3]_DNB_AIS - (STEP 3)/metadata_S-NPP.csv
708K	[3]_DNB_AIS - (STEP 3)/eez
1.3M	[3]_DNB_AIS - (STEP 3)/[C]_metadata_analyzer.ipynb
 13M	[3]_DNB_AIS - (STEP 3)/DRUID
 20M	[3]_DNB_AIS - (STEP 3)/RESULTS_TEST_5_A2025001_1754_021_batch_3.tif
 77M	[3]_DNB_AIS - (STEP 3)/outputs
402M	[3]_DNB_AIS - (STEP 3)/pixel_overlap.csv
417M	[3]_DNB_AIS - (STEP 3)/pixel_overlap-allITPL.csv
506M	[3]_DNB_AIS - (STEP 3)/A2025001_1754_021.tif
642M	[3]_DNB_AIS - (STEP 3)/DRUID_TESTING
1.1G	[3]_DNB_AIS - (STEP 3)/toTW.zip
4.3G	[3]_DNB_AIS - (STEP 3)/AIS
 11G	[3]_DNB_AIS - (STEP 3)/bboxes_JPSS-2

```

## File Count by Step Subdirectory

```text
 513 [1]_DNB_AIS - (STEP 1)/results_4326_OVERLAP_O
 359 [3]_DNB_AIS - (STEP 3)/bboxes_JPSS-2
 327 [3]_DNB_AIS - (STEP 3)/AIS
 327 [2]_DNB_AIS - (STEP 2)/AIS
  22 [2]_DNB_AIS - (STEP 2)/bboxes_JPSS-2
  11 [3]_DNB_AIS - (STEP 3)/ships_radiance
   9 [3]_DNB_AIS - (STEP 3)/cross-check
   9 [2]_DNB_AIS - (STEP 2)/cross-check
   8 [3]_DNB_AIS - (STEP 3)/DRUID
   7 [3]_DNB_AIS - (STEP 3)/DRUID_TESTING
   7 [1]_DNB_AIS - (STEP 1)/references
   6 [1]_DNB_AIS - (STEP 1)/eez
   5 [3]_DNB_AIS - (STEP 3)/eez_12nm
   5 [3]_DNB_AIS - (STEP 3)/eez
   5 [2]_DNB_AIS - (STEP 2)/eez_12nm
   5 [2]_DNB_AIS - (STEP 2)/eez
   5 [1]_DNB_AIS - (STEP 1)/eez_12nm
   3 [2]_DNB_AIS - (STEP 2)/qgz_archieve
   3 [2]_DNB_AIS - (STEP 2)/IMG_min:max
   3 [1]_DNB_AIS - (STEP 1)/IMG_min:max
   2 [3]_DNB_AIS - (STEP 3)/QGIS_prj_files
   1 [3]_DNB_AIS - (STEP 3)/toTW.zip
   1 [3]_DNB_AIS - (STEP 3)/ship_heatmap_dl_trend_summary.pdf
   1 [3]_DNB_AIS - (STEP 3)/pixel_overlap.csv
   1 [3]_DNB_AIS - (STEP 3)/pixel_overlap-allITPL.csv
   1 [3]_DNB_AIS - (STEP 3)/percentiles.mat
   1 [3]_DNB_AIS - (STEP 3)/percentile_mk2.m
   1 [3]_DNB_AIS - (STEP 3)/percentile_mk1.m
   1 [3]_DNB_AIS - (STEP 3)/metadata_S-NPP.csv
   1 [3]_DNB_AIS - (STEP 3)/metadata_JPSS-2.csv
   1 [3]_DNB_AIS - (STEP 3)/metadata_JPSS-1.csv
   1 [3]_DNB_AIS - (STEP 3)/__pycache__
   1 [3]_DNB_AIS - (STEP 3)/[sub4]_refresh.py
   1 [3]_DNB_AIS - (STEP 3)/[sub3]_fill_tif_name.py
   1 [3]_DNB_AIS - (STEP 3)/[sub2]_crosscheck.py
   1 [3]_DNB_AIS - (STEP 3)/[sub1]_preprocessing.ipynb
   1 [3]_DNB_AIS - (STEP 3)/[H]_pixel_analyzer_v2.ipynb
   1 [3]_DNB_AIS - (STEP 3)/[G]_pixel_analyzer.ipynb
   1 [3]_DNB_AIS - (STEP 3)/[F]_pixel_overlap.py
   1 [3]_DNB_AIS - (STEP 3)/[E]_bounding_box.py
   1 [3]_DNB_AIS - (STEP 3)/[D]_ship_class_SQL_fast.py
   1 [3]_DNB_AIS - (STEP 3)/[D]_ship_class_SQL.py
   1 [3]_DNB_AIS - (STEP 3)/[C]_metadata_analyzer.ipynb
   1 [3]_DNB_AIS - (STEP 3)/[B]_polygon_handler.py
   1 [3]_DNB_AIS - (STEP 3)/[A]_dnb2geotif_v2_modified_4326_metadata.py
   1 [3]_DNB_AIS - (STEP 3)/RESULTS_TEST_5_A2025001_1754_021_batch_3.tif
   1 [3]_DNB_AIS - (STEP 3)/RESULTS_TEST_5_A2025001_1754_021_batch_1.tif
   1 [3]_DNB_AIS - (STEP 3)/PH_masked_density_unet_design.md
   1 [3]_DNB_AIS - (STEP 3)/OnlyIn.txt
   1 [3]_DNB_AIS - (STEP 3)/A2025001_1754_021.tif
   1 [3]_DNB_AIS - (STEP 3)/.DS_Store
   1 [2]_DNB_AIS - (STEP 2)/toTW.zip
   1 [2]_DNB_AIS - (STEP 2)/pixel_overlap.csv
   1 [2]_DNB_AIS - (STEP 2)/pixel_overlap-allITPL.csv
   1 [2]_DNB_AIS - (STEP 2)/percentile.m
   1 [2]_DNB_AIS - (STEP 2)/metadata_S-NPP.csv
   1 [2]_DNB_AIS - (STEP 2)/metadata_JPSS-2.csv
   1 [2]_DNB_AIS - (STEP 2)/metadata_JPSS-1.csv
   1 [2]_DNB_AIS - (STEP 2)/layers_JPSS-2
   1 [2]_DNB_AIS - (STEP 2)/[sub4]_refresh.py
   1 [2]_DNB_AIS - (STEP 2)/[sub3]_fill_tif_name.py
   1 [2]_DNB_AIS - (STEP 2)/[sub2]_crosscheck.py
   1 [2]_DNB_AIS - (STEP 2)/[sub1]_preprocessing.ipynb
   1 [2]_DNB_AIS - (STEP 2)/[G]_pixel_analyzer.ipynb
   1 [2]_DNB_AIS - (STEP 2)/[F]_pixel_overlap.py
   1 [2]_DNB_AIS - (STEP 2)/[E]_bounding_box.py
   1 [2]_DNB_AIS - (STEP 2)/[D]_ship_class_SQL.py
   1 [2]_DNB_AIS - (STEP 2)/[C]_metadata_analyzer.ipynb
   1 [2]_DNB_AIS - (STEP 2)/[B]_polygon_handler.py
   1 [2]_DNB_AIS - (STEP 2)/[A]_dnb2geotif_v2_modified_4326_metadata.py
   1 [2]_DNB_AIS - (STEP 2)/OnlyIn.txt
   1 [2]_DNB_AIS - (STEP 2)/.DS_Store
   1 [1]_DNB_AIS - (STEP 1)/metadata.csv
   1 [1]_DNB_AIS - (STEP 1)/layers
   1 [1]_DNB_AIS - (STEP 1)/dnb_03.tar.gz
   1 [1]_DNB_AIS - (STEP 1)/dnb_02.tar.gz
   1 [1]_DNB_AIS - (STEP 1)/OnlyIn.txt
   1 [1]_DNB_AIS - (STEP 1)/D_metadata_analyzer.ipynb
   1 [1]_DNB_AIS - (STEP 1)/C_polygon_handler.py
   1 [1]_DNB_AIS - (STEP 1)/B_metadata_generator.py
   1 [1]_DNB_AIS - (STEP 1)/A_dnb2geotif_v2_modified_4326.py
   1 [1]_DNB_AIS - (STEP 1)/2025년_S-NPP_DNB영상_관측통계.pptx
   1 [1]_DNB_AIS - (STEP 1)/2025년_S-NPP_DNB영상_관측통계.pdf
   1 [1]_DNB_AIS - (STEP 1)/.DS_Store
```

## Git Status Snapshot

```text
 M .gitignore
 D 0408_1640.md
 D 0408_1834.md
 D 0408_1909.md
 M DNB_GAT_v1.ipynb
 M codex_stamp.md
 M sub_module/dnb_gat_pipeline.py
 M sub_module/generate_dnb_gat_notebook.py
?? "[1]_DNB_AIS - (STEP 1)/"
?? "[2]_DNB_AIS - (STEP 2)/"
?? "[3]_DNB_AIS - (STEP 3)/OnlyIn.txt"
?? "[3]_DNB_AIS - (STEP 3)/[A]_dnb2geotif_v2_modified_4326_metadata.py"
?? "[3]_DNB_AIS - (STEP 3)/[B]_polygon_handler.py"
?? "[3]_DNB_AIS - (STEP 3)/[C]_metadata_analyzer.ipynb"
?? "[3]_DNB_AIS - (STEP 3)/[D]_ship_class_SQL.py"
?? "[3]_DNB_AIS - (STEP 3)/[D]_ship_class_SQL_fast.py"
?? "[3]_DNB_AIS - (STEP 3)/[E]_bounding_box.py"
?? "[3]_DNB_AIS - (STEP 3)/[F]_pixel_overlap.py"
?? "[3]_DNB_AIS - (STEP 3)/[G]_pixel_analyzer.ipynb"
?? "[3]_DNB_AIS - (STEP 3)/[H]_pixel_analyzer_v2.ipynb"
?? "[3]_DNB_AIS - (STEP 3)/[sub1]_preprocessing.ipynb"
?? "[3]_DNB_AIS - (STEP 3)/[sub2]_crosscheck.py"
?? "[3]_DNB_AIS - (STEP 3)/[sub3]_fill_tif_name.py"
?? "[3]_DNB_AIS - (STEP 3)/[sub4]_refresh.py"
?? "[3]_DNB_AIS - (STEP 3)/eez/"
?? "[3]_DNB_AIS - (STEP 3)/eez_12nm/"
?? "[3]_DNB_AIS - (STEP 3)/metadata_JPSS-1.csv"
?? "[3]_DNB_AIS - (STEP 3)/metadata_JPSS-2.csv"
?? "[3]_DNB_AIS - (STEP 3)/metadata_S-NPP.csv"
?? "[3]_DNB_AIS - (STEP 3)/outputs/"
?? "[3]_DNB_AIS - (STEP 3)/percentile_mk1.m"
?? "[3]_DNB_AIS - (STEP 3)/percentile_mk2.m"
?? "[3]_DNB_AIS - (STEP 3)/percentiles.mat"
?? "[3]_DNB_AIS - (STEP 3)/pixel_overlap-allITPL.csv"
?? "[3]_DNB_AIS - (STEP 3)/pixel_overlap.csv"
?? _archive/
?? codex_logs/
?? scripts/archive_legacy_steps.sh
?? sub_module/pipeline_runtime.py
?? sub_module/run_count_sum_lambda_sweep.py
?? sub_module/run_druid_spsa_search.py
?? sub_module/run_sum_preserving_validation.py
?? sub_module/run_sum_preserving_weighting_grid.py
?? sub_module/torchview_export.py
?? sub_module/torchviz_export.py
```

## Classification

- Active code/config/docs: `sub_module/`, `configs/`, `docs/`, `scripts/`.
- Active step workspace: `[3]_DNB_AIS - (STEP 3)`; keep path stable for current scripts.
- Legacy archive candidates: `[1]_DNB_AIS - (STEP 1)`, `[2]_DNB_AIS - (STEP 2)`.
- Generated runtime outputs: `outputs/`, `[3]_DNB_AIS - (STEP 3)/outputs/`; keep out of git except README/curated summaries.
- Heavy raw/geospatial data: `AIS/`, `bboxes_*`, TIFF/DB/ZIP/PDF artifacts; keep out of git.

## Follow-Up Command

```sh
./scripts/archive_legacy_steps.sh
```
