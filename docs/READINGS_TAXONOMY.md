# Readings Taxonomy

`_Readings/` is a local, git-ignored literature folder. On 2026-05-29, its 39 PDFs were reorganized by reading only each document's title page, abstract, preface, or opening section.

## Categories

| Folder | Count | Purpose |
|---|---:|---|
| `_Readings/01_viirs_dnb_instrument_and_processing/` | 9 | VIIRS/DNB sensor, SDR/L1B, calibration, JPSS/S-NPP ground systems, KOSC DNB archive/processing, lunar correction, DNB flicker/NTL behavior. |
| `_Readings/02_dnb_ais_maritime_vessel_detection/` | 2 | DNB nighttime vessel/fishing detection and AIS-based validation. |
| `_Readings/03_density_heatmap_and_geospatial_deep_learning/` | 5 | Density-map counting, Gaussian heatmap regression, NTL downscaling, land-cover segmentation, geospatial AI benchmarks. |
| `_Readings/04_ais_trajectory_and_maritime_rules/` | 5 | AIS interpolation, imputation, trajectory reconstruction, reporting intervals, and navigation light rules. |
| `_Readings/05_sar_radar_and_isac/` | 12 | SAR/radar imaging, SAR ATR/detection, radar enhancement, GPU radar processing, ISAC, rotational Doppler/sensing physics. |
| `_Readings/06_persistent_homology_and_astro_image_processing/` | 2 | DRUID/persistent homology and astronomy image preprocessing/source detection references. |
| `_Readings/07_general_ml_statistics_and_compression/` | 3 | General ML fitting, statistics, nonlinear transforms, and compression references. |
| `_Readings/08_ocean_disaster_forecasting/` | 1 | Storm surge and ocean/disaster forecasting references. |

## Local Files

- `_Readings/README.md`: category descriptions and cleanup notes.
- `_Readings/classification_manifest.tsv`: per-PDF category and short classification basis.

## Policy

- Keep `_Readings/` ignored by git because it contains copyrighted PDFs and local reference copies.
- Promote only short notes, taxonomy summaries, or citation metadata into `docs/` when useful for reproducibility.
