# Final Report Prep - IEEE-Style EESRL Report

This document prepares the final-report writing workflow for the active DNB/AIS pipeline.

## Source Midterm Report

Located EESRL report folder:

```text
/Users/jungtaeuk/Desktop/26-1 Semester/EESRL_Earth_and_Environmental_Science_Research_Lab
```

Midterm final PDF:

```text
/Users/jungtaeuk/Desktop/26-1 Semester/EESRL_Earth_and_Environmental_Science_Research_Lab/정태욱_지구환경과학연구실습_중간보고서_최종.pdf
```

Important pivot:

```text
The midterm report was written around a DRUID + GAT direction.
The active final-report method is now PH-assisted OccupancySpatial U-Net: patch-level ship O/X plus positive-patch spatial localization.
GAT and direct count regression are historical design paths only.
```

## Final Report Target Form

Use an IEEE-like technical paper structure even if the final submission is not required to be exact IEEEtran.

Recommended LaTeX base:

```latex
\documentclass[conference]{IEEEtran}
\usepackage{graphicx}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{cite}
\usepackage{url}
```

If Korean body text is required, use a XeLaTeX/LuaLaTeX-compatible Korean template instead of forcing Korean into plain `IEEEtran`.

## Active Method Summary

```text
problem: VIIRS DNB pixels are coarse enough that multiple vessel lights can overlap.
data: 2025 JPSS-2/VIIRS DNB GeoTIFF + AIS-derived GeoJSON point/bbox supervision.
preprocessing: arctan-encoded DNB brightness, KR EEZ + 12 nm sea mask, day-level split.
structure prior: H0 persistent-homology anchors from encoded brightness, hierarchical PH rerun for oversized anchors, exact-cover sea partition with fallback tiles.
model: OccupancySpatialUNet.
output: occupancy evidence heatmap, not expected integer ship-count density.
training target: patch O/X label plus normalized positive-patch spatial density target.
```

Model decomposition:

```text
P = sigmoid(occupancy_logit)                         # probability that the patch contains at least one ship
S = softmax(spatial_logits over valid owner pixels)  # pixel distribution conditional on a positive patch
Y_pred = P * S                                       # occupancy evidence heatmap
```

Target construction:

```text
raw AIS/GeoJSON points -> Gaussian kernel target over valid sea pixels
positive patch: target_count > 0
T = target_density / sum(target_density) for positive patches
negative patch: T = 0 and O/X target = 0
PH masks are input/attention priors, not GT censoring masks.
```

Current input channels from `configs/dnb_density_unet_occupancy_spatial.json`:

```text
0 brightness
1 parent_ph_mask
2 child_ph_union_mask
3 ph_seed_map
4 ph_persistence_map
5 ph_soft_attention
6 anchor_lifetime_map
```

## Proposed Final Report Outline

### 1. Abstract

Include:

```text
problem: low-resolution nighttime vessel-light overlap in VIIRS DNB scenes
data: 2025 JPSS-2/VIIRS DNB GeoTIFF and AIS-derived supervision
method: sea masking + PH-assisted exact-cover partitioning + OccupancySpatial U-Net
output: full-scene occupancy evidence heatmap
result: final train/validation/test O/X and spatial-localization metrics
```

### 2. Introduction

Keep the midterm motivation:

```text
AIS-only monitoring has blind spots.
VIIRS DNB directly observes nighttime lights over large sea areas.
Low spatial resolution makes individual-vessel detection hard in dense clusters.
Therefore the current final task is ship-presence heatmap estimation and localization, with count reintroduction deferred until O/X and localization are reliable.
```

### 3. Data

Subsections:

```text
3.1 VIIRS DNB GeoTIFF full-scene imagery
3.2 AIS interpolation and GeoJSON-derived supervision
3.3 KR EEZ + 12 nm sea mask
3.4 Day-level train/validation/test split
```

Required tables:

```text
Table I: final split by days and scenes
Table II: scene/patch counts after PH-anchor-zero filtering
Table III: positive/negative patch distribution by split
```

### 4. Method

Subsections:

```text
4.1 DNB radiance preprocessing and arctan encoding
4.2 AIS-to-image time/position matching
4.3 KR sea masking
4.4 PH-assisted full-scene partitioning
4.5 PH feature-channel construction
4.6 OccupancySpatial U-Net architecture
4.7 O/X + spatial-distribution loss
4.8 Patch inference and full-scene prediction merge
4.9 Count head as deferred extension
```

Core equations:

```text
encoded = (2 / pi) * arctan(radiance / 1e-9)

O_target = 1[sum(target_density over valid owner pixels) > 0]

T_pixel = target_density_pixel / sum(target_density over valid owner pixels)  for positive patches

P = sigmoid(occupancy_logit)
S_pixel = softmax(spatial_logits over valid owner pixels)
Y_pred_pixel = P * S_pixel

L = lambda_occ * BCE(P, O_target) + lambda_spatial * KL(T || S) on positive patches
```

Lifetime use:

```text
PH lifetime is currently used as an input channel and as optional sample weighting through normalized log1p(anchor_lifetime).
It is not used to rescale final prediction mass, because the active output mass is an occupancy probability rather than an integer count.
```

### 5. Experiments

Include:

```text
hardware: Apple Silicon Mac, MPS backend
training config: epochs, batch size, patch caps, optimizer, lr
split policy: day-level split to avoid same-day leakage
checkpoint policy: last, best validation loss, best validation occupancy F1, best validation occupancy mass ratio
```

Baseline comparison protocol:

```text
Do not call the comparison model a legacy method unless a true historical implementation is being reproduced.
Use the name "rule-based brightness threshold baseline".

Baseline rule:
  pred_positive_patch = any(encoded_brightness >= threshold over valid owner pixels)
  pred_spatial_map = normalize(binary_threshold_mask over valid owner pixels)

Threshold candidates:
  primary sweep: 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.90, 0.95 over arctan-encoded DNB brightness
  include 0.85, 0.90, 0.95 as high-threshold cases, but do not restrict the baseline to them because encoded DNB brightness often peaks below 0.85 in the current GeoTIFF products

Selection:
  evaluate all threshold candidates on validation
  select the threshold with best validation occupancy_f1
  break ties by spatial_overlap_mean_positive, target_explained, then lower occupancy_brier
  report the selected threshold on test without retuning

Fairness constraint:
  use the exact same scene_split.csv and patch-selection settings as the active U-Net run
```

Runnable baseline command:

```sh
SCENE_SPLIT_CSV=outputs/dnb_density/splits/<run_tag>/scene_split.csv \
RUN_TAG=brightness_threshold_<run_tag> \
bash scripts/run_brightness_threshold_baseline.sh
```

Outputs:

```text
outputs/dnb_density/baselines/<run_tag>/brightness_threshold_metrics_by_split.csv
outputs/dnb_density/baselines/<run_tag>/brightness_threshold_selected_threshold_metrics.csv
outputs/dnb_density/baselines/<run_tag>/brightness_threshold_scene_build_metrics.csv
outputs/dnb_density/baselines/<run_tag>/brightness_threshold_summary.json
```

### 6. Results

Primary metrics:

```text
occupancy_accuracy
occupancy_precision
occupancy_recall
occupancy_f1
occupancy_brier
spatial_overlap_mean_positive
target_explained
pred_matched
occupancy_mass_ratio_abs_log_error
```

Interpretation:

```text
occupancy metrics evaluate whether the model knows a patch contains any vessel signal.
spatial_overlap_mean_positive evaluates whether positive-patch mass is placed in the right pixels.
target_explained and pred_matched describe how much predicted/target mass overlaps after masking.
occupancy_mass_* is diagnostic only; it is not integer ship-count error.
```

Required figures:

```text
Figure 1: overall active pipeline diagram
Figure 2: VIIRS DNB full scene + KR sea mask
Figure 3: PH anchor / child / fallback partition visualization
Figure 4: patch-level input-channel preview showing all PH features
Figure 5: patch-level target/prediction/error preview
Figure 6: full-scene merged occupancy evidence heatmap
Figure 7: train/validation loss, occupancy F1, and spatial-overlap curves
Figure 8: selected qualitative success/failure cases
```

Required comparison table:

```text
Table IV: rule-based brightness threshold baseline vs PH-assisted OccupancySpatial U-Net

Rows:
  brightness >= 0.35
  brightness >= 0.45
  brightness >= 0.55
  brightness >= 0.65
  brightness >= 0.75
  brightness >= 0.85
  brightness >= 0.90
  brightness >= 0.95
  best brightness threshold selected on validation
  PH-assisted OccupancySpatial U-Net

Columns:
  occupancy_precision
  occupancy_recall
  occupancy_f1
  occupancy_brier
  spatial_overlap_mean_positive
  target_explained
  pred_matched
```

### 7. Discussion

Discuss:

```text
why O/X + localization is better posed than direct count regression at this stage
why PH is used as structure/input prior, not as a hard label mask
why count is deferred and how a conditional count head can be reintroduced
cloud/bright outlier issue from arctan-preserved imagery
limitations of AIS-derived supervision
limitations caused by sparse positive examples and patch-scale variation
```

### 8. Conclusion

State:

```text
The completed active pipeline predicts full-scene vessel-presence evidence from DNB brightness and PH-derived topological features.
It is evaluated first as O/X detection plus spatial localization, not as final integer count estimation.
```

## Citation File

Use:

```text
docs/FINAL_REPORT_CITATIONS_IEEE.md
```

When citation information changes, update that file first.

## Full-Scene Prediction Figure Generation

Patch previews explain local model behavior. For report figures, use the full-scene merge utility:

```sh
PYTHONPATH=. /Users/jungtaeuk/anaconda3/envs/DNB_AIS/bin/python \
  -m sub_module.render_density_full_scene_predictions \
  --run-dir outputs/dnb_density/runs/<run_tag> \
  --split test \
  --limit-scenes 1 \
  --checkpoint-kind best_val_occupancy_f1 \
  --device mps
```

Outputs:

```text
outputs/dnb_density/runs/<run_tag>/full_scene_predictions/<scene_key>/<scene_key>_full_scene_prediction.png
outputs/dnb_density/runs/<run_tag>/full_scene_predictions/<scene_key>/<scene_key>_full_scene_metrics.json
outputs/dnb_density/runs/<run_tag>/full_scene_predictions/full_scene_prediction_metrics.csv
```

Do not commit generated PNG/JSON/CSV outputs unless a small curated metric summary is explicitly needed.

## Next Report-Readiness Tasks

Before writing the final report:

```text
1. run the overnight O/X + spatial train/validation/test experiment
2. generate patch previews with all PH input channels
3. generate full-scene merged predictions for representative test scenes
4. freeze final metric table
5. draft final LaTeX from this outline
```
