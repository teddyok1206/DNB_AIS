# Final Report Prep - IEEE-Style EESRL Report

This document prepares the final-report writing workflow.

## Source Midterm Report

Located EESRL report folder:

```text
/Users/jungtaeuk/Desktop/26-1 Semester/EESRL_Earth_and_Environmental_Science_Research_Lab
```

Midterm final PDF:

```text
/Users/jungtaeuk/Desktop/26-1 Semester/EESRL_Earth_and_Environmental_Science_Research_Lab/정태욱_지구환경과학연구실습_중간보고서_최종.pdf
```

Midterm structure note:

```text
/Users/jungtaeuk/Desktop/26-1 Semester/EESRL_Earth_and_Environmental_Science_Research_Lab/중간보고서 작성 보조/중간보고서_구조안.md
```

Important pivot:

```text
The midterm report was written around a DRUID + GAT direction.
The active final-report method is now PH-assisted U-Net density heatmap prediction.
```

GAT should be mentioned only as a discarded/intermediate design path if the narrative needs it.

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

## Proposed Final Report Outline

### 1. Abstract

Include:

```text
problem: VIIRS DNB low-resolution nighttime vessel lights overlap
data: 2025 JPSS-2/VIIRS DNB GeoTIFF + AIS-derived bbox/point supervision
method: KR sea mask + PH partition + sum-preserving density target + CountSpatial U-Net
output: full-scene per-pixel expected ship-count density heatmap
result: final train/val/test metrics after full run
```

### 2. Introduction

Keep the midterm motivation:

```text
AIS-only monitoring has blind spots.
VIIRS DNB directly observes nighttime lights over large sea areas.
Low spatial resolution makes individual-vessel detection hard in dense clusters.
Therefore the final task is density heatmap prediction, not simple binary vessel detection.
```

### 3. Data

Subsections:

```text
3.1 VIIRS DNB GeoTIFF full-scene imagery
3.2 AIS interpolation and bbox-derived supervision
3.3 KR EEZ + 12 nm sea mask
3.4 Day-level train/validation/test split
```

Required tables:

```text
Table I: final split by days and scenes
Table II: scene/patch counts after PH-anchor filtering
Table III: target count distribution by split
```

### 4. Method

Subsections:

```text
4.1 DNB radiance preprocessing and arctan encoding
4.2 AIS-to-image time/position matching
4.3 PH-assisted full-scene partitioning
4.4 Sum-preserving Gaussian density target
4.5 CountSpatial U-Net architecture
4.6 Loss function and count-by-integral interpretation
4.7 Patch inference and full-scene prediction merge
```

Core equations:

```text
encoded = (2 / pi) * arctan(radiance / 1e-9)

sum(target_density over owner pixels) = AIS ship count in partition

density_pred = count_head(x) * softmax(spatial_logits over valid owner pixels)

ship_count(region) = sum(density_pred over region)
```

### 5. Experiments

Include:

```text
hardware: M2 Max MacBook, MPS backend
training config: epochs, batch size, patch caps, optimizer, lr
split policy: day-level split to avoid same-day leakage
checkpoint policy: last/best-val-loss/best-count-calibrated if implemented
```

### 6. Results

Required metrics:

```text
count ratio: pred_sum / target_sum
MAE or absolute count error per scene/patch
target_explained: sum(min(pred,target)) / sum(target)
pred_matched: sum(min(pred,target)) / sum(pred)
normalized spatial overlap: sum(min(pred/sum(pred), target/sum(target)))
```

Required figures:

```text
Figure 1: overall pipeline diagram
Figure 2: VIIRS DNB full scene + KR sea mask
Figure 3: PH partition / anchor / fallback-grid visualization
Figure 4: patch-level enhanced preview
Figure 5: full-scene merged prediction preview
Figure 6: train/validation loss and count-ratio curve
Figure 7: error map or selected qualitative cases
```

### 7. Discussion

Discuss:

```text
why density heatmap is better aligned than object detection for 750 m DNB pixels
why PH is used as structure/partition prior, not GT censor
why count calibration remains the main model-risk
cloud/bright outlier issue from arctan-preserved imagery
limitations of AIS-derived supervision
```

### 8. Conclusion

State:

```text
The completed pipeline predicts continuous expected ship-count density from DNB brightness and PH features.
The final output is evaluated by density-map integration, not by integer class softmax.
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
1. add best-checkpoint selection
2. run full train/val/test split
3. generate full-scene merged predictions for representative test scenes
4. freeze final metric table
5. draft final LaTeX from this outline
```
