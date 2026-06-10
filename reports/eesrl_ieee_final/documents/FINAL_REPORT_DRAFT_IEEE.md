# Draft Final Report - PH-Assisted VIIRS DNB Ship-Presence Probability Mapping

> Working draft. Numerical result paragraphs and tables should be finalized only after the active full-split experiment completes.

## Title

PH-Assisted Ship-Presence Probability Mapping from VIIRS Day/Night Band Imagery with AIS-Derived Supervision

## Abstract

Nighttime maritime monitoring with VIIRS Day/Night Band (DNB) imagery is limited by coarse spatial resolution, light blooming, and overlapping vessel lights. A bright DNB pixel is therefore not always a reliable indicator of vessel presence, especially in dense or saturated regions. This study proposes a PH-assisted image-to-image pipeline that converts DNB radiance imagery into a per-pixel ship-presence probability map using AIS-derived supervision. Persistent homology (PH) is used to construct topology-aware image partitions and compact feature channels, while a U-Net predicts a continuous probability field from arctan-encoded brightness, PH persistence, and PH seed maps. Instead of treating AIS pixels as exact hard labels or attempting integer vessel counting, AIS-positive pixels seed a Gaussian proximity field that reflects the spatial uncertainty between AIS points and coarse DNB observations. The model is evaluated by comparing the ranking quality of predicted probabilities against raw DNB brightness using average precision and top-k precision over held-out AIS-derived targets. Preliminary experiments suggest that the learned probability map can rank ship-presence pixels more reliably than raw brightness on expanded held-out splits. Final full-split metrics will be inserted after the ongoing active experiment completes.

## I. Introduction

Nighttime vessel activity can be observed by low-light satellite sensors such as the VIIRS Day/Night Band (DNB). In principle, vessel lights provide a useful wide-area signal for maritime monitoring, including fishing activity and other nighttime operations. In practice, however, DNB imagery is difficult to interpret at the pixel level. The sensor footprint is much larger than individual vessels, light can bloom across neighboring pixels, and multiple vessels can contribute to a single bright region. As a result, a simple threshold on DNB brightness is often insufficient: some bright regions are not vessel-related, while dense vessel groups may appear as saturated blobs with limited internal structure.

Automatic Identification System (AIS) records provide sparse supervision for vessels that broadcast their location. AIS is not a complete observation of all vessels, but it offers a practical reference for validating whether nighttime brightness patterns correspond to known ship presence. The central question of this report is therefore not whether DNB can recover exact vessel counts, but whether a learned probability map can make bright DNB imagery more semantically meaningful: when a pixel has a high predicted value, it should correspond to AIS-derived ship presence more reliably than a high raw DNB brightness value.

This project evolved from an earlier graph-based direction into a simpler image-to-image probability mapping pipeline. The active method uses persistent homology (PH) as a structural prior over DNB brightness patterns. PH-derived anchors guide recursive patch partitioning, and two physically interpretable PH channels are provided to the U-Net: a persistence map and a seed map. The model output is a per-pixel ship-presence probability field, trained with a target-weighted SmoothL1 regression loss against a Gaussian proximity target derived from AIS-positive pixels.

The contributions of this work are:

1. A PH-assisted recursive exact-cover patching pipeline for coarse nighttime maritime imagery.
2. A minimal PH feature design using only brightness, PH persistence, and PH seed channels.
3. A probability-field supervision strategy that avoids overclaiming exact AIS-pixel localization or integer vessel counts.
4. A direct evaluation protocol comparing learned probability rankings against raw DNB brightness on the same AIS-derived target.

## II. Related Work

### A. VIIRS DNB and Nighttime Maritime Observation

The VIIRS DNB sensor enables low-light nighttime imaging and has been widely studied for nighttime illumination monitoring [1], [2]. Maritime applications use DNB brightness to identify vessel lights and fishing activity, often with AIS or other vessel records as validation sources [3], [4]. These studies motivate the use of DNB imagery for vessel monitoring, but they also highlight a key challenge: raw brightness is not a calibrated vessel-presence probability.

### B. Density and Point-Supervised Heatmap Learning

Object counting and density-map learning commonly transform point annotations into spatial target maps, then train convolutional networks to predict those maps [5]. Related work in crowd counting shows that exact point matching is often too brittle, motivating probabilistic or distributional supervision when objects are dense or annotation alignment is uncertain [7]-[10]. This report borrows the idea of point-derived spatial supervision, but does not claim to solve vessel count regression. The active target is a ship-presence probability field, not an integer density map.

### C. U-Net Image-to-Image Prediction

U-Net-style encoder-decoder networks are effective for pixelwise prediction tasks because they combine local detail with multi-scale context [6]. The proposed model follows this image-to-image framing: it maps a small stack of DNB/PH input channels to one ship-presence probability channel.

### D. Persistent Homology for Source Structure

Persistent homology provides a way to describe connected components and topological persistence across intensity thresholds. Cubical persistent homology tools such as Cubical Ripser support image-based PH computation [11]. DRUID demonstrates how PH can support source detection and deblending in astronomical imagery [12]. In this project, PH is not used as a final detector. Instead, it provides image-structure anchors for patch partitioning and lightweight feature channels for the U-Net.

## III. Data and Preprocessing

### A. DNB Imagery

The input imagery consists of JPSS-2/VIIRS DNB GeoTIFF scenes from 2025. Each scene is converted to an encoded brightness image using an arctan transform:

```text
B = (2 / pi) * arctan(radiance / 1e-9)
```

This compresses the high dynamic range of nighttime radiance while preserving relative brightness structure. The encoded brightness channel is the first model input.

### B. AIS-Derived Supervision

AIS-derived GeoJSON records are rasterized into image coordinates. Pixels with one or more AIS-associated vessel observations are treated as source pixels:

```text
Y_seed(x, y) = 1[raw_count(x, y) > 0]
```

The raw count map is not used as an integer count target in the active method. Instead, the positive source pixels define a spatial proximity target.

### C. Sea Mask

A KR EEZ plus 12 nautical mile sea mask is applied to restrict training and evaluation to valid maritime regions. All model outputs and target fields are masked by the valid owner mask of each partition.

### D. Splits

Scene splits are performed at the day/scene level to reduce leakage between train, validation, and test data. The active split for the current full experiment is:

```text
outputs/dnb_density/splits/ox_spatial_25pct_63_15_14_20260609_011421/scene_split.csv
```

Final split statistics will be inserted after the active full run completes.

## IV. Method

### A. PH-Assisted Recursive Exact-Cover Partitioning

The pipeline first computes PH candidates from encoded DNB brightness. PH anchors claim valid sea pixels, and oversized anchors are recursively subdivided by rerunning PH within large regions. Remaining valid sea pixels are assigned to fallback grid tiles. This creates an exact-cover patch set: each valid owner pixel is assigned to one partition for supervision and evaluation.

The motivation is practical. Direct fixed tiling can create very large or poorly aligned patches, while pure PH anchors may leave gaps. The hybrid policy keeps PH structure where available and uses fallback tiles only for uncovered regions.

### B. PH Input Channels

The active input tensor has three channels:

```text
0: encoded DNB brightness
1: PH persistence map
2: PH seed map
```

The PH persistence map assigns each PH component a normalized lifetime score within its local patch. The PH seed map marks PH seed locations with small binary disks. Retired channels such as parent masks, child masks, soft attention, and lifetime maps are no longer used as model inputs.

### C. Probability Target

AIS-positive pixels seed a Gaussian proximity field:

```text
Y_field(x, y) = exp(-0.5 * d(x, y)^2 / sigma^2)
```

where `d(x, y)` is the distance to the nearest AIS-positive source pixel within the valid owner mask. The active configuration uses:

```text
sigma = 4 pixels
radius = 12 pixels
presence threshold for metrics = 0.25
```

This target represents proximity to AIS-observed ship presence. It should not be interpreted as a smoothed vessel count or as a guarantee that every high target pixel contains a vessel center.

### D. PixelProbabilityUNet

The model is a compact U-Net with three input channels and one output channel. The output logits are converted to probabilities with a sigmoid:

```text
P(x, y) = sigmoid(logit(x, y)) * valid_owner_mask(x, y)
```

### E. Loss Function

The active training objective is target-weighted SmoothL1 probability-field regression:

```text
L = mean_valid((1 + alpha * Y_field) * SmoothL1(P, Y_field; beta))
```

with:

```text
alpha = 8.0
beta = 0.1
```

The weighting increases the contribution of AIS-proximal pixels without turning the task into a hard binary classification problem.

### F. Evaluation Protocol

The primary evaluation asks:

```text
Does model probability rank AIS-derived presence pixels above non-presence pixels better than raw DNB brightness does?
```

The target field is binarized for ranking evaluation:

```text
Y_eval = 1[Y_field >= 0.25]
```

Within valid sea pixels, two scores are compared against the same binary target:

```text
score_model = P(x, y)
score_brightness = B(x, y)
```

Primary metrics are average precision (AP), top-k precision, AP lift over brightness, validation-calibrated threshold F1, and model reliability bins. A radius-tolerant sensitivity check repeats the same ranking evaluation using broader Gaussian targets such as `sigma=8`.

## V. Experiments

### A. Implementation

Experiments are implemented in Python/PyTorch and run on Apple Silicon with the MPS backend. The active configuration is:

```text
configs/dnb_density_unet_probability_field_recursive_ph_20260610.json
```

### B. Training Setup

The current active full experiment uses:

```text
split: outputs/dnb_density/splits/ox_spatial_25pct_63_15_14_20260609_011421/scene_split.csv
epochs: 24
batch size: 8
optimizer: AdamW
learning rate: 5e-5
weight decay: 3e-4
max patches per scene: 32
input channels: brightness, PH persistence, PH seed
```

A reusable patch cache is generated for the run. Old caches are preserved to support reproducibility.

### C. Baseline

The main baseline is raw DNB brightness. This is the correct baseline for the project claim because the goal is to improve the semantic reliability of high-brightness pixels. Brightness is not treated as a calibrated probability, so Brier and reliability diagnostics are reported only for the model unless a separate brightness calibration model is fitted.

### D. Tables to Fill After Full Run

Table I. Scene and patch statistics by split.

| Split | Scenes | Kept scenes | Patches | AIS seed count | Valid pixels | Positive eval pixels |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | TODO_RESULT | TODO_RESULT | TODO_RESULT | TODO_RESULT | TODO_RESULT | TODO_RESULT |
| Val | TODO_RESULT | TODO_RESULT | TODO_RESULT | TODO_RESULT | TODO_RESULT | TODO_RESULT |
| Test | TODO_RESULT | TODO_RESULT | TODO_RESULT | TODO_RESULT | TODO_RESULT | TODO_RESULT |

Table II. Model probability vs raw brightness on the test split.

| Score | AP | Top-0.5% precision | Top-1% precision | Top-5% precision | Top-10% precision |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw DNB brightness | TODO_RESULT | TODO_RESULT | TODO_RESULT | TODO_RESULT | TODO_RESULT |
| Model probability | TODO_RESULT | TODO_RESULT | TODO_RESULT | TODO_RESULT | TODO_RESULT |
| Model / brightness lift | TODO_RESULT | TODO_RESULT | TODO_RESULT | TODO_RESULT | TODO_RESULT |

Table III. Radius-tolerant sensitivity analysis.

| Target radius | Model AP | Brightness AP | AP lift | Top-1% lift | Top-5% lift |
| --- | ---: | ---: | ---: | ---: | ---: |
| sigma=4 | TODO_RESULT | TODO_RESULT | TODO_RESULT | TODO_RESULT | TODO_RESULT |
| sigma=8 | TODO_RESULT | TODO_RESULT | TODO_RESULT | TODO_RESULT | TODO_RESULT |

Table IV. Thresholded field metrics.

| Checkpoint | Threshold policy | Threshold | Precision | Recall | F1 | IoU | Brier |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| best validation pixel F1 | fixed 0.5 | 0.500 | TODO_RESULT | TODO_RESULT | TODO_RESULT | TODO_RESULT | TODO_RESULT |
| best validation pixel F1 | val-calibrated | TODO_RESULT | TODO_RESULT | TODO_RESULT | TODO_RESULT | TODO_RESULT | TODO_RESULT |
| best validation loss | fixed 0.5 | 0.500 | TODO_RESULT | TODO_RESULT | TODO_RESULT | TODO_RESULT | TODO_RESULT |
| best validation loss | val-calibrated | TODO_RESULT | TODO_RESULT | TODO_RESULT | TODO_RESULT | TODO_RESULT | TODO_RESULT |

## VI. Preliminary Results Placeholder

The 20-scene expanded experiment already showed that the SmoothL1 probability-field formulation can outperform raw brightness on held-out test patches. The active full-split experiment should determine whether this behavior scales to the larger 157/40/31 scene split.

Final text should be inserted here after the active full run completes:

```text
TODO_RESULT_ACTIVE_FULL_SUMMARY
```

The strongest result statement should only be used if the full run preserves model lift over brightness on AP and top-k precision.

## VII. Discussion

### A. Why Probability Mapping Instead of Count Regression?

At the current DNB resolution, exact count regression is ill-posed. Multiple vessels may fall within one bright region, vessel light intensity varies by vessel type and direction, and saturation can flatten radiance differences. A probability map is therefore a more defensible first target: it asks whether the image contains evidence of ship presence, not how many ships are present in each pixel.

### B. Why Not Exact AIS Pixel Hits?

AIS points and DNB pixels are not perfectly aligned in time or space. Treating the exact rasterized AIS pixel as the only positive target would over-penalize near misses caused by geolocation uncertainty, sensor footprint, and temporal mismatch. The Gaussian proximity field preserves the AIS source pixels while giving the model a spatially tolerant target.

### C. Role of PH

PH is used as a structural prior, not as a label. It helps identify bright connected structures, subdivide large regions, and provide minimal topology-informed feature channels. This design is intentionally conservative: only PH persistence and PH seed maps are retained as inputs, while more complex masks and soft attention channels are retired.

### D. Interpreting Curvature and Gated DNB Products

Gradient magnitude and negative Laplacian visualizations can reveal whether the predicted probability field forms meaningful edges, ridges, or peaks inside saturated DNB blobs. These are qualitative diagnostics, not primary metrics. A derived product such as `raw_DNB * probability^gamma` may be explored later as a ship-presence-gated radiance map, but it should remain a future visualization or correction-filter idea rather than the main claim.

### E. Limitations

This study relies on AIS-derived labels, which are incomplete because not all vessels broadcast AIS. DNB brightness can include non-vessel light sources, atmospheric effects, clouds, coastal contamination, or sensor artifacts. The probability map is not yet calibrated as a true physical probability and should not be interpreted as an integer vessel count. Final claims should be limited to ranking improvement over raw brightness unless additional external validation is added.

## VIII. Conclusion

This report presents a PH-assisted U-Net pipeline for converting coarse VIIRS DNB nighttime imagery into a ship-presence probability map using AIS-derived supervision. The method avoids brittle exact-pixel classification and premature count regression by training against a Gaussian proximity probability field. The primary evaluation directly compares the learned probability score with raw DNB brightness on held-out AIS-derived targets. If the active full-split experiment confirms the preliminary brightness-lift results, the final conclusion will be that PH-assisted probability mapping provides a more semantically reliable ship-presence score than raw DNB brightness alone.

## References

Use the IEEE-formatted references maintained in:

```text
docs/FINAL_REPORT_CITATIONS_IEEE.md
```
