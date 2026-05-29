# DNB Density Model Scaffold

This scaffold now uses the PH hierarchy as a rigid region/structure prior and trains a U-Net density regressor on crop-level targets.

## Active Direction

- Active model: `MaskedDensityUNet`.
- Input: 6-channel PH-hierarchical DNB crop.
- Target: crop-level sum-preserving ship-count density map.
- Loss: soft PH-attention weighted loss over the valid crop.
- Fast dilated CNN and GAT are retained only as design records or future references; they are not the active path.

## U-Net Input Channels

```text
channel 0: normalized DNB brightness crop
channel 1: parent PH mask
channel 2: child PH union mask
channel 3: PH seed map
channel 4: PH persistence score map
channel 5: PH soft attention map
```

## Target Policy

PH no longer censors the GT target.

```text
crop 내부 모든 GT point 사용
-> Gaussian density kernel 생성
-> crop boundary 밖으로 나간 kernel만 제거
-> 남은 kernel을 sum-preserving renormalize
-> target density에 누적
```

Default target flags:

```text
require_source_in_roi = false
renormalize_after_roi_mask = false
```

This keeps GT points that are inside the parent crop even when strict PH masks miss them.

## Loss Policy

The active loss is a structured density loss:

```text
L = 0.45 * L_pixel
  + 0.22 * L_partition_count
  + 0.08 * L_batch_count
  + 0.20 * L_local_count
  + 0.05 * L_background
```

`L_pixel` is a PH-soft-attention weighted Huber density loss over the valid
crop:

```text
loss_weight = valid_crop * (0.25 + 0.75 * ph_soft_attention)
```

This preserves supervision outside strict PH masks while still telling the
U-Net where PH thinks the important light structure is.

`L_partition_count` enforces the physical integral constraint on each owned
partition crop:

```text
sum(pred_density over owned valid pixels) ~= sum(target_density over owned valid pixels)
```

Because active full-scene partitioning gives every valid sea pixel exactly one
owner, this is a partition-level count-conservation term, not just an arbitrary
regularizer. The default uses relative Huber error normalized by
`target_count + 1`, so dense and sparse partitions both contribute without
exploding on empty partitions.

`L_batch_count` applies the same integral constraint after summing all owned
pixels in the mini-batch:

```text
sum_batch(pred_density over owned valid pixels) ~= sum_batch(target_density over owned valid pixels)
```

This is a count-calibration term. It directly encodes the density-map definition
that the heatmap integral should recover the number of ships, and it reduces
systematic over-counting or under-counting even when per-pixel density shape is
reasonable.

`L_local_count` applies the same integral-count idea over multiscale local
windows:

```text
windows = 16, 32, 64 pixels
stride = window / 2
```

This term is the logical bridge between pixelwise density fitting and pure total
count fitting: it discourages solutions that match the partition total count but
spread the density into a blurry or physically misplaced heatmap.

`L_background` penalizes predicted density on valid ocean pixels whose target
density is effectively zero. This keeps fallback-grid background partitions from
becoming unconstrained false-positive areas.

## Count Output Interpretation

The model output is a continuous non-negative density map, not a softmax over
integer count classes.

```text
model(input) -> density_pred[h, w] >= 0
count_pred(partition) = sum(density_pred * valid_owner_mask)
```

The true ship count is integer-valued, but the training signal is handled as a
continuous expected count. This is consistent with density-map crowd/object
counting literature: the integer count is recovered by integrating/summing the
predicted density, while losses remain differentiable with respect to continuous
pixel values. Rounding is therefore an evaluation/reporting operation, not a
training operation.

Do not apply softmax over classes such as `0 ships`, `1 ship`, `2 ships` for
the active heatmap path. A softmax count classifier would make each crop choose
a discrete count bin and would discard the spatial density structure needed for
pixel-level heatmap inference. It may be useful only as an auxiliary head in a
future multi-task model.

Bayesian Loss uses probability, but not as a softmax distribution over integer
count classes. It constructs a posterior contribution probability from each
pixel to each point annotation, multiplies that probability by the continuous
predicted density, and supervises the expected count at each annotation point to
be one. DM-Count similarly treats predicted output as a real-valued density map:
it compares total mass for count conservation and compares normalized density
distributions with optimal transport.

## Loss References

- Victor Lempitsky and Andrew Zisserman, "Learning To Count Objects in Images",
  NeurIPS 2010. Core idea used here: estimate a density whose integral over an
  image region gives the object count.
  <https://papers.neurips.cc/paper/4043-learning-to-count-objects-in-images>
- Yuhong Li, Xiaofan Zhang, and Deming Chen, "CSRNet: Dilated Convolutional
  Neural Networks for Understanding the Highly Congested Scenes", CVPR 2018.
  Reference baseline for CNN density-map regression and count-by-sum inference.
  <https://arxiv.org/abs/1802.10062>
- Zhiheng Ma, Xing Wei, Xiaopeng Hong, and Yihong Gong, "Bayesian Loss for
  Crowd Count Estimation With Point Supervision", ICCV 2019. Motivation for
  supervising expected count rather than trusting a strict pixelwise Gaussian
  target everywhere.
  <https://openaccess.thecvf.com/content_ICCV_2019/html/Ma_Bayesian_Loss_for_Crowd_Count_Estimation_With_Point_Supervision_ICCV_2019_paper.html>
- Boyu Wang, Huidong Liu, Dimitris Samaras, and Minh Hoai Nguyen,
  "Distribution Matching for Crowd Counting", NeurIPS 2020. Motivation for
  explicitly separating count/mass conservation from spatial distribution
  matching.
  <https://proceedings.neurips.cc/paper/2020/hash/118bd558033a1016fcc82560c65cca5f-Abstract.html>
- Haroon Idrees, Muhmmad Tayyab, Kishan Athrey, Dong Zhang, Somaya Al-Maadeed,
  Nasir Rajpoot, and Mubarak Shah, "Composition Loss for Counting, Density Map
  Estimation and Localization in Dense Crowds", ECCV 2018. Motivation for a
  composed loss that jointly supports counting, density shape, and localization.
  <https://www.ecva.net/papers/eccv_2018/papers_ECCV/html/Haroon_Idrees_Composition_Loss_for_ECCV_2018_paper.php>

## Shared Data Path

```text
SceneRaster + GeoJSON GT
-> raw count map
-> DnbCandidateDetector with drop_nested=false
-> outer PH components selected as parent crops
-> contained PH components assigned as children
-> parent/child/seed/persistence/attention maps generated
-> crop-level sum-preserving Gaussian target from all crop GT
-> MaskedDensityUNet forward/backward with soft weighted loss
```

## Scene Split Policy

Train/validation/test splitting must be done by day group, not by patch.

```text
day_key = A2025DDD
scene_key = A2025DDD_HHMM_021
```

Patch-level splitting is leakage-prone because multiple partitions from the
same full-scene image share the same illumination, scan geometry, clouds, and
AIS distribution. Scene-level random splitting is also weaker than day-level
splitting because scenes from the same day are only minutes apart and can
overlap spatially.

The final intended split is:

```text
train = 250 days
val   = 60 days
test  = 55 days
```

Do not generate or freeze this final split until the full 365-day GeoTIFF and
bbox-complete set is stable.

For pipeline smoke testing, use the current bbox-complete snapshot plus existing
GeoTIFF files only:

```sh
PYTHONPATH=. /Users/jungtaeuk/anaconda3/envs/DNB_AIS/bin/python -m sub_module.build_density_scene_split \
  --output-dir "[3]_DNB_AIS - (STEP 3)/outputs/density_smoke_split_10_3_2" \
  --train-days 10 \
  --val-days 3 \
  --test-days 2 \
  --seed 20260529
```

Then generate visual checks:

```sh
PYTHONPATH=. /Users/jungtaeuk/anaconda3/envs/DNB_AIS/bin/python -m sub_module.visualize_density_split \
  --scene-split-csv "[3]_DNB_AIS - (STEP 3)/outputs/density_smoke_split_10_3_2/scene_split.csv" \
  --output-dir "[3]_DNB_AIS - (STEP 3)/outputs/density_smoke_split_10_3_2/visuals"
```

Smoke split artifacts under `outputs/` are runtime diagnostics, not final
manifests. Keep them out of git unless a small curated metric summary is
explicitly needed.

### MPS Smoke Train

Use `sub_module.run_density_split_smoke_train` to verify that the U-Net training
and inference path works end-to-end on a day-grouped split. For Mac MPS, this
command must be run outside the sandbox because sandboxed processes can fail to
see the Metal device even when the hardware supports it.

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 \
PYTHONPATH=. \
/Users/jungtaeuk/anaconda3/envs/DNB_AIS/bin/python -m sub_module.run_density_split_smoke_train \
  --scene-split-csv "[3]_DNB_AIS - (STEP 3)/outputs/density_smoke_split_10_3_2/scene_split_ph_positive_minimal.csv" \
  --config configs/dnb_density_unet_main.json \
  --output-dir "[3]_DNB_AIS - (STEP 3)/outputs/density_split_smoke_train_mps_ph_positive_minimal" \
  --device mps \
  --epochs 1 \
  --batch-size 2 \
  --max-patches-per-scene 8 \
  --max-patch-height 512 \
  --max-patch-width 512 \
  --preview-patches 8
```

`PYTORCH_ENABLE_MPS_FALLBACK=0` is intentional for this check. Unsupported MPS
ops should fail instead of silently falling back to CPU. Raster IO, KR sea
masking, PH extraction, and partition construction are still CPU/IO
preprocessing steps; the GPU requirement applies to the PyTorch model
forward/backward/inference tensors.

The split-smoke runner refuses CPU execution. `--device auto` resolves to MPS
and raises if `torch.backends.mps.is_available()` is false.

## Smoke Test

From repository root:

```sh
PYTHONPATH=. /Users/jungtaeuk/anaconda3/envs/DNB_AIS/bin/python -m sub_module.run_hierarchical_unet_smoke --steps 2 --max-patches 24
```

Equivalent generic runner:

```sh
PYTHONPATH=. /Users/jungtaeuk/anaconda3/envs/DNB_AIS/bin/python -m sub_module.run_density_smoke --config configs/dnb_density_unet_main.json --model main --steps 2 --max-patches 24
```

From `[3]_DNB_AIS - (STEP 3)`:

```sh
PYTHONPATH=.. /Users/jungtaeuk/anaconda3/envs/DNB_AIS/bin/python -m sub_module.run_hierarchical_unet_smoke --steps 2 --max-patches 24
```

## Visual Preview

Generate PNG previews of the 6 U-Net input channels plus target/loss maps:

```sh
PYTHONPATH=. /Users/jungtaeuk/anaconda3/envs/DNB_AIS/bin/python -m sub_module.run_hierarchical_unet_smoke \
  --steps 1 \
  --max-patches 8 \
  --preview-dir outputs/hierarchical_unet_preview \
  --preview-patches 3
```

The preview images show:

```text
brightness
parent PH mask
child PH union
seed map
persistence map
soft attention
raw GT count
crop-level density target
loss weight
```

## PH Threshold Diagnostic

Run the threshold/coverage sweep:

```sh
PYTHONPATH=. /Users/jungtaeuk/anaconda3/envs/DNB_AIS/bin/python -m sub_module.analyze_ph_threshold_sweep \
  --output-dir outputs/ph_threshold_sweep_median \
  --top-n 24 \
  --preview-patches 3
```

The current active parent PH threshold is median-referenced:

```text
detection_threshold = 1.0
analysis_threshold = 0.25
threshold_reference = median
area_limit = 0
remove_edge = false
```

`remove_edge=false` is intentional for the active full-scene feature path: edge-touching PH components are kept as model information instead of being deleted.

## Full-Scene Partitioning Direction

The first split should be computed on the full scene, not on already-cropped tiles.

```text
full DNB scene
-> full-scene PH hierarchy with remove_edge=false
-> sea-domain partition that covers the full valid ocean area
-> each partition becomes a rectangular model crop with padding/halo
-> U-Net consumes brightness + PH hierarchy channels
-> overlapping predictions are merged back to the full-scene density map
```

The partitioning step should use PH results as adaptive anchors, but it must not leave ocean pixels uncovered. This means PH components are region-proposal seeds, not the full partition by themselves. Background ocean between PH components still needs assignment to a partition, for example by bounding-box expansion, Voronoi/watershed-style assignment from PH seeds, or a fixed fallback grid where PH coverage is weak.

Implemented partitioning path:

- `sub_module.dnb_scene_partition.build_scene_partitions` assigns every valid sea pixel exactly once.
- PH anchor partitions claim padded PH parent boxes first.
- Fallback grid partitions claim the remaining valid sea pixels.
- Each partition keeps halo/context pixels for the U-Net input, but the patch `valid_mask` marks only owned output/loss pixels.
- Active KR sea masking uses `all_touched=true` so boundary sea pixels do not drop GT points.

Current smoke result on `TEST_5_A2025001_1754_021_batch_1.tif`:

```text
partition_count = 21
ph_anchor_count = 18
fallback_grid_count = 3
valid_sea_pixels = 24770
missed_valid_pixels = 0
overlap_valid_pixels = 0
raw_count_sum = 160.0
target_density_sum = 160.0
```

For true full-scene input, PH anchor extraction is downsampled before being
projected back to the full-resolution grid. This keeps target/loss/partition
ownership on the original pixels while avoiding full-resolution PH runtime.

Current full-scene smoke result on `A2025001_1754_021.tif` with
`ph_downsample.factor=4`, `ph_downsample.reducer=max`:

```text
masked_scene_shape = 2495 x 3638
ph_downsampled_shape = 624 x 910
runtime = 43.06s
partition_count = 687
ph_anchor_count = 376
fallback_grid_count = 311
valid_sea_pixels = 3640885
missed_valid_pixels = 0
overlap_valid_pixels = 0
gt_count_sum_in_scene_crop = 5380.0
target_density_sum_inside_kr_sea_mask = 3467.0
```

Locked active baseline as of 2026-05-29:

- Use `configs/dnb_density_unet_main.json` for the main U-Net density path.
- Keep KR sea mask as `eez + eez_12nm`, `crop_to_bounds=true`, `all_touched=true`.
- Keep PH anchor extraction downsampled with `factor=4`, `reducer=max`.
- Keep exact-cover partitioning as PH-anchor-first plus fallback grid.
- Keep fallback tile size `96`, halo `16`, anchor padding `16`.
- Do not change these defaults unless a new full-scene diagnostic improves both coverage and qualitative partition structure.

## Archival Baselines

`MaskedDilatedDensityNet` and the old GAT direction remain in the repository as references. The active implementation path is U-Net only until the PH hierarchy coverage and target policy are stable.
