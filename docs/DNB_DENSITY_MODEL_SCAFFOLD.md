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

The loss uses the valid crop, with stronger weight near PH hierarchy structure:

```text
loss_weight = valid_crop * (0.25 + 0.75 * ph_soft_attention)
```

This preserves supervision outside strict PH masks while still telling the U-Net where PH thinks the important light structure is.

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

## Archival Baselines

`MaskedDilatedDensityNet` and the old GAT direction remain in the repository as references. The active implementation path is U-Net only until the PH hierarchy coverage and target policy are stable.
