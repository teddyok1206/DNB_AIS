# PH Threshold Sweep Evaluation

## Scene

```text
scene_key: A2025001_1754_021
scene_gt_sum: 160.0
top_n_patches: 24
parent_min_nodes: 32
child_min_nodes: 4
```

## Key Finding

The old zero-referenced threshold range around `0.1-1.0` barely changes the output because the DNB image is already normalized to roughly:

```text
min: 0.1667
median: 0.4140
p90: 0.7895
p99: 0.9834
robust_sigma: 0.1765
zero_ref floor at threshold 1.0: 0.1765
```

So `threshold_reference=zero` with threshold near `1.0` is too low to act as a meaningful cutoff. Median-referenced thresholds change the PH proposal behavior much more clearly.

## Best Single-Scene Candidates

| case | top24 crop GT recall | top24 crop area ratio | all crop GT recall | patch count | child count | note |
|---|---:|---:|---:|---:|---:|---|
| `median_single_d200_a050` | 0.7875 | 0.7513 | 0.9062 | 27 | 9 | best efficiency among recall >= 0.75 |
| `median_single_d150_a025` | 0.8063 | 0.7884 | 0.9375 | 31 | 13 | best top24 recall; selected active default |
| `median_single_d100_a000` | 0.7937 | 0.7964 | 0.9625 | 35 | 16 | higher all-crop recall, slightly more area |
| `dual_parent_median_d100_a000_child_median_d200_a050` | 0.7937 | 0.7964 | 0.9625 | 35 | 25 | more child structure, same parent coverage |
| `single_d100_a100` | 0.6000 | 0.8687 | 0.9938 | 55 | 33 | old zero-ref default; poor top24 efficiency |

## Padding Check

Reducing padding controls area but loses many GTs for top24 crops.

| padding | case | top24 crop GT recall | top24 crop area ratio | all crop GT recall |
|---:|---|---:|---:|---:|
| 8 | `median_d150_a025` | 0.5000 | 0.5249 | 0.7125 |
| 12 | `median_d150_a025` | 0.6813 | 0.6750 | 0.8313 |
| 16 | `median_single_d150_a025` | 0.8063 | 0.7884 | 0.9375 |

For coverage-first training, `padding_pixels=16` remains the safer starting point.

## Qualitative Outputs

Main comparison grid:

```text
outputs/ph_threshold_sweep_median/threshold_sweep_overlay_grid.png
```

Metrics:

```text
outputs/ph_threshold_sweep_median/threshold_sweep_metrics.csv
outputs/ph_threshold_sweep_median/threshold_sweep_metrics.json
```

Representative patch previews are under each case directory, for example:

```text
outputs/ph_threshold_sweep_median/median_single_d150_a025/patch_previews/
outputs/ph_threshold_sweep_median/dual_parent_median_d100_a000_child_median_d200_a050/patch_previews/
```

## Recommendation

Use the median-referenced single detector as the active default first:

```text
threshold_reference = median
detection_threshold = 1.0
analysis_threshold = 0.25
area_limit = 0
remove_edge = false
padding_pixels = 16
```

Reasoning:

- It keeps top24 crop GT recall high at `0.78125` on the test scene.
- It keeps all-crop GT recall at `0.9375` without exploding parent patch count.
- Removing `area_limit` increases child hierarchy detail from `13` to `27` child components.
- Keeping edge components raises available PH/GT information for the full-scene feature path. In the quick check, `remove_edge=false` raised candidate GT mass from `93` to `113` and child components from `27` to `76`.
- It is simpler than the dual-threshold path while the rest of the training pipeline is still being stabilized.

The dual-threshold path is still useful later if we want sharper child/seed maps without changing parent crop coverage.
