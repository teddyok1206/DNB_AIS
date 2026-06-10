# SmoothL1 Probability Field Runs - 2026-06-10

## Purpose

These runs test the current active DNB/AIS density pipeline after changing the objective from binary presence classification to probability-field regression.

The project claim being tested is narrow:

- Raw DNB brightness should be replaced by a learned 0..1 ship-presence probability map.
- A high model probability pixel should rank AIS-derived ship presence better than a high raw DNB brightness pixel.
- Count regression is not the current target. Dense ship count and saturation effects remain future work.

## Active Setup

- Model: `PixelProbabilityUNet`
- Input channels: `brightness`, `ph_persistence_map`, `ph_seed_map`
- Target: AIS seed pixels converted to a clipped Gaussian probability field
- Target radius config: `sigma=4 px`, `radius=12 px`, target threshold for presence metrics `0.25`
- Loss: weighted SmoothL1 field regression
- Loss form: `mean_valid((1 + 8 * target_probability) * SmoothL1(sigmoid(logits), target_probability, beta=0.1))`
- Config: `configs/dnb_density_unet_probability_field_recursive_ph_20260610.json`
- Code commit used by runs: `a0da5b457fb1367358304c046f5ee2ae47f71d9f`

## Runs

| Run | Split size limit | Epochs | Cache | Run directory |
| --- | ---: | ---: | --- | --- |
| Smoke | 5 scenes/split | 6 | `outputs/dnb_density/patch_caches/probability_field_posweight_probe_5scene_20260610_024330` | `outputs/dnb_density/runs/smoothl1_probability_smoke_20260610_110554` |
| Expanded | 20 scenes/split | 12 | `outputs/dnb_density/patch_caches/smoothl1_probability_20scene_20260610_111141` | `outputs/dnb_density/runs/smoothl1_probability_20scene_e12_20260610_111141` |

The expanded run created a new cache and did not delete the earlier probe cache.

Expanded cache sizes:

| Split | Input scenes | Kept scenes | Patches | Selected target sum |
| --- | ---: | ---: | ---: | ---: |
| Train | 20 | 13 | 416 | 1187.0 |
| Val | 20 | 14 | 448 | 1100.0 |
| Test | 20 | 15 | 473 | 1053.0 |

## Primary Test Results

Primary target: `target_probability >= 0.25` inside valid sea mask.

| Run | Model AP | Brightness AP | AP ratio | Model Top-1% | Brightness Top-1% | Top-1% ratio | Model Top-10% | Brightness Top-10% | Top-10% ratio | Calibrated F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Smoke | 0.1231 | 0.1985 | 0.6199 | 0.2566 | 0.6449 | 0.3979 | 0.1633 | 0.1534 | 1.0642 | 0.1961 |
| Expanded | 0.1237 | 0.0854 | 1.4482 | 0.3403 | 0.2130 | 1.5976 | 0.1331 | 0.0896 | 1.4857 | 0.1701 |

## Radius Presence Cross-Check

This evaluates the same model probability against a broader AIS radius target with `sigma=8 px`, then compares rank quality against raw DNB brightness.

| Run | Model AP | Brightness AP | AP ratio | Top-1% ratio | Top-5% ratio | Top-10% ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Smoke | 0.2463 | 0.2922 | 0.8429 | 0.5136 | 0.9100 | 1.0934 |
| Expanded | 0.2539 | 0.2038 | 1.2463 | 1.1688 | 1.5206 | 1.3821 |

## Reliability Check

The expanded run's reliability bins are not calibrated as true probabilities yet, but the ordering is meaningful. Empirical presence rate generally rises as model score increases:

| Score bin | Empirical presence rate |
| --- | ---: |
| 0.0 to 0.1 | 0.0559 |
| 0.1 to 0.2 | 0.0642 |
| 0.2 to 0.3 | 0.0895 |
| 0.3 to 0.4 | 0.1239 |
| 0.4 to 0.5 | 0.1608 |
| 0.5 to 0.6 | 0.1987 |
| 0.6 to 0.7 | 0.2250 |
| 0.7 to 0.8 | 0.3191 |
| 0.8 to 0.9 | 0.4357 |
| 0.9 to 1.0 | 0.4173 |

The final bin has only 139 pixels, so the slight drop from the 0.8 to 0.9 bin should not be overinterpreted.

## Visual Inspection

Analysis previews were generated under:

```text
outputs/dnb_density/runs/smoothl1_probability_20scene_e12_20260610_111141/analysis_previews
```

Most useful example:

```text
analysis_preview_03_idx118_ph_child_824.png
```

This patch has `raw_count_sum=60`, `valid_pixels=798`, `target_mean_valid=0.5132`, `pred_mean_valid=0.6191`, and `pred_p95_valid=0.7702`.

Interpretation:

1. Raw DNB brightness is saturated and blob-like, so it does not isolate the AIS ship chain cleanly.
2. The AIS Gaussian target forms a ship-presence ridge along the observed AIS points.
3. The learned model probability produces a broad but structured probability field that follows the main occupied region rather than acting as a pure copy of raw brightness.
4. The `-Laplacian(prob)` and `grad magnitude(prob)` panels show usable curvature and edge structure, but this is not yet a precise detector.

A weaker example is `analysis_preview_07_idx362_ph_anchor_1270.png`: the model probability is low in absolute magnitude, but still places elevated response near the AIS/target clusters.

## Interpretation

The 5-scene smoke result was not enough. It was below raw brightness on AP and most top-k metrics, although it showed some weak reliability structure.

The 20-scene expanded run is meaningfully better:

1. It beats raw DNB brightness on primary AP by `1.45x`.
2. It beats raw DNB brightness on primary Top-0.5%, Top-1%, Top-5%, and Top-10% precision.
3. It also beats raw DNB brightness on the broader `sigma=8` radius-presence AP and top-k metrics.
4. It still loses at the ultra-selective Top-0.1% precision. This suggests raw brightness remains strong for the brightest saturated pixels, while the model improves broader ranking over the high-probability field.
5. The learned field is not yet calibrated as an exact probability and should not be presented as a ship-count estimator.

## Decision

This direction is worth scaling. Weighted SmoothL1 probability-field regression is a better match to the stated project goal than hard exact binary targets or old BCE-style presence training.

The current evidence supports the following report-safe claim:

> On a 20-scene cached split, the PH-assisted U-Net probability field ranks AIS-derived ship-presence pixels better than raw DNB brightness across AP and most top-k precision metrics, while also producing visually interpretable probability curvature in dense/saturated patches.

## Next Experimental Step

Scale the same setup further before adding new modeling ideas:

1. Keep the current probability-field target and SmoothL1 objective.
2. Increase scene coverage using a new cache, preserving all old caches.
3. Run a controlled brightness-only version after the larger PH-assisted run, so PH's contribution can be isolated from the loss/target change.
4. Keep AP, top-k precision lift over raw brightness, reliability bins, and selected visual previews as the main reporting metrics.

## Future Idea: Probability-Gated DNB

Treat the learned probability map as a possible radiance correction gate:

```text
gated_DNB = raw_DNB * probability^gamma
```

This should remain a future/derived-product idea, not the primary metric. The primary claim is still that model probability ranks AIS-derived ship presence better than raw DNB brightness. The gated map may be useful as a ship-presence-weighted DNB visualization or correction filter, especially when raw DNB is bright but not all bright pixels are equally ship-like.

Initial visual probe:

```text
outputs/dnb_density/runs/smoothl1_probability_20scene_e12_20260610_111141/analysis_previews/analysis_preview_03_idx118_ph_child_824_gamma_gated_dnb.png
```

Use `gamma < 1` for a softer correction, `gamma = 1` for direct gating, and `gamma > 1` for aggressive suppression of low-probability bright pixels.
