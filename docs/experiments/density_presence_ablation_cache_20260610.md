# Presence Probability Cache Ablation - 2026-06-10

## Purpose

This smoke ablation checks four low-code-change training variants against the same cached 5-scene patch split. The goal is to test whether the current pixel probability U-Net can produce a better ship-presence probability map than raw DNB brightness ranking.

This is a fast direction-finding run, not a final full-scale result.

## Shared Setup

- Run root: `outputs/dnb_density/runs/presence_ablation_cache_20260610_033531`
- Cache reused: `outputs/dnb_density/patch_caches/probability_field_posweight_probe_5scene_20260610_024330`
- Train/val/test scenes: 5 / 5 / 4 after filtering
- Epochs per condition: 6
- Model: `PixelProbabilityUNet`
- Training target: pixel-level presence probability only
- Primary evaluation: AP and top-k precision of model probability vs raw DNB brightness, using AIS-derived pixel/radius presence targets

## Conditions

| Run | Input channels | Target/loss setting | Config |
| --- | --- | --- | --- |
| 01 | brightness | Gaussian radius target, sigma=4, radius=12, posw=6 | `configs/ablation_presence_brightness_only_gaussian_s4_posw6_20260610.json` |
| 02 | brightness + persistence + seed | Gaussian radius target, sigma=2, radius=6, posw=6 | `configs/ablation_presence_brightness_ph_gaussian_s2_posw6_20260610.json` |
| 03 | brightness | hard exact pixel target, posw=256 | `configs/ablation_presence_brightness_only_hard_posw256_20260610.json` |
| 04 | brightness + persistence + seed | hard exact pixel target, posw=256 | `configs/ablation_presence_brightness_ph_hard_posw256_20260610.json` |

## Primary Test Results

Primary target means the same target definition used by each condition's main evaluation.

| Run | Model AP | Brightness AP | AP ratio | Model Top-1% precision | Brightness Top-1% precision | Top-1% ratio | Model Brier | Calibrated F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 01 brightness Gaussian s4 | 0.1474 | 0.1985 | 0.7427 | 0.4241 | 0.6449 | 0.6576 | 0.0750 | 0.1733 |
| 02 PH Gaussian s2 | 0.0512 | 0.1425 | 0.3591 | 0.1187 | 0.4012 | 0.2959 | 0.0321 | 0.1164 |
| 03 brightness hard | 0.0080 | 0.0271 | 0.2971 | 0.0166 | 0.0332 | 0.5000 | 0.0371 | 0.0195 |
| 04 PH hard | 0.0035 | 0.0271 | 0.1308 | 0.0062 | 0.0332 | 0.1875 | 0.0362 | 0.0131 |

## Radius Presence Cross-Check

These rows re-evaluate the trained model probabilities against common radius-presence targets. Ratios above 1.0 would mean the model beats raw brightness for that metric.

| Run | Sigma 4 AP ratio | Sigma 4 Top-1% ratio | Sigma 8 AP ratio | Sigma 8 Top-1% ratio | Sigma 8 Top-5% ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| 01 brightness Gaussian s4 | 0.7427 | 0.6576 | 0.8853 | 0.6500 | 1.0224 |
| 02 PH Gaussian s2 | 0.5788 | 0.3834 | 0.8396 | 0.5275 | 0.8547 |
| 03 brightness hard | 0.6410 | 0.4799 | 0.8080 | 0.5100 | 0.8679 |
| 04 PH hard | 0.5891 | 0.3778 | 0.7876 | 0.5100 | 0.8773 |

## Interpretation

1. Raw DNB brightness is still the stronger ranker in this cached smoke test. All four model variants are below brightness baseline on AP and Top-1% precision.
2. The least bad condition is brightness-only with the broader Gaussian target (`01`). It gets closest to brightness AP and only slightly beats brightness in the coarse sigma=8 Top-5% precision check.
3. Narrowing the Gaussian target from sigma=4 to sigma=2 hurts substantially in this short run. The target becomes too sparse for the current small-data setup.
4. Hard exact pixel targets are not viable as the main supervision in this setup. The positive rate is about 0.14%, and both hard-target runs collapse to weak rankers.
5. Adding PH channels in these two quick conditions does not help; it makes both the narrowed Gaussian and hard-target variants worse. This does not prove PH is useless, but it says the current PH-channel injection is not enough to beat raw brightness under short cached training.

## Decision

Do not move to hard exact target training. Keep the probability-field framing, but the next useful change should address ranking against brightness directly instead of only changing the target radius.

## Follow-Up Loss Change

The active probability-field config now uses `weighted_smooth_l1_probability_loss` instead of weighted BCE. The target remains a clipped Gaussian AIS-presence field, but the loss is now field regression:

`loss = mean_valid((1 + field_weight_strength * target_probability) * SmoothL1(sigmoid(logits), target_probability))`

This makes the training objective match the project target more directly: the model should emit a continuous 0..1 ship-presence probability field, not solve a binary classification problem.

Existing BCE-trained checkpoints should not be treated as equivalent after this change. A new training run is required for fair model-vs-brightness comparison under the SmoothL1 objective.

Recommended next experiments:

1. Use brightness-only Gaussian sigma=4 as the current minimal reference.
2. Run the weighted SmoothL1 probability-field config and compare against the cached BCE probes.
3. Add a ranking or brightness-lift objective only if SmoothL1 still fails to beat raw brightness on AP and top-k precision.
4. Treat PH as a controlled optional input, not a default, until it shows positive lift over brightness-only.
5. Keep AP and top-k precision against raw brightness as the primary reporting metrics.
