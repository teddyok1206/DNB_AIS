# DNB Density Output Workspace - 2026-06-02

## Decision

Use the repository-level output directory for density-model runtime artifacts:

```text
outputs/dnb_density/
```

Keep `[3]_DNB_AIS - (STEP 3)` as the development/data-preprocessing workspace.

The old path below is now legacy for already generated pilot outputs:

```text
[3]_DNB_AIS - (STEP 3)/outputs/
```

Do not move old pilot outputs automatically, because the locked good-baseline note records paths to those historical preview PNGs.

## Standard Layout

```text
outputs/dnb_density/
  splits/
    density_smoke_split_10_3_2/
      scene_split.csv
      day_split.csv
      split_summary.json
      visuals/
  runs/
    <run_tag>/
      run_summary.json
      scene_metrics.csv
      filtered_scene_split.csv
      inference_previews/
      checkpoints/
      run_git_dirty.patch
  preprocessed_scene_masks/
    density/
```

The `outputs/` tree is ignored by git. Curated results should be summarized in `docs/`, not committed as bulk runtime artifacts.

## Path Controls

Default root:

```text
outputs/dnb_density/
```

Optional environment overrides:

```text
DNB_AIS_OUTPUT_ROOT=/path/to/outputs
DNB_DENSITY_OUTPUT_ROOT=/path/to/dnb_density_outputs
```

## Updated Scripts

These now default to the repo-level density output root:

```text
python -m sub_module.build_density_scene_split
python -m sub_module.visualize_density_split
python -m sub_module.run_density_split_smoke_train
bash scripts/run_density_count_spatial_scaled_patchmix.sh
```

Explicit `--output-dir` and `--scene-split-csv` still work, so legacy outputs can still be inspected.

## Next Scaled Experiment

Run:

```sh
bash scripts/run_density_count_spatial_scaled_patchmix.sh
```

Default experiment settings:

```text
model_config=configs/dnb_density_unet_count_spatial.json
device=mps
epochs=20
batch_size=2
max_scenes_per_split=30
max_patches_per_scene=64
max_ph_patches_per_scene=48
max_fallback_patches_per_scene=16
max_patch_height=512
max_patch_width=512
preview_patches=24
save_checkpoint=true
```

This is intentionally larger than the good-baseline pilot:

```text
previous pilot: 6 train scenes, 16 patches/scene, all selected patches were PH anchors
next scaled run: up to 30 scenes/split, 64 patches/scene, forced PH+fallback mixture
```

The fallback patches are included to train negative/background behavior instead of only learning from high-signal PH anchor regions.

## Reproducibility

`run_density_split_smoke_train` now records:

```text
run_summary.json
scene_metrics.csv
filtered_scene_split.csv
inference_previews/
checkpoints/checkpoint_last.pt, when --save-checkpoint is used
run_git_dirty.patch, when the git worktree is dirty
```

The current worktree often contains unrelated local changes. The dirty patch is saved into each run directory so model artifacts can still be traced to the exact code state used at runtime.

## Legacy Baseline

The current best qualitative baseline remains:

```text
git tag: density-count-spatial-good-20260602
note: [3]_DNB_AIS - (STEP 3)/density_count_spatial_good_baseline_20260602.md
```

Do not delete the old `[3]/outputs` pilot directory until the baseline preview PNGs are regenerated under the new layout or no longer needed.
