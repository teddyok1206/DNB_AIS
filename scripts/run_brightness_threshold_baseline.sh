#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${DNB_AIS_PYTHON:-/Users/jungtaeuk/anaconda3/envs/DNB_AIS/bin/python}"

: "${SCENE_SPLIT_CSV:?Set SCENE_SPLIT_CSV to the same scene_split.csv used by the model run.}"

RUN_TAG="${RUN_TAG:-brightness_threshold_baseline_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/dnb_density/baselines/${RUN_TAG}}"

export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplcache}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/dnb_pycache}"

mkdir -p "${OUTPUT_DIR}"

"${PYTHON_BIN}" -m sub_module.evaluate_brightness_threshold_baseline \
  --scene-split-csv "${SCENE_SPLIT_CSV}" \
  --config "${ROOT}/configs/dnb_density_unet_occupancy_spatial.json" \
  --output-dir "${OUTPUT_DIR}" \
  --thresholds "${THRESHOLDS:-0.85,0.90,0.95}" \
  --splits "${SPLITS:-train,val,test}" \
  --select-threshold-split "${SELECT_THRESHOLD_SPLIT:-val}" \
  --limit-scenes-per-split "${LIMIT_SCENES_PER_SPLIT:-0}" \
  --seed "${SEED:-20260529}" \
  --max-patches-per-scene "${MAX_PATCHES_PER_SCENE:-48}" \
  --max-ph-patches-per-scene "${MAX_PH_PATCHES_PER_SCENE:-36}" \
  --max-fallback-patches-per-scene "${MAX_FALLBACK_PATCHES_PER_SCENE:-12}" \
  --positive-patches-per-scene "${POSITIVE_PATCHES_PER_SCENE:-24}" \
  --negative-patches-per-scene "${NEGATIVE_PATCHES_PER_SCENE:-24}" \
  --selection-seed "${SELECTION_SEED:-20260609}" \
  --max-patch-height "${MAX_PATCH_HEIGHT:-256}" \
  --max-patch-width "${MAX_PATCH_WIDTH:-256}"
