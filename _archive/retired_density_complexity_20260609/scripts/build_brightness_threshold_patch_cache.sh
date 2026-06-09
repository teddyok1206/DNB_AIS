#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${DNB_AIS_PYTHON:-/Users/jungtaeuk/anaconda3/envs/DNB_AIS/bin/python}"

: "${SCENE_SPLIT_CSV:?Set SCENE_SPLIT_CSV to the model scene_split.csv.}"

CACHE_TAG="${CACHE_TAG:-brightness_threshold_patch_cache_$(date +%Y%m%d_%H%M%S)}"
CACHE_DIR="${CACHE_DIR:-/Volumes/SAMSUNG/dnb_density_patch_cache/${CACHE_TAG}}"

export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplcache}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/dnb_pycache}"

mkdir -p "${CACHE_DIR}"

"${PYTHON_BIN}" -m sub_module.build_brightness_threshold_patch_cache \
  --scene-split-csv "${SCENE_SPLIT_CSV}" \
  --config "${ROOT}/configs/dnb_density_unet_occupancy_spatial.json" \
  --cache-dir "${CACHE_DIR}" \
  --splits "${SPLITS:-val,test}" \
  --limit-scenes-per-split "${LIMIT_SCENES_PER_SPLIT:-0}" \
  --seed "${SEED:-20260529}" \
  --max-patches-per-scene "${MAX_PATCHES_PER_SCENE:-48}" \
  --max-ph-patches-per-scene "${MAX_PH_PATCHES_PER_SCENE:-36}" \
  --max-fallback-patches-per-scene "${MAX_FALLBACK_PATCHES_PER_SCENE:-12}" \
  --positive-patches-per-scene "${POSITIVE_PATCHES_PER_SCENE:-24}" \
  --negative-patches-per-scene "${NEGATIVE_PATCHES_PER_SCENE:-24}" \
  --selection-seed "${SELECTION_SEED:-20260609}" \
  --max-patch-height "${MAX_PATCH_HEIGHT:-256}" \
  --max-patch-width "${MAX_PATCH_WIDTH:-256}" \
  --image-dtype "${IMAGE_DTYPE:-float16}" \
  --target-dtype "${TARGET_DTYPE:-float32}"
