#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${DNB_AIS_PYTHON:-/Users/jungtaeuk/anaconda3/envs/DNB_AIS/bin/python}"
RUN_TAG="${RUN_TAG:-count_spatial_lifetime_patchmix64_$(date +%Y%m%d_%H%M%S)}"

export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-0}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplcache}"

SPLIT_DIR="${SPLIT_DIR:-${ROOT}/outputs/dnb_density/splits/density_smoke_split_10_3_2}"
RUN_DIR="${RUN_DIR:-${ROOT}/outputs/dnb_density/runs/${RUN_TAG}}"

mkdir -p "${SPLIT_DIR}" "${RUN_DIR}"

if [[ ! -f "${SPLIT_DIR}/scene_split.csv" ]]; then
  "${PYTHON_BIN}" -m sub_module.build_density_scene_split \
    --output-dir "${SPLIT_DIR}" \
    --train-days "${TRAIN_DAYS:-10}" \
    --val-days "${VAL_DAYS:-3}" \
    --test-days "${TEST_DAYS:-2}"
fi

"${PYTHON_BIN}" -m sub_module.run_density_split_smoke_train \
  --scene-split-csv "${SPLIT_DIR}/scene_split.csv" \
  --config "${ROOT}/configs/dnb_density_unet_count_spatial_lifetime.json" \
  --output-dir "${RUN_DIR}" \
  --device mps \
  --epochs "${EPOCHS:-20}" \
  --batch-size "${BATCH_SIZE:-2}" \
  --max-scenes-per-split "${MAX_SCENES_PER_SPLIT:-30}" \
  --max-patches-per-scene "${MAX_PATCHES_PER_SCENE:-64}" \
  --max-ph-patches-per-scene "${MAX_PH_PATCHES_PER_SCENE:-48}" \
  --max-fallback-patches-per-scene "${MAX_FALLBACK_PATCHES_PER_SCENE:-16}" \
  --max-patch-height "${MAX_PATCH_HEIGHT:-512}" \
  --max-patch-width "${MAX_PATCH_WIDTH:-512}" \
  --preview-patches "${PREVIEW_PATCHES:-24}" \
  --num-workers "${NUM_WORKERS:-0}" \
  --save-checkpoint
