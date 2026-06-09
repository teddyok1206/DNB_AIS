#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${DNB_AIS_PYTHON:-/Users/jungtaeuk/anaconda3/envs/DNB_AIS/bin/python}"
CONFIG="${CONFIG:-configs/dnb_density_unet_pixel_binary_recursive_ph_hardtarget_20260609.json}"
SCENE_SPLIT_CSV="${SCENE_SPLIT_CSV:-outputs/dnb_density/splits/ox_spatial_25pct_63_15_14_20260609_011421/scene_split.csv}"
RUN_TAG="${RUN_TAG:-pixel_binary_recursive_ph_min3_hardtarget_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/dnb_density/runs/${RUN_TAG}}"
PATCH_CACHE_DIR="${PATCH_CACHE_DIR:-outputs/dnb_density/patch_caches/pixel_binary_recursive_ph_min3_hardtarget_20260609}"
PATCH_CACHE_MODE="${PATCH_CACHE_MODE:-readwrite}"

DEVICE="${DEVICE:-mps}"
SEED="${SEED:-20260609}"
SELECTION_SEED="${SELECTION_SEED:-20260609}"
EPOCHS="${EPOCHS:-18}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-5e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-3e-4}"
MAX_SCENES_PER_SPLIT="${MAX_SCENES_PER_SPLIT:-0}"
MAX_PATCHES_PER_SCENE="${MAX_PATCHES_PER_SCENE:-64}"
MAX_PH_PATCHES_PER_SCENE="${MAX_PH_PATCHES_PER_SCENE:-48}"
MAX_FALLBACK_PATCHES_PER_SCENE="${MAX_FALLBACK_PATCHES_PER_SCENE:-16}"
POSITIVE_PATCHES_PER_SCENE="${POSITIVE_PATCHES_PER_SCENE:-32}"
NEGATIVE_PATCHES_PER_SCENE="${NEGATIVE_PATCHES_PER_SCENE:-32}"
MAX_PATCH_HEIGHT="${MAX_PATCH_HEIGHT:-160}"
MAX_PATCH_WIDTH="${MAX_PATCH_WIDTH:-160}"
PREVIEW_PATCHES="${PREVIEW_PATCHES:-32}"
NUM_WORKERS="${NUM_WORKERS:-0}"
RUN_RADIUS_EVAL="${RUN_RADIUS_EVAL:-1}"
EVAL_CHECKPOINTS="${EVAL_CHECKPOINTS:-best_val_pixel_f1}"
RADIUS_SIGMAS="${RADIUS_SIGMAS:-1,2,4,8}"
RADIUS_TARGET_THRESHOLD="${RADIUS_TARGET_THRESHOLD:-0.25}"
RADIUS_TRUNCATE="${RADIUS_TRUNCATE:-3.0}"

mkdir -p "${OUTPUT_DIR}" "${PATCH_CACHE_DIR}"

check_writable_dir() {
  local dir="$1"
  local probe="${dir}/.write_test_$$"
  if ! ( : > "${probe}" ) 2>/dev/null; then
    printf '[error] directory is not writable: %s\n' "${dir}" >&2
    exit 1
  fi
  rm -f "${probe}"
}

check_writable_dir "${OUTPUT_DIR}"
if [[ "${PATCH_CACHE_MODE}" == "write" || "${PATCH_CACHE_MODE}" == "readwrite" ]]; then
  check_writable_dir "${PATCH_CACHE_DIR}"
fi

printf '[run] output_dir=%s\n' "${OUTPUT_DIR}"
printf '[run] log=%s\n' "${OUTPUT_DIR}/run.log"
printf '[run] config=%s\n' "${CONFIG}"
printf '[run] scene_split=%s\n' "${SCENE_SPLIT_CSV}"

PYTHONUNBUFFERED=1 PYTHONPATH=. PYTORCH_ENABLE_MPS_FALLBACK=0 "${PYTHON_BIN}" \
  -m sub_module.run_density_split_smoke_train \
  --config "${CONFIG}" \
  --scene-split-csv "${SCENE_SPLIT_CSV}" \
  --output-dir "${OUTPUT_DIR}" \
  --device "${DEVICE}" \
  --seed "${SEED}" \
  --selection-seed "${SELECTION_SEED}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --max-scenes-per-split "${MAX_SCENES_PER_SPLIT}" \
  --max-patches-per-scene "${MAX_PATCHES_PER_SCENE}" \
  --max-ph-patches-per-scene "${MAX_PH_PATCHES_PER_SCENE}" \
  --max-fallback-patches-per-scene "${MAX_FALLBACK_PATCHES_PER_SCENE}" \
  --positive-patches-per-scene "${POSITIVE_PATCHES_PER_SCENE}" \
  --negative-patches-per-scene "${NEGATIVE_PATCHES_PER_SCENE}" \
  --max-patch-height "${MAX_PATCH_HEIGHT}" \
  --max-patch-width "${MAX_PATCH_WIDTH}" \
  --preview-patches "${PREVIEW_PATCHES}" \
  --num-workers "${NUM_WORKERS}" \
  --patch-cache-dir "${PATCH_CACHE_DIR}" \
  --patch-cache-mode "${PATCH_CACHE_MODE}" \
  --save-checkpoint \
  2>&1 | tee "${OUTPUT_DIR}/run.log"

if [[ "${RUN_RADIUS_EVAL}" == "1" || "${RUN_RADIUS_EVAL}" == "true" || "${RUN_RADIUS_EVAL}" == "yes" ]]; then
  for checkpoint in ${EVAL_CHECKPOINTS}; do
    printf '[eval] checkpoint=%s radius_sigmas=%s\n' "${checkpoint}" "${RADIUS_SIGMAS}" | tee -a "${OUTPUT_DIR}/run.log"
    PYTHONUNBUFFERED=1 PYTHONPATH=. PYTORCH_ENABLE_MPS_FALLBACK=0 "${PYTHON_BIN}" \
      -m sub_module.evaluate_density_checkpoint \
      --run-dir "${OUTPUT_DIR}" \
      --checkpoint "${checkpoint}" \
      --split test \
      --calibration-split val \
      --device "${DEVICE}" \
      --batch-size "${BATCH_SIZE}" \
      --num-workers "${NUM_WORKERS}" \
      --radius-sigmas "${RADIUS_SIGMAS}" \
      --radius-target-threshold "${RADIUS_TARGET_THRESHOLD}" \
      --radius-truncate "${RADIUS_TRUNCATE}" \
      2>&1 | tee -a "${OUTPUT_DIR}/run.log"
  done
fi
