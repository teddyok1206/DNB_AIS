#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${DNB_AIS_PYTHON:-/Users/jungtaeuk/anaconda3/envs/DNB_AIS/bin/python}"

ACTION="${ACTION:-help}"
SPLIT_DIR="${SPLIT_DIR:-${ROOT}/outputs/dnb_density/splits/ox_spatial_25pct_63_15_14_20260609_011421}"
BASE_CONFIG="${BASE_CONFIG:-${ROOT}/configs/dnb_density_unet_occupancy_spatial_ow08_sw02.json}"
VARIANT_CONFIG="${VARIANT_CONFIG:-${ROOT}/configs/dnb_density_unet_occupancy_spatial_ow08_sw02_sigma1p0_r4.json}"
SOURCE_CACHE_DIR="${SOURCE_CACHE_DIR:-/Volumes/SAMSUNG/dnb_density_training_patch_cache/ox_spatial_25pct_48p_20260609}"
VARIANT_CACHE_DIR="${VARIANT_CACHE_DIR:-/Volumes/SAMSUNG/dnb_density_training_patch_cache/ox_spatial_25pct_48p_sigma1p0_r4_20260609}"
BASELINE_RUN_DIR="${BASELINE_RUN_DIR:-${ROOT}/outputs/dnb_density/runs/ox_spatial_25pct_63_15_14_20260609_011421}"
RUN_TAG="${RUN_TAG:-ox_spatial_ow08_sw02_sigma1p0_r4_25pct_cached_$(date +%Y%m%d_%H%M%S)}"
VARIANT_RUN_DIR="${VARIANT_RUN_DIR:-${ROOT}/outputs/dnb_density/runs/${RUN_TAG}}"
COMPARE_DIR="${COMPARE_DIR:-${ROOT}/outputs/dnb_density/comparisons/kernel_variant_25pct_$(date +%Y%m%d_%H%M%S)}"

export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-0}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplcache}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/dnb_pycache}"

common_train_args=(
  --scene-split-csv "${SPLIT_DIR}/scene_split.csv"
  --device mps
  --batch-size "${BATCH_SIZE:-4}"
  --max-scenes-per-split "${MAX_SCENES_PER_SPLIT:-0}"
  --max-patches-per-scene "${MAX_PATCHES_PER_SCENE:-48}"
  --max-ph-patches-per-scene "${MAX_PH_PATCHES_PER_SCENE:-36}"
  --max-fallback-patches-per-scene "${MAX_FALLBACK_PATCHES_PER_SCENE:-12}"
  --positive-patches-per-scene "${POSITIVE_PATCHES_PER_SCENE:-24}"
  --negative-patches-per-scene "${NEGATIVE_PATCHES_PER_SCENE:-24}"
  --selection-seed "${SELECTION_SEED:-20260609}"
  --max-patch-height "${MAX_PATCH_HEIGHT:-256}"
  --max-patch-width "${MAX_PATCH_WIDTH:-256}"
  --preview-patches "${PREVIEW_PATCHES:-32}"
  --num-workers "${NUM_WORKERS:-0}"
)

usage() {
  cat <<EOF
Usage:
  ACTION=build-cache   bash scripts/run_density_kernel_variant_25pct.sh
  ACTION=retarget      bash scripts/run_density_kernel_variant_25pct.sh
  ACTION=train-variant bash scripts/run_density_kernel_variant_25pct.sh
  ACTION=compare VARIANT_RUN_DIR=/path/to/run bash scripts/run_density_kernel_variant_25pct.sh
  ACTION=all           bash scripts/run_density_kernel_variant_25pct.sh

Defaults:
  SPLIT_DIR=${SPLIT_DIR}
  SOURCE_CACHE_DIR=${SOURCE_CACHE_DIR}
  VARIANT_CACHE_DIR=${VARIANT_CACHE_DIR}
  BASELINE_RUN_DIR=${BASELINE_RUN_DIR}
  VARIANT_RUN_DIR=${VARIANT_RUN_DIR}
EOF
}

build_cache() {
  local cache_run_dir="${CACHE_RUN_DIR:-${ROOT}/outputs/dnb_density/runs/cache_build_ox_spatial_25pct_48p_$(date +%Y%m%d_%H%M%S)}"
  mkdir -p "${cache_run_dir}"
  echo "[kernel-variant] build source cache"
  echo "  run_dir=${cache_run_dir}"
  echo "  cache_dir=${SOURCE_CACHE_DIR}"
  "${PYTHON_BIN}" -m sub_module.run_density_split_smoke_train \
    "${common_train_args[@]}" \
    --config "${BASE_CONFIG}" \
    --output-dir "${cache_run_dir}" \
    --epochs 0 \
    --patch-cache-dir "${SOURCE_CACHE_DIR}" \
    --patch-cache-mode readwrite \
    --cache-only
}

retarget_cache() {
  echo "[kernel-variant] retarget source cache to sigma=1.0/radius=4"
  echo "  source=${SOURCE_CACHE_DIR}"
  echo "  output=${VARIANT_CACHE_DIR}"
  SOURCE_CACHE_DIR="${SOURCE_CACHE_DIR}" \
  OUTPUT_CACHE_DIR="${VARIANT_CACHE_DIR}" \
  CONFIG_PATH="${VARIANT_CONFIG}" \
  SPLITS="${SPLITS:-train,val,test}" \
  bash "${ROOT}/scripts/retarget_density_patch_cache.sh"
}

train_variant() {
  mkdir -p "${VARIANT_RUN_DIR}"
  echo "[kernel-variant] train sigma=1.0/radius=4 variant"
  echo "  run_dir=${VARIANT_RUN_DIR}"
  echo "  cache_dir=${VARIANT_CACHE_DIR}"
  "${PYTHON_BIN}" -m sub_module.run_density_split_smoke_train \
    "${common_train_args[@]}" \
    --config "${VARIANT_CONFIG}" \
    --output-dir "${VARIANT_RUN_DIR}" \
    --epochs "${EPOCHS:-12}" \
    --patch-cache-dir "${VARIANT_CACHE_DIR}" \
    --patch-cache-mode read \
    --save-checkpoint
}

compare_runs() {
  mkdir -p "${COMPARE_DIR}"
  echo "[kernel-variant] compare run summaries"
  echo "  output=${COMPARE_DIR}"
  "${PYTHON_BIN}" -m sub_module.compare_density_kernel_runs \
    --run "baseline_sigma1p5_r5=${BASELINE_RUN_DIR}" \
    --run "variant_sigma1p0_r4=${VARIANT_RUN_DIR}" \
    --output-md "${COMPARE_DIR}/kernel_variant_comparison.md" \
    --output-json "${COMPARE_DIR}/kernel_variant_comparison.json"
}

case "${ACTION}" in
  build-cache)
    build_cache
    ;;
  retarget)
    retarget_cache
    ;;
  train-variant)
    train_variant
    ;;
  compare)
    compare_runs
    ;;
  all)
    build_cache
    retarget_cache
    train_variant
    compare_runs
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    usage
    echo "Unknown ACTION=${ACTION}" >&2
    exit 2
    ;;
esac
