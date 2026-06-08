#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${DNB_AIS_PYTHON:-/Users/jungtaeuk/anaconda3/envs/DNB_AIS/bin/python}"

: "${CACHE_DIR:?Set CACHE_DIR to the patch cache directory.}"

OUTPUT_DIR="${OUTPUT_DIR:-${CACHE_DIR}/threshold_sweeps/$(date +%Y%m%d_%H%M%S)}"

export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplcache}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/dnb_pycache}"

"${PYTHON_BIN}" -m sub_module.sweep_brightness_threshold_patch_cache \
  --cache-dir "${CACHE_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --thresholds "${THRESHOLDS:-0.35,0.45,0.55,0.65,0.75,0.85,0.90,0.95}" \
  --splits "${SPLITS:-val,test}" \
  --select-threshold-split "${SELECT_THRESHOLD_SPLIT:-val}"
