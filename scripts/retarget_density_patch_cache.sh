#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${DNB_AIS_PYTHON:-/Users/jungtaeuk/anaconda3/envs/DNB_AIS/bin/python}"

: "${SOURCE_CACHE_DIR:?Set SOURCE_CACHE_DIR to an existing density patch cache directory.}"
: "${OUTPUT_CACHE_DIR:?Set OUTPUT_CACHE_DIR for the retargeted cache.}"
: "${CONFIG_PATH:?Set CONFIG_PATH to the target-kernel config.}"

export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplcache}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/dnb_pycache}"

"${PYTHON_BIN}" -m sub_module.retarget_density_patch_cache \
  --source-cache-dir "${SOURCE_CACHE_DIR}" \
  --output-cache-dir "${OUTPUT_CACHE_DIR}" \
  --config "${CONFIG_PATH}" \
  --splits "${SPLITS:-train,val,test}"
