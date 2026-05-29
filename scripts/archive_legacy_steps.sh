#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/archive_legacy_steps.sh [--force] [--target-dir PATH]

Creates compressed archives for legacy [1] and [2] step folders without deleting
sources. By default the script refuses to run while active [A]/[D]/[E] jobs are
running, because archiving ~100GB can contend for disk IO.
USAGE
}

force=0
target_dir="/Volumes/SAMSUNG/DNB_AIS_archive/legacy_steps"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --force)
      force=1
      shift
      ;;
    --target-dir)
      target_dir="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$script_dir/.." && pwd)"
archive_date="${DNB_AIS_ARCHIVE_DATE:-$(date +%Y%m%d_%H%M%S)}"
manifest_dir="$root/_archive/manifests"
manifest="$manifest_dir/legacy_archive_${archive_date}.md"
zstd_bin="$(command -v zstd || true)"

if [[ -z "$zstd_bin" ]]; then
  echo "zstd is required but was not found on PATH." >&2
  exit 1
fi

active_jobs="$(ps -ax -o command | grep -E '\[A\]_dnb2geotif|\[D\]_ship_class|\[E\]_bounding_box' | grep -v grep || true)"
if [[ "$force" -ne 1 && -n "$active_jobs" ]]; then
  echo "Refusing to archive while active DNB jobs are running. Re-run with --force only if IO contention is acceptable." >&2
  echo "$active_jobs" >&2
  exit 3
fi

mkdir -p "$target_dir" "$manifest_dir"
renice -n 19 -p $$ >/dev/null 2>&1 || true

sha256_file() {
  shasum -a 256 "$1" | awk '{print $1}'
}

size_bytes() {
  stat -f %z "$1" 2>/dev/null || stat -c %s "$1"
}

write_header() {
  {
    echo "# Legacy Archive Manifest - ${archive_date}"
    echo
    echo "- repo_root: \`$root\`"
    echo "- target_dir: \`$target_dir\`"
    echo "- compression: \`zstd -T1 -1\`"
    echo "- source_delete_policy: sources are not deleted by this script"
    echo "- active_job_check: $([[ "$force" -eq 1 ]] && echo forced || echo passed)"
    echo
    echo "## Archives"
    echo
  } > "$manifest"
}

archive_one() {
  local label="$1"
  local folder="$2"
  local source_path="$root/$folder"
  local output="$target_dir/${label}_legacy_${archive_date}.tar.zst"
  local list_head="$output.list_head.txt"
  local checksum
  local bytes

  if [[ ! -d "$source_path" ]]; then
    echo "Skipping missing source: $source_path" >&2
    return 0
  fi

  if [[ -e "$output" ]]; then
    echo "Archive already exists: $output" >&2
    return 1
  fi

  echo "[archive] $folder -> $output"
  tar -C "$root" -cf - "$folder" | "$zstd_bin" -T1 -1 -o "$output"
  checksum="$(sha256_file "$output")"
  bytes="$(size_bytes "$output")"
  echo "$checksum  $output" > "$output.sha256"

  set +o pipefail
  "$zstd_bin" -dc "$output" | tar -tf - | head -n 200 > "$list_head"
  set -o pipefail

  {
    echo "### $label"
    echo
    echo "- source: \`$source_path\`"
    echo "- archive: \`$output\`"
    echo "- size_bytes: $bytes"
    echo "- sha256: \`$checksum\`"
    echo "- listing_head: \`$list_head\`"
    echo
  } >> "$manifest"
}

write_header
archive_one "step1" "[1]_DNB_AIS - (STEP 1)"
archive_one "step2" "[2]_DNB_AIS - (STEP 2)"

echo "[done] manifest: $manifest"
