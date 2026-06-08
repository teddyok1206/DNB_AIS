from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .dnb_density_common import DensityPatch


SCHEMA_VERSION = 1


def split_cache_path(cache_dir: Path, split: str) -> Path:
    return cache_dir.expanduser().resolve() / f"{split}_density_patches.pkl"


def save_patch_split_cache(
    cache_dir: Path,
    split: str,
    patches: list[DensityPatch],
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    cache_dir = cache_dir.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = split_cache_path(cache_dir, split)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "density_patch_pickle_cache",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": str(split),
        "patch_count": int(len(patches)),
        "patches": patches,
        "metadata": dict(metadata or {}),
    }
    with tmp_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(path)
    manifest_path = cache_dir / f"{split}_density_patches_manifest.json"
    manifest = {key: value for key, value in payload.items() if key != "patches"}
    manifest["path"] = str(path)
    manifest["bytes"] = int(path.stat().st_size)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_patch_split_cache(cache_dir: Path, split: str) -> tuple[list[DensityPatch], dict[str, Any]]:
    path = split_cache_path(cache_dir, split)
    if not path.exists():
        raise FileNotFoundError(f"Missing density patch cache for split={split}: {path}")
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid density patch cache payload: {path}")
    if int(payload.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError(f"Unsupported density patch cache schema: {payload.get('schema_version')} in {path}")
    patches = payload.get("patches")
    if not isinstance(patches, list):
        raise ValueError(f"Density patch cache does not contain a patch list: {path}")
    return patches, {key: value for key, value in payload.items() if key != "patches"}
