from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .dnb_density_common import DensityTargetConfig, make_sum_preserving_density_target
from .dnb_density_patch_pickle_cache import load_patch_split_cache, save_patch_split_cache
from .run_density_split_smoke_train import git_metadata, read_json, stable_json_hash, target_config_from_config


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reuse cached PH/partition DensityPatch objects while regenerating only target_density "
            "for a new Gaussian target kernel."
        )
    )
    parser.add_argument("--source-cache-dir", type=Path, required=True)
    parser.add_argument("--output-cache-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--splits", default="train,val,test")
    return parser


def parse_splits(text: str) -> list[str]:
    values = [part.strip() for part in str(text).split(",") if part.strip()]
    valid = {"train", "val", "test"}
    bad = [value for value in values if value not in valid]
    if bad:
        raise ValueError(f"Unsupported split(s): {bad}")
    if not values:
        raise ValueError("At least one split is required")
    return values


def retarget_patch(patch: Any, target_config: DensityTargetConfig) -> float:
    old_sum = float(patch.target_sum)
    patch.target_density = make_sum_preserving_density_target(
        np.asarray(patch.raw_count, dtype=np.float32),
        np.asarray(patch.parent_mask, dtype=np.float32),
        target_config,
        domain_mask=np.asarray(patch.valid_mask, dtype=np.float32),
    ).astype(np.float32, copy=False)
    return abs(float(patch.target_sum) - old_sum)


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    source_cache_dir = args.source_cache_dir.expanduser().resolve()
    output_cache_dir = args.output_cache_dir.expanduser().resolve()
    config = read_json(args.config)
    target_config = target_config_from_config(config)
    splits = parse_splits(str(args.splits))
    output_cache_dir.mkdir(parents=True, exist_ok=True)

    split_summaries: dict[str, Any] = {}
    for split in splits:
        patches, source_metadata = load_patch_split_cache(source_cache_dir, split)
        max_sum_delta = 0.0
        total_sum_delta = 0.0
        for patch in patches:
            delta = retarget_patch(patch, target_config)
            max_sum_delta = max(max_sum_delta, float(delta))
            total_sum_delta += float(delta)
        output_path = save_patch_split_cache(
            output_cache_dir,
            split,
            patches,
            metadata={
                "retargeted_from_cache_dir": str(source_cache_dir),
                "source_cache_metadata": source_metadata,
                "config_path": str(args.config.expanduser().resolve()),
                "config_hash": stable_json_hash(config),
                "target_config": {
                    "kernel": str(target_config.kernel),
                    "sigma_pixels": float(target_config.sigma_pixels),
                    "radius_pixels": int(target_config.radius_pixels),
                    "per_ship_mass": float(target_config.per_ship_mass),
                    "renormalize_after_roi_mask": bool(target_config.renormalize_after_roi_mask),
                    "require_source_in_roi": bool(target_config.require_source_in_roi),
                },
                "target_sum_delta_abs_max": float(max_sum_delta),
                "target_sum_delta_abs_total": float(total_sum_delta),
            },
        )
        split_summaries[split] = {
            "path": str(output_path),
            "patch_count": int(len(patches)),
            "selected_target_sum": float(sum(float(patch.target_sum) for patch in patches)),
            "target_sum_delta_abs_max": float(max_sum_delta),
            "target_sum_delta_abs_total": float(total_sum_delta),
        }
        print(json.dumps({"split": split, **split_summaries[split]}, ensure_ascii=False), flush=True)

    manifest = {
        "schema_version": 1,
        "kind": "retargeted_density_patch_cache",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_cache_dir": str(source_cache_dir),
        "output_cache_dir": str(output_cache_dir),
        "config_path": str(args.config.expanduser().resolve()),
        "config_hash": stable_json_hash(config),
        "target_config": {
            "kernel": str(target_config.kernel),
            "sigma_pixels": float(target_config.sigma_pixels),
            "radius_pixels": int(target_config.radius_pixels),
        },
        "splits": splits,
        "split_summaries": split_summaries,
        "git": git_metadata(),
    }
    manifest_path = output_cache_dir / "retarget_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"manifest_json": str(manifest_path), "split_summaries": split_summaries}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
