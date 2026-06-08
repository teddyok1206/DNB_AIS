from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .dnb_pipeline_core import GroundTruthResolver
from .dnb_project_paths import ROOT, STEP3
from .evaluate_brightness_threshold_baseline import (
    build_patch_args,
    parse_split_list,
    records_by_split,
    write_csv,
)
from .run_density_split_smoke_train import (
    SceneBuildResult,
    build_scene,
    git_metadata,
    read_json,
    read_scene_split,
    stable_json_hash,
)


def default_cache_root() -> Path:
    ssd_root = Path("/Volumes/SAMSUNG")
    if ssd_root.exists():
        return ssd_root / "dnb_density_patch_cache"
    return ROOT / "outputs" / "dnb_density" / "patch_cache"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a reusable SSD patch cache for rule-based brightness-threshold sweeps. "
            "This stores the selected patches after PH/partitioning so later threshold changes "
            "do not rebuild PH anchors or patches."
        )
    )
    parser.add_argument("--scene-split-csv", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "dnb_density_unet_occupancy_spatial.json")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--cache-tag", default=f"brightness_threshold_patch_cache_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--splits", default="val,test")
    parser.add_argument("--limit-scenes-per-split", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260529)
    parser.add_argument("--max-patches-per-scene", type=int, default=48)
    parser.add_argument("--max-ph-patches-per-scene", type=int, default=36)
    parser.add_argument("--max-fallback-patches-per-scene", type=int, default=12)
    parser.add_argument("--positive-patches-per-scene", type=int, default=24)
    parser.add_argument("--negative-patches-per-scene", type=int, default=24)
    parser.add_argument("--selection-seed", type=int, default=20260609)
    parser.add_argument("--max-patch-height", type=int, default=256)
    parser.add_argument("--max-patch-width", type=int, default=256)
    parser.add_argument("--skip-ph-anchor-zero", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--image-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--target-dtype", choices=["float16", "float32"], default="float32")
    parser.add_argument("--compressed", action=argparse.BooleanOptionalAction, default=True)
    return parser


def cache_dir_from_args(args: argparse.Namespace) -> Path:
    if args.cache_dir is not None:
        return args.cache_dir.expanduser().resolve()
    return (default_cache_root() / str(args.cache_tag)).expanduser().resolve()


def build_split_patches(
    *,
    split: str,
    records: list[Any],
    config: dict[str, Any],
    args: argparse.Namespace,
    resolver: GroundTruthResolver,
) -> tuple[list[Any], list[dict[str, Any]]]:
    patch_args = build_patch_args(args)
    patches: list[Any] = []
    rows: list[dict[str, Any]] = []
    for record in records:
        print(f"[cache-build] {split} {record.scene_key}", flush=True)
        try:
            result = build_scene(record, config=config, args=patch_args, resolver=resolver)
        except Exception as exc:
            result = SceneBuildResult(
                record=record,
                patches=[],
                metrics={"split": record.split, "scene_key": record.scene_key},
                excluded_reason=f"exception:{type(exc).__name__}:{exc}",
            )

        patches.extend(result.patches)
        metrics = result.metrics or {}
        partition = metrics.get("partition_summary", {}) if isinstance(metrics.get("partition_summary", {}), dict) else {}
        selection = metrics.get("selection", {}) if isinstance(metrics.get("selection", {}), dict) else {}
        row = {
            "split": record.split,
            "day_key": record.day_key,
            "scene_key": record.scene_key,
            "tif_path": str(record.tif_path),
            "geojson_path": str(record.geojson_path),
            "excluded_reason": result.excluded_reason or "",
            "selected_patch_count": int(len(result.patches)),
            "selected_target_sum": float(sum(float(patch.target_sum) for patch in result.patches)),
            "ph_anchor_count": int(partition.get("ph_anchor_count", 0) or 0),
            "fallback_grid_count": int(partition.get("fallback_grid_count", 0) or 0),
            "selected_positive_patch_count": int(selection.get("selected_positive_patch_count", 0) or 0),
            "selected_negative_patch_count": int(selection.get("selected_negative_patch_count", 0) or 0),
        }
        if result.excluded_reason:
            print(f"  [skip] {record.scene_key}: {result.excluded_reason}", flush=True)
        else:
            print(
                "  [ok] "
                f"patches={len(result.patches)} "
                f"positive={row['selected_positive_patch_count']} "
                f"negative={row['selected_negative_patch_count']}",
                flush=True,
            )
        rows.append(row)
    return patches, rows


def patch_rows(patches: list[Any], split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, patch in enumerate(patches):
        h, w = patch.shape
        rows.append(
            {
                "split": split,
                "patch_index": index,
                "cluster_id": int(patch.cluster_id),
                "partition_id": "" if patch.partition_id is None else int(patch.partition_id),
                "partition_kind": str(patch.partition_kind),
                "anchor_cluster_id": "" if patch.anchor_cluster_id is None else int(patch.anchor_cluster_id),
                "height": int(h),
                "width": int(w),
                "target_sum": float(patch.target_sum),
                "raw_count_sum": float(patch.raw_count_sum),
                "valid_pixels": int(patch.valid_pixels),
                "lifetime": float(patch.lifetime),
                "bbox_rc": json.dumps([int(v) for v in patch.bbox_rc], separators=(",", ":")),
                "crop_rc": json.dumps([int(v) for v in patch.crop_rc], separators=(",", ":")),
            }
        )
    return rows


def save_split_npz(
    path: Path,
    *,
    split: str,
    patches: list[Any],
    image_dtype: str,
    target_dtype: str,
    compressed: bool,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = int(len(patches))
    max_h = max((int(patch.shape[0]) for patch in patches), default=0)
    max_w = max((int(patch.shape[1]) for patch in patches), default=0)
    image = np.zeros((count, max_h, max_w), dtype=np.dtype(image_dtype))
    valid_mask = np.zeros((count, max_h, max_w), dtype=np.uint8)
    target_density = np.zeros((count, max_h, max_w), dtype=np.dtype(target_dtype))
    heights = np.zeros((count,), dtype=np.uint16)
    widths = np.zeros((count,), dtype=np.uint16)
    target_sum = np.zeros((count,), dtype=np.float32)
    raw_count_sum = np.zeros((count,), dtype=np.float32)
    valid_pixels = np.zeros((count,), dtype=np.uint32)

    for index, patch in enumerate(patches):
        h, w = patch.shape
        heights[index] = int(h)
        widths[index] = int(w)
        image[index, :h, :w] = np.asarray(patch.image, dtype=np.dtype(image_dtype))
        valid_mask[index, :h, :w] = (np.asarray(patch.valid_mask) > 0).astype(np.uint8)
        target_density[index, :h, :w] = np.asarray(patch.target_density, dtype=np.dtype(target_dtype))
        target_sum[index] = float(patch.target_sum)
        raw_count_sum[index] = float(patch.raw_count_sum)
        valid_pixels[index] = int(patch.valid_pixels)

    payload = {
        "split": np.asarray([split], dtype=object),
        "image": image,
        "valid_mask": valid_mask,
        "target_density": target_density,
        "height": heights,
        "width": widths,
        "target_sum": target_sum,
        "raw_count_sum": raw_count_sum,
        "valid_pixels": valid_pixels,
    }
    if compressed:
        np.savez_compressed(path, **payload)
    else:
        np.savez(path, **payload)
    return {
        "path": str(path),
        "patch_count": count,
        "shape": [count, max_h, max_w],
        "image_dtype": image_dtype,
        "target_dtype": target_dtype,
        "compressed": bool(compressed),
        "bytes": int(path.stat().st_size),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    splits = parse_split_list(str(args.splits))
    config = read_json(args.config)
    records = read_scene_split(args.scene_split_csv)
    grouped = records_by_split(records, splits, int(args.limit_scenes_per_split))
    cache_dir = cache_dir_from_args(args)
    cache_dir.mkdir(parents=True, exist_ok=True)
    resolver = GroundTruthResolver(STEP3 / "bboxes_JPSS-2")

    scene_rows: list[dict[str, Any]] = []
    split_summaries: dict[str, Any] = {}
    patch_metadata_rows: list[dict[str, Any]] = []
    for split in splits:
        patches, rows = build_split_patches(
            split=split,
            records=grouped.get(split, []),
            config=config,
            args=args,
            resolver=resolver,
        )
        scene_rows.extend(rows)
        patch_metadata_rows.extend(patch_rows(patches, split))
        split_summaries[split] = save_split_npz(
            cache_dir / f"{split}_patch_cache.npz",
            split=split,
            patches=patches,
            image_dtype=str(args.image_dtype),
            target_dtype=str(args.target_dtype),
            compressed=bool(args.compressed),
        )

    scene_csv = cache_dir / "scene_build_metrics.csv"
    patch_csv = cache_dir / "patch_metadata.csv"
    write_csv(scene_csv, scene_rows)
    write_csv(patch_csv, patch_metadata_rows)

    summary = {
        "schema_version": 1,
        "kind": "brightness_threshold_patch_cache",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cache_dir": str(cache_dir),
        "scene_split_csv": str(args.scene_split_csv.expanduser().resolve()),
        "config_path": str(args.config.expanduser().resolve()),
        "config_hash": stable_json_hash(config),
        "git": git_metadata(),
        "splits": splits,
        "patch_selection": {
            "skip_ph_anchor_zero": bool(args.skip_ph_anchor_zero),
            "max_patches_per_scene": int(args.max_patches_per_scene),
            "max_ph_patches_per_scene": int(args.max_ph_patches_per_scene),
            "max_fallback_patches_per_scene": int(args.max_fallback_patches_per_scene),
            "positive_patches_per_scene": int(args.positive_patches_per_scene),
            "negative_patches_per_scene": int(args.negative_patches_per_scene),
            "selection_seed": int(args.selection_seed),
            "max_patch_height": int(args.max_patch_height),
            "max_patch_width": int(args.max_patch_width),
        },
        "storage": {
            "image_dtype": str(args.image_dtype),
            "target_dtype": str(args.target_dtype),
            "compressed": bool(args.compressed),
        },
        "split_caches": split_summaries,
        "outputs": {
            "scene_build_metrics_csv": str(scene_csv),
            "patch_metadata_csv": str(patch_csv),
            "manifest_json": str(cache_dir / "manifest.json"),
        },
    }
    manifest_path = cache_dir / "manifest.json"
    manifest_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"manifest_json": str(manifest_path), "split_caches": split_summaries}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
