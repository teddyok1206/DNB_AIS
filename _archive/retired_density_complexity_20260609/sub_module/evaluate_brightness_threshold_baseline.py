from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .dnb_pipeline_core import GroundTruthResolver
from .dnb_project_paths import ROOT, STEP3
from .run_density_split_smoke_train import (
    SceneBuildResult,
    SceneSplitRecord,
    build_scene,
    read_json,
    read_scene_split,
)


@dataclass(frozen=True)
class PatchEval:
    pred_positive: bool
    target_positive: bool
    overlap: float
    raw_target_count: float


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate brightness-threshold O/X baselines on the same scene split "
            "and patch-selection policy used by the active OccupancySpatial U-Net."
        )
    )
    parser.add_argument("--scene-split-csv", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "dnb_density_unet_occupancy_spatial.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "dnb_density" / "baseline_evaluations" / "brightness_threshold")
    parser.add_argument("--thresholds", default="0.85,0.90,0.95", help="Comma-separated encoded-brightness thresholds.")
    parser.add_argument("--splits", default="train,val,test", help="Comma-separated split names to evaluate.")
    parser.add_argument("--select-threshold-split", choices=["train", "val", "test"], default="val")
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
    parser.add_argument("--eps", type=float, default=1.0e-8)
    return parser


def parse_float_list(text: str) -> list[float]:
    values = [float(item.strip()) for item in str(text).split(",") if item.strip()]
    if not values:
        raise ValueError("At least one threshold is required")
    for value in values:
        if not math.isfinite(value):
            raise ValueError(f"Non-finite threshold: {value}")
    return values


def parse_split_list(text: str) -> list[str]:
    values = [item.strip() for item in str(text).split(",") if item.strip()]
    valid = {"train", "val", "test"}
    bad = [item for item in values if item not in valid]
    if bad:
        raise ValueError(f"Unsupported splits: {bad}")
    if not values:
        raise ValueError("At least one split is required")
    return values


def records_by_split(records: Iterable[SceneSplitRecord], splits: list[str], limit: int) -> dict[str, list[SceneSplitRecord]]:
    grouped = {split: [] for split in splits}
    for record in records:
        if record.split in grouped:
            grouped[record.split].append(record)
    if int(limit) > 0:
        grouped = {split: items[: int(limit)] for split, items in grouped.items()}
    return grouped


def build_patch_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        max_patches_per_scene=int(args.max_patches_per_scene),
        max_ph_patches_per_scene=int(args.max_ph_patches_per_scene),
        max_fallback_patches_per_scene=int(args.max_fallback_patches_per_scene),
        positive_patches_per_scene=int(args.positive_patches_per_scene),
        negative_patches_per_scene=int(args.negative_patches_per_scene),
        selection_seed=int(args.selection_seed),
        seed=int(args.seed),
        max_patch_height=int(args.max_patch_height),
        max_patch_width=int(args.max_patch_width),
        skip_ph_anchor_zero=bool(args.skip_ph_anchor_zero),
    )


def build_split_patches(
    *,
    records: list[SceneSplitRecord],
    config: dict[str, Any],
    args: argparse.Namespace,
    resolver: GroundTruthResolver,
) -> tuple[list[Any], list[dict[str, Any]]]:
    patch_args = build_patch_args(args)
    patches: list[Any] = []
    scene_rows: list[dict[str, Any]] = []
    for record in records:
        print(f"[baseline-build] {record.split} {record.scene_key}")
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
        row = {
            "split": record.split,
            "day_key": record.day_key,
            "scene_key": record.scene_key,
            "excluded_reason": result.excluded_reason or "",
            "selected_patch_count": int(len(result.patches)),
            "selected_target_sum": float(sum(float(patch.target_sum) for patch in result.patches)),
        }
        metrics = result.metrics or {}
        partition = metrics.get("partition_summary", {}) if isinstance(metrics.get("partition_summary", {}), dict) else {}
        selection = metrics.get("selection", {}) if isinstance(metrics.get("selection", {}), dict) else {}
        row.update(
            {
                "ph_anchor_count": int(partition.get("ph_anchor_count", 0) or 0),
                "fallback_grid_count": int(partition.get("fallback_grid_count", 0) or 0),
                "selected_positive_patch_count": int(selection.get("selected_positive_patch_count", 0) or 0),
                "selected_negative_patch_count": int(selection.get("selected_negative_patch_count", 0) or 0),
            }
        )
        if result.excluded_reason:
            print(f"  [skip] {record.scene_key}: {result.excluded_reason}")
        else:
            print(
                "  [ok] "
                f"patches={len(result.patches)} "
                f"positive={row['selected_positive_patch_count']} "
                f"negative={row['selected_negative_patch_count']}"
            )
        scene_rows.append(row)
    return patches, scene_rows


def evaluate_patch_threshold(patch: Any, threshold: float, eps: float) -> PatchEval:
    valid = np.asarray(patch.valid_mask, dtype=np.float32) > 0
    image = np.asarray(patch.image, dtype=np.float32)
    pred_mask = (image >= np.float32(threshold)) & valid
    pred_positive = bool(np.any(pred_mask))

    target = np.asarray(patch.target_density, dtype=np.float32) * valid.astype(np.float32)
    target_count = float(target.sum())
    target_positive = target_count > float(eps)
    raw_target_count = float(patch.target_sum)

    if not pred_positive or not target_positive:
        return PatchEval(
            pred_positive=pred_positive,
            target_positive=target_positive,
            overlap=0.0,
            raw_target_count=raw_target_count,
        )

    pred_prob = pred_mask.astype(np.float32)
    pred_prob /= max(float(pred_prob.sum()), float(eps))
    target_prob = target / max(target_count, float(eps))
    return PatchEval(
        pred_positive=True,
        target_positive=True,
        overlap=float(np.minimum(pred_prob, target_prob).sum()),
        raw_target_count=raw_target_count,
    )


def summarize_patch_evals(split: str, threshold: float, evals: list[PatchEval], eps: float) -> dict[str, Any]:
    patch_count = int(len(evals))
    target_positive = np.asarray([item.target_positive for item in evals], dtype=bool)
    pred_positive = np.asarray([item.pred_positive for item in evals], dtype=bool)
    overlaps = np.asarray([item.overlap for item in evals], dtype=np.float64)
    raw_target_count_sum = float(sum(float(item.raw_target_count) for item in evals))

    tp = int(np.logical_and(pred_positive, target_positive).sum())
    fp = int(np.logical_and(pred_positive, ~target_positive).sum())
    fn = int(np.logical_and(~pred_positive, target_positive).sum())
    tn = int(np.logical_and(~pred_positive, ~target_positive).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, float(eps))

    positive_count = int(target_positive.sum())
    pred_positive_count = int(pred_positive.sum())
    overlap_sum = float(overlaps.sum())
    positive_overlap_sum = float(overlaps[target_positive].sum()) if patch_count else 0.0
    brier = float(np.mean((pred_positive.astype(np.float64) - target_positive.astype(np.float64)) ** 2)) if patch_count else None
    mass_error = pred_positive.astype(np.float64) - target_positive.astype(np.float64)
    smape = (
        2.0 * np.abs(mass_error) / np.maximum(np.abs(pred_positive.astype(np.float64)) + np.abs(target_positive.astype(np.float64)), float(eps))
        if patch_count
        else np.asarray([], dtype=np.float64)
    )
    return {
        "split": split,
        "baseline": "brightness_threshold",
        "threshold": float(threshold),
        "patch_count": patch_count,
        "positive_patch_count": positive_count,
        "zero_target_patch_count": int(patch_count - positive_count),
        "pred_positive_count": pred_positive_count,
        "pred_sum": float(pred_positive_count),
        "target_sum": float(positive_count),
        "raw_target_count_sum": raw_target_count_sum,
        "pred_target_ratio": float(pred_positive_count / max(positive_count, float(eps))),
        "target_explained": float(overlap_sum / max(positive_count, float(eps))),
        "pred_matched": float(overlap_sum / max(pred_positive_count, float(eps))),
        "spatial_overlap_mean_positive": float(positive_overlap_sum / max(positive_count, 1)),
        "occupancy_mass_ratio_abs_log_error": float(abs(np.log((pred_positive_count + float(eps)) / (positive_count + float(eps))))),
        "patch_occupancy_mass_mae": float(np.mean(np.abs(mass_error))) if patch_count else None,
        "patch_occupancy_mass_rmse": float(np.sqrt(np.mean(mass_error**2))) if patch_count else None,
        "patch_occupancy_mass_bias_mean": float(np.mean(mass_error)) if patch_count else None,
        "patch_occupancy_mass_smape": float(np.mean(smape)) if smape.size else None,
        "occupancy_accuracy": float((tp + tn) / max(patch_count, 1)),
        "occupancy_precision": float(precision),
        "occupancy_recall": float(recall),
        "occupancy_f1": float(f1),
        "occupancy_brier": brier,
        "occupancy_positive_target_count": positive_count,
        "occupancy_negative_target_count": int(patch_count - positive_count),
        "occupancy_pred_positive_count": pred_positive_count,
        "occupancy_tp": tp,
        "occupancy_fp": fp,
        "occupancy_fn": fn,
        "occupancy_tn": tn,
    }


def evaluate_thresholds_for_split(split: str, patches: list[Any], thresholds: list[float], eps: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        evals = [evaluate_patch_threshold(patch, threshold, eps) for patch in patches]
        rows.append(summarize_patch_evals(split, threshold, evals, eps))
    return rows


def select_best_threshold(rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
    candidates = [row for row in rows if row["split"] == split]
    if not candidates:
        raise RuntimeError(f"No threshold metrics found for selection split: {split}")
    return max(
        candidates,
        key=lambda row: (
            float(row.get("occupancy_f1") or 0.0),
            float(row.get("spatial_overlap_mean_positive") or 0.0),
            float(row.get("target_explained") or 0.0),
            -float(row.get("occupancy_brier") or 1.0e9),
        ),
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    thresholds = parse_float_list(str(args.thresholds))
    splits = parse_split_list(str(args.splits))
    if str(args.select_threshold_split) not in splits:
        splits.append(str(args.select_threshold_split))

    config = read_json(args.config)
    records = read_scene_split(args.scene_split_csv)
    grouped = records_by_split(records, splits, int(args.limit_scenes_per_split))
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    resolver = GroundTruthResolver(STEP3 / "bboxes_JPSS-2")

    split_patches: dict[str, list[Any]] = {}
    scene_rows: list[dict[str, Any]] = []
    for split in splits:
        patches, rows = build_split_patches(records=grouped.get(split, []), config=config, args=args, resolver=resolver)
        split_patches[split] = patches
        scene_rows.extend(rows)

    metric_rows: list[dict[str, Any]] = []
    for split in splits:
        metric_rows.extend(evaluate_thresholds_for_split(split, split_patches.get(split, []), thresholds, float(args.eps)))

    best = select_best_threshold(metric_rows, str(args.select_threshold_split))
    best_threshold = float(best["threshold"])
    best_rows = [dict(row, selected_by_val=(float(row["threshold"]) == best_threshold)) for row in metric_rows if float(row["threshold"]) == best_threshold]
    for row in metric_rows:
        row["selected_by_val"] = bool(float(row["threshold"]) == best_threshold)

    scene_csv = output_dir / "brightness_threshold_scene_build_metrics.csv"
    metrics_csv = output_dir / "brightness_threshold_metrics_by_split.csv"
    selected_csv = output_dir / "brightness_threshold_selected_threshold_metrics.csv"
    summary_json = output_dir / "brightness_threshold_summary.json"
    write_csv(scene_csv, scene_rows)
    write_csv(metrics_csv, metric_rows)
    write_csv(selected_csv, best_rows)
    summary = {
        "schema_version": 1,
        "kind": "brightness_threshold_baseline",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scene_split_csv": str(args.scene_split_csv.expanduser().resolve()),
        "config_path": str(args.config.expanduser().resolve()),
        "thresholds": thresholds,
        "splits": splits,
        "select_threshold_split": str(args.select_threshold_split),
        "selected_threshold": best_threshold,
        "selected_threshold_metric": best,
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
        "outputs": {
            "scene_build_metrics_csv": str(scene_csv),
            "metrics_by_split_csv": str(metrics_csv),
            "selected_threshold_metrics_csv": str(selected_csv),
            "summary_json": str(summary_json),
        },
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary_json": str(summary_json), "selected_threshold": best_threshold, "selected_rows": best_rows}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
