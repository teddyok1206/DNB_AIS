from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .evaluate_brightness_threshold_baseline import parse_float_list, parse_split_list, select_best_threshold, write_csv


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate brightness-threshold baselines from a cached patch dataset. "
            "This avoids rebuilding PH anchors/patches when only thresholds change."
        )
    )
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--thresholds", default="0.35,0.45,0.55,0.65,0.75,0.85,0.90,0.95")
    parser.add_argument("--splits", default="val,test")
    parser.add_argument("--select-threshold-split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--eps", type=float, default=1.0e-8)
    return parser


def output_dir_from_args(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir.expanduser().resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return args.cache_dir.expanduser().resolve() / "threshold_sweeps" / stamp


def evaluate_cached_split(split: str, cache_path: Path, thresholds: list[float], eps: float) -> list[dict[str, Any]]:
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing split cache: {cache_path}")
    data = np.load(cache_path, allow_pickle=True)
    image = np.asarray(data["image"], dtype=np.float32)
    valid = np.asarray(data["valid_mask"]) > 0
    target = np.asarray(data["target_density"], dtype=np.float32) * valid.astype(np.float32)
    raw_target_sum = np.asarray(data["raw_count_sum"], dtype=np.float64)

    target_count = target.reshape(target.shape[0], -1).sum(axis=1, dtype=np.float64) if target.size else np.asarray([], dtype=np.float64)
    target_positive = target_count > float(eps)
    positive_count = int(target_positive.sum())
    patch_count = int(image.shape[0])
    target_prob = np.zeros_like(target, dtype=np.float32)
    if positive_count:
        target_prob[target_positive] = target[target_positive] / np.maximum(target_count[target_positive], float(eps)).reshape(-1, 1, 1)

    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        pred_mask = (image >= np.float32(threshold)) & valid
        pred_count = pred_mask.reshape(patch_count, -1).sum(axis=1, dtype=np.float64) if patch_count else np.asarray([], dtype=np.float64)
        pred_positive = pred_count > 0
        pred_positive_count = int(pred_positive.sum())
        overlap = np.zeros((patch_count,), dtype=np.float64)
        both = pred_positive & target_positive
        if bool(both.any()):
            pred_prob = np.zeros_like(target, dtype=np.float32)
            pred_prob[both] = pred_mask[both].astype(np.float32) / np.maximum(pred_count[both], float(eps)).reshape(-1, 1, 1)
            overlap[both] = np.minimum(pred_prob[both], target_prob[both]).reshape(int(both.sum()), -1).sum(axis=1, dtype=np.float64)

        tp = int(np.logical_and(pred_positive, target_positive).sum())
        fp = int(np.logical_and(pred_positive, ~target_positive).sum())
        fn = int(np.logical_and(~pred_positive, target_positive).sum())
        tn = int(np.logical_and(~pred_positive, ~target_positive).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, float(eps))
        pred_binary = pred_positive.astype(np.float64)
        target_binary = target_positive.astype(np.float64)
        mass_error = pred_binary - target_binary
        smape = 2.0 * np.abs(mass_error) / np.maximum(np.abs(pred_binary) + np.abs(target_binary), float(eps)) if patch_count else np.asarray([], dtype=np.float64)
        overlap_sum = float(overlap.sum())
        positive_overlap_sum = float(overlap[target_positive].sum()) if patch_count else 0.0

        rows.append(
            {
                "split": split,
                "baseline": "brightness_threshold_cache",
                "threshold": float(threshold),
                "patch_count": patch_count,
                "positive_patch_count": positive_count,
                "zero_target_patch_count": int(patch_count - positive_count),
                "pred_positive_count": pred_positive_count,
                "pred_sum": float(pred_positive_count),
                "target_sum": float(positive_count),
                "raw_target_count_sum": float(raw_target_sum.sum()),
                "pred_target_ratio": float(pred_positive_count / max(positive_count, float(eps))),
                "target_explained": float(overlap_sum / max(positive_count, float(eps))),
                "pred_matched": float(overlap_sum / max(pred_positive_count, float(eps))),
                "spatial_overlap_mean_positive": float(positive_overlap_sum / max(positive_count, 1)),
                "occupancy_mass_ratio_abs_log_error": float(abs(math.log((pred_positive_count + float(eps)) / (positive_count + float(eps))))),
                "patch_occupancy_mass_mae": float(np.mean(np.abs(mass_error))) if patch_count else None,
                "patch_occupancy_mass_rmse": float(np.sqrt(np.mean(mass_error**2))) if patch_count else None,
                "patch_occupancy_mass_bias_mean": float(np.mean(mass_error)) if patch_count else None,
                "patch_occupancy_mass_smape": float(np.mean(smape)) if smape.size else None,
                "occupancy_accuracy": float((tp + tn) / max(patch_count, 1)),
                "occupancy_precision": float(precision),
                "occupancy_recall": float(recall),
                "occupancy_f1": float(f1),
                "occupancy_brier": float(np.mean((pred_binary - target_binary) ** 2)) if patch_count else None,
                "occupancy_positive_target_count": positive_count,
                "occupancy_negative_target_count": int(patch_count - positive_count),
                "occupancy_pred_positive_count": pred_positive_count,
                "occupancy_tp": tp,
                "occupancy_fp": fp,
                "occupancy_fn": fn,
                "occupancy_tn": tn,
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cache_dir = args.cache_dir.expanduser().resolve()
    output_dir = output_dir_from_args(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = parse_float_list(str(args.thresholds))
    splits = parse_split_list(str(args.splits))
    if str(args.select_threshold_split) not in splits:
        splits.append(str(args.select_threshold_split))

    metric_rows: list[dict[str, Any]] = []
    for split in splits:
        metric_rows.extend(evaluate_cached_split(split, cache_dir / f"{split}_patch_cache.npz", thresholds, float(args.eps)))

    best = select_best_threshold(metric_rows, str(args.select_threshold_split))
    best_threshold = float(best["threshold"])
    for row in metric_rows:
        row["selected_by_val"] = bool(float(row["threshold"]) == best_threshold)
    selected_rows = [dict(row) for row in metric_rows if float(row["threshold"]) == best_threshold]

    metrics_csv = output_dir / "brightness_threshold_metrics_by_split.csv"
    selected_csv = output_dir / "brightness_threshold_selected_threshold_metrics.csv"
    summary_json = output_dir / "brightness_threshold_summary.json"
    write_csv(metrics_csv, metric_rows)
    write_csv(selected_csv, selected_rows)
    summary = {
        "schema_version": 1,
        "kind": "brightness_threshold_cache_sweep",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cache_dir": str(cache_dir),
        "manifest_json": str(cache_dir / "manifest.json"),
        "thresholds": thresholds,
        "splits": splits,
        "select_threshold_split": str(args.select_threshold_split),
        "selected_threshold": best_threshold,
        "selected_threshold_metric": best,
        "outputs": {
            "metrics_by_split_csv": str(metrics_csv),
            "selected_threshold_metrics_csv": str(selected_csv),
            "summary_json": str(summary_json),
        },
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary_json": str(summary_json), "selected_threshold": best_threshold, "selected_rows": selected_rows}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
