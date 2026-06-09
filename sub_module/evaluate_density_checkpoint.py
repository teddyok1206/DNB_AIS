from __future__ import annotations

import argparse
import json
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

from .dnb_density_common import density_patch_collate, move_density_batch_to_device
from .dnb_density_losses import build_density_loss
from .dnb_density_patch_pickle_cache import load_patch_split_cache
from .run_density_split_smoke_train import (
    build_arg_parser as build_train_arg_parser,
    build_scene,
    build_model_from_config,
    group_records,
    input_channels_from_config,
    loss_config_from_config,
    make_loader,
    read_scene_split,
    read_json,
    resolve_required_device,
    run_batches,
    seed_everything,
    stable_json_hash,
)
from .dnb_pipeline_core import GroundTruthResolver
from .dnb_project_paths import ROOT, STEP3


CHECKPOINT_SELECTIONS = {
    "last": ("outputs", "checkpoint_last"),
    "best_val_loss": ("best_checkpoints", "best_val_loss"),
    "best_val_pixel_f1": ("best_checkpoints", "best_val_pixel_f1"),
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a saved density checkpoint on cached patch splits.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Completed run directory containing run_summary.json.")
    parser.add_argument(
        "--checkpoint",
        choices=sorted(CHECKPOINT_SELECTIONS),
        default="best_val_pixel_f1",
        help="Checkpoint selection from run_summary.json.",
    )
    parser.add_argument("--checkpoint-path", type=Path, default=None, help="Explicit checkpoint path. Overrides --checkpoint.")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--calibration-split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threshold-grid-size", type=int, default=201)
    parser.add_argument(
        "--radius-sigmas",
        default="1,2,4,8",
        help="Comma-separated Gaussian sigma sweep in pixels for radius-tolerant probability evaluation. Empty disables radius evaluation.",
    )
    parser.add_argument("--radius-target-threshold", type=float, default=0.25, help="Threshold applied to the soft radius target for F1/IoU.")
    parser.add_argument("--radius-truncate", type=float, default=3.0, help="Gaussian cutoff radius is ceil(sigma * truncate).")
    parser.add_argument("--patch-cache-dir", type=Path, default=None, help="Override patch cache directory when an older run summary did not record it.")
    parser.add_argument("--output-json", type=Path, default=None)
    return parser


def checkpoint_path_from_summary(summary: dict[str, Any], checkpoint: str, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    section_name, key = CHECKPOINT_SELECTIONS[str(checkpoint)]
    section = summary.get(section_name, {})
    if not isinstance(section, dict):
        raise KeyError(f"run_summary.json does not contain section {section_name!r}")
    if checkpoint == "last":
        raw_path = section.get(key)
    else:
        item = section.get(key, {})
        raw_path = item.get("path") if isinstance(item, dict) else None
    if not raw_path:
        raise KeyError(f"run_summary.json does not contain checkpoint selection {checkpoint!r}")
    return Path(str(raw_path)).expanduser().resolve()


def checkpoint_path_from_run_dir(run_dir: Path, checkpoint: str, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    if checkpoint == "last":
        candidate = run_dir / "checkpoints" / "checkpoint_last.pt"
    else:
        candidate = run_dir / "checkpoints" / f"checkpoint_{checkpoint}.pt"
    if not candidate.exists():
        raise FileNotFoundError(f"Checkpoint selection {checkpoint!r} is missing: {candidate}")
    return candidate.expanduser().resolve()


def patch_cache_dir_from_summary(summary: dict[str, Any]) -> Path:
    raw_dir = summary.get("patch_cache", {}).get("dir") if isinstance(summary.get("patch_cache"), dict) else None
    if not raw_dir:
        raise KeyError("run_summary.json does not record patch_cache.dir")
    return Path(str(raw_dir)).expanduser().resolve()


def train_args_from_command(run_dir: Path) -> argparse.Namespace:
    command_path = run_dir / "command.txt"
    if not command_path.exists():
        raise FileNotFoundError(f"No patch cache is recorded and command.txt is missing: {command_path}")
    command_text = command_path.read_text(encoding="utf-8").replace("\\\n", " ")
    tokens = shlex.split(command_text)
    module = "sub_module.run_density_split_smoke_train"
    if module not in tokens:
        raise ValueError(f"command.txt does not invoke {module}: {command_path}")
    argv = tokens[tokens.index(module) + 1 :]
    return build_train_arg_parser().parse_args(argv)


def resolve_repo_relative(path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def rebuild_split_patches_from_run(run_dir: Path, config: dict[str, Any], split: str) -> tuple[list[Any], dict[str, Any]]:
    train_args = train_args_from_command(run_dir)
    train_args.scene_split_csv = resolve_repo_relative(train_args.scene_split_csv)
    seed_everything(int(train_args.seed))
    records = read_scene_split(train_args.scene_split_csv)
    grouped = group_records(records, int(train_args.max_scenes_per_split))
    resolver = GroundTruthResolver(STEP3 / "bboxes_JPSS-2")
    patches: list[Any] = []
    kept_scene_count = 0
    skipped: list[dict[str, str]] = []
    for record in grouped[str(split)]:
        print(f"[patch-rebuild] {split} {record.scene_key}")
        try:
            result = build_scene(record, config=config, args=train_args, resolver=resolver)
        except Exception as exc:
            skipped.append({"scene_key": record.scene_key, "reason": f"exception:{type(exc).__name__}:{exc}"})
            continue
        if result.excluded_reason:
            skipped.append({"scene_key": record.scene_key, "reason": str(result.excluded_reason)})
            continue
        kept_scene_count += 1
        patches.extend(result.patches)
    metadata = {
        "schema_version": None,
        "kind": "rebuilt_density_patch_split",
        "source": str(run_dir / "command.txt"),
        "cache": "none",
        "split": str(split),
        "input_scene_count": int(len(grouped[str(split)])),
        "kept_scene_count": int(kept_scene_count),
        "patch_count": int(len(patches)),
        "skipped": skipped,
    }
    return patches, metadata


def load_model_and_loss(checkpoint_path: Path, config: dict[str, Any], device: torch.device) -> tuple[torch.nn.Module, torch.nn.Module, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint does not contain model_state_dict: {checkpoint_path}")
    checkpoint_config = checkpoint.get("config")
    if isinstance(checkpoint_config, dict):
        config = checkpoint_config
    model = build_model_from_config(config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(device)
    loss_fn = build_density_loss(loss_config_from_config(config)).to(device)
    return model, loss_fn, config


def brightness_channel_index(input_channels: list[str] | None, batch_channels: list[str] | None = None) -> int:
    channel_names = batch_channels if batch_channels else input_channels
    if not channel_names:
        return 0
    normalized = [str(name).strip().lower() for name in channel_names]
    for candidate in ("brightness", "radiance", "dnb", "image"):
        if candidate in normalized:
            return int(normalized.index(candidate))
    return 0


def collect_presence_scores(
    *,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    patches: list[Any],
    batch_size: int,
    size_divisor: int,
    device: torch.device,
    num_workers: int,
    input_channels: list[str] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not hasattr(loss_fn, "pixel_occupancy_from_output"):
        return (
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=bool),
        )
    loader = make_loader(
        patches,
        batch_size,
        size_divisor,
        shuffle=False,
        num_workers=num_workers,
        input_channels=input_channels,
    )
    model_scores: list[np.ndarray] = []
    brightness_scores: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = move_density_batch_to_device(batch, device)
            output = model(batch["x"])
            pixel_prob, pixel_target, pixel_valid = loss_fn.pixel_occupancy_from_output(output, batch)
            target_threshold = float(getattr(loss_fn, "pixel_metric_target_threshold", 0.5))
            valid_np = pixel_valid.detach().cpu().numpy()
            model_np = pixel_prob.detach().cpu().numpy()
            target_np = pixel_target.detach().cpu().numpy()
            x_np = batch["x"].detach().cpu().numpy()
            batch_channels = [str(name) for name in batch.get("input_channels", input_channels or [])]
            brightness_idx = brightness_channel_index(input_channels, batch_channels)
            for idx, meta in enumerate(batch["metadata"]):
                height, width = [int(v) for v in meta["shape"]]
                valid = valid_np[idx, 0, :height, :width] > 0
                if not bool(np.any(valid)):
                    continue
                model_arr = model_np[idx, 0, :height, :width]
                target_arr = target_np[idx, 0, :height, :width]
                brightness_arr = x_np[idx, brightness_idx, :height, :width]
                model_scores.append(model_arr[valid].astype(np.float32, copy=False))
                brightness_scores.append(brightness_arr[valid].astype(np.float32, copy=False))
                targets.append((target_arr[valid] >= target_threshold).astype(bool, copy=False))
    if not model_scores:
        return (
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=bool),
        )
    return (
        np.concatenate(model_scores).astype(np.float64, copy=False),
        np.concatenate(brightness_scores).astype(np.float64, copy=False),
        np.concatenate(targets).astype(bool, copy=False),
    )


def threshold_metrics(probs: np.ndarray, targets: np.ndarray, threshold: float) -> dict[str, Any]:
    if probs.size == 0:
        return {}
    pred = probs >= float(threshold)
    tp = int(np.logical_and(pred, targets).sum())
    fp = int(np.logical_and(pred, ~targets).sum())
    fn = int(np.logical_and(~pred, targets).sum())
    tn = int(np.logical_and(~pred, ~targets).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1.0e-8)
    iou = tp / max(tp + fp + fn, 1)
    return {
        "threshold": float(threshold),
        "accuracy": float((tp + tn) / max(len(targets), 1)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
        "brier": float(np.mean((probs - targets.astype(np.float64)) ** 2)),
        "positive_target_count": int(targets.sum()),
        "negative_target_count": int((~targets).sum()),
        "pred_positive_count": int(pred.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def best_threshold_by_f1(probs: np.ndarray, targets: np.ndarray, grid_size: int) -> dict[str, Any]:
    if probs.size == 0:
        return {}
    thresholds = np.linspace(0.0, 1.0, max(int(grid_size), 2), dtype=np.float64)
    rows = [threshold_metrics(probs, targets, float(threshold)) for threshold in thresholds]
    # Prefer thresholds close to 0.5 when F1 ties, so calibration does not
    # drift unnecessarily on small validation sets.
    best = max(rows, key=lambda row: (float(row["f1"]), -abs(float(row["threshold"]) - 0.5)))
    return dict(best)


def parse_float_list(raw: str) -> list[float]:
    values: list[float] = []
    for item in str(raw).split(","):
        item = item.strip()
        if not item:
            continue
        values.append(float(item))
    return values


def average_precision_score(scores: np.ndarray, targets: np.ndarray) -> float | None:
    if scores.size == 0:
        return None
    positives = int(targets.sum())
    if positives <= 0:
        return None
    order = np.argsort(-scores, kind="mergesort")
    sorted_targets = targets[order].astype(np.float64, copy=False)
    tp = np.cumsum(sorted_targets)
    fp = np.cumsum(1.0 - sorted_targets)
    precision = tp / np.maximum(tp + fp, 1.0)
    recall = tp / max(float(positives), 1.0)
    recall_delta = np.diff(np.concatenate(([0.0], recall)))
    return float(np.sum(precision * recall_delta))


def top_fraction_key(fraction: float) -> str:
    percent = 100.0 * float(fraction)
    return f"top_{percent:g}pct".replace(".", "p")


def precision_at_top_fractions(
    scores: np.ndarray,
    targets: np.ndarray,
    fractions: tuple[float, ...] = (0.001, 0.005, 0.01, 0.05, 0.1),
) -> dict[str, Any]:
    if scores.size == 0:
        return {}
    order = np.argsort(-scores, kind="mergesort")
    sorted_targets = targets[order].astype(bool, copy=False)
    positives = int(targets.sum())
    rows: dict[str, Any] = {}
    for fraction in fractions:
        take = min(int(np.ceil(scores.size * float(fraction))), int(scores.size))
        take = max(take, 1)
        selected = sorted_targets[:take]
        hit_count = int(selected.sum())
        rows[top_fraction_key(float(fraction))] = {
            "fraction": float(fraction),
            "sample_count": int(take),
            "precision": float(hit_count / max(take, 1)),
            "recall": float(hit_count / max(positives, 1)),
            "hit_count": int(hit_count),
        }
    return rows


def calibration_bins(scores: np.ndarray, targets: np.ndarray, bin_count: int = 10) -> list[dict[str, Any]]:
    if scores.size == 0:
        return []
    clipped = np.clip(scores.astype(np.float64, copy=False), 0.0, 1.0)
    targets_float = targets.astype(np.float64, copy=False)
    edges = np.linspace(0.0, 1.0, max(int(bin_count), 1) + 1, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for idx in range(len(edges) - 1):
        left = float(edges[idx])
        right = float(edges[idx + 1])
        if idx == len(edges) - 2:
            mask = (clipped >= left) & (clipped <= right)
        else:
            mask = (clipped >= left) & (clipped < right)
        count = int(mask.sum())
        if count <= 0:
            rows.append({"bin_left": left, "bin_right": right, "count": 0})
            continue
        rows.append(
            {
                "bin_left": left,
                "bin_right": right,
                "count": count,
                "mean_score": float(clipped[mask].mean()),
                "empirical_presence_rate": float(targets_float[mask].mean()),
            }
        )
    return rows


def ranking_metrics(scores: np.ndarray, targets: np.ndarray, *, include_brier: bool) -> dict[str, Any]:
    if scores.size == 0:
        return {}
    targets_bool = targets.astype(bool, copy=False)
    positive_mask = targets_bool
    negative_mask = ~targets_bool
    result: dict[str, Any] = {
        "sample_count": int(scores.size),
        "positive_target_count": int(positive_mask.sum()),
        "negative_target_count": int(negative_mask.sum()),
        "positive_rate": float(positive_mask.mean()) if scores.size else None,
        "average_precision": average_precision_score(scores.astype(np.float64, copy=False), targets_bool),
        "score_mean_positive": float(scores[positive_mask].mean()) if bool(positive_mask.any()) else None,
        "score_mean_negative": float(scores[negative_mask].mean()) if bool(negative_mask.any()) else None,
        "precision_at_top": precision_at_top_fractions(scores.astype(np.float64, copy=False), targets_bool),
    }
    if include_brier:
        clipped = np.clip(scores.astype(np.float64, copy=False), 0.0, 1.0)
        result["brier"] = float(np.mean((clipped - targets_bool.astype(np.float64)) ** 2))
    return result


def ratio_or_none(numerator: Any, denominator: Any) -> float | None:
    if numerator is None or denominator is None:
        return None
    denominator_float = float(denominator)
    if abs(denominator_float) <= 1.0e-12:
        return None
    return float(float(numerator) / denominator_float)


def ranking_lift(model_metrics: dict[str, Any], baseline_metrics: dict[str, Any]) -> dict[str, Any]:
    lift: dict[str, Any] = {
        "average_precision_ratio": ratio_or_none(
            model_metrics.get("average_precision"),
            baseline_metrics.get("average_precision"),
        )
    }
    model_top = model_metrics.get("precision_at_top", {})
    baseline_top = baseline_metrics.get("precision_at_top", {})
    if isinstance(model_top, dict) and isinstance(baseline_top, dict):
        for key, model_row in model_top.items():
            baseline_row = baseline_top.get(key)
            if isinstance(model_row, dict) and isinstance(baseline_row, dict):
                lift[f"{key}_precision_ratio"] = ratio_or_none(
                    model_row.get("precision"),
                    baseline_row.get("precision"),
                )
    return lift


def gaussian_radius_target(target: np.ndarray, valid: np.ndarray, *, sigma: float, radius_pixels: int) -> np.ndarray:
    target_bool = (np.asarray(target, dtype=np.float32) > 0.5) & (np.asarray(valid, dtype=np.float32) > 0)
    valid_bool = np.asarray(valid, dtype=np.float32) > 0
    if not bool(np.any(target_bool)):
        return np.zeros_like(np.asarray(target, dtype=np.float32), dtype=np.float32)
    if float(sigma) <= 0.0:
        return target_bool.astype(np.float32) * valid_bool.astype(np.float32)
    distance = distance_transform_edt(~target_bool)
    radius = np.exp(-0.5 * (distance / max(float(sigma), 1.0e-6)) ** 2).astype(np.float32)
    if int(radius_pixels) > 0:
        radius[distance > int(radius_pixels)] = 0.0
    return np.clip(radius * valid_bool.astype(np.float32), 0.0, 1.0).astype(np.float32, copy=False)


def collect_radius_scores(
    *,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    patches: list[Any],
    batch_size: int,
    size_divisor: int,
    device: torch.device,
    num_workers: int,
    input_channels: list[str] | None,
    sigma: float,
    radius_pixels: int,
    target_threshold: float,
    truncate: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not hasattr(loss_fn, "pixel_occupancy_from_output"):
        empty_scores = np.asarray([], dtype=np.float64)
        empty_binary = np.asarray([], dtype=bool)
        return empty_scores, empty_scores, empty_binary
    loader = make_loader(
        patches,
        batch_size,
        size_divisor,
        shuffle=False,
        num_workers=num_workers,
        input_channels=input_channels,
    )
    model_values: list[np.ndarray] = []
    brightness_values: list[np.ndarray] = []
    binary_values: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = move_density_batch_to_device(batch, device)
            output = model(batch["x"])
            pixel_prob, pixel_target, pixel_valid = loss_fn.pixel_occupancy_from_output(output, batch)
            if hasattr(loss_fn, "source_pixel_target_from_batch"):
                radius_source_target = loss_fn.source_pixel_target_from_batch(batch)
            else:
                radius_source_target = pixel_target
            probs_np = pixel_prob.detach().cpu().numpy()
            targets_np = radius_source_target.detach().cpu().numpy()
            valid_np = pixel_valid.detach().cpu().numpy()
            x_np = batch["x"].detach().cpu().numpy()
            batch_channels = [str(name) for name in batch.get("input_channels", input_channels or [])]
            brightness_idx = brightness_channel_index(input_channels, batch_channels)
            for idx, meta in enumerate(batch["metadata"]):
                height, width = [int(v) for v in meta["shape"]]
                prob_arr = probs_np[idx, 0, :height, :width]
                target_arr = targets_np[idx, 0, :height, :width]
                valid_arr = valid_np[idx, 0, :height, :width]
                brightness_arr = x_np[idx, brightness_idx, :height, :width]
                mask = valid_arr > 0
                if not bool(np.any(mask)):
                    continue
                radius_target = gaussian_radius_target(
                    target_arr,
                    valid_arr,
                    sigma=float(sigma),
                    radius_pixels=int(radius_pixels),
                )
                model_values.append(prob_arr[mask].astype(np.float32, copy=False))
                brightness_values.append(brightness_arr[mask].astype(np.float32, copy=False))
                binary_values.append((radius_target[mask] >= float(target_threshold)).astype(bool, copy=False))
    if not model_values:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64), np.asarray([], dtype=bool)
    return (
        np.concatenate(model_values).astype(np.float64, copy=False),
        np.concatenate(brightness_values).astype(np.float64, copy=False),
        np.concatenate(binary_values).astype(bool, copy=False),
    )


def evaluate_radius_probability_sweep(
    *,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    calibration_patches: list[Any],
    eval_patches: list[Any],
    batch_size: int,
    size_divisor: int,
    device: torch.device,
    num_workers: int,
    input_channels: list[str] | None,
    sigmas: list[float],
    target_threshold: float,
    threshold_grid_size: int,
    truncate: float,
) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for sigma in sigmas:
        radius_pixels = int(np.ceil(max(float(sigma), 0.0) * max(float(truncate), 0.0)))
        calibration_model, _calibration_brightness, calibration_target_binary = collect_radius_scores(
            model=model,
            loss_fn=loss_fn,
            patches=calibration_patches,
            batch_size=batch_size,
            size_divisor=size_divisor,
            device=device,
            num_workers=num_workers,
            input_channels=input_channels,
            sigma=float(sigma),
            radius_pixels=radius_pixels,
            target_threshold=float(target_threshold),
            truncate=float(truncate),
        )
        eval_model, eval_brightness, eval_target_binary = collect_radius_scores(
            model=model,
            loss_fn=loss_fn,
            patches=eval_patches,
            batch_size=batch_size,
            size_divisor=size_divisor,
            device=device,
            num_workers=num_workers,
            input_channels=input_channels,
            sigma=float(sigma),
            radius_pixels=radius_pixels,
            target_threshold=float(target_threshold),
            truncate=float(truncate),
        )
        calibration_best = best_threshold_by_f1(calibration_model, calibration_target_binary, int(threshold_grid_size))
        calibrated_eval = (
            threshold_metrics(eval_model, eval_target_binary, float(calibration_best["threshold"]))
            if calibration_best
            else {}
        )
        fixed_eval = threshold_metrics(eval_model, eval_target_binary, 0.5) if eval_model.size else {}
        model_ranking = ranking_metrics(eval_model, eval_target_binary, include_brier=True)
        brightness_ranking = ranking_metrics(eval_brightness, eval_target_binary, include_brier=False)
        label = f"sigma_{str(float(sigma)).replace('.', 'p')}"
        rows[label] = {
            "sigma_pixels": float(sigma),
            "radius_pixels": int(radius_pixels),
            "target_threshold": float(target_threshold),
            "truncate": float(truncate),
            "model_probability": model_ranking,
            "brightness_baseline": brightness_ranking,
            "model_vs_brightness_lift": ranking_lift(model_ranking, brightness_ranking),
            "model_fixed_threshold_0p5": fixed_eval,
            "model_calibrated_threshold": calibrated_eval,
            "model_calibration": {
                "method": "maximize_f1_on_calibration_split",
                "threshold_grid_size": int(threshold_grid_size),
                "best": calibration_best,
            },
        }
    return {
        "method": "rank raw model probability and raw DNB brightness against AIS presence dilated by a Gaussian radius target",
        "sigmas": [float(value) for value in sigmas],
        "target_threshold": float(target_threshold),
        "threshold_grid_size": int(threshold_grid_size),
        "truncate": float(truncate),
        "by_sigma": rows,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    run_dir = args.run_dir.expanduser().resolve()
    summary_path = run_dir / "run_summary.json"
    summary = read_json(summary_path) if summary_path.exists() else {}
    checkpoint_path = (
        checkpoint_path_from_summary(summary, str(args.checkpoint), args.checkpoint_path)
        if summary
        else checkpoint_path_from_run_dir(run_dir, str(args.checkpoint), args.checkpoint_path)
    )
    config_path = run_dir / "config_snapshot.json"
    if config_path.exists():
        config = read_json(config_path)
    elif summary:
        config = read_json(Path(str(summary["config_path"])))
    else:
        raise FileNotFoundError(f"Missing config_snapshot.json and run_summary.json in {run_dir}")
    device = resolve_required_device(str(args.device))
    model, loss_fn, config = load_model_and_loss(checkpoint_path, config, device)
    patch_source = "cache"
    cache_dir: Path | None = None
    if args.patch_cache_dir is not None:
        cache_dir = args.patch_cache_dir.expanduser().resolve()
    elif summary:
        try:
            cache_dir = patch_cache_dir_from_summary(summary)
        except KeyError:
            cache_dir = None
    if cache_dir is not None:
        eval_patches, eval_cache_metadata = load_patch_split_cache(cache_dir, str(args.split))
        calibration_patches, calibration_cache_metadata = load_patch_split_cache(cache_dir, str(args.calibration_split))
    elif summary:
        patch_source = "rebuilt_no_cache"
        eval_patches, eval_cache_metadata = rebuild_split_patches_from_run(run_dir, config, str(args.split))
        calibration_patches, calibration_cache_metadata = rebuild_split_patches_from_run(
            run_dir,
            config,
            str(args.calibration_split),
        )
    else:
        raise KeyError("Interrupted runs without run_summary.json require --patch-cache-dir.")
    batch_size = int(args.batch_size if args.batch_size is not None else summary.get("batch_size", 4))
    size_divisor = int(config.get("patching", {}).get("size_divisor", 16))
    input_channels = input_channels_from_config(config)

    eval_metrics = run_batches(
        model=model,
        loss_fn=loss_fn,
        patches=eval_patches,
        batch_size=batch_size,
        size_divisor=size_divisor,
        device=device,
        train=False,
        num_workers=int(args.num_workers),
        input_channels=input_channels,
    )

    presence_calibration_model, _presence_calibration_brightness, presence_calibration_targets = collect_presence_scores(
        model=model,
        loss_fn=loss_fn,
        patches=calibration_patches,
        batch_size=batch_size,
        size_divisor=size_divisor,
        device=device,
        num_workers=int(args.num_workers),
        input_channels=input_channels,
    )
    presence_eval_model, presence_eval_brightness, presence_eval_targets = collect_presence_scores(
        model=model,
        loss_fn=loss_fn,
        patches=eval_patches,
        batch_size=batch_size,
        size_divisor=size_divisor,
        device=device,
        num_workers=int(args.num_workers),
        input_channels=input_channels,
    )
    presence_calibration_best = best_threshold_by_f1(
        presence_calibration_model,
        presence_calibration_targets,
        int(args.threshold_grid_size),
    )
    presence_calibrated_eval = (
        threshold_metrics(presence_eval_model, presence_eval_targets, float(presence_calibration_best["threshold"]))
        if presence_calibration_best
        else {}
    )
    presence_fixed_eval = threshold_metrics(presence_eval_model, presence_eval_targets, 0.5) if presence_eval_model.size else {}
    model_ranking = ranking_metrics(presence_eval_model, presence_eval_targets, include_brier=True)
    brightness_ranking = ranking_metrics(presence_eval_brightness, presence_eval_targets, include_brier=False)
    presence_probability = {
        "target": {
            "definition": "pixel target >= pixel_metric_target_threshold within valid sea mask",
            "pixel_metric_target_threshold": float(getattr(loss_fn, "pixel_metric_target_threshold", 0.5)),
        },
        "model_probability": model_ranking,
        "brightness_baseline": brightness_ranking,
        "model_vs_brightness_lift": ranking_lift(model_ranking, brightness_ranking),
        "model_fixed_threshold_0p5": presence_fixed_eval,
        "model_calibrated_threshold": presence_calibrated_eval,
        "model_calibration": {
            "method": "maximize_f1_on_calibration_split",
            "threshold_grid_size": int(args.threshold_grid_size),
            "best": presence_calibration_best,
            "reliability_bins": calibration_bins(presence_eval_model, presence_eval_targets, bin_count=10),
        },
    }
    radius_sigmas = parse_float_list(str(args.radius_sigmas))
    radius_presence = (
        evaluate_radius_probability_sweep(
            model=model,
            loss_fn=loss_fn,
            calibration_patches=calibration_patches,
            eval_patches=eval_patches,
            batch_size=batch_size,
            size_divisor=size_divisor,
            device=device,
            num_workers=int(args.num_workers),
            input_channels=input_channels,
            sigmas=radius_sigmas,
            target_threshold=float(args.radius_target_threshold),
            threshold_grid_size=int(args.threshold_grid_size),
            truncate=float(args.radius_truncate),
        )
        if radius_sigmas
        else {}
    )

    result = {
        "schema_version": 3,
        "kind": "density_checkpoint_evaluation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "checkpoint_selection": str(args.checkpoint),
        "checkpoint_path": str(checkpoint_path),
        "config_hash": stable_json_hash(config),
        "device": str(device),
        "batch_size": batch_size,
        "split": str(args.split),
        "calibration_split": str(args.calibration_split),
        "patch_source": patch_source,
        "patch_cache_dir": None if cache_dir is None else str(cache_dir),
        "patch_cache_metadata": {
            str(args.split): {key: value for key, value in eval_cache_metadata.items() if key != "patches"},
            str(args.calibration_split): {key: value for key, value in calibration_cache_metadata.items() if key != "patches"},
        },
        "train_style_fixed_threshold_0p5": eval_metrics,
        "presence_probability": presence_probability,
        "radius_presence": radius_presence,
    }
    output_json = args.output_json
    if output_json is None:
        output_json = run_dir / "evaluations" / f"{args.checkpoint}_{args.split}_eval.json"
    output_json = output_json.expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_json": str(output_json),
                "train_style_fixed_threshold_0p5": eval_metrics,
                "presence_probability": presence_probability,
                "radius_presence": radius_presence,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
