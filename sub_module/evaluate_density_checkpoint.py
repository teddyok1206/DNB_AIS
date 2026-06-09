from __future__ import annotations

import argparse
import json
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt, gaussian_filter

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


def collect_occupancy_scores(
    *,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    patches: list[Any],
    batch_size: int,
    size_divisor: int,
    device: torch.device,
    num_workers: int,
    input_channels: list[str] | None,
) -> tuple[np.ndarray, np.ndarray]:
    if not hasattr(loss_fn, "occupancy_from_output"):
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=bool)
    loader = make_loader(
        patches,
        batch_size,
        size_divisor,
        shuffle=False,
        num_workers=num_workers,
        input_channels=input_channels,
    )
    probs: list[float] = []
    targets: list[bool] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = move_density_batch_to_device(batch, device)
            output = model(batch["x"])
            occ_prob, occ_target = loss_fn.occupancy_from_output(output, batch)
            probs.extend(float(v) for v in occ_prob.detach().cpu().tolist())
            targets.extend(bool(v > 0.5) for v in occ_target.detach().cpu().tolist())
    return np.asarray(probs, dtype=np.float64), np.asarray(targets, dtype=bool)


def collect_pixel_scores(
    *,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    patches: list[Any],
    batch_size: int,
    size_divisor: int,
    device: torch.device,
    num_workers: int,
    input_channels: list[str] | None,
) -> tuple[np.ndarray, np.ndarray]:
    if not hasattr(loss_fn, "pixel_occupancy_from_output"):
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=bool)
    loader = make_loader(
        patches,
        batch_size,
        size_divisor,
        shuffle=False,
        num_workers=num_workers,
        input_channels=input_channels,
    )
    probs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = move_density_batch_to_device(batch, device)
            output = model(batch["x"])
            pixel_prob, pixel_target, pixel_valid = loss_fn.pixel_occupancy_from_output(output, batch)
            target_threshold = float(getattr(loss_fn, "pixel_metric_target_threshold", 0.5))
            valid = (pixel_valid.detach() > 0).cpu().numpy().reshape(-1)
            probs.append(pixel_prob.detach().cpu().numpy().reshape(-1)[valid].astype(np.float32, copy=False))
            targets.append((pixel_target.detach().cpu().numpy().reshape(-1)[valid] >= target_threshold))
    if not probs:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=bool)
    return np.concatenate(probs).astype(np.float64, copy=False), np.concatenate(targets).astype(bool, copy=False)


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


def gaussian_smooth_probability(prob: np.ndarray, valid: np.ndarray, *, sigma: float, truncate: float, eps: float = 1.0e-8) -> np.ndarray:
    prob = np.asarray(prob, dtype=np.float32)
    valid = (np.asarray(valid, dtype=np.float32) > 0).astype(np.float32)
    if float(sigma) <= 0.0:
        return np.clip(prob * valid, 0.0, 1.0).astype(np.float32, copy=False)
    numerator = gaussian_filter(prob * valid, sigma=float(sigma), mode="constant", cval=0.0, truncate=float(truncate))
    denominator = gaussian_filter(valid, sigma=float(sigma), mode="constant", cval=0.0, truncate=float(truncate))
    smoothed = np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=np.float32), where=denominator > float(eps))
    return np.clip(smoothed * valid, 0.0, 1.0).astype(np.float32, copy=False)


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


def soft_probability_metrics(pred: np.ndarray, target: np.ndarray, target_binary: np.ndarray) -> dict[str, Any]:
    if pred.size == 0:
        return {}
    eps = 1.0e-8
    pred = pred.astype(np.float64, copy=False)
    target = target.astype(np.float64, copy=False)
    overlap = np.minimum(pred, target)
    pred_sum = float(pred.sum())
    target_sum = float(target.sum())
    return {
        "soft_brier": float(np.mean((pred - target) ** 2)),
        "soft_mae": float(np.mean(np.abs(pred - target))),
        "soft_target_sum": target_sum,
        "soft_pred_sum": pred_sum,
        "soft_target_explained": float(overlap.sum() / max(target_sum, eps)),
        "soft_pred_matched": float(overlap.sum() / max(pred_sum, eps)),
        "average_precision": average_precision_score(pred.astype(np.float32, copy=False), target_binary.astype(bool, copy=False)),
        "target_positive_count": int(target_binary.sum()),
        "target_negative_count": int((~target_binary).sum()),
        "sample_count": int(pred.size),
    }


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
        empty_scores = np.asarray([], dtype=np.float32)
        empty_targets = np.asarray([], dtype=np.float32)
        empty_binary = np.asarray([], dtype=bool)
        return empty_scores, empty_targets, empty_binary
    loader = make_loader(
        patches,
        batch_size,
        size_divisor,
        shuffle=False,
        num_workers=num_workers,
        input_channels=input_channels,
    )
    pred_values: list[np.ndarray] = []
    target_values: list[np.ndarray] = []
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
            for idx, meta in enumerate(batch["metadata"]):
                height, width = [int(v) for v in meta["shape"]]
                prob_arr = probs_np[idx, 0, :height, :width]
                target_arr = targets_np[idx, 0, :height, :width]
                valid_arr = valid_np[idx, 0, :height, :width]
                mask = valid_arr > 0
                if not bool(np.any(mask)):
                    continue
                radius_pred = gaussian_smooth_probability(
                    prob_arr,
                    valid_arr,
                    sigma=float(sigma),
                    truncate=float(truncate),
                )
                radius_target = gaussian_radius_target(
                    target_arr,
                    valid_arr,
                    sigma=float(sigma),
                    radius_pixels=int(radius_pixels),
                )
                pred_values.append(radius_pred[mask].astype(np.float32, copy=False))
                target_flat = radius_target[mask].astype(np.float32, copy=False)
                target_values.append(target_flat)
                binary_values.append(target_flat >= float(target_threshold))
    if not pred_values:
        return np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32), np.asarray([], dtype=bool)
    return (
        np.concatenate(pred_values).astype(np.float32, copy=False),
        np.concatenate(target_values).astype(np.float32, copy=False),
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
        calibration_pred, calibration_target_soft, calibration_target_binary = collect_radius_scores(
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
        eval_pred, eval_target_soft, eval_target_binary = collect_radius_scores(
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
        calibration_best = best_threshold_by_f1(calibration_pred, calibration_target_binary, int(threshold_grid_size))
        calibrated_eval = (
            threshold_metrics(eval_pred, eval_target_binary, float(calibration_best["threshold"]))
            if calibration_best
            else {}
        )
        fixed_eval = threshold_metrics(eval_pred, eval_target_binary, 0.5) if eval_pred.size else {}
        label = f"sigma_{str(float(sigma)).replace('.', 'p')}"
        rows[label] = {
            "sigma_pixels": float(sigma),
            "radius_pixels": int(radius_pixels),
            "target_threshold": float(target_threshold),
            "truncate": float(truncate),
            "calibration_soft": soft_probability_metrics(calibration_pred, calibration_target_soft, calibration_target_binary),
            "calibration_best_threshold": calibration_best,
            "eval_soft": soft_probability_metrics(eval_pred, eval_target_soft, eval_target_binary),
            "eval_fixed_threshold_0p5": fixed_eval,
            "eval_calibrated_threshold": calibrated_eval,
        }
    return {
        "method": "smooth sigmoid probability with masked Gaussian; compare to distance-transform Gaussian radius target",
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

    calibration_probs, calibration_targets = collect_occupancy_scores(
        model=model,
        loss_fn=loss_fn,
        patches=calibration_patches,
        batch_size=batch_size,
        size_divisor=size_divisor,
        device=device,
        num_workers=int(args.num_workers),
        input_channels=input_channels,
    )
    eval_probs, eval_targets = collect_occupancy_scores(
        model=model,
        loss_fn=loss_fn,
        patches=eval_patches,
        batch_size=batch_size,
        size_divisor=size_divisor,
        device=device,
        num_workers=int(args.num_workers),
        input_channels=input_channels,
    )
    calibration_best = best_threshold_by_f1(calibration_probs, calibration_targets, int(args.threshold_grid_size))
    calibrated_eval = (
        threshold_metrics(eval_probs, eval_targets, float(calibration_best["threshold"]))
        if calibration_best
        else {}
    )
    fixed_eval = threshold_metrics(eval_probs, eval_targets, 0.5) if eval_probs.size else {}

    pixel_calibration_probs, pixel_calibration_targets = collect_pixel_scores(
        model=model,
        loss_fn=loss_fn,
        patches=calibration_patches,
        batch_size=batch_size,
        size_divisor=size_divisor,
        device=device,
        num_workers=int(args.num_workers),
        input_channels=input_channels,
    )
    pixel_eval_probs, pixel_eval_targets = collect_pixel_scores(
        model=model,
        loss_fn=loss_fn,
        patches=eval_patches,
        batch_size=batch_size,
        size_divisor=size_divisor,
        device=device,
        num_workers=int(args.num_workers),
        input_channels=input_channels,
    )
    pixel_calibration_best = best_threshold_by_f1(
        pixel_calibration_probs,
        pixel_calibration_targets,
        int(args.threshold_grid_size),
    )
    pixel_calibrated_eval = (
        threshold_metrics(pixel_eval_probs, pixel_eval_targets, float(pixel_calibration_best["threshold"]))
        if pixel_calibration_best
        else {}
    )
    pixel_fixed_eval = threshold_metrics(pixel_eval_probs, pixel_eval_targets, 0.5) if pixel_eval_probs.size else {}
    radius_sigmas = parse_float_list(str(args.radius_sigmas))
    radius_probability = (
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
        "schema_version": 2,
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
        "metrics_fixed_threshold_0p5": eval_metrics,
        "occupancy_fixed_threshold_0p5": fixed_eval,
        "pixel_fixed_threshold_0p5": pixel_fixed_eval,
        "calibration": {
            "method": "maximize_f1_on_calibration_split",
            "threshold_grid_size": int(args.threshold_grid_size),
            "best": calibration_best,
        },
        "metrics_calibrated_threshold": calibrated_eval,
        "pixel_calibration": {
            "method": "maximize_f1_on_calibration_split",
            "threshold_grid_size": int(args.threshold_grid_size),
            "best": pixel_calibration_best,
        },
        "pixel_metrics_calibrated_threshold": pixel_calibrated_eval,
        "radius_probability": radius_probability,
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
                "fixed": eval_metrics,
                "calibrated": calibrated_eval,
                "calibration": calibration_best,
                "pixel_fixed": pixel_fixed_eval,
                "pixel_calibrated": pixel_calibrated_eval,
                "pixel_calibration": pixel_calibration_best,
                "radius_probability": radius_probability,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
