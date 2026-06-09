from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .dnb_density_common import density_patch_collate, move_density_batch_to_device
from .dnb_density_losses import build_density_loss
from .dnb_density_patch_pickle_cache import load_patch_split_cache
from .run_density_split_smoke_train import (
    build_model_from_config,
    input_channels_from_config,
    loss_config_from_config,
    make_loader,
    read_json,
    resolve_required_device,
    run_batches,
    stable_json_hash,
)


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
            valid = (pixel_valid.detach() > 0).cpu().numpy().reshape(-1)
            probs.append(pixel_prob.detach().cpu().numpy().reshape(-1)[valid].astype(np.float32, copy=False))
            targets.append((pixel_target.detach().cpu().numpy().reshape(-1)[valid] > 0.5))
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
    if args.patch_cache_dir is not None:
        cache_dir = args.patch_cache_dir.expanduser().resolve()
    elif summary:
        cache_dir = patch_cache_dir_from_summary(summary)
    else:
        raise KeyError("Interrupted runs without run_summary.json require --patch-cache-dir.")
    eval_patches, eval_cache_metadata = load_patch_split_cache(cache_dir, str(args.split))
    calibration_patches, calibration_cache_metadata = load_patch_split_cache(cache_dir, str(args.calibration_split))
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

    result = {
        "schema_version": 1,
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
        "patch_cache_dir": str(cache_dir),
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
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
