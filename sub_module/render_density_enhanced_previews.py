from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .dnb_density_common import density_patch_collate, move_density_batch_to_device
from .dnb_density_models import build_density_model
from .run_density_split_smoke_train import (
    SceneSplitRecord,
    build_scene,
    infer_density_from_output,
    input_channels_from_config,
    read_json,
)
from .run_density_smoke import DEFAULT_METADATA, DEFAULT_SHIPS_DB, STEP3
from .dnb_pipeline_core import GroundTruthResolver


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render enhanced density-map explanation previews for a completed run.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--checkpoint-kind", choices=["last", "best_val_loss", "best_val_count_ratio"], default="last")
    parser.add_argument("--checkpoint-path", type=Path, default=None, help="Explicit checkpoint override.")
    return parser


def read_filtered_records(path: Path, split: str) -> list[SceneSplitRecord]:
    records: list[SceneSplitRecord] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if str(row["split"]) != split:
                continue
            records.append(
                SceneSplitRecord(
                    split=str(row["split"]),
                    day_key=str(row["day_key"]),
                    scene_key=str(row["scene_key"]),
                    tif_path=Path(row["tif_path"]).expanduser(),
                    geojson_path=Path(row["geojson_path"]).expanduser(),
                )
            )
    return records


def namespace_from_summary(summary: dict[str, Any]) -> argparse.Namespace:
    filters = summary.get("smoke_filters", {})
    return argparse.Namespace(
        max_patches_per_scene=int(filters.get("max_patches_per_scene", 16)),
        max_ph_patches_per_scene=int(filters.get("max_ph_patches_per_scene", 0)),
        max_fallback_patches_per_scene=int(filters.get("max_fallback_patches_per_scene", 0)),
        max_patch_height=int(filters.get("max_patch_height", 512)),
        max_patch_width=int(filters.get("max_patch_width", 512)),
        skip_ph_anchor_zero=bool(filters.get("skip_ph_anchor_zero", True)),
    )


def build_model_from_checkpoint(config: dict[str, Any], checkpoint: dict[str, Any], device: torch.device) -> torch.nn.Module:
    model_config = dict(config.get("model", {}))
    name = str(model_config.pop("name", "MaskedDensityUNet"))
    model_config.pop("out_channels", None)
    model = build_density_model(name, out_channels=1, **model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def load_run_config(run_dir: Path, summary: dict[str, Any], checkpoint: dict[str, Any]) -> tuple[dict[str, Any], str]:
    if isinstance(checkpoint.get("config"), dict):
        return dict(checkpoint["config"]), "checkpoint.config"
    snapshot = summary.get("outputs", {}).get("config_snapshot")
    if snapshot:
        snapshot_path = Path(snapshot).expanduser()
        if snapshot_path.exists():
            return read_json(snapshot_path), str(snapshot_path)
    fallback_snapshot = run_dir / "config_snapshot.json"
    if fallback_snapshot.exists():
        return read_json(fallback_snapshot), str(fallback_snapshot)
    return read_json(Path(summary["config_path"])), str(summary["config_path"])


def resolve_checkpoint_path(run_dir: Path, summary: dict[str, Any], *, checkpoint_kind: str, checkpoint_path: Path | None) -> Path:
    if checkpoint_path is not None:
        return checkpoint_path.expanduser().resolve()
    outputs = summary.get("outputs", {})
    if checkpoint_kind == "last":
        key = "checkpoint_last"
    else:
        key = f"checkpoint_{checkpoint_kind}"
    value = outputs.get(key)
    if value:
        return Path(value).expanduser().resolve()
    if checkpoint_kind != "last":
        best = summary.get("best_checkpoints", {}).get(checkpoint_kind, {})
        if best.get("path"):
            return Path(best["path"]).expanduser().resolve()
    fallback = outputs.get("checkpoint_last")
    if fallback:
        return Path(fallback).expanduser().resolve()
    return (run_dir / "checkpoints" / "checkpoint_last.pt").resolve()


def robust_max(arrays: list[np.ndarray], eps: float = 1.0e-8) -> float:
    values = [float(np.nanpercentile(np.asarray(arr, dtype=np.float32), 99.5)) for arr in arrays if np.asarray(arr).size]
    return max(max(values) if values else 1.0, eps)


def normalized_density(arr: np.ndarray, valid: np.ndarray, eps: float = 1.0e-8) -> np.ndarray:
    masked = np.asarray(arr, dtype=np.float32) * np.asarray(valid, dtype=np.float32)
    total = float(masked.sum())
    if total <= eps:
        return np.zeros_like(masked, dtype=np.float32)
    return masked / total


def save_enhanced_preview(
    path: Path,
    *,
    brightness: np.ndarray,
    attention: np.ndarray,
    target: np.ndarray,
    pred: np.ndarray,
    valid: np.ndarray,
    title: str,
) -> dict[str, float]:
    path.parent.mkdir(parents=True, exist_ok=True)
    valid_bool = np.asarray(valid) > 0
    target = np.where(valid_bool, target, 0.0).astype(np.float32, copy=False)
    pred = np.where(valid_bool, pred, 0.0).astype(np.float32, copy=False)
    overlap = np.minimum(pred, target)
    abs_error = np.abs(pred - target)
    signed_error = pred - target
    target_explained_fraction = np.divide(
        overlap,
        np.maximum(target, 1.0e-8),
        out=np.zeros_like(target, dtype=np.float32),
        where=target > 1.0e-8,
    )
    target_prob = normalized_density(target, valid)
    pred_prob = normalized_density(pred, valid)
    spatial_overlap = np.minimum(target_prob, pred_prob)

    target_sum = float(target.sum())
    pred_sum = float(pred.sum())
    overlap_sum = float(overlap.sum())
    spatial_overlap_sum = float(spatial_overlap.sum())
    target_explained = overlap_sum / max(target_sum, 1.0e-8)
    pred_matched = overlap_sum / max(pred_sum, 1.0e-8)

    density_vmax = robust_max([target, pred])
    error_vmax = max(float(np.nanpercentile(np.abs(signed_error), 99.5)), 1.0e-8)
    overlap_vmax = max(float(np.nanpercentile(overlap, 99.5)), 1.0e-8)
    prob_vmax = robust_max([target_prob, pred_prob, spatial_overlap])

    panels = [
        ("brightness", brightness, "magma", None, None),
        ("soft attention", attention, "viridis", 0.0, 1.0),
        ("target density", target, "viridis", 0.0, density_vmax),
        ("pred density", pred, "viridis", 0.0, density_vmax),
        ("abs error", abs_error, "inferno", 0.0, error_vmax),
        ("signed error\nblue=under red=over", signed_error, "coolwarm", -error_vmax, error_vmax),
        ("explained target mass\nmin(pred,target)", overlap, "Greens", 0.0, overlap_vmax),
        ("target explained fraction\nmin(pred,target)/target", target_explained_fraction, "Greens", 0.0, 1.0),
        ("normalized spatial overlap\nmin(pred/sum,target/sum)", spatial_overlap, "Greens", 0.0, prob_vmax),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(15, 12), constrained_layout=True)
    for ax, (panel_title, arr, cmap, vmin, vmax) in zip(axes.ravel(), panels):
        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(panel_title, fontsize=10)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        (
            f"{title} | pred_sum={pred_sum:.2f} target_sum={target_sum:.2f} "
            f"| target_explained={target_explained:.3f} pred_matched={pred_matched:.3f} "
            f"| spatial_overlap={spatial_overlap_sum:.3f}"
        ),
        fontsize=12,
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {
        "pred_sum": pred_sum,
        "target_sum": target_sum,
        "target_explained": float(target_explained),
        "pred_matched": float(pred_matched),
        "spatial_overlap": float(spatial_overlap_sum),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    run_dir = args.run_dir.expanduser().resolve()
    summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    checkpoint_path = resolve_checkpoint_path(
        run_dir,
        summary,
        checkpoint_kind=str(args.checkpoint_kind),
        checkpoint_path=args.checkpoint_path,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config, config_source = load_run_config(run_dir, summary, checkpoint)
    device = torch.device(args.device)
    model = build_model_from_checkpoint(config, checkpoint, device)

    filtered_split = run_dir / "filtered_scene_split.csv"
    records = read_filtered_records(filtered_split, str(args.split))
    resolver = GroundTruthResolver(DEFAULT_METADATA, DEFAULT_SHIPS_DB, STEP3 / "bboxes_JPSS-2")
    ns = namespace_from_summary(summary)
    patches = []
    for record in records:
        result = build_scene(record, config=config, args=ns, resolver=resolver)
        if not result.excluded_reason:
            patches.extend(result.patches)
    patches = patches[: max(int(args.limit), 0)]

    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else run_dir / "enhanced_previews"
    size_divisor = int(config.get("patching", {}).get("size_divisor", 16))
    input_channels = input_channels_from_config(config)
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for idx, patch in enumerate(patches):
            batch = density_patch_collate([patch], size_divisor=size_divisor, input_channels=input_channels)
            batch = move_density_batch_to_device(batch, device)
            output = model(batch["x"])
            pred = infer_density_from_output(output, batch).detach().cpu().numpy()[0, 0]
            x = batch["x"].detach().cpu().numpy()[0]
            target = batch["target"].detach().cpu().numpy()[0, 0]
            valid = batch["valid_mask"].detach().cpu().numpy()[0, 0]
            attention = batch["soft_attention"].detach().cpu().numpy()[0, 0]
            height, width = patch.shape
            meta = batch["metadata"][0]
            title = f"{args.split} | {meta['partition_kind']} | scene-partition {meta.get('partition_id')}"
            out_path = output_dir / f"enhanced_preview_{idx:03d}_{meta['partition_kind']}_{meta.get('partition_id')}.png"
            metrics = save_enhanced_preview(
                out_path,
                brightness=x[0, :height, :width],
                attention=attention[:height, :width],
                target=target[:height, :width],
                pred=pred[:height, :width],
                valid=valid[:height, :width],
                title=title,
            )
            rows.append(
                {
                    "index": idx,
                    "path": str(out_path),
                    "partition_kind": str(meta["partition_kind"]),
                    "partition_id": meta.get("partition_id"),
                    "config_source": config_source,
                    "checkpoint_path": str(checkpoint_path),
                    "checkpoint_kind": str(args.checkpoint_kind),
                    **metrics,
                }
            )

    metrics_path = output_dir / "enhanced_preview_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "index",
            "path",
            "partition_kind",
            "partition_id",
            "config_source",
            "checkpoint_path",
            "checkpoint_kind",
            "pred_sum",
            "target_sum",
            "target_explained",
            "pred_matched",
            "spatial_overlap",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps({"output_dir": str(output_dir), "preview_count": len(rows), "metrics_csv": str(metrics_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
