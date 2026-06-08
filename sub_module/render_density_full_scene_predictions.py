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

from .dnb_density_common import DensityPatch, density_patch_collate, move_density_batch_to_device
from .dnb_density_losses import build_density_loss
from .dnb_density_models import build_density_model
from .dnb_ph_downsample import PHDownsampleConfig, build_ph_anchor_store
from .dnb_pipeline_core import GroundTruthResolver, SceneRaster
from .dnb_scene_partition import build_partitioned_density_patches
from .kr_sea_mask import apply_kr_sea_mask
from .run_density_smoke import DEFAULT_METADATA, DEFAULT_SHIPS_DB, STEP3
from .run_density_split_smoke_train import (
    SceneSplitRecord,
    detector_config_from_config,
    infer_density_from_output,
    input_channels_from_config,
    loss_config_from_config,
    partition_config_from_config,
    patch_config_from_config,
    read_json,
    read_scene_split,
    target_density_for_metrics,
    target_config_from_config,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge patch-level density predictions back into full-scene canvases.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Completed density run directory containing run_summary.json and checkpoint_last.pt.")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--scene-key", default=None, help="Optional exact scene key, e.g. A2025001_1754_021.")
    parser.add_argument("--scene-split-csv", type=Path, default=None, help="Override scene split CSV. Defaults to filtered_scene_split.csv when present.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--metadata-csv", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--ships-db", type=Path, default=DEFAULT_SHIPS_DB)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--checkpoint-kind", choices=["last", "best_val_loss", "best_val_count_ratio", "best_val_occupancy_f1"], default="last")
    parser.add_argument("--checkpoint-path", type=Path, default=None, help="Explicit checkpoint override.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--limit-scenes", type=int, default=1)
    parser.add_argument("--max-patch-height", type=int, default=0, help="Skip larger patches when >0. Default keeps all patches.")
    parser.add_argument("--max-patch-width", type=int, default=0, help="Skip larger patches when >0. Default keeps all patches.")
    parser.add_argument("--save-npz", action=argparse.BooleanOptionalAction, default=False)
    return parser


def build_model_from_checkpoint(config: dict[str, Any], checkpoint: dict[str, Any], device: torch.device) -> torch.nn.Module:
    model_config = dict(config.get("model", {}))
    name = str(model_config.pop("name", "MaskedDensityUNet"))
    model_config.pop("out_channels", None)
    model = build_density_model(name, out_channels=1, **model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def read_records(path: Path, *, split: str, scene_key: str | None, limit: int) -> list[SceneSplitRecord]:
    records = [record for record in read_scene_split(path) if record.split == split]
    if scene_key:
        records = [record for record in records if record.scene_key == scene_key]
    if int(limit) > 0:
        records = records[: int(limit)]
    return records


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


def build_all_scene_patches(
    record: SceneSplitRecord,
    *,
    config: dict[str, Any],
    resolver: GroundTruthResolver,
) -> tuple[SceneRaster, np.ndarray, np.ndarray, list[DensityPatch], dict[str, Any]]:
    sea_mask = config.get("sea_mask", {})
    mask_result = apply_kr_sea_mask(
        record.tif_path,
        step3_dir=STEP3,
        crop_to_bounds=bool(sea_mask.get("crop_to_bounds", True)),
        segment_policy=str(sea_mask.get("segment_policy", "single_scene")),
        write_masked_tif=False,
        all_touched=bool(sea_mask.get("all_touched", True)),
    )
    scene = mask_result.scene
    valid_mask = mask_result.valid_mask
    gt_path = resolver.resolve_geojson(scene, record.geojson_path)
    gt_points = resolver.load_points(gt_path)
    gt_count_map = resolver.rasterize_counts(scene, gt_points)

    ph_cfg = config.get("ph_downsample", {})
    ph_anchor_result = build_ph_anchor_store(
        scene,
        gt_count_map,
        detector_config_from_config(config),
        valid_mask=valid_mask,
        downsample_config=PHDownsampleConfig(
            factor=int(ph_cfg.get("factor", 4)),
            reducer=str(ph_cfg.get("reducer", "max")),
        ),
    )
    patches, _, partition_summary = build_partitioned_density_patches(
        scene,
        gt_count_map,
        ph_anchor_result.store,
        valid_mask=valid_mask,
        patch_config=patch_config_from_config(config, max_patches=0),
        target_config=target_config_from_config(config),
        partition_config=partition_config_from_config(config),
    )
    metrics = {
        "scene_key": record.scene_key,
        "split": record.split,
        "tif_path": str(record.tif_path),
        "geojson_path": str(gt_path),
        "sea_mask": mask_result.metadata,
        "ph_downsample": ph_anchor_result.metadata,
        "partition_summary": partition_summary,
        "patch_count_total": len(patches),
    }
    return scene, valid_mask, gt_count_map, patches, metrics


def _robust_vmax(arrays: list[np.ndarray], percentile: float = 99.5, eps: float = 1.0e-8) -> float:
    values: list[float] = []
    for arr in arrays:
        data = np.asarray(arr, dtype=np.float32)
        finite = data[np.isfinite(data)]
        if finite.size:
            values.append(float(np.nanpercentile(finite, float(percentile))))
    return max(max(values) if values else 1.0, eps)


def _normalized(arr: np.ndarray, valid_mask: np.ndarray, eps: float = 1.0e-8) -> np.ndarray:
    masked = np.asarray(arr, dtype=np.float32) * (np.asarray(valid_mask) > 0)
    total = float(masked.sum())
    if total <= eps:
        return np.zeros_like(masked, dtype=np.float32)
    return masked / total


def merge_scene_predictions(
    *,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    patches: list[DensityPatch],
    scene_shape: tuple[int, int],
    device: torch.device,
    batch_size: int,
    size_divisor: int,
    input_channels: list[str],
    max_patch_height: int = 0,
    max_patch_width: int = 0,
) -> dict[str, Any]:
    pred_canvas = np.zeros(scene_shape, dtype=np.float32)
    target_canvas = np.zeros(scene_shape, dtype=np.float32)
    raw_canvas = np.zeros(scene_shape, dtype=np.float32)
    owner_coverage = np.zeros(scene_shape, dtype=np.uint16)

    kept: list[DensityPatch] = []
    skipped: list[dict[str, Any]] = []
    for patch in patches:
        height, width = patch.shape
        too_tall = int(max_patch_height) > 0 and int(height) > int(max_patch_height)
        too_wide = int(max_patch_width) > 0 and int(width) > int(max_patch_width)
        if too_tall or too_wide:
            skipped.append(
                {
                    "partition_id": patch.partition_id,
                    "partition_kind": patch.partition_kind,
                    "shape": [int(height), int(width)],
                    "crop_rc": list(patch.crop_rc),
                }
            )
            continue
        kept.append(patch)

    model.eval()
    with torch.no_grad():
        for start in range(0, len(kept), max(int(batch_size), 1)):
            batch_patches = kept[start : start + max(int(batch_size), 1)]
            batch = density_patch_collate(batch_patches, size_divisor=size_divisor, input_channels=input_channels)
            batch = move_density_batch_to_device(batch, device)
            output = model(batch["x"])
            pred_batch = infer_density_from_output(output, batch).detach().cpu().numpy()
            target_batch = target_density_for_metrics(loss_fn, batch).detach().cpu().numpy()
            for idx, patch in enumerate(batch_patches):
                height, width = patch.shape
                r0, r1, c0, c1 = [int(v) for v in patch.crop_rc]
                valid = np.asarray(patch.valid_mask, dtype=np.float32)[:height, :width] > 0
                pred_local = np.asarray(pred_batch[idx, 0, :height, :width], dtype=np.float32)
                target_local = np.asarray(target_batch[idx, 0, :height, :width], dtype=np.float32)
                raw_local = np.asarray(patch.raw_count, dtype=np.float32)[:height, :width]

                pred_view = pred_canvas[r0 : r1 + 1, c0 : c1 + 1]
                target_view = target_canvas[r0 : r1 + 1, c0 : c1 + 1]
                raw_view = raw_canvas[r0 : r1 + 1, c0 : c1 + 1]
                coverage_view = owner_coverage[r0 : r1 + 1, c0 : c1 + 1]

                pred_view[valid] += pred_local[valid]
                target_view[valid] += target_local[valid]
                raw_view[valid] += raw_local[valid]
                coverage_view[valid] += np.uint16(1)

    overlap = np.minimum(pred_canvas, target_canvas)
    pred_prob = _normalized(pred_canvas, owner_coverage)
    target_prob = _normalized(target_canvas, owner_coverage)
    spatial_overlap = np.minimum(pred_prob, target_prob)
    pred_sum = float(pred_canvas.sum())
    target_sum = float(target_canvas.sum())
    raw_sum = float(raw_canvas.sum())
    overlap_sum = float(overlap.sum())
    metrics = {
        "patch_count_input": int(len(patches)),
        "patch_count_kept": int(len(kept)),
        "patch_count_skipped": int(len(skipped)),
        "pred_sum": pred_sum,
        "target_sum": target_sum,
        "raw_sum": raw_sum,
        "pred_target_ratio": float(pred_sum / max(target_sum, 1.0e-8)),
        "target_explained": float(overlap_sum / max(target_sum, 1.0e-8)),
        "pred_matched": float(overlap_sum / max(pred_sum, 1.0e-8)),
        "spatial_overlap": float(spatial_overlap.sum()),
        "covered_pixels": int((owner_coverage > 0).sum()),
        "overlap_pixels": int((owner_coverage > 1).sum()),
        "max_owner_coverage": int(owner_coverage.max()) if owner_coverage.size else 0,
        "skipped_patches": skipped,
    }
    return {
        "pred": pred_canvas,
        "target": target_canvas,
        "raw": raw_canvas,
        "owner_coverage": owner_coverage,
        "metrics": metrics,
    }


def save_scene_png(
    path: Path,
    *,
    scene: SceneRaster,
    valid_mask: np.ndarray,
    merged: dict[str, Any],
    title: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pred = np.asarray(merged["pred"], dtype=np.float32)
    target = np.asarray(merged["target"], dtype=np.float32)
    raw = np.asarray(merged["raw"], dtype=np.float32)
    coverage = np.asarray(merged["owner_coverage"], dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=np.float32) > 0
    brightness = np.where(valid, np.asarray(scene.image, dtype=np.float32), 0.0)
    abs_error = np.abs(pred - target)
    signed_error = pred - target
    spatial_overlap = np.minimum(_normalized(pred, coverage), _normalized(target, coverage))

    density_vmax = _robust_vmax([target, pred])
    raw_vmax = _robust_vmax([raw])
    error_vmax = _robust_vmax([np.abs(signed_error)])
    overlap_vmax = _robust_vmax([spatial_overlap])
    brightness_vmax = _robust_vmax([brightness])

    panels = [
        ("brightness", brightness, "magma", 0.0, brightness_vmax),
        ("valid sea mask", valid.astype(np.float32), "gray", 0.0, 1.0),
        ("raw count", raw, "viridis", 0.0, raw_vmax),
        ("target density", target, "viridis", 0.0, density_vmax),
        ("pred density", pred, "viridis", 0.0, density_vmax),
        ("abs error", abs_error, "inferno", 0.0, error_vmax),
        ("signed error\nblue=under red=over", signed_error, "coolwarm", -error_vmax, error_vmax),
        ("owner coverage", coverage, "gray", 0.0, max(float(coverage.max()), 1.0)),
        ("normalized spatial overlap", spatial_overlap, "Greens", 0.0, overlap_vmax),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(15, 12), constrained_layout=True)
    for ax, (panel_title, arr, cmap, vmin, vmax) in zip(axes.ravel(), panels):
        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(panel_title, fontsize=10)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    m = merged["metrics"]
    fig.suptitle(
        (
            f"{title} | pred_sum={m['pred_sum']:.2f} target_sum={m['target_sum']:.2f} "
            f"ratio={m['pred_target_ratio']:.3f} spatial_overlap={m['spatial_overlap']:.3f}"
        ),
        fontsize=12,
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)


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
    device = torch.device(str(args.device))
    model = build_model_from_checkpoint(config, checkpoint, device)
    loss_fn = build_density_loss(loss_config_from_config(config)).to(device)
    input_channels = input_channels_from_config(config)
    size_divisor = int(config.get("patching", {}).get("size_divisor", 16))
    batch_size = int(args.batch_size if args.batch_size is not None else summary.get("batch_size", 2))

    split_csv = args.scene_split_csv
    if split_csv is None:
        filtered = run_dir / "filtered_scene_split.csv"
        split_csv = filtered if filtered.exists() else Path(summary["scene_split_csv"])
    records = read_records(split_csv.expanduser().resolve(), split=str(args.split), scene_key=args.scene_key, limit=int(args.limit_scenes))
    if not records:
        raise RuntimeError(f"No records found for split={args.split!r}, scene_key={args.scene_key!r} in {split_csv}")

    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else run_dir / "full_scene_predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    resolver = GroundTruthResolver(args.metadata_csv, args.ships_db, STEP3 / "bboxes_JPSS-2")

    metric_rows: list[dict[str, Any]] = []
    for record in records:
        print(f"[full-scene] build {record.split} {record.scene_key}")
        scene, valid_mask, _gt_count_map, patches, build_metrics = build_all_scene_patches(record, config=config, resolver=resolver)
        print(f"  [patches] {len(patches)}")
        merged = merge_scene_predictions(
            model=model,
            loss_fn=loss_fn,
            patches=patches,
            scene_shape=scene.shape,
            device=device,
            batch_size=batch_size,
            size_divisor=size_divisor,
            input_channels=input_channels,
            max_patch_height=int(args.max_patch_height),
            max_patch_width=int(args.max_patch_width),
        )
        scene_dir = output_dir / record.scene_key
        scene_dir.mkdir(parents=True, exist_ok=True)
        png_path = scene_dir / f"{record.scene_key}_full_scene_prediction.png"
        save_scene_png(png_path, scene=scene, valid_mask=valid_mask, merged=merged, title=f"{record.split} | {record.scene_key}")

        metrics = {
            "split": record.split,
            "scene_key": record.scene_key,
            "config_source": config_source,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_kind": str(args.checkpoint_kind),
            "png_path": str(png_path),
            **{k: v for k, v in merged["metrics"].items() if k != "skipped_patches"},
            "partition_ph_anchor_count": build_metrics["partition_summary"].get("ph_anchor_count"),
            "partition_ph_child_count": build_metrics["partition_summary"].get("ph_child_count"),
            "partition_fallback_grid_count": build_metrics["partition_summary"].get("fallback_grid_count"),
            "partition_coverage_ratio": build_metrics["partition_summary"].get("coverage_ratio"),
        }
        metric_rows.append(metrics)
        (scene_dir / f"{record.scene_key}_full_scene_metrics.json").write_text(
            json.dumps({"metrics": metrics, "build": build_metrics, "skipped_patches": merged["metrics"]["skipped_patches"]}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if bool(args.save_npz):
            np.savez_compressed(
                scene_dir / f"{record.scene_key}_full_scene_prediction.npz",
                pred=merged["pred"],
                target=merged["target"],
                raw=merged["raw"],
                owner_coverage=merged["owner_coverage"],
                valid_mask=np.asarray(valid_mask, dtype=np.uint8),
            )
        print(json.dumps(metrics, ensure_ascii=False, indent=2))

    metrics_csv = output_dir / "full_scene_prediction_metrics.csv"
    with metrics_csv.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(metric_rows[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metric_rows)

    print(json.dumps({"output_dir": str(output_dir), "scene_count": len(metric_rows), "metrics_csv": str(metrics_csv)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
