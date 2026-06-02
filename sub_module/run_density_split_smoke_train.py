from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from .dnb_candidate_detector import DnbCandidateDetectorConfig, candidate_store_summary
from .dnb_density_common import (
    DensityPatch,
    DensityPatchConfig,
    DensityPatchDataset,
    DensityTargetConfig,
    density_patch_collate,
    move_density_batch_to_device,
)
from .dnb_density_losses import build_density_loss
from .dnb_density_models import build_density_model
from .dnb_gat_pipeline import GroundTruthResolver, SceneRaster
from .dnb_ph_downsample import PHDownsampleConfig, build_ph_anchor_store
from .dnb_scene_partition import ScenePartitionConfig, build_partitioned_density_patches
from .kr_sea_mask import apply_kr_sea_mask
from .run_density_smoke import DEFAULT_METADATA, DEFAULT_SHIPS_DB, ROOT, STEP3, seed_everything


@dataclass(frozen=True)
class SceneSplitRecord:
    split: str
    day_key: str
    scene_key: str
    tif_path: Path
    geojson_path: Path


@dataclass
class SceneBuildResult:
    record: SceneSplitRecord
    patches: list[DensityPatch]
    metrics: dict[str, Any]
    excluded_reason: str | None = None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a short split-level U-Net train/inference smoke test.")
    parser.add_argument("--scene-split-csv", type=Path, default=STEP3 / "outputs" / "density_smoke_split_10_3_2" / "scene_split.csv")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "dnb_density_unet_main.json")
    parser.add_argument("--output-dir", type=Path, default=STEP3 / "outputs" / "density_split_smoke_train")
    parser.add_argument("--metadata-csv", type=Path, default=STEP3 / "metadata_JPSS-2.csv")
    parser.add_argument("--ships-db", type=Path, default=DEFAULT_SHIPS_DB)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=20260529)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--max-scenes-per-split", type=int, default=0)
    parser.add_argument("--max-patches-per-scene", type=int, default=16)
    parser.add_argument("--max-patch-height", type=int, default=512)
    parser.add_argument("--max-patch-width", type=int, default=512)
    parser.add_argument("--skip-ph-anchor-zero", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preview-patches", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.expanduser().read_text(encoding="utf-8"))


def read_scene_split(path: Path) -> list[SceneSplitRecord]:
    records: list[SceneSplitRecord] = []
    with path.expanduser().open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
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


def group_records(records: Iterable[SceneSplitRecord], max_per_split: int) -> dict[str, list[SceneSplitRecord]]:
    grouped = {"train": [], "val": [], "test": []}
    for record in records:
        if record.split in grouped:
            grouped[record.split].append(record)
    if int(max_per_split) > 0:
        grouped = {split: items[: int(max_per_split)] for split, items in grouped.items()}
    return grouped


def git_metadata() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        status = subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True)
        return {"git_commit": commit, "git_dirty": bool(status.strip()), "git_status_short": status.splitlines()}
    except Exception as exc:
        return {"git_error": str(exc)}


def resolve_required_device(requested: str) -> torch.device:
    normalized = str(requested).lower()
    if normalized == "auto":
        normalized = "mps"
    if normalized == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but torch.backends.mps.is_available() is False. Run outside the sandbox/escalated.")
        return torch.device("mps")
    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    if normalized == "cpu":
        raise RuntimeError("CPU execution is disabled for this split smoke runner. Use --device mps.")
    return torch.device(normalized)


def detector_config_from_config(config: dict[str, Any]) -> DnbCandidateDetectorConfig:
    detector = config.get("detector", {})
    return DnbCandidateDetectorConfig(
        detection_threshold=float(detector.get("detection_threshold", 1.0)),
        analysis_threshold=float(detector.get("analysis_threshold", 0.25)),
        threshold_reference=str(detector.get("threshold_reference", "median")),
        smooth_sigma=float(detector.get("smooth_sigma", 0.0)),
        lifetime_limit=float(detector.get("lifetime_limit", 0.0)),
        lifetime_limit_fraction=float(detector.get("lifetime_limit_fraction", 1.001)),
        area_limit=int(detector.get("area_limit", 0)),
        min_nodes=int(detector.get("min_nodes", 4)),
        max_nodes=int(detector.get("max_nodes", 2500)),
        max_candidates=detector.get("max_candidates"),
        connectivity=int(detector.get("connectivity", 1)),
        remove_edge=bool(detector.get("remove_edge", False)),
        drop_nested=bool(detector.get("drop_nested", False)),
    )


def patch_config_from_config(config: dict[str, Any], max_patches: int) -> DensityPatchConfig:
    patching = config.get("patching", {})
    return DensityPatchConfig(
        padding_pixels=int(patching.get("padding_pixels", 16)),
        size_divisor=int(patching.get("size_divisor", 16)),
        max_patches=max_patches if max_patches > 0 else None,
        sort_by=str(patching.get("sort_by", "node_count")),
        parent_min_nodes=int(patching.get("parent_min_nodes", 32)),
        parent_max_nodes=patching.get("parent_max_nodes"),
        child_min_nodes=int(patching.get("child_min_nodes", 4)),
        child_max_nodes=patching.get("child_max_nodes"),
        max_children=patching.get("max_children"),
        seed_radius_pixels=int(patching.get("seed_radius_pixels", 1)),
        attention_distance_sigma=float(patching.get("attention_distance_sigma", 4.0)),
        attention_base_weight=float(patching.get("attention_base_weight", 0.25)),
        attention_ph_weight=float(patching.get("attention_ph_weight", 0.75)),
    )


def target_config_from_config(config: dict[str, Any]) -> DensityTargetConfig:
    target = config.get("target", {})
    return DensityTargetConfig(
        kernel=str(target.get("kernel", "gaussian")),
        sigma_pixels=float(target.get("sigma_pixels", 1.5)),
        radius_pixels=int(target.get("radius_pixels", 5)),
        per_ship_mass=float(target.get("per_ship_mass", 1.0)),
        renormalize_after_roi_mask=bool(target.get("renormalize_after_roi_mask", False)),
        require_source_in_roi=bool(target.get("require_source_in_roi", False)),
    )


def partition_config_from_config(config: dict[str, Any]) -> ScenePartitionConfig:
    partition = config.get("partitioning", {})
    return ScenePartitionConfig(
        enabled=True,
        fallback_tile_pixels=int(partition.get("fallback_tile_pixels", 96)),
        halo_pixels=int(partition.get("halo_pixels", 16)),
        anchor_padding_pixels=int(partition.get("anchor_padding_pixels", 16)),
        min_owner_pixels=int(partition.get("min_owner_pixels", 1)),
        min_fallback_owner_pixels=int(partition.get("min_fallback_owner_pixels", 1)),
    )


def loss_config_from_config(config: dict[str, Any]) -> dict[str, Any]:
    loss = config.get("training", {}).get("loss", {})
    return dict(loss) if isinstance(loss, dict) else {"name": str(loss)}


def build_model_from_config(config: dict[str, Any]) -> torch.nn.Module:
    model_config = dict(config.get("model", {}))
    name = str(model_config.pop("name", "MaskedDensityUNet"))
    model_config.pop("out_channels", None)
    return build_density_model(name, out_channels=1, **model_config)


def select_smoke_patches(
    patches: list[DensityPatch],
    *,
    max_patches: int,
    max_height: int,
    max_width: int,
) -> tuple[list[DensityPatch], dict[str, Any]]:
    kept_size = [patch for patch in patches if patch.shape[0] <= int(max_height) and patch.shape[1] <= int(max_width)]
    dropped_too_large = len(patches) - len(kept_size)
    if int(max_patches) <= 0 or len(kept_size) <= int(max_patches):
        return kept_size, {"dropped_too_large": int(dropped_too_large), "dropped_by_cap": 0}

    def priority(patch: DensityPatch) -> tuple[int, float, int]:
        return (
            1 if patch.partition_kind == "ph_anchor" else 0,
            float(patch.raw_count_sum),
            int(patch.valid_pixels),
        )

    selected = sorted(kept_size, key=priority, reverse=True)[: int(max_patches)]
    return selected, {
        "dropped_too_large": int(dropped_too_large),
        "dropped_by_cap": int(len(kept_size) - len(selected)),
    }


def build_scene(
    record: SceneSplitRecord,
    *,
    config: dict[str, Any],
    args: argparse.Namespace,
    resolver: GroundTruthResolver,
) -> SceneBuildResult:
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
    patch_config = patch_config_from_config(config, max_patches=0)
    patches, _, partition_summary = build_partitioned_density_patches(
        scene,
        gt_count_map,
        ph_anchor_result.store,
        valid_mask=valid_mask,
        patch_config=patch_config,
        target_config=target_config_from_config(config),
        partition_config=partition_config_from_config(config),
    )
    selected, selection_metrics = select_smoke_patches(
        patches,
        max_patches=int(args.max_patches_per_scene),
        max_height=int(args.max_patch_height),
        max_width=int(args.max_patch_width),
    )
    metrics = {
        "split": record.split,
        "day_key": record.day_key,
        "scene_key": record.scene_key,
        "tif_path": str(record.tif_path),
        "geojson_path": str(gt_path),
        "sea_mask": mask_result.metadata,
        "ph_downsample": ph_anchor_result.metadata,
        "detector_summary": candidate_store_summary(ph_anchor_result.store),
        "partition_summary": partition_summary,
        "patch_count_total": int(len(patches)),
        "patch_count_selected": int(len(selected)),
        "selection": selection_metrics,
        "selected_target_sum": float(sum(patch.target_sum for patch in selected)),
        "selected_raw_sum": float(sum(patch.raw_count_sum for patch in selected)),
    }
    if bool(args.skip_ph_anchor_zero) and int(partition_summary.get("ph_anchor_count", 0)) <= 0:
        return SceneBuildResult(record=record, patches=[], metrics=metrics, excluded_reason="ph_anchor_count_zero")
    if not selected:
        return SceneBuildResult(record=record, patches=[], metrics=metrics, excluded_reason="no_selected_patches_after_smoke_filters")
    return SceneBuildResult(record=record, patches=selected, metrics=metrics)


def make_loader(patches: list[DensityPatch], batch_size: int, size_divisor: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        DensityPatchDataset(patches),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        collate_fn=lambda batch: density_patch_collate(batch, size_divisor=int(size_divisor)),
    )


def average_dicts(values: list[dict[str, float]]) -> dict[str, float]:
    if not values:
        return {}
    keys = sorted({key for item in values for key in item})
    return {key: float(np.mean([item[key] for item in values if key in item])) for key in keys}


def run_batches(
    *,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    patches: list[DensityPatch],
    batch_size: int,
    size_divisor: int,
    device: torch.device,
    train: bool,
    optimizer: torch.optim.Optimizer | None = None,
    num_workers: int = 0,
    grad_clip_norm: float = 0.0,
) -> dict[str, Any]:
    loader = make_loader(patches, batch_size, size_divisor, shuffle=train, num_workers=num_workers)
    losses: list[float] = []
    components: list[dict[str, float]] = []
    pred_sum = 0.0
    target_sum = 0.0
    patch_count = 0
    if train:
        model.train()
    else:
        model.eval()
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in loader:
            batch = move_density_batch_to_device(batch, device)
            if train:
                if optimizer is None:
                    raise ValueError("optimizer is required in train mode")
                optimizer.zero_grad(set_to_none=True)
            pred = model(batch["x"])
            loss = loss_fn(pred, batch)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss encountered in {'train' if train else 'eval'} mode")
            if train:
                loss.backward()
                if float(grad_clip_norm) > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))
                optimizer.step()
            losses.append(float(loss.detach().cpu()))
            if getattr(loss_fn, "last_components", None):
                components.append(dict(loss_fn.last_components))
            pred_sum += float((pred.detach() * batch["valid_mask"]).sum().cpu())
            target_sum += float((batch["target"] * batch["valid_mask"]).sum().cpu())
            patch_count += len(batch["metadata"])
    return {
        "patch_count": int(patch_count),
        "batch_count": int(len(losses)),
        "loss_mean": float(np.mean(losses)) if losses else None,
        "loss_last": float(losses[-1]) if losses else None,
        "loss_components_mean": average_dicts(components),
        "pred_sum": float(pred_sum),
        "target_sum": float(target_sum),
    }


def save_scene_metrics(path: Path, results: list[SceneBuildResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "day_key",
        "scene_key",
        "excluded_reason",
        "ph_anchor_count",
        "fallback_grid_count",
        "patch_count_total",
        "patch_count_selected",
        "selected_target_sum",
        "selected_raw_sum",
        "dropped_too_large",
        "dropped_by_cap",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            summary = result.metrics.get("partition_summary", {})
            selection = result.metrics.get("selection", {})
            writer.writerow(
                {
                    "split": result.record.split,
                    "day_key": result.record.day_key,
                    "scene_key": result.record.scene_key,
                    "excluded_reason": result.excluded_reason or "",
                    "ph_anchor_count": int(summary.get("ph_anchor_count", 0)),
                    "fallback_grid_count": int(summary.get("fallback_grid_count", 0)),
                    "patch_count_total": int(result.metrics.get("patch_count_total", 0)),
                    "patch_count_selected": int(result.metrics.get("patch_count_selected", 0)),
                    "selected_target_sum": float(result.metrics.get("selected_target_sum", 0.0)),
                    "selected_raw_sum": float(result.metrics.get("selected_raw_sum", 0.0)),
                    "dropped_too_large": int(selection.get("dropped_too_large", 0)),
                    "dropped_by_cap": int(selection.get("dropped_by_cap", 0)),
                }
            )


def save_filtered_scene_split(path: Path, results: list[SceneBuildResult]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "day_key", "scene_key", "tif_path", "geojson_path"])
        writer.writeheader()
        for result in results:
            if result.excluded_reason:
                continue
            record = result.record
            writer.writerow(
                {
                    "split": record.split,
                    "day_key": record.day_key,
                    "scene_key": record.scene_key,
                    "tif_path": str(record.tif_path),
                    "geojson_path": str(record.geojson_path),
                }
            )


def save_inference_previews(
    *,
    model: torch.nn.Module,
    patches: list[DensityPatch],
    output_dir: Path,
    device: torch.device,
    batch_size: int,
    size_divisor: int,
    limit: int,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    if int(limit) <= 0 or not patches:
        return paths
    loader = make_loader(patches[: int(limit)], batch_size, size_divisor, shuffle=False, num_workers=0)
    model.eval()
    saved = 0
    with torch.no_grad():
        for batch in loader:
            batch = move_density_batch_to_device(batch, device)
            pred = model(batch["x"]).detach().cpu().numpy()
            x = batch["x"].detach().cpu().numpy()
            target = batch["target"].detach().cpu().numpy()
            valid = batch["valid_mask"].detach().cpu().numpy()
            attention = batch["soft_attention"].detach().cpu().numpy()
            for idx, meta in enumerate(batch["metadata"]):
                height, width = [int(v) for v in meta["shape"]]
                panels = [
                    ("brightness", x[idx, 0, :height, :width], "magma"),
                    ("soft attention", attention[idx, 0, :height, :width], "viridis"),
                    ("target density", target[idx, 0, :height, :width], "viridis"),
                    ("pred density", pred[idx, 0, :height, :width], "viridis"),
                    ("abs error", np.abs(pred[idx, 0, :height, :width] - target[idx, 0, :height, :width]), "inferno"),
                    ("valid owner mask", valid[idx, 0, :height, :width], "gray"),
                ]
                fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
                for ax, (title, arr, cmap) in zip(axes.ravel(), panels):
                    im = ax.imshow(arr, cmap=cmap)
                    ax.set_title(title)
                    ax.axis("off")
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                pred_sum = float((pred[idx, 0, :height, :width] * valid[idx, 0, :height, :width]).sum())
                target_sum = float((target[idx, 0, :height, :width] * valid[idx, 0, :height, :width]).sum())
                fig.suptitle(
                    f"{meta['partition_kind']} | scene-partition {meta.get('partition_id')} | pred_sum={pred_sum:.2f} | target_sum={target_sum:.2f}",
                    fontsize=12,
                )
                path = output_dir / f"inference_preview_{saved:03d}_{meta['partition_kind']}_{meta.get('partition_id')}.png"
                fig.savefig(path, dpi=150)
                plt.close(fig)
                paths.append(str(path))
                saved += 1
                if saved >= int(limit):
                    return paths
    return paths


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    seed_everything(int(args.seed))
    random.seed(int(args.seed))
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = read_json(args.config)
    records = read_scene_split(args.scene_split_csv)
    grouped = group_records(records, int(args.max_scenes_per_split))
    device = resolve_required_device(str(args.device))
    resolver = GroundTruthResolver(args.metadata_csv, args.ships_db, STEP3 / "bboxes_JPSS-2")

    print(f"[split-smoke] device={device} output_dir={output_dir}")
    print(f"[split-smoke] records train={len(grouped['train'])} val={len(grouped['val'])} test={len(grouped['test'])}")

    model = build_model_from_config(config).to(device)
    loss_fn = build_density_loss(loss_config_from_config(config)).to(device)
    training = config.get("training", {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr if args.lr is not None else training.get("lr", 1.0e-3)),
        weight_decay=float(args.weight_decay if args.weight_decay is not None else training.get("weight_decay", 1.0e-4)),
    )
    size_divisor = int(config.get("patching", {}).get("size_divisor", 16))
    grad_clip_norm = float(training.get("grad_clip_norm", 0.0) or 0.0)

    all_scene_results: list[SceneBuildResult] = []
    selected_by_split: dict[str, list[DensityPatch]] = {"train": [], "val": [], "test": []}
    for split in ("train", "val", "test"):
        for record in grouped[split]:
            print(f"[build] {split} {record.scene_key}")
            try:
                result = build_scene(record, config=config, args=args, resolver=resolver)
            except Exception as exc:
                result = SceneBuildResult(record=record, patches=[], metrics={"split": split, "scene_key": record.scene_key}, excluded_reason=f"exception:{type(exc).__name__}:{exc}")
            all_scene_results.append(result)
            if result.excluded_reason:
                print(f"  [skip] {record.scene_key}: {result.excluded_reason}")
                continue
            selected_by_split[split].extend(result.patches)
            ph_count = int(result.metrics["partition_summary"]["ph_anchor_count"])
            print(f"  [ok] ph={ph_count} selected_patches={len(result.patches)} target_sum={result.metrics['selected_target_sum']:.1f}")

    save_scene_metrics(output_dir / "scene_metrics.csv", all_scene_results)
    save_filtered_scene_split(output_dir / "filtered_scene_split.csv", all_scene_results)

    train_history: list[dict[str, Any]] = []
    for epoch in range(int(args.epochs)):
        train_metrics = run_batches(
            model=model,
            loss_fn=loss_fn,
            patches=selected_by_split["train"],
            batch_size=int(args.batch_size),
            size_divisor=size_divisor,
            device=device,
            train=True,
            optimizer=optimizer,
            num_workers=int(args.num_workers),
            grad_clip_norm=grad_clip_norm,
        )
        val_metrics = run_batches(
            model=model,
            loss_fn=loss_fn,
            patches=selected_by_split["val"],
            batch_size=int(args.batch_size),
            size_divisor=size_divisor,
            device=device,
            train=False,
            num_workers=int(args.num_workers),
        )
        epoch_metrics = {"epoch": int(epoch + 1), "train": train_metrics, "val": val_metrics}
        train_history.append(epoch_metrics)
        print(json.dumps(epoch_metrics, ensure_ascii=False, indent=2))

    test_metrics = run_batches(
        model=model,
        loss_fn=loss_fn,
        patches=selected_by_split["test"],
        batch_size=int(args.batch_size),
        size_divisor=size_divisor,
        device=device,
        train=False,
        num_workers=int(args.num_workers),
    )
    preview_paths = save_inference_previews(
        model=model,
        patches=selected_by_split["test"] or selected_by_split["val"],
        output_dir=output_dir / "inference_previews",
        device=device,
        batch_size=int(args.batch_size),
        size_divisor=size_divisor,
        limit=int(args.preview_patches),
    )

    summary = {
        "schema_version": 1,
        "kind": "density_split_smoke_train",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(args.config.expanduser().resolve()),
        "scene_split_csv": str(args.scene_split_csv.expanduser().resolve()),
        "device": str(device),
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "grad_clip_norm": float(grad_clip_norm),
        "smoke_filters": {
            "skip_ph_anchor_zero": bool(args.skip_ph_anchor_zero),
            "max_patches_per_scene": int(args.max_patches_per_scene),
            "max_patch_height": int(args.max_patch_height),
            "max_patch_width": int(args.max_patch_width),
        },
        "git": git_metadata(),
        "scene_counts": {
            split: {
                "input_scenes": int(len(grouped[split])),
                "kept_scenes": int(sum((not item.excluded_reason) and item.record.split == split for item in all_scene_results)),
                "selected_patches": int(len(selected_by_split[split])),
                "selected_target_sum": float(sum(patch.target_sum for patch in selected_by_split[split])),
            }
            for split in ("train", "val", "test")
        },
        "train_history": train_history,
        "test": test_metrics,
        "preview_paths": preview_paths,
        "outputs": {
            "scene_metrics_csv": str(output_dir / "scene_metrics.csv"),
            "filtered_scene_split_csv": str(output_dir / "filtered_scene_split.csv"),
            "inference_preview_dir": str(output_dir / "inference_previews"),
        },
        "checkpoint_saved": False,
    }
    summary_path = output_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary_path": str(summary_path), "scene_counts": summary["scene_counts"], "test": test_metrics, "preview_count": len(preview_paths)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
