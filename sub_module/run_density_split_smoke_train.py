from __future__ import annotations

import argparse
import csv
import hashlib
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
from .dnb_density_patch_pickle_cache import (
    SCHEMA_VERSION as PATCH_CACHE_SCHEMA_VERSION,
    load_patch_split_cache,
    save_patch_split_cache,
    split_cache_path,
)
from .dnb_density_preview import preview_cmap_for_name, preview_limits_for_name, robust_vmax, save_panel_grid
from .dnb_pipeline_core import GroundTruthResolver, SceneRaster
from .dnb_project_paths import DENSITY_OUTPUT_ROOT, ROOT, STEP3
from .dnb_ph_downsample import PHDownsampleConfig, build_ph_anchor_store
from .dnb_scene_partition import ScenePartitionConfig, build_partitioned_density_patches
from .kr_sea_mask import apply_kr_sea_mask


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


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
    parser = argparse.ArgumentParser(description="Run split-level PH-assisted pixel-binary U-Net training.")
    parser.add_argument("--scene-split-csv", type=Path, default=DENSITY_OUTPUT_ROOT / "splits" / "ox_spatial_25pct_63_15_14_20260609_011421" / "scene_split.csv")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "dnb_density_unet_pixel_binary_recursive_ph_hardtarget_20260609.json")
    parser.add_argument("--output-dir", type=Path, default=DENSITY_OUTPUT_ROOT / "runs" / "density_split_smoke_train")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=20260529)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--max-scenes-per-split", type=int, default=0)
    parser.add_argument("--max-patches-per-scene", type=int, default=16)
    parser.add_argument("--max-ph-patches-per-scene", type=int, default=0)
    parser.add_argument("--max-fallback-patches-per-scene", type=int, default=0)
    parser.add_argument("--positive-patches-per-scene", type=int, default=0, help="When >0, sample this many target-positive patches per scene for O/X tests.")
    parser.add_argument("--negative-patches-per-scene", type=int, default=0, help="When >0, sample this many target-empty patches per scene for O/X tests.")
    parser.add_argument("--selection-seed", type=int, default=None, help="Optional deterministic patch-selection seed. Defaults to --seed.")
    parser.add_argument("--max-patch-height", type=int, default=512)
    parser.add_argument("--max-patch-width", type=int, default=512)
    parser.add_argument("--skip-ph-anchor-zero", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preview-patches", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-checkpoint", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--cache-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Build/load split patch caches and write cache metadata, then exit before model training/evaluation.",
    )
    parser.add_argument("--patch-cache-dir", type=Path, default=None, help="Optional directory for reusable full DensityPatch caches.")
    parser.add_argument(
        "--patch-cache-mode",
        choices=["off", "read", "write", "readwrite"],
        default="off",
        help="Patch cache behavior. readwrite loads existing split caches and writes missing ones.",
    )
    return parser


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.expanduser().read_text(encoding="utf-8"))


def stable_json_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
    hierarchical = partition.get("hierarchical_ph", {})
    recursive = hierarchical.get("recursive", {})
    return ScenePartitionConfig(
        enabled=True,
        fallback_tile_pixels=int(partition.get("fallback_tile_pixels", 96)),
        halo_pixels=int(partition.get("halo_pixels", 16)),
        anchor_padding_pixels=int(partition.get("anchor_padding_pixels", 16)),
        min_owner_pixels=int(partition.get("min_owner_pixels", 1)),
        min_fallback_owner_pixels=int(partition.get("min_fallback_owner_pixels", 1)),
        hierarchical_ph_enabled=bool(hierarchical.get("enabled", False)),
        hierarchical_large_min_pixels=int(hierarchical.get("large_min_pixels", 65536)),
        hierarchical_large_min_height=int(hierarchical.get("large_min_height", 384)),
        hierarchical_large_min_width=int(hierarchical.get("large_min_width", 384)),
        hierarchical_child_anchor_padding_pixels=int(hierarchical.get("child_anchor_padding_pixels", 8)),
        hierarchical_child_detection_threshold=float(hierarchical.get("child_detection_threshold", 0.5)),
        hierarchical_child_analysis_threshold=float(hierarchical.get("child_analysis_threshold", 0.25)),
        hierarchical_child_threshold_reference=str(hierarchical.get("child_threshold_reference", "median")),
        hierarchical_child_smooth_sigma=float(hierarchical.get("child_smooth_sigma", 0.0)),
        hierarchical_child_lifetime_limit=float(hierarchical.get("child_lifetime_limit", 0.0)),
        hierarchical_child_lifetime_limit_fraction=float(hierarchical.get("child_lifetime_limit_fraction", 1.0005)),
        hierarchical_child_area_limit=int(hierarchical.get("child_area_limit", 0)),
        hierarchical_child_min_nodes=int(hierarchical.get("child_min_nodes", 3)),
        hierarchical_child_max_nodes=int(hierarchical.get("child_max_nodes", 2048)),
        hierarchical_child_max_candidates_per_parent=int(hierarchical.get("child_max_candidates_per_parent", 64)),
        hierarchical_keep_large_parent=bool(hierarchical.get("keep_large_parent", False)),
        hierarchical_recursive_enabled=bool(recursive.get("enabled", False)),
        hierarchical_recursive_max_depth=int(recursive.get("max_depth", 1)),
        hierarchical_recursive_target_max_height=int(recursive.get("target_max_height", 0)),
        hierarchical_recursive_target_max_width=int(recursive.get("target_max_width", 0)),
        hierarchical_recursive_target_max_pixels=int(recursive.get("target_max_pixels", 0)),
    )


def loss_config_from_config(config: dict[str, Any]) -> dict[str, Any]:
    loss = config.get("training", {}).get("loss", {})
    return dict(loss) if isinstance(loss, dict) else {"name": str(loss)}


def input_channels_from_config(config: dict[str, Any]) -> list[str] | None:
    channels = config.get("patching", {}).get("input_channels")
    if channels is None:
        return None
    return [str(channel) for channel in channels]


def build_model_from_config(config: dict[str, Any]) -> torch.nn.Module:
    model_config = dict(config.get("model", {}))
    name = str(model_config.pop("name", "PixelBinaryOccupancyUNet"))
    model_config.pop("out_channels", None)
    return build_density_model(name, out_channels=1, **model_config)


def select_smoke_patches(
    patches: list[DensityPatch],
    *,
    max_patches: int,
    max_ph_patches: int = 0,
    max_fallback_patches: int = 0,
    positive_patches: int = 0,
    negative_patches: int = 0,
    selection_seed: int = 0,
    selection_key: str = "",
    max_height: int,
    max_width: int,
) -> tuple[list[DensityPatch], dict[str, Any]]:
    kept_size = [patch for patch in patches if patch.shape[0] <= int(max_height) and patch.shape[1] <= int(max_width)]
    dropped_too_large = len(patches) - len(kept_size)

    ph_kinds = {"ph_anchor", "ph_child"}

    def priority(patch: DensityPatch) -> tuple[int, float, int]:
        return (
            1 if patch.partition_kind in ph_kinds else 0,
            float(patch.raw_count_sum),
            int(patch.valid_pixels),
        )

    def within_total_limit(selected: list[DensityPatch]) -> list[DensityPatch]:
        if int(max_patches) <= 0 or len(selected) <= int(max_patches):
            return selected
        return sorted(selected, key=priority, reverse=True)[: int(max_patches)]

    by_kind = {
        "ph_anchor": [patch for patch in kept_size if patch.partition_kind in ph_kinds],
        "fallback_grid": [patch for patch in kept_size if patch.partition_kind == "fallback_grid"],
    }
    positive_candidates = [patch for patch in kept_size if float(patch.target_sum) > 1.0e-6]
    negative_candidates = [patch for patch in kept_size if float(patch.target_sum) <= 1.0e-6]
    candidate_ph_anchor = sum(1 for patch in kept_size if patch.partition_kind == "ph_anchor")
    candidate_ph_child = sum(1 for patch in kept_size if patch.partition_kind == "ph_child")
    for kind in by_kind:
        by_kind[kind] = sorted(by_kind[kind], key=lambda patch: (float(patch.raw_count_sum), int(patch.valid_pixels)), reverse=True)

    use_ox_quotas = int(positive_patches) > 0 or int(negative_patches) > 0
    if use_ox_quotas:
        seed_material = f"{int(selection_seed)}:{selection_key}".encode("utf-8")
        seed_value = int(hashlib.sha256(seed_material).hexdigest()[:16], 16)
        rng = random.Random(seed_value)

        def sampled(items: list[DensityPatch], limit: int) -> list[DensityPatch]:
            shuffled = list(items)
            rng.shuffle(shuffled)
            if int(limit) <= 0:
                return shuffled
            return shuffled[: min(int(limit), len(shuffled))]

        selected = []
        selected.extend(sampled(positive_candidates, int(positive_patches)))
        selected.extend(sampled(negative_candidates, int(negative_patches)))
        rng.shuffle(selected)
        if int(max_patches) > 0 and len(selected) > int(max_patches):
            selected = selected[: int(max_patches)]
    else:
        use_kind_quotas = int(max_ph_patches) > 0 or int(max_fallback_patches) > 0
    if not use_ox_quotas and use_kind_quotas:
        selected: list[DensityPatch] = []
        selected.extend(by_kind["ph_anchor"][: max(int(max_ph_patches), 0)])
        selected.extend(by_kind["fallback_grid"][: max(int(max_fallback_patches), 0)])

        if int(max_patches) > 0 and len(selected) < int(max_patches):
            selected_ids = {id(patch) for patch in selected}
            remaining = [patch for patch in kept_size if id(patch) not in selected_ids]
            fill_count = int(max_patches) - len(selected)
            selected.extend(sorted(remaining, key=priority, reverse=True)[:fill_count])
        selected = within_total_limit(selected)
    elif not use_ox_quotas and (int(max_patches) <= 0 or len(kept_size) <= int(max_patches)):
        selected = kept_size
    elif not use_ox_quotas:
        selected = sorted(kept_size, key=priority, reverse=True)[: int(max_patches)]

    selected_ph = sum(1 for patch in selected if patch.partition_kind in ph_kinds)
    selected_ph_anchor = sum(1 for patch in selected if patch.partition_kind == "ph_anchor")
    selected_ph_child = sum(1 for patch in selected if patch.partition_kind == "ph_child")
    selected_fallback = sum(1 for patch in selected if patch.partition_kind == "fallback_grid")
    selected_positive = sum(1 for patch in selected if float(patch.target_sum) > 1.0e-6)
    selected_negative = sum(1 for patch in selected if float(patch.target_sum) <= 1.0e-6)
    return selected, {
        "dropped_too_large": int(dropped_too_large),
        "dropped_by_cap": int(len(kept_size) - len(selected)),
        "kept_size_count": int(len(kept_size)),
        "candidate_positive_patch_count": int(len(positive_candidates)),
        "candidate_negative_patch_count": int(len(negative_candidates)),
        "candidate_ph_patch_count": int(len(by_kind["ph_anchor"])),
        "candidate_ph_anchor_count": int(candidate_ph_anchor),
        "candidate_ph_child_count": int(candidate_ph_child),
        "candidate_fallback_grid_count": int(len(by_kind["fallback_grid"])),
        "selected_positive_patch_count": int(selected_positive),
        "selected_negative_patch_count": int(selected_negative),
        "selected_ph_patch_count": int(selected_ph),
        "selected_ph_anchor_count": int(selected_ph_anchor),
        "selected_ph_child_count": int(selected_ph_child),
        "selected_fallback_grid_count": int(selected_fallback),
        "max_ph_patches_per_scene": int(max_ph_patches),
        "max_fallback_patches_per_scene": int(max_fallback_patches),
        "positive_patches_per_scene": int(positive_patches),
        "negative_patches_per_scene": int(negative_patches),
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
        max_ph_patches=int(args.max_ph_patches_per_scene),
        max_fallback_patches=int(args.max_fallback_patches_per_scene),
        positive_patches=int(args.positive_patches_per_scene),
        negative_patches=int(args.negative_patches_per_scene),
        selection_seed=int(args.selection_seed if args.selection_seed is not None else args.seed),
        selection_key=f"{record.split}:{record.scene_key}",
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
    ph_partition_count = int(partition_summary.get("ph_anchor_count", 0)) + int(partition_summary.get("ph_child_count", 0))
    if bool(args.skip_ph_anchor_zero) and ph_partition_count <= 0:
        return SceneBuildResult(record=record, patches=[], metrics=metrics, excluded_reason="ph_partition_count_zero")
    if not selected:
        return SceneBuildResult(record=record, patches=[], metrics=metrics, excluded_reason="no_selected_patches_after_smoke_filters")
    return SceneBuildResult(record=record, patches=selected, metrics=metrics)


def infer_density_from_output(output: torch.Tensor | dict[str, torch.Tensor], batch: dict[str, Any]) -> torch.Tensor:
    if torch.is_tensor(output):
        return torch.sigmoid(output) * batch["valid_mask"]
    if "density" in output:
        return output["density"] * batch["valid_mask"]
    if "pixel_logits" in output:
        return torch.sigmoid(output["pixel_logits"]) * batch["valid_mask"]
    raise ValueError("Current density runner expects pixel_logits or density output")


def target_density_for_metrics(loss_fn: torch.nn.Module, batch: dict[str, Any]) -> torch.Tensor:
    if hasattr(loss_fn, "target_density_from_batch"):
        return loss_fn.target_density_from_batch(batch)
    return batch["target"]


def make_loader(
    patches: list[DensityPatch],
    batch_size: int,
    size_divisor: int,
    shuffle: bool,
    num_workers: int,
    input_channels: list[str] | None,
) -> DataLoader:
    return DataLoader(
        DensityPatchDataset(patches),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        collate_fn=lambda batch: density_patch_collate(
            batch,
            size_divisor=int(size_divisor),
            input_channels=input_channels,
        ),
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
    input_channels: list[str] | None = None,
) -> dict[str, Any]:
    loader = make_loader(
        patches,
        batch_size,
        size_divisor,
        shuffle=train,
        num_workers=num_workers,
        input_channels=input_channels,
    )
    losses: list[float] = []
    components: list[dict[str, float]] = []
    pred_sum = 0.0
    target_sum = 0.0
    overlap_sum = 0.0
    patch_count = 0
    pred_patch_masses: list[float] = []
    target_patch_masses: list[float] = []
    spatial_overlaps: list[float] = []
    occupancy_probs: list[float] = []
    occupancy_targets: list[float] = []
    pixel_tp = 0
    pixel_fp = 0
    pixel_fn = 0
    pixel_tn = 0
    pixel_valid_count = 0
    pixel_target_positive_count = 0
    pixel_pred_positive_count = 0
    pixel_brier_sum = 0.0
    uses_occupancy_loss = hasattr(loss_fn, "occupancy_from_output")
    uses_pixel_occupancy_loss = hasattr(loss_fn, "pixel_occupancy_from_output")
    eps = 1.0e-8
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
            pred_output = model(batch["x"])
            loss = loss_fn(pred_output, batch)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss encountered in {'train' if train else 'eval'} mode")
            if train:
                loss.backward()
                if float(grad_clip_norm) > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))
                optimizer.step()
            pred_density = loss_fn.density_from_output(pred_output, batch) if hasattr(loss_fn, "density_from_output") else pred_output
            losses.append(float(loss.detach().cpu()))
            if getattr(loss_fn, "last_components", None):
                components.append(dict(loss_fn.last_components))
            pred_masked = pred_density.detach() * batch["valid_mask"]
            target_metric_density = target_density_for_metrics(loss_fn, batch)
            target_masked = target_metric_density.detach() * batch["valid_mask"]
            pred_sum += float(pred_masked.sum().cpu())
            target_sum += float(target_masked.sum().cpu())
            overlap_sum += float(torch.minimum(pred_masked, target_masked).sum().cpu())
            pred_mass_batch = pred_masked.flatten(1).sum(dim=1)
            target_mass_batch = target_masked.flatten(1).sum(dim=1)
            pred_patch_masses.extend(float(v) for v in pred_mass_batch.detach().cpu().tolist())
            target_patch_masses.extend(float(v) for v in target_mass_batch.detach().cpu().tolist())
            target_positive = target_mass_batch > eps
            if bool(target_positive.any()):
                pred_prob = pred_masked / torch.clamp(pred_mass_batch.reshape(-1, 1, 1, 1), min=eps)
                target_prob = target_masked / torch.clamp(target_mass_batch.reshape(-1, 1, 1, 1), min=eps)
                spatial_batch = torch.minimum(pred_prob, target_prob).flatten(1).sum(dim=1)
                spatial_overlaps.extend(float(v) for v in spatial_batch[target_positive].detach().cpu().tolist())
            if hasattr(loss_fn, "occupancy_from_output"):
                occ_prob, occ_target = loss_fn.occupancy_from_output(pred_output, batch)
                occupancy_probs.extend(float(v) for v in occ_prob.detach().cpu().tolist())
                occupancy_targets.extend(float(v) for v in occ_target.detach().cpu().tolist())
            if hasattr(loss_fn, "pixel_occupancy_from_output"):
                pixel_prob, pixel_target, pixel_valid = loss_fn.pixel_occupancy_from_output(pred_output, batch)
                valid_bool = pixel_valid.detach() > 0
                target_bool = (pixel_target.detach() > 0.5) & valid_bool
                pred_bool = (pixel_prob.detach() >= 0.5) & valid_bool
                pixel_tp += int((pred_bool & target_bool).sum().cpu())
                pixel_fp += int((pred_bool & ~target_bool & valid_bool).sum().cpu())
                pixel_fn += int((~pred_bool & target_bool).sum().cpu())
                pixel_tn += int((~pred_bool & ~target_bool & valid_bool).sum().cpu())
                pixel_valid_count += int(valid_bool.sum().cpu())
                pixel_target_positive_count += int(target_bool.sum().cpu())
                pixel_pred_positive_count += int(pred_bool.sum().cpu())
                pixel_brier_sum += float((((pixel_prob.detach() - pixel_target.detach()) ** 2) * pixel_valid.detach()).sum().cpu())
            patch_count += len(batch["metadata"])
    pred_masses = np.asarray(pred_patch_masses, dtype=np.float64)
    target_masses = np.asarray(target_patch_masses, dtype=np.float64)
    mass_error = pred_masses - target_masses if pred_masses.size else np.asarray([], dtype=np.float64)
    positive_mask = target_masses > eps if target_masses.size else np.asarray([], dtype=bool)
    smape = (
        2.0 * np.abs(mass_error) / np.maximum(np.abs(pred_masses) + np.abs(target_masses), eps)
        if pred_masses.size
        else np.asarray([], dtype=np.float64)
    )
    target_positive_count = int(positive_mask.sum()) if positive_mask.size else 0
    pred_target_ratio = float(pred_sum / max(target_sum, eps))
    metrics = {
        "patch_count": int(patch_count),
        "batch_count": int(len(losses)),
        "loss_mean": float(np.mean(losses)) if losses else None,
        "loss_last": float(losses[-1]) if losses else None,
        "loss_components_mean": average_dicts(components),
        "pred_sum": float(pred_sum),
        "target_sum": float(target_sum),
        "pred_target_ratio": pred_target_ratio,
        "positive_patch_count": target_positive_count,
        "zero_target_patch_count": int((~positive_mask).sum()) if positive_mask.size else 0,
        "target_explained": float(overlap_sum / max(target_sum, eps)),
        "pred_matched": float(overlap_sum / max(pred_sum, eps)),
        "spatial_overlap_mean_positive": float(np.mean(spatial_overlaps)) if spatial_overlaps else None,
    }
    if uses_occupancy_loss:
        metrics.update(
            {
                "occupancy_mass_ratio_abs_log_error": float(abs(np.log((pred_sum + eps) / (target_sum + eps)))),
                "occupancy_mass_bias": float(pred_sum - target_sum),
                "patch_occupancy_mass_mae": float(np.mean(np.abs(mass_error))) if mass_error.size else None,
                "patch_occupancy_mass_rmse": float(np.sqrt(np.mean(mass_error**2))) if mass_error.size else None,
                "patch_occupancy_mass_bias_mean": float(np.mean(mass_error)) if mass_error.size else None,
                "patch_occupancy_mass_smape": float(np.mean(smape)) if smape.size else None,
            }
        )
    else:
        metrics.update(
            {
                "count_ratio_abs_log_error": float(abs(np.log((pred_sum + eps) / (target_sum + eps)))),
                "count_bias": float(pred_sum - target_sum),
                "patch_count_mae": float(np.mean(np.abs(mass_error))) if mass_error.size else None,
                "patch_count_rmse": float(np.sqrt(np.mean(mass_error**2))) if mass_error.size else None,
                "patch_count_bias_mean": float(np.mean(mass_error)) if mass_error.size else None,
                "patch_count_smape": float(np.mean(smape)) if smape.size else None,
            }
        )
    if occupancy_targets:
        probs = np.asarray(occupancy_probs, dtype=np.float64)
        targets = np.asarray(occupancy_targets, dtype=np.float64) > 0.5
        pred_positive = probs >= 0.5
        tp = int(np.logical_and(pred_positive, targets).sum())
        fp = int(np.logical_and(pred_positive, ~targets).sum())
        fn = int(np.logical_and(~pred_positive, targets).sum())
        tn = int(np.logical_and(~pred_positive, ~targets).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        metrics.update(
            {
                "occupancy_accuracy": float((tp + tn) / max(len(targets), 1)),
                "occupancy_precision": float(precision),
                "occupancy_recall": float(recall),
                "occupancy_f1": float(2.0 * precision * recall / max(precision + recall, eps)),
                "occupancy_brier": float(np.mean((probs - targets.astype(np.float64)) ** 2)),
                "occupancy_positive_target_count": int(targets.sum()),
                "occupancy_negative_target_count": int((~targets).sum()),
                "occupancy_pred_positive_count": int(pred_positive.sum()),
                "occupancy_tp": tp,
                "occupancy_fp": fp,
                "occupancy_fn": fn,
                "occupancy_tn": tn,
            }
        )
    if uses_pixel_occupancy_loss and pixel_valid_count > 0:
        pixel_precision = pixel_tp / max(pixel_tp + pixel_fp, 1)
        pixel_recall = pixel_tp / max(pixel_tp + pixel_fn, 1)
        pixel_f1 = 2.0 * pixel_precision * pixel_recall / max(pixel_precision + pixel_recall, eps)
        pixel_iou = pixel_tp / max(pixel_tp + pixel_fp + pixel_fn, 1)
        metrics.update(
            {
                "pixel_accuracy": float((pixel_tp + pixel_tn) / max(pixel_valid_count, 1)),
                "pixel_precision": float(pixel_precision),
                "pixel_recall": float(pixel_recall),
                "pixel_f1": float(pixel_f1),
                "pixel_iou": float(pixel_iou),
                "pixel_brier": float(pixel_brier_sum / max(pixel_valid_count, 1)),
                "pixel_valid_count": int(pixel_valid_count),
                "pixel_positive_target_count": int(pixel_target_positive_count),
                "pixel_negative_target_count": int(pixel_valid_count - pixel_target_positive_count),
                "pixel_pred_positive_count": int(pixel_pred_positive_count),
                "pixel_tp": int(pixel_tp),
                "pixel_fp": int(pixel_fp),
                "pixel_fn": int(pixel_fn),
                "pixel_tn": int(pixel_tn),
                "pixel_threshold": 0.5,
            }
        )
    return metrics


def save_scene_metrics(path: Path, results: list[SceneBuildResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "day_key",
        "scene_key",
        "excluded_reason",
        "ph_anchor_count",
        "ph_child_count",
        "fallback_grid_count",
        "patch_count_total",
        "patch_count_selected",
        "selected_target_sum",
        "selected_raw_sum",
        "dropped_too_large",
        "dropped_by_cap",
        "kept_size_count",
        "candidate_positive_patch_count",
        "candidate_negative_patch_count",
        "candidate_ph_anchor_count",
        "candidate_ph_child_count",
        "candidate_fallback_grid_count",
        "selected_positive_patch_count",
        "selected_negative_patch_count",
        "selected_ph_anchor_count",
        "selected_ph_child_count",
        "selected_fallback_grid_count",
        "max_ph_patches_per_scene",
        "max_fallback_patches_per_scene",
        "positive_patches_per_scene",
        "negative_patches_per_scene",
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
                    "ph_child_count": int(summary.get("ph_child_count", 0)),
                    "fallback_grid_count": int(summary.get("fallback_grid_count", 0)),
                    "patch_count_total": int(result.metrics.get("patch_count_total", 0)),
                    "patch_count_selected": int(result.metrics.get("patch_count_selected", 0)),
                    "selected_target_sum": float(result.metrics.get("selected_target_sum", 0.0)),
                    "selected_raw_sum": float(result.metrics.get("selected_raw_sum", 0.0)),
                    "dropped_too_large": int(selection.get("dropped_too_large", 0)),
                    "dropped_by_cap": int(selection.get("dropped_by_cap", 0)),
                    "kept_size_count": int(selection.get("kept_size_count", 0)),
                    "candidate_positive_patch_count": int(selection.get("candidate_positive_patch_count", 0)),
                    "candidate_negative_patch_count": int(selection.get("candidate_negative_patch_count", 0)),
                    "candidate_ph_anchor_count": int(selection.get("candidate_ph_anchor_count", 0)),
                    "candidate_ph_child_count": int(selection.get("candidate_ph_child_count", 0)),
                    "candidate_fallback_grid_count": int(selection.get("candidate_fallback_grid_count", 0)),
                    "selected_positive_patch_count": int(selection.get("selected_positive_patch_count", 0)),
                    "selected_negative_patch_count": int(selection.get("selected_negative_patch_count", 0)),
                    "selected_ph_anchor_count": int(selection.get("selected_ph_anchor_count", 0)),
                    "selected_ph_child_count": int(selection.get("selected_ph_child_count", 0)),
                    "selected_fallback_grid_count": int(selection.get("selected_fallback_grid_count", 0)),
                    "max_ph_patches_per_scene": int(selection.get("max_ph_patches_per_scene", 0)),
                    "max_fallback_patches_per_scene": int(selection.get("max_fallback_patches_per_scene", 0)),
                    "positive_patches_per_scene": int(selection.get("positive_patches_per_scene", 0)),
                    "negative_patches_per_scene": int(selection.get("negative_patches_per_scene", 0)),
                }
            )


def write_dirty_patch(output_dir: Path, git_info: dict[str, Any]) -> str | None:
    if not bool(git_info.get("git_dirty")):
        return None
    try:
        diff = subprocess.check_output(["git", "diff", "HEAD", "--"], cwd=ROOT, text=True)
    except Exception as exc:
        path = output_dir / "run_git_dirty.patch.error.txt"
        path.write_text(str(exc), encoding="utf-8")
        return str(path)
    if not diff.strip():
        return None
    path = output_dir / "run_git_dirty.patch"
    path.write_text(diff, encoding="utf-8")
    return str(path)


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    args: argparse.Namespace,
    train_history: list[dict[str, Any]],
    test_metrics: dict[str, Any],
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "schema_version": 1,
        "kind": "density_model_checkpoint",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "config_hash": stable_json_hash(config),
        "args": vars(args),
        "train_history": train_history,
        "test": test_metrics,
    }
    torch.save(checkpoint, path)
    return str(path)


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
    loss_fn: torch.nn.Module,
    patches: list[DensityPatch],
    output_dir: Path,
    device: torch.device,
    batch_size: int,
    size_divisor: int,
    limit: int,
    input_channels: list[str] | None,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    if int(limit) <= 0 or not patches:
        return paths
    loader = make_loader(
        patches[: int(limit)],
        batch_size,
        size_divisor,
        shuffle=False,
        num_workers=0,
        input_channels=input_channels,
    )
    model.eval()
    saved = 0
    with torch.no_grad():
        for batch in loader:
            batch = move_density_batch_to_device(batch, device)
            pred_output = model(batch["x"])
            pred_tensor = infer_density_from_output(pred_output, batch)
            pred = pred_tensor.detach().cpu().numpy()
            x = batch["x"].detach().cpu().numpy()
            target_tensor = target_density_for_metrics(loss_fn, batch)
            target = target_tensor.detach().cpu().numpy()
            valid = batch["valid_mask"].detach().cpu().numpy()
            channel_names = [str(name) for name in batch.get("input_channels", input_channels or [])]
            if not channel_names:
                channel_names = [f"channel_{channel_idx}" for channel_idx in range(x.shape[1])]
            for idx, meta in enumerate(batch["metadata"]):
                height, width = [int(v) for v in meta["shape"]]
                pred_arr = pred[idx, 0, :height, :width]
                target_arr = target[idx, 0, :height, :width]
                valid_arr = valid[idx, 0, :height, :width]
                density_vmax = robust_vmax([pred_arr, target_arr])
                error_arr = np.abs(pred_arr - target_arr)
                error_vmax = robust_vmax([error_arr])
                panels = []
                for channel_idx, channel_name in enumerate(channel_names[: x.shape[1]]):
                    arr = x[idx, channel_idx, :height, :width]
                    vmin, vmax = preview_limits_for_name(channel_name, arr)
                    panels.append((f"input {channel_idx}: {channel_name}", arr, preview_cmap_for_name(channel_name), vmin, vmax))
                panels.extend(
                    [
                        ("target hard pixel occupancy", target_arr, "viridis", 0.0, density_vmax),
                        ("pred pixel probability", pred_arr, "viridis", 0.0, density_vmax),
                        ("abs prob error", error_arr, "inferno", 0.0, error_vmax),
                        ("valid owner mask", valid_arr, "gray", 0.0, 1.0),
                    ]
                )
                pred_sum = float((pred_arr * valid_arr).sum())
                target_sum = float((target_arr * valid_arr).sum())
                path = output_dir / f"inference_preview_{saved:03d}_{meta['partition_kind']}_{meta.get('partition_id')}.png"
                save_panel_grid(
                    path,
                    panels,
                    title=(
                        f"{meta['partition_kind']} | scene-partition {meta.get('partition_id')} "
                        f"| pred_sum={pred_sum:.2f} | target_sum={target_sum:.2f}"
                    ),
                    cols=4,
                    dpi=150,
                )
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
    config_hash = stable_json_hash(config)
    config_snapshot_path = output_dir / "config_snapshot.json"
    config_snapshot_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    records = read_scene_split(args.scene_split_csv)
    grouped = group_records(records, int(args.max_scenes_per_split))
    device = resolve_required_device(str(args.device))
    resolver = GroundTruthResolver(STEP3 / "bboxes_JPSS-2")

    print(f"[split-smoke] device={device} output_dir={output_dir}")
    print(f"[split-smoke] records train={len(grouped['train'])} val={len(grouped['val'])} test={len(grouped['test'])}")

    model = build_model_from_config(config)
    model = model.to(device)
    loss_fn = build_density_loss(loss_config_from_config(config)).to(device)
    training = config.get("training", {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr if args.lr is not None else training.get("lr", 1.0e-3)),
        weight_decay=float(args.weight_decay if args.weight_decay is not None else training.get("weight_decay", 1.0e-4)),
    )
    size_divisor = int(config.get("patching", {}).get("size_divisor", 16))
    input_channels = input_channels_from_config(config)
    grad_clip_norm = float(training.get("grad_clip_norm", 0.0) or 0.0)

    all_scene_results: list[SceneBuildResult] = []
    selected_by_split: dict[str, list[DensityPatch]] = {"train": [], "val": [], "test": []}
    cache_metadata_by_split: dict[str, Any] = {}
    patch_cache_dir = args.patch_cache_dir.expanduser().resolve() if args.patch_cache_dir is not None else None
    patch_cache_mode = str(args.patch_cache_mode)
    if bool(args.cache_only) and patch_cache_dir is None:
        raise ValueError("--cache-only requires --patch-cache-dir so the built patches have a persistent destination.")
    for split in ("train", "val", "test"):
        loaded_from_cache = False
        if patch_cache_dir is not None and patch_cache_mode in {"read", "readwrite"}:
            cache_path = split_cache_path(patch_cache_dir, split)
            if cache_path.exists():
                patches, cache_metadata = load_patch_split_cache(patch_cache_dir, split)
                selected_by_split[split] = patches
                cache_metadata_by_split[split] = cache_metadata
                loaded_from_cache = True
                print(f"[cache-load] {split} patches={len(patches)} path={cache_path}")
            elif patch_cache_mode == "read":
                raise FileNotFoundError(f"--patch-cache-mode=read but split cache is missing: {cache_path}")
        if loaded_from_cache:
            continue

        split_scene_results: list[SceneBuildResult] = []
        for record in grouped[split]:
            print(f"[build] {split} {record.scene_key}")
            try:
                result = build_scene(record, config=config, args=args, resolver=resolver)
            except Exception as exc:
                result = SceneBuildResult(record=record, patches=[], metrics={"split": split, "scene_key": record.scene_key}, excluded_reason=f"exception:{type(exc).__name__}:{exc}")
            all_scene_results.append(result)
            split_scene_results.append(result)
            if result.excluded_reason:
                print(f"  [skip] {record.scene_key}: {result.excluded_reason}")
                continue
            selected_by_split[split].extend(result.patches)
            partition_summary = result.metrics["partition_summary"]
            ph_anchor_count = int(partition_summary.get("ph_anchor_count", 0))
            ph_child_count = int(partition_summary.get("ph_child_count", 0))
            fallback_count = int(partition_summary.get("fallback_grid_count", 0))
            print(
                "  [ok] "
                f"ph_anchor={ph_anchor_count} ph_child={ph_child_count} fallback={fallback_count} "
                f"selected_patches={len(result.patches)} target_sum={result.metrics['selected_target_sum']:.1f}"
            )
        if patch_cache_dir is not None and patch_cache_mode in {"write", "readwrite"}:
            cache_metadata = {
                "scene_split_csv": str(args.scene_split_csv.expanduser().resolve()),
                "config_path": str(args.config.expanduser().resolve()),
                "config_hash": config_hash,
                "input_scene_count": int(len(grouped[split])),
                "kept_scene_count": int(sum(not item.excluded_reason for item in split_scene_results)),
                "selected_target_sum": float(sum(patch.target_sum for patch in selected_by_split[split])),
                "selection_seed": int(args.selection_seed if args.selection_seed is not None else args.seed),
            }
            cache_path = save_patch_split_cache(
                patch_cache_dir,
                split,
                selected_by_split[split],
                metadata=cache_metadata,
            )
            cache_metadata_by_split[split] = {
                "schema_version": PATCH_CACHE_SCHEMA_VERSION,
                "kind": "density_patch_pickle_cache",
                "split": split,
                "patch_count": int(len(selected_by_split[split])),
                "metadata": cache_metadata,
                "path": str(cache_path),
            }
            print(f"[cache-save] {split} patches={len(selected_by_split[split])} path={cache_path}")

    save_scene_metrics(output_dir / "scene_metrics.csv", all_scene_results)
    save_filtered_scene_split(output_dir / "filtered_scene_split.csv", all_scene_results)

    if bool(args.cache_only):
        summary = {
            "schema_version": 1,
            "kind": "density_patch_cache_build",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "config_path": str(args.config.expanduser().resolve()),
            "config_hash": config_hash,
            "scene_split_csv": str(args.scene_split_csv.expanduser().resolve()),
            "seed": int(args.seed),
            "smoke_filters": {
                "skip_ph_anchor_zero": bool(args.skip_ph_anchor_zero),
                "max_patches_per_scene": int(args.max_patches_per_scene),
                "max_ph_patches_per_scene": int(args.max_ph_patches_per_scene),
                "max_fallback_patches_per_scene": int(args.max_fallback_patches_per_scene),
                "positive_patches_per_scene": int(args.positive_patches_per_scene),
                "negative_patches_per_scene": int(args.negative_patches_per_scene),
                "selection_seed": int(args.selection_seed if args.selection_seed is not None else args.seed),
                "max_patch_height": int(args.max_patch_height),
                "max_patch_width": int(args.max_patch_width),
            },
            "patch_cache": {
                "mode": patch_cache_mode,
                "dir": str(patch_cache_dir),
                "metadata_by_split": cache_metadata_by_split,
            },
            "git": git_metadata(),
            "scene_counts": {
                split: {
                    "input_scenes": int(len(grouped[split])),
                    "kept_scenes": (
                        cache_metadata_by_split.get(split, {}).get("metadata", {}).get("kept_scene_count")
                        if split in cache_metadata_by_split
                        else int(sum((not item.excluded_reason) and item.record.split == split for item in all_scene_results))
                    ),
                    "selected_patches": int(len(selected_by_split[split])),
                    "selected_target_sum": float(sum(patch.target_sum for patch in selected_by_split[split])),
                }
                for split in ("train", "val", "test")
            },
            "outputs": {
                "config_snapshot": str(config_snapshot_path),
                "scene_metrics_csv": str(output_dir / "scene_metrics.csv"),
                "filtered_scene_split_csv": str(output_dir / "filtered_scene_split.csv"),
                "cache_build_summary": str(output_dir / "cache_build_summary.json"),
            },
        }
        dirty_patch_path = write_dirty_patch(output_dir, summary["git"])
        if dirty_patch_path is not None:
            summary["outputs"]["run_git_dirty_patch"] = dirty_patch_path
        summary_path = output_dir / "cache_build_summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"summary_path": str(summary_path), "scene_counts": summary["scene_counts"]}, ensure_ascii=False, indent=2))
        return 0

    train_history: list[dict[str, Any]] = []
    best_checkpoints: dict[str, dict[str, Any]] = {}
    best_val_loss = float("inf")
    best_val_pixel_f1 = -float("inf")
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
            input_channels=input_channels,
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
            input_channels=input_channels,
        )
        epoch_metrics = {"epoch": int(epoch + 1), "train": train_metrics, "val": val_metrics}
        train_history.append(epoch_metrics)
        val_loss = val_metrics.get("loss_mean")
        if bool(args.save_checkpoint) and val_loss is not None and float(val_loss) < best_val_loss:
            best_val_loss = float(val_loss)
            checkpoint_path = save_checkpoint(
                output_dir / "checkpoints" / "checkpoint_best_val_loss.pt",
                model=model,
                optimizer=optimizer,
                config=config,
                args=args,
                train_history=train_history,
                test_metrics={"selection": "best_val_loss", "epoch": int(epoch + 1), "val": val_metrics},
            )
            best_checkpoints["best_val_loss"] = {
                "path": checkpoint_path,
                "epoch": int(epoch + 1),
                "val_loss": float(val_loss),
            }
        val_pixel_f1 = val_metrics.get("pixel_f1")
        if bool(args.save_checkpoint) and val_pixel_f1 is not None and float(val_pixel_f1) > best_val_pixel_f1:
            best_val_pixel_f1 = float(val_pixel_f1)
            checkpoint_path = save_checkpoint(
                output_dir / "checkpoints" / "checkpoint_best_val_pixel_f1.pt",
                model=model,
                optimizer=optimizer,
                config=config,
                args=args,
                train_history=train_history,
                test_metrics={
                    "selection": "best_val_pixel_f1",
                    "epoch": int(epoch + 1),
                    "val": val_metrics,
                    "pixel_f1": float(val_pixel_f1),
                },
            )
            best_checkpoints["best_val_pixel_f1"] = {
                "path": checkpoint_path,
                "epoch": int(epoch + 1),
                "pixel_f1": float(val_pixel_f1),
            }
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
        input_channels=input_channels,
    )
    preview_paths = save_inference_previews(
        model=model,
        loss_fn=loss_fn,
        patches=selected_by_split["test"] or selected_by_split["val"],
        output_dir=output_dir / "inference_previews",
        device=device,
        batch_size=int(args.batch_size),
        size_divisor=size_divisor,
        limit=int(args.preview_patches),
        input_channels=input_channels,
    )

    summary = {
        "schema_version": 1,
        "kind": "density_split_smoke_train",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(args.config.expanduser().resolve()),
        "config_hash": config_hash,
        "scene_split_csv": str(args.scene_split_csv.expanduser().resolve()),
        "device": str(device),
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "input_channels": input_channels,
        "grad_clip_norm": float(grad_clip_norm),
        "smoke_filters": {
            "skip_ph_anchor_zero": bool(args.skip_ph_anchor_zero),
            "max_patches_per_scene": int(args.max_patches_per_scene),
            "max_ph_patches_per_scene": int(args.max_ph_patches_per_scene),
            "max_fallback_patches_per_scene": int(args.max_fallback_patches_per_scene),
            "positive_patches_per_scene": int(args.positive_patches_per_scene),
            "negative_patches_per_scene": int(args.negative_patches_per_scene),
            "selection_seed": int(args.selection_seed if args.selection_seed is not None else args.seed),
            "max_patch_height": int(args.max_patch_height),
            "max_patch_width": int(args.max_patch_width),
        },
        "patch_cache": {
            "mode": patch_cache_mode,
            "dir": None if patch_cache_dir is None else str(patch_cache_dir),
            "metadata_by_split": cache_metadata_by_split,
        },
        "git": git_metadata(),
        "scene_counts": {
            split: {
                "input_scenes": int(len(grouped[split])),
                "kept_scenes": (
                    cache_metadata_by_split.get(split, {}).get("metadata", {}).get("kept_scene_count")
                    if split in cache_metadata_by_split
                    else int(sum((not item.excluded_reason) and item.record.split == split for item in all_scene_results))
                ),
                "selected_patches": int(len(selected_by_split[split])),
                "selected_target_sum": float(sum(patch.target_sum for patch in selected_by_split[split])),
            }
            for split in ("train", "val", "test")
        },
        "train_history": train_history,
        "test": test_metrics,
        "preview_paths": preview_paths,
        "outputs": {
            "config_snapshot": str(config_snapshot_path),
            "scene_metrics_csv": str(output_dir / "scene_metrics.csv"),
            "filtered_scene_split_csv": str(output_dir / "filtered_scene_split.csv"),
            "inference_preview_dir": str(output_dir / "inference_previews"),
        },
        "checkpoint_saved": False,
        "best_checkpoints": best_checkpoints,
    }
    dirty_patch_path = write_dirty_patch(output_dir, summary["git"])
    if dirty_patch_path is not None:
        summary["outputs"]["run_git_dirty_patch"] = dirty_patch_path
    if bool(args.save_checkpoint):
        checkpoint_path = save_checkpoint(
            output_dir / "checkpoints" / "checkpoint_last.pt",
            model=model,
            optimizer=optimizer,
            config=config,
            args=args,
            train_history=train_history,
            test_metrics=test_metrics,
        )
        summary["checkpoint_saved"] = True
        summary["outputs"]["checkpoint_last"] = checkpoint_path
        for name, info in best_checkpoints.items():
            summary["outputs"][f"checkpoint_{name}"] = str(info["path"])
    summary_path = output_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary_path": str(summary_path), "scene_counts": summary["scene_counts"], "test": test_metrics, "preview_count": len(preview_paths)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
