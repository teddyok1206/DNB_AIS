from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dnb_candidate_detector import DnbCandidateDetector, DnbCandidateDetectorConfig, candidate_store_summary
from .dnb_density_common import (
    DensityPatchConfig,
    DensityPatchDataset,
    DensityTargetConfig,
    build_density_patches,
    density_patch_collate,
    masked_mse_loss,
    masked_poisson_nll_loss,
    move_density_batch_to_device,
    summarize_density_patches,
)
from .dnb_density_losses import build_density_loss
from .dnb_density_models import build_density_model
from .dnb_pipeline_core import GroundTruthResolver, SceneRaster
from .dnb_project_paths import DENSITY_OUTPUT_ROOT, project_path
from .kr_sea_mask import apply_kr_sea_mask
from .dnb_scene_partition import ScenePartitionConfig, build_partitioned_density_patches
from .dnb_ph_downsample import PHDownsampleConfig, build_ph_anchor_store


ROOT = Path(__file__).resolve().parents[1]
STEP3 = ROOT / "[3]_DNB_AIS - (STEP 3)"
DEFAULT_SCENE_TIF = STEP3 / "A2025001_1754_021.tif"
DEFAULT_GEOJSON = STEP3 / "bboxes_JPSS-2" / "A2025001_1754_021.geojson"
DEFAULT_METADATA = STEP3 / "metadata_JPSS-2.csv"


def _loss_name_from_config(value: Any) -> str:
    if isinstance(value, dict):
        return _loss_name_from_config(value.get("name", "structured_density_loss"))
    normalized = str(value).strip().lower()
    if "structured" in normalized or "composite" in normalized:
        return "structured"
    if "poisson" in normalized:
        return "poisson_nll"
    if "mse" in normalized:
        return "mse"
    return normalized


def _config_defaults(config_path: Path | None) -> dict[str, Any]:
    if config_path is None:
        return {}
    with Path(config_path).expanduser().open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    defaults: dict[str, Any] = {}
    role = config.get("role")
    if role in {"main", "fast"}:
        defaults["model"] = role
    if "device" in config:
        defaults["device"] = config["device"]

    sea_mask = config.get("sea_mask", {})
    sea_mask_map = {
        "enabled": "kr_eez_mask",
        "step3_dir": "kr_eez_step3_dir",
        "crop_to_bounds": "kr_eez_crop_to_bounds",
        "segment_policy": "kr_eez_segment_policy",
        "write_masked_tif": "kr_eez_write_masked_tif",
        "output_dir": "kr_eez_mask_output_dir",
        "all_touched": "kr_eez_all_touched",
    }
    for source_key, target_key in sea_mask_map.items():
        if source_key in sea_mask and sea_mask[source_key] is not None:
            defaults[target_key] = sea_mask[source_key]

    detector = config.get("detector", {})
    detector_map = {
        "detection_threshold": "detector_detection_threshold",
        "analysis_threshold": "detector_analysis_threshold",
        "threshold_reference": "detector_threshold_reference",
        "smooth_sigma": "detector_smooth_sigma",
        "area_limit": "detector_area_limit",
        "min_nodes": "detector_min_nodes",
        "max_nodes": "detector_max_nodes",
        "max_candidates": "detector_max_candidates",
        "connectivity": "detector_connectivity",
        "remove_edge": "detector_remove_edge",
        "drop_nested": "detector_drop_nested",
        "lifetime_limit": "detector_lifetime_limit",
        "lifetime_limit_fraction": "detector_lifetime_limit_fraction",
    }
    for source_key, target_key in detector_map.items():
        if source_key in detector and detector[source_key] is not None:
            defaults[target_key] = detector[source_key]

    ph_downsample = config.get("ph_downsample", {})
    ph_downsample_map = {
        "factor": "ph_downsample_factor",
        "reducer": "ph_downsample_reducer",
    }
    for source_key, target_key in ph_downsample_map.items():
        if source_key in ph_downsample and ph_downsample[source_key] is not None:
            defaults[target_key] = ph_downsample[source_key]

    patching = config.get("patching", {})
    patching_map = {
        "padding_pixels": "padding_pixels",
        "size_divisor": "size_divisor",
        "sort_by": "sort_by",
        "parent_min_nodes": "parent_min_nodes",
        "parent_max_nodes": "parent_max_nodes",
        "child_min_nodes": "child_min_nodes",
        "child_max_nodes": "child_max_nodes",
        "max_children": "max_children",
        "seed_radius_pixels": "seed_radius_pixels",
        "attention_distance_sigma": "attention_distance_sigma",
        "attention_base_weight": "attention_base_weight",
        "attention_ph_weight": "attention_ph_weight",
    }
    for source_key, target_key in patching_map.items():
        if source_key in patching and patching[source_key] is not None:
            defaults[target_key] = patching[source_key]

    partitioning = config.get("partitioning", {})
    partitioning_map = {
        "enabled": "partitioning",
        "fallback_tile_pixels": "partition_fallback_tile_pixels",
        "halo_pixels": "partition_halo_pixels",
        "anchor_padding_pixels": "partition_anchor_padding_pixels",
        "min_owner_pixels": "partition_min_owner_pixels",
        "min_fallback_owner_pixels": "partition_min_fallback_owner_pixels",
    }
    for source_key, target_key in partitioning_map.items():
        if source_key in partitioning and partitioning[source_key] is not None:
            defaults[target_key] = partitioning[source_key]
    hierarchical = partitioning.get("hierarchical_ph", {})
    hierarchical_map = {
        "enabled": "partition_hierarchical_ph",
        "large_min_pixels": "partition_hierarchical_large_min_pixels",
        "large_min_height": "partition_hierarchical_large_min_height",
        "large_min_width": "partition_hierarchical_large_min_width",
        "child_anchor_padding_pixels": "partition_hierarchical_child_anchor_padding_pixels",
        "child_detection_threshold": "partition_hierarchical_child_detection_threshold",
        "child_analysis_threshold": "partition_hierarchical_child_analysis_threshold",
        "child_threshold_reference": "partition_hierarchical_child_threshold_reference",
        "child_smooth_sigma": "partition_hierarchical_child_smooth_sigma",
        "child_lifetime_limit": "partition_hierarchical_child_lifetime_limit",
        "child_lifetime_limit_fraction": "partition_hierarchical_child_lifetime_limit_fraction",
        "child_area_limit": "partition_hierarchical_child_area_limit",
        "child_min_nodes": "partition_hierarchical_child_min_nodes",
        "child_max_nodes": "partition_hierarchical_child_max_nodes",
        "child_max_candidates_per_parent": "partition_hierarchical_child_max_candidates_per_parent",
        "keep_large_parent": "partition_hierarchical_keep_large_parent",
    }
    for source_key, target_key in hierarchical_map.items():
        if source_key in hierarchical and hierarchical[source_key] is not None:
            defaults[target_key] = hierarchical[source_key]

    target = config.get("target", {})
    target_map = {
        "sigma_pixels": "target_sigma",
        "radius_pixels": "target_radius",
        "per_ship_mass": "target_per_ship_mass",
        "renormalize_after_roi_mask": "target_renormalize_after_roi_mask",
        "require_source_in_roi": "target_require_source_in_roi",
    }
    for source_key, target_key in target_map.items():
        if source_key in target and target[source_key] is not None:
            defaults[target_key] = target[source_key]

    model = config.get("model", {})
    model_name = str(model.get("name", "")).lower()
    if "unet" in model_name:
        defaults["model"] = "main"
    elif "dilated" in model_name or "csrnet" in model_name:
        defaults["model"] = "fast"
    for key in ["in_channels", "base_channels", "depth", "activation", "dropout"]:
        if key in model and model[key] is not None:
            target_key = "input_channels" if key == "in_channels" else key
            defaults[target_key] = model[key]
    if "dilations" in model:
        defaults["dilations"] = ",".join(str(int(v)) for v in model["dilations"])

    training = config.get("training", {})
    training_map = {"batch_size": "batch_size", "lr": "lr", "weight_decay": "weight_decay"}
    for source_key, target_key in training_map.items():
        if source_key in training:
            defaults[target_key] = training[source_key]
    if "loss" in training:
        loss_config = training["loss"]
        defaults["loss"] = _loss_name_from_config(loss_config)
        if isinstance(loss_config, dict):
            loss_map = {
                "pixel_weight": "loss_pixel_weight",
                "count_weight": "loss_count_weight",
                "batch_count_weight": "loss_batch_count_weight",
                "local_count_weight": "loss_local_count_weight",
                "background_weight": "loss_background_weight",
                "pixel_loss": "loss_pixel_loss",
                "huber_delta": "loss_huber_delta",
                "count_loss": "loss_count_loss",
                "count_huber_delta": "loss_count_huber_delta",
                "count_normalizer": "loss_count_normalizer",
                "local_count_windows": "loss_local_count_windows",
                "local_count_stride_factor": "loss_local_count_stride_factor",
                "background_target_threshold": "loss_background_target_threshold",
            }
            for source_key, target_key in loss_map.items():
                if source_key in loss_config and loss_config[source_key] is not None:
                    value = loss_config[source_key]
                    if source_key == "local_count_windows" and not isinstance(value, str):
                        value = ",".join(str(int(item)) for item in value)
                    defaults[target_key] = value
    return defaults


def _parse_args_with_config(argv: list[str] | None) -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=None)
    config_args, _ = config_parser.parse_known_args(argv)

    parser = build_arg_parser()
    defaults = _config_defaults(config_args.config)
    if defaults:
        parser.set_defaults(**defaults)
    return parser.parse_args(argv)


def resolve_device(requested: str) -> torch.device:
    requested = str(requested).lower()
    if requested == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        print("[warn] MPS requested but unavailable; falling back to CPU")
        return torch.device("cpu")
    if requested == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("[warn] CUDA requested but unavailable; falling back to CPU")
        return torch.device("cpu")
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(requested)


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    total = int(sum(p.numel() for p in model.parameters()))
    trainable = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    return {"parameter_count": total, "trainable_parameter_count": trainable}


def parse_dilations(value: str) -> tuple[int, ...]:
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if not parts:
        raise ValueError("dilations must contain at least one integer")
    return tuple(int(part) for part in parts)


def save_density_patch_previews(
    patches: list[Any],
    *,
    scene_key: str,
    output_dir: Path,
    limit: int,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for patch in patches[: max(int(limit), 0)]:
        panels = [
            ("brightness", patch.image, "magma"),
            ("parent PH mask", patch.parent_mask, "gray"),
            ("child PH union", patch.child_union_mask, "gray"),
            ("seed map", patch.seed_map, "gray"),
            ("persistence map", patch.persistence_map, "viridis"),
            ("soft attention", patch.soft_attention, "viridis"),
            ("raw GT count", patch.raw_count, "viridis"),
            ("crop-level density target", patch.target_density, "viridis"),
            ("loss weight", patch.loss_weight, "viridis"),
        ]
        fig, axes = plt.subplots(3, 3, figsize=(14, 11), constrained_layout=True)
        for ax, (title, arr, cmap) in zip(axes.ravel(), panels):
            im = ax.imshow(arr, cmap=cmap)
            ax.set_title(title)
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(
            (
                f"{scene_key} | parent={patch.cluster_id} | children={len(patch.child_ids)} | "
                f"raw_sum={patch.raw_count_sum:.1f} | target_sum={patch.target_sum:.1f}"
            ),
            fontsize=13,
        )
        path = output_dir / f"{scene_key}_parent_{int(patch.cluster_id)}_hierarchical_unet_preview.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(str(path))
    return paths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke-test PH-hierarchical U-Net density models on a DNB batch scene.")
    parser.add_argument("--config", type=Path, default=None, help="Optional JSON config used as parser defaults.")
    parser.add_argument("--model", choices=["main", "fast", "both"], default="main")
    parser.add_argument("--scene-tif", type=Path, default=DEFAULT_SCENE_TIF)
    parser.add_argument("--gt-geojson", type=Path, default=DEFAULT_GEOJSON)
    parser.add_argument("--kr-eez-mask", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--kr-eez-step3-dir", type=Path, default=STEP3)
    parser.add_argument("--kr-eez-crop-to-bounds", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--kr-eez-segment-policy", choices=["single_scene", "largest_segment"], default="single_scene")
    parser.add_argument("--kr-eez-write-masked-tif", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--kr-eez-mask-output-dir", type=Path, default=DENSITY_OUTPUT_ROOT / "preprocessed_scene_masks" / "density")
    parser.add_argument("--kr-eez-all-touched", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--max-patches", type=int, default=24)
    parser.add_argument("--padding-pixels", type=int, default=16)
    parser.add_argument("--size-divisor", type=int, default=16)
    parser.add_argument("--partitioning", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--partition-fallback-tile-pixels", type=int, default=96)
    parser.add_argument("--partition-halo-pixels", type=int, default=16)
    parser.add_argument("--partition-anchor-padding-pixels", type=int, default=16)
    parser.add_argument("--partition-min-owner-pixels", type=int, default=1)
    parser.add_argument("--partition-min-fallback-owner-pixels", type=int, default=1)
    parser.add_argument("--partition-hierarchical-ph", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--partition-hierarchical-large-min-pixels", type=int, default=65536)
    parser.add_argument("--partition-hierarchical-large-min-height", type=int, default=384)
    parser.add_argument("--partition-hierarchical-large-min-width", type=int, default=384)
    parser.add_argument("--partition-hierarchical-child-anchor-padding-pixels", type=int, default=8)
    parser.add_argument("--partition-hierarchical-child-detection-threshold", type=float, default=0.5)
    parser.add_argument("--partition-hierarchical-child-analysis-threshold", type=float, default=0.25)
    parser.add_argument("--partition-hierarchical-child-threshold-reference", choices=["zero", "median"], default="median")
    parser.add_argument("--partition-hierarchical-child-smooth-sigma", type=float, default=0.0)
    parser.add_argument("--partition-hierarchical-child-lifetime-limit", type=float, default=0.0)
    parser.add_argument("--partition-hierarchical-child-lifetime-limit-fraction", type=float, default=1.0005)
    parser.add_argument("--partition-hierarchical-child-area-limit", type=int, default=0)
    parser.add_argument("--partition-hierarchical-child-min-nodes", type=int, default=3)
    parser.add_argument("--partition-hierarchical-child-max-nodes", type=int, default=2048)
    parser.add_argument("--partition-hierarchical-child-max-candidates-per-parent", type=int, default=64)
    parser.add_argument("--partition-hierarchical-keep-large-parent", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--sort-by", choices=["lifetime", "cluster_id", "node_count"], default="node_count")
    parser.add_argument("--parent-min-nodes", type=int, default=32)
    parser.add_argument("--parent-max-nodes", type=int, default=0)
    parser.add_argument("--child-min-nodes", type=int, default=4)
    parser.add_argument("--child-max-nodes", type=int, default=0)
    parser.add_argument("--max-children", type=int, default=0)
    parser.add_argument("--seed-radius-pixels", type=int, default=1)
    parser.add_argument("--attention-distance-sigma", type=float, default=4.0)
    parser.add_argument("--attention-base-weight", type=float, default=0.25)
    parser.add_argument("--attention-ph-weight", type=float, default=0.75)
    parser.add_argument("--input-channels", type=int, default=6)
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--dilations", default="1,2,4,8,4,2")
    parser.add_argument("--activation", choices=["softplus", "relu", "identity", "linear", "none"], default="softplus")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--loss", choices=["mse", "poisson_nll", "structured"], default="mse")
    parser.add_argument("--loss-pixel-weight", type=float, default=0.45)
    parser.add_argument("--loss-count-weight", type=float, default=0.22)
    parser.add_argument("--loss-batch-count-weight", type=float, default=0.08)
    parser.add_argument("--loss-local-count-weight", type=float, default=0.20)
    parser.add_argument("--loss-background-weight", type=float, default=0.05)
    parser.add_argument("--loss-pixel-loss", choices=["mse", "mae", "huber"], default="huber")
    parser.add_argument("--loss-huber-delta", type=float, default=0.05)
    parser.add_argument("--loss-count-loss", choices=["relative_mse", "relative_mae", "relative_huber"], default="relative_huber")
    parser.add_argument("--loss-count-huber-delta", type=float, default=0.25)
    parser.add_argument("--loss-count-normalizer", type=float, default=1.0)
    parser.add_argument("--loss-local-count-windows", default="16,32,64")
    parser.add_argument("--loss-local-count-stride-factor", type=int, default=2)
    parser.add_argument("--loss-background-target-threshold", type=float, default=1.0e-6)
    parser.add_argument("--target-sigma", type=float, default=1.5)
    parser.add_argument("--target-radius", type=int, default=5)
    parser.add_argument("--target-per-ship-mass", type=float, default=1.0)
    parser.add_argument("--target-renormalize-after-roi-mask", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--target-require-source-in-roi", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--detector-detection-threshold", type=float, default=1.0)
    parser.add_argument("--detector-analysis-threshold", type=float, default=1.0)
    parser.add_argument("--detector-threshold-reference", choices=["zero", "median"], default="zero")
    parser.add_argument("--detector-smooth-sigma", type=float, default=0.0)
    parser.add_argument("--detector-area-limit", type=int, default=12)
    parser.add_argument("--detector-min-nodes", type=int, default=16)
    parser.add_argument("--detector-max-nodes", type=int, default=2500)
    parser.add_argument("--detector-max-candidates", type=int, default=0)
    parser.add_argument("--detector-connectivity", type=int, choices=[1, 2], default=1)
    parser.add_argument("--detector-remove-edge", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--detector-lifetime-limit", type=float, default=0.0)
    parser.add_argument("--detector-lifetime-limit-fraction", type=float, default=1.001)
    parser.add_argument("--detector-drop-nested", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ph-downsample-factor", type=int, default=1)
    parser.add_argument("--ph-downsample-reducer", choices=["max", "mean"], default="max")
    parser.add_argument("--preview-dir", type=Path, default=None)
    parser.add_argument("--preview-patches", type=int, default=3)
    return parser


def make_models_to_run(model_arg: str) -> list[str]:
    if model_arg == "both":
        return ["main", "fast"]
    return [model_arg]


def parse_loss_windows(value: str) -> tuple[int, ...]:
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if not parts:
        raise ValueError("loss local-count windows must contain at least one integer")
    return tuple(int(part) for part in parts)


def structured_loss_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "name": "structured_density_loss",
        "pixel_weight": float(args.loss_pixel_weight),
        "count_weight": float(args.loss_count_weight),
        "batch_count_weight": float(args.loss_batch_count_weight),
        "local_count_weight": float(args.loss_local_count_weight),
        "background_weight": float(args.loss_background_weight),
        "pixel_loss": str(args.loss_pixel_loss),
        "huber_delta": float(args.loss_huber_delta),
        "count_loss": str(args.loss_count_loss),
        "count_huber_delta": float(args.loss_count_huber_delta),
        "count_normalizer": float(args.loss_count_normalizer),
        "local_count_windows": parse_loss_windows(str(args.loss_local_count_windows)),
        "local_count_stride_factor": int(args.loss_local_count_stride_factor),
        "background_target_threshold": float(args.loss_background_target_threshold),
    }


def run_model_smoke(model_name: str, loader: DataLoader, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    model_kwargs: dict[str, Any] = {
        "in_channels": int(args.input_channels),
        "out_channels": 1,
        "base_channels": int(args.base_channels),
        "depth": int(args.depth),
        "activation": str(args.activation),
        "dropout": float(args.dropout),
    }
    if model_name == "fast":
        model_kwargs["dilations"] = parse_dilations(str(args.dilations))
    model = build_density_model(
        model_name,
        **model_kwargs,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    structured_loss = None
    if str(args.loss) == "structured":
        structured_loss = build_density_loss(structured_loss_config_from_args(args))
    losses: list[float] = []
    loss_components: list[dict[str, float]] = []
    pred_sum = 0.0
    target_sum = 0.0
    roi_pixels = 0.0
    last_shape: list[int] = []

    model.train()
    step = 0
    while step < int(args.steps):
        for batch in loader:
            batch = move_density_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch["x"])
            if structured_loss is not None:
                loss = structured_loss(pred, batch)
                if structured_loss.last_components:
                    loss_components.append(dict(structured_loss.last_components))
            elif str(args.loss) == "poisson_nll":
                loss = masked_poisson_nll_loss(pred, batch["target"], batch["loss_weight"])
            else:
                loss = masked_mse_loss(pred, batch["target"], batch["loss_weight"])
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            pred_sum = float((pred.detach() * batch["valid_mask"]).sum().cpu())
            target_sum = float((batch["target"] * batch["valid_mask"]).sum().cpu())
            roi_pixels = float(batch["loss_weight"].sum().cpu())
            last_shape = [int(v) for v in pred.shape]
            step += 1
            if step >= int(args.steps):
                break

    result = {
        "model": model_name,
        **count_parameters(model),
        "device": str(device),
        "loss_name": str(args.loss),
        "loss_config": structured_loss_config_from_args(args) if structured_loss is not None else None,
        "losses": losses,
        "loss_components": loss_components,
        "last_pred_shape": last_shape,
        "last_valid_pred_sum": pred_sum,
        "last_valid_target_sum": target_sum,
        "last_loss_weight_sum": roi_pixels,
    }
    if hasattr(model, "architecture_dict"):
        result["architecture"] = model.architecture_dict()
    return result


def main(argv: list[str] | None = None) -> int:
    args = _parse_args_with_config(argv)
    seed_everything(int(args.seed))
    device = resolve_device(str(args.device))

    sea_mask_result: dict[str, Any] = {"enabled": False}
    valid_sea_mask: np.ndarray | None = None
    if bool(args.kr_eez_mask):
        mask_result = apply_kr_sea_mask(
            args.scene_tif,
            step3_dir=args.kr_eez_step3_dir,
            output_dir=project_path(args.kr_eez_mask_output_dir),
            crop_to_bounds=bool(args.kr_eez_crop_to_bounds),
            segment_policy=str(args.kr_eez_segment_policy),
            write_masked_tif=bool(args.kr_eez_write_masked_tif),
            all_touched=bool(args.kr_eez_all_touched),
        )
        scene = mask_result.scene
        valid_sea_mask = mask_result.valid_mask
        sea_mask_result = mask_result.metadata
    else:
        scene = SceneRaster.load(args.scene_tif)
    resolver = GroundTruthResolver(args.gt_geojson.parent)
    gt_path = resolver.resolve_geojson(scene, args.gt_geojson)
    gt_points = resolver.load_points(gt_path)
    gt_count_map = resolver.rasterize_counts(scene, gt_points)

    detector_config = DnbCandidateDetectorConfig(
        detection_threshold=float(args.detector_detection_threshold),
        analysis_threshold=float(args.detector_analysis_threshold),
        threshold_reference=str(args.detector_threshold_reference),
        smooth_sigma=float(args.detector_smooth_sigma),
        lifetime_limit=float(args.detector_lifetime_limit),
        lifetime_limit_fraction=float(args.detector_lifetime_limit_fraction),
        area_limit=int(args.detector_area_limit),
        min_nodes=int(args.detector_min_nodes),
        max_nodes=int(args.detector_max_nodes),
        max_candidates=int(args.detector_max_candidates) if int(args.detector_max_candidates) > 0 else None,
        connectivity=int(args.detector_connectivity),
        remove_edge=bool(args.detector_remove_edge),
        drop_nested=bool(args.detector_drop_nested),
    )
    ph_anchor_result = build_ph_anchor_store(
        scene,
        gt_count_map,
        detector_config,
        valid_mask=valid_sea_mask,
        downsample_config=PHDownsampleConfig(
            factor=int(args.ph_downsample_factor),
            reducer=str(args.ph_downsample_reducer),
        ),
    )
    store = ph_anchor_result.store

    patch_config = DensityPatchConfig(
        padding_pixels=int(args.padding_pixels),
        size_divisor=int(args.size_divisor),
        max_patches=int(args.max_patches) if int(args.max_patches) > 0 else None,
        sort_by=str(args.sort_by),
        parent_min_nodes=int(args.parent_min_nodes),
        parent_max_nodes=int(args.parent_max_nodes) if int(args.parent_max_nodes) > 0 else None,
        child_min_nodes=int(args.child_min_nodes),
        child_max_nodes=int(args.child_max_nodes) if int(args.child_max_nodes) > 0 else None,
        max_children=int(args.max_children) if int(args.max_children) > 0 else None,
        seed_radius_pixels=int(args.seed_radius_pixels),
        attention_distance_sigma=float(args.attention_distance_sigma),
        attention_base_weight=float(args.attention_base_weight),
        attention_ph_weight=float(args.attention_ph_weight),
    )
    target_config = DensityTargetConfig(
        sigma_pixels=float(args.target_sigma),
        radius_pixels=int(args.target_radius),
        per_ship_mass=float(args.target_per_ship_mass),
        renormalize_after_roi_mask=bool(args.target_renormalize_after_roi_mask),
        require_source_in_roi=bool(args.target_require_source_in_roi),
    )
    partition_summary: dict[str, Any] | None = None
    if bool(args.partitioning):
        partition_config = ScenePartitionConfig(
            enabled=True,
            fallback_tile_pixels=int(args.partition_fallback_tile_pixels),
            halo_pixels=int(args.partition_halo_pixels),
            anchor_padding_pixels=int(args.partition_anchor_padding_pixels),
            min_owner_pixels=int(args.partition_min_owner_pixels),
            min_fallback_owner_pixels=int(args.partition_min_fallback_owner_pixels),
            hierarchical_ph_enabled=bool(args.partition_hierarchical_ph),
            hierarchical_large_min_pixels=int(args.partition_hierarchical_large_min_pixels),
            hierarchical_large_min_height=int(args.partition_hierarchical_large_min_height),
            hierarchical_large_min_width=int(args.partition_hierarchical_large_min_width),
            hierarchical_child_anchor_padding_pixels=int(args.partition_hierarchical_child_anchor_padding_pixels),
            hierarchical_child_detection_threshold=float(args.partition_hierarchical_child_detection_threshold),
            hierarchical_child_analysis_threshold=float(args.partition_hierarchical_child_analysis_threshold),
            hierarchical_child_threshold_reference=str(args.partition_hierarchical_child_threshold_reference),
            hierarchical_child_smooth_sigma=float(args.partition_hierarchical_child_smooth_sigma),
            hierarchical_child_lifetime_limit=float(args.partition_hierarchical_child_lifetime_limit),
            hierarchical_child_lifetime_limit_fraction=float(args.partition_hierarchical_child_lifetime_limit_fraction),
            hierarchical_child_area_limit=int(args.partition_hierarchical_child_area_limit),
            hierarchical_child_min_nodes=int(args.partition_hierarchical_child_min_nodes),
            hierarchical_child_max_nodes=int(args.partition_hierarchical_child_max_nodes),
            hierarchical_child_max_candidates_per_parent=int(args.partition_hierarchical_child_max_candidates_per_parent),
            hierarchical_keep_large_parent=bool(args.partition_hierarchical_keep_large_parent),
        )
        patches, _, partition_summary = build_partitioned_density_patches(
            scene,
            gt_count_map,
            store,
            valid_mask=valid_sea_mask,
            patch_config=patch_config,
            target_config=target_config,
            partition_config=partition_config,
        )
    else:
        patches = build_density_patches(
            scene,
            gt_count_map,
            store,
            valid_mask=valid_sea_mask,
            patch_config=patch_config,
            target_config=target_config,
        )
    if not patches:
        raise RuntimeError("No density patches were built")
    preview_paths: list[str] = []
    if args.preview_dir is not None:
        preview_paths = save_density_patch_previews(
            patches,
            scene_key=scene.key,
            output_dir=args.preview_dir,
            limit=int(args.preview_patches),
        )

    dataset = DensityPatchDataset(patches)
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        collate_fn=lambda batch: density_patch_collate(batch, size_divisor=int(args.size_divisor)),
    )

    model_results = [run_model_smoke(name, loader, args, device) for name in make_models_to_run(str(args.model))]
    report = {
        "scene_key": scene.key,
        "scene_shape": [int(scene.height), int(scene.width)],
        "gt_path": str(gt_path),
        "gt_point_count": int(len(gt_points)),
        "gt_count_sum": float(gt_count_map.sum()),
        "sea_mask": sea_mask_result,
        "ph_downsample": ph_anchor_result.metadata,
        "partitioning_enabled": bool(args.partitioning),
        "partition_summary": partition_summary,
        "detector_summary": candidate_store_summary(store),
        "patch_summary": summarize_density_patches(patches),
        "target_config": target_config.__dict__,
        "patch_config": patch_config.__dict__,
        "preview_paths": preview_paths,
        "model_results": model_results,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
