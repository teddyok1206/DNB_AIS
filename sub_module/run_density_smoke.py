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
from .dnb_density_models import build_density_model
from .dnb_gat_pipeline import GroundTruthResolver, SceneRaster


ROOT = Path(__file__).resolve().parents[1]
STEP3 = ROOT / "[3]_DNB_AIS - (STEP 3)"
DEFAULT_SCENE_TIF = STEP3 / "DRUID_TESTING" / "TEST_5_A2025001_1754_021_batch_1.tif"
DEFAULT_GEOJSON = STEP3 / "bboxes_JPSS-2" / "A2025001_1754_021.geojson"
DEFAULT_METADATA = STEP3 / "metadata_JPSS-2.csv"
DEFAULT_SHIPS_DB = STEP3 / "ships.db"


def _loss_name_from_config(value: Any) -> str:
    normalized = str(value).strip().lower()
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
        if source_key in detector:
            defaults[target_key] = detector[source_key]

    patching = config.get("patching", {})
    patching_map = {"padding_pixels": "padding_pixels", "size_divisor": "size_divisor"}
    for source_key, target_key in patching_map.items():
        if source_key in patching:
            defaults[target_key] = patching[source_key]

    target = config.get("target", {})
    target_map = {
        "sigma_pixels": "target_sigma",
        "radius_pixels": "target_radius",
        "per_ship_mass": "target_per_ship_mass",
        "renormalize_after_roi_mask": "target_renormalize_after_roi_mask",
        "require_source_in_roi": "target_require_source_in_roi",
    }
    for source_key, target_key in target_map.items():
        if source_key in target:
            defaults[target_key] = target[source_key]

    model = config.get("model", {})
    model_name = str(model.get("name", "")).lower()
    if "unet" in model_name:
        defaults["model"] = "main"
    elif "dilated" in model_name or "csrnet" in model_name:
        defaults["model"] = "fast"
    for key in ["base_channels", "depth", "activation", "dropout"]:
        if key in model:
            defaults[key] = model[key]
    if "dilations" in model:
        defaults["dilations"] = ",".join(str(int(v)) for v in model["dilations"])

    training = config.get("training", {})
    training_map = {"batch_size": "batch_size", "lr": "lr", "weight_decay": "weight_decay"}
    for source_key, target_key in training_map.items():
        if source_key in training:
            defaults[target_key] = training[source_key]
    if "loss" in training:
        defaults["loss"] = _loss_name_from_config(training["loss"])
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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke-test PH-masked density models on a DNB batch scene.")
    parser.add_argument("--config", type=Path, default=None, help="Optional JSON config used as parser defaults.")
    parser.add_argument("--model", choices=["main", "fast", "both"], default="both")
    parser.add_argument("--scene-tif", type=Path, default=DEFAULT_SCENE_TIF)
    parser.add_argument("--gt-geojson", type=Path, default=DEFAULT_GEOJSON)
    parser.add_argument("--metadata-csv", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--ships-db", type=Path, default=DEFAULT_SHIPS_DB)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--max-patches", type=int, default=24)
    parser.add_argument("--padding-pixels", type=int, default=16)
    parser.add_argument("--size-divisor", type=int, default=16)
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--dilations", default="1,2,4,8,4,2")
    parser.add_argument("--activation", choices=["softplus", "relu", "identity", "linear", "none"], default="softplus")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--loss", choices=["mse", "poisson_nll"], default="mse")
    parser.add_argument("--target-sigma", type=float, default=1.5)
    parser.add_argument("--target-radius", type=int, default=5)
    parser.add_argument("--target-per-ship-mass", type=float, default=1.0)
    parser.add_argument("--target-renormalize-after-roi-mask", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--target-require-source-in-roi", action=argparse.BooleanOptionalAction, default=True)
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
    parser.add_argument("--detector-drop-nested", action=argparse.BooleanOptionalAction, default=True)
    return parser


def make_models_to_run(model_arg: str) -> list[str]:
    if model_arg == "both":
        return ["main", "fast"]
    return [model_arg]


def run_model_smoke(model_name: str, loader: DataLoader, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    model_kwargs: dict[str, Any] = {
        "in_channels": 2,
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
    losses: list[float] = []
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
            if str(args.loss) == "poisson_nll":
                loss = masked_poisson_nll_loss(pred, batch["target"], batch["roi_mask"])
            else:
                loss = masked_mse_loss(pred, batch["target"], batch["roi_mask"])
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            pred_sum = float((pred.detach() * batch["roi_mask"]).sum().cpu())
            target_sum = float((batch["target"] * batch["roi_mask"]).sum().cpu())
            roi_pixels = float(batch["roi_mask"].sum().cpu())
            last_shape = [int(v) for v in pred.shape]
            step += 1
            if step >= int(args.steps):
                break

    result = {
        "model": model_name,
        **count_parameters(model),
        "device": str(device),
        "loss_name": str(args.loss),
        "losses": losses,
        "last_pred_shape": last_shape,
        "last_roi_pred_sum": pred_sum,
        "last_roi_target_sum": target_sum,
        "last_roi_pixels": roi_pixels,
    }
    if hasattr(model, "architecture_dict"):
        result["architecture"] = model.architecture_dict()
    return result


def main(argv: list[str] | None = None) -> int:
    args = _parse_args_with_config(argv)
    seed_everything(int(args.seed))
    device = resolve_device(str(args.device))

    scene = SceneRaster.load(args.scene_tif)
    resolver = GroundTruthResolver(args.metadata_csv, args.ships_db, args.gt_geojson.parent)
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
    store = DnbCandidateDetector(detector_config).build_store(scene, gt_count_map)

    patch_config = DensityPatchConfig(
        padding_pixels=int(args.padding_pixels),
        size_divisor=int(args.size_divisor),
        max_patches=int(args.max_patches) if int(args.max_patches) > 0 else None,
    )
    target_config = DensityTargetConfig(
        sigma_pixels=float(args.target_sigma),
        radius_pixels=int(args.target_radius),
        per_ship_mass=float(args.target_per_ship_mass),
        renormalize_after_roi_mask=bool(args.target_renormalize_after_roi_mask),
        require_source_in_roi=bool(args.target_require_source_in_roi),
    )
    patches = build_density_patches(scene, gt_count_map, store, patch_config=patch_config, target_config=target_config)
    if not patches:
        raise RuntimeError("No density patches were built")

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
        "detector_summary": candidate_store_summary(store),
        "patch_summary": summarize_density_patches(patches),
        "target_config": target_config.__dict__,
        "patch_config": patch_config.__dict__,
        "model_results": model_results,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
