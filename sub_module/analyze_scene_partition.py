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

from .dnb_candidate_detector import DnbCandidateDetectorConfig, candidate_store_summary
from .dnb_density_common import DensityPatchConfig, DensityTargetConfig
from .dnb_pipeline_core import GroundTruthResolver, SceneRaster
from .dnb_ph_downsample import PHDownsampleConfig, build_ph_anchor_store
from .dnb_scene_partition import ScenePartition, ScenePartitionConfig, build_partitioned_density_patches
from .kr_sea_mask import apply_kr_sea_mask
from .dnb_project_paths import ROOT, STEP3


DEFAULT_SCENE_TIF = STEP3 / "A2025001_1754_021.tif"
DEFAULT_GEOJSON = STEP3 / "bboxes_JPSS-2" / "A2025001_1754_021.geojson"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect PH-anchor plus fallback-grid sea-domain partitioning.")
    parser.add_argument("--scene-tif", type=Path, default=DEFAULT_SCENE_TIF)
    parser.add_argument("--gt-geojson", type=Path, default=DEFAULT_GEOJSON)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "scene_partition")
    parser.add_argument("--kr-eez-mask", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--kr-eez-step3-dir", type=Path, default=STEP3)
    parser.add_argument("--kr-eez-crop-to-bounds", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--kr-eez-segment-policy", choices=["single_scene", "largest_segment"], default="single_scene")
    parser.add_argument("--kr-eez-all-touched", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--detector-detection-threshold", type=float, default=1.0)
    parser.add_argument("--detector-analysis-threshold", type=float, default=0.25)
    parser.add_argument("--detector-threshold-reference", choices=["zero", "median"], default="median")
    parser.add_argument("--detector-area-limit", type=int, default=0)
    parser.add_argument("--detector-min-nodes", type=int, default=4)
    parser.add_argument("--detector-max-nodes", type=int, default=2500)
    parser.add_argument("--detector-remove-edge", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ph-downsample-factor", type=int, default=4)
    parser.add_argument("--ph-downsample-reducer", choices=["max", "mean"], default="max")
    parser.add_argument("--parent-min-nodes", type=int, default=32)
    parser.add_argument("--child-min-nodes", type=int, default=4)
    parser.add_argument("--fallback-tile-pixels", type=int, default=96)
    parser.add_argument("--halo-pixels", type=int, default=16)
    parser.add_argument("--anchor-padding-pixels", type=int, default=16)
    parser.add_argument("--target-sigma", type=float, default=1.5)
    parser.add_argument("--target-radius", type=int, default=5)
    return parser


def robust_norm(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    lo = float(np.nanpercentile(valid, 1))
    hi = float(np.nanpercentile(valid, 99))
    return np.clip((arr - lo) / max(hi - lo, 1.0e-6), 0.0, 1.0)


def partition_coverage(shape: tuple[int, int], partitions: list[ScenePartition]) -> np.ndarray:
    coverage = np.zeros(shape, dtype=np.int16)
    for partition in partitions:
        r0, r1, c0, c1 = partition.crop_rc
        coverage[r0 : r1 + 1, c0 : c1 + 1] += (partition.owner_mask > 0).astype(np.int16)
    return coverage


def overlay_partitions(
    scene: SceneRaster,
    partitions: list[ScenePartition],
    gt_count_map: np.ndarray,
    valid_mask: np.ndarray | None,
) -> np.ndarray:
    base = robust_norm(scene.image)
    rgb = np.repeat(base[..., None], 3, axis=2).astype(np.float32)
    if valid_mask is not None:
        outside = ~np.asarray(valid_mask, dtype=bool)
        rgb[outside] *= 0.18
    for partition in partitions:
        r0, r1, c0, c1 = partition.crop_rc
        owner = partition.owner_mask > 0
        if not owner.any():
            continue
        color = np.array([0.15, 0.95, 0.65], dtype=np.float32) if partition.kind == "ph_anchor" else np.array([1.0, 0.55, 0.12], dtype=np.float32)
        view = rgb[r0 : r1 + 1, c0 : c1 + 1]
        view[owner] = 0.42 * view[owner] + 0.58 * color
    gt_rc = np.argwhere(gt_count_map > 0)
    if gt_rc.size:
        rgb[gt_rc[:, 0], gt_rc[:, 1], 0] = 1.0
        rgb[gt_rc[:, 0], gt_rc[:, 1], 1] = 0.0
        rgb[gt_rc[:, 0], gt_rc[:, 1], 2] = 0.0
    return np.clip(rgb, 0.0, 1.0)


def write_manifest(path: Path, partitions: list[ScenePartition]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "partition_id",
                "kind",
                "anchor_cluster_id",
                "owner_pixels",
                "core_r0",
                "core_r1",
                "core_c0",
                "core_c1",
                "crop_r0",
                "crop_r1",
                "crop_c0",
                "crop_c1",
            ],
        )
        writer.writeheader()
        for item in partitions:
            writer.writerow(
                {
                    "partition_id": int(item.partition_id),
                    "kind": item.kind,
                    "anchor_cluster_id": "" if item.anchor_cluster_id is None else int(item.anchor_cluster_id),
                    "owner_pixels": int(item.owner_pixels),
                    "core_r0": int(item.core_rc[0]),
                    "core_r1": int(item.core_rc[1]),
                    "core_c0": int(item.core_rc[2]),
                    "core_c1": int(item.core_rc[3]),
                    "crop_r0": int(item.crop_rc[0]),
                    "crop_r1": int(item.crop_rc[1]),
                    "crop_c0": int(item.crop_rc[2]),
                    "crop_c1": int(item.crop_rc[3]),
                }
            )


def main() -> int:
    args = build_arg_parser().parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    sea_mask_metadata: dict[str, Any] = {"enabled": False}
    valid_sea_mask: np.ndarray | None = None
    if bool(args.kr_eez_mask):
        mask_result = apply_kr_sea_mask(
            args.scene_tif,
            step3_dir=args.kr_eez_step3_dir,
            crop_to_bounds=bool(args.kr_eez_crop_to_bounds),
            segment_policy=str(args.kr_eez_segment_policy),
            all_touched=bool(args.kr_eez_all_touched),
        )
        scene = mask_result.scene
        valid_sea_mask = mask_result.valid_mask
        sea_mask_metadata = mask_result.metadata
    else:
        scene = SceneRaster.load(args.scene_tif)

    resolver = GroundTruthResolver(args.gt_geojson.parent)
    gt_path = resolver.resolve_geojson(scene, args.gt_geojson)
    gt_count_map = resolver.rasterize_counts(scene, resolver.load_points(gt_path))

    detector_config = DnbCandidateDetectorConfig(
        detection_threshold=float(args.detector_detection_threshold),
        analysis_threshold=float(args.detector_analysis_threshold),
        threshold_reference=str(args.detector_threshold_reference),
        area_limit=int(args.detector_area_limit),
        min_nodes=int(args.detector_min_nodes),
        max_nodes=int(args.detector_max_nodes),
        remove_edge=bool(args.detector_remove_edge),
        drop_nested=False,
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
        padding_pixels=int(args.halo_pixels),
        parent_min_nodes=int(args.parent_min_nodes),
        child_min_nodes=int(args.child_min_nodes),
    )
    target_config = DensityTargetConfig(sigma_pixels=float(args.target_sigma), radius_pixels=int(args.target_radius))
    partition_config = ScenePartitionConfig(
        fallback_tile_pixels=int(args.fallback_tile_pixels),
        halo_pixels=int(args.halo_pixels),
        anchor_padding_pixels=int(args.anchor_padding_pixels),
    )
    patches, partitions, partition_summary = build_partitioned_density_patches(
        scene,
        gt_count_map,
        store,
        valid_mask=valid_sea_mask,
        patch_config=patch_config,
        target_config=target_config,
        partition_config=partition_config,
    )

    coverage = partition_coverage(scene.shape, partitions)
    domain = np.asarray(valid_sea_mask, dtype=bool) if valid_sea_mask is not None else np.ones(scene.shape, dtype=bool)
    metrics = {
        "scene_key": scene.key,
        "scene_shape": [int(scene.height), int(scene.width)],
        "gt_path": str(gt_path),
        "gt_count_sum": float(gt_count_map.sum()),
        "sea_mask": sea_mask_metadata,
        "ph_downsample": ph_anchor_result.metadata,
        "detector_summary": candidate_store_summary(store),
        "partition_summary": partition_summary,
        "coverage_max": int(coverage[domain].max()) if domain.any() else 0,
        "coverage_min": int(coverage[domain].min()) if domain.any() else 0,
        "patch_raw_count_sum": float(sum(patch.raw_count_sum for patch in patches)),
        "patch_target_density_sum": float(sum(patch.target_sum for patch in patches)),
    }
    metrics_path = output_dir / "partition_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    write_manifest(output_dir / "partition_manifest.csv", partitions)

    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    ax.imshow(overlay_partitions(scene, partitions, gt_count_map, valid_sea_mask))
    ax.set_title(
        (
            f"{scene.key} | PH anchors={partition_summary['ph_anchor_count']} | "
            f"fallback={partition_summary['fallback_grid_count']} | "
            f"coverage={partition_summary['coverage_ratio']:.3f}"
        )
    )
    ax.axis("off")
    overlay_path = output_dir / "partition_overlay.png"
    fig.savefig(overlay_path, dpi=160)
    plt.close(fig)

    print(json.dumps({"metrics_path": str(metrics_path), "overlay_path": str(overlay_path), **partition_summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
