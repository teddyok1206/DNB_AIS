from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .dnb_candidate_detector import DnbCandidateDetector, DnbCandidateDetectorConfig
from .dnb_density_common import DensityPatch, DensityPatchConfig, DensityTargetConfig, build_density_patches
from .dnb_pipeline_core import GroundTruthResolver, SceneRaster
from .kr_sea_mask import apply_kr_sea_mask
from .run_density_smoke import ROOT, STEP3, save_density_patch_previews


DEFAULT_SCENE_TIF = STEP3 / "DRUID_TESTING" / "TEST_5_A2025001_1754_021_batch_1.tif"
DEFAULT_GEOJSON = STEP3 / "bboxes_JPSS-2" / "A2025001_1754_021.geojson"
DEFAULT_METADATA = STEP3 / "metadata_JPSS-2.csv"
DEFAULT_SHIPS_DB = STEP3 / "ships.db"


@dataclass(frozen=True)
class ThresholdCase:
    name: str
    parent_detection: float
    parent_analysis: float
    parent_reference: str = "zero"
    child_detection: float | None = None
    child_analysis: float | None = None
    child_reference: str | None = None

    @property
    def is_dual(self) -> bool:
        return self.child_detection is not None and self.child_analysis is not None

    @property
    def effective_child_reference(self) -> str:
        return self.child_reference if self.child_reference is not None else self.parent_reference


def default_cases() -> list[ThresholdCase]:
    return [
        ThresholdCase("single_d100_a100", 1.00, 1.00),
        ThresholdCase("zero_single_d300_a100", 3.00, 1.00, "zero"),
        ThresholdCase("zero_single_d200_a075", 2.00, 0.75, "zero"),
        ThresholdCase("median_single_d200_a050", 2.00, 0.50, "median"),
        ThresholdCase("median_single_d150_a025", 1.50, 0.25, "median"),
        ThresholdCase("median_single_d100_a000", 1.00, 0.00, "median"),
        ThresholdCase("median_single_d075_a000", 0.75, 0.00, "median"),
        ThresholdCase("median_single_d050_a000", 0.50, 0.00, "median"),
        ThresholdCase("median_single_d050_am050", 0.50, -0.50, "median"),
        ThresholdCase("dual_parent_median_d100_a000_child_median_d200_a050", 1.00, 0.00, "median", 2.00, 0.50, "median"),
        ThresholdCase("dual_parent_median_d050_am050_child_median_d150_a025", 0.50, -0.50, "median", 1.50, 0.25, "median"),
        ThresholdCase("dual_parent_zero_d200_a075_child_median_d150_a025", 2.00, 0.75, "zero", 1.50, 0.25, "median"),
    ]


def parse_case(text: str) -> ThresholdCase:
    parts = [part.strip() for part in text.split(",")]
    if len(parts) not in {3, 4, 5, 7}:
        raise ValueError(
            "case must be name,parent_detection,parent_analysis or "
            "name,parent_reference,parent_detection,parent_analysis or "
            "name,parent_detection,parent_analysis,child_detection,child_analysis or "
            "name,parent_reference,parent_detection,parent_analysis,child_reference,child_detection,child_analysis"
        )
    if len(parts) == 3:
        return ThresholdCase(parts[0], float(parts[1]), float(parts[2]))
    if len(parts) == 4:
        return ThresholdCase(parts[0], float(parts[2]), float(parts[3]), parts[1])
    if len(parts) == 5:
        return ThresholdCase(parts[0], float(parts[1]), float(parts[2]), "zero", float(parts[3]), float(parts[4]))
    return ThresholdCase(parts[0], float(parts[2]), float(parts[3]), parts[1], float(parts[5]), float(parts[6]), parts[4])


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sweep PH thresholds for hierarchical U-Net region proposals.")
    parser.add_argument("--scene-tif", type=Path, default=DEFAULT_SCENE_TIF)
    parser.add_argument("--gt-geojson", type=Path, default=DEFAULT_GEOJSON)
    parser.add_argument("--metadata-csv", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--ships-db", type=Path, default=DEFAULT_SHIPS_DB)
    parser.add_argument("--kr-eez-mask", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--kr-eez-step3-dir", type=Path, default=STEP3)
    parser.add_argument("--kr-eez-crop-to-bounds", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--kr-eez-segment-policy", choices=["single_scene", "largest_segment"], default="single_scene")
    parser.add_argument("--kr-eez-write-masked-tif", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--kr-eez-mask-output-dir", type=Path, default=ROOT / "outputs" / "preprocessed_scene_masks" / "density")
    parser.add_argument("--kr-eez-all-touched", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "ph_threshold_sweep")
    parser.add_argument("--case", action="append", default=None, help="name,parent_detection,parent_analysis[,child_detection,child_analysis]")
    parser.add_argument("--top-n", type=int, default=24)
    parser.add_argument("--preview-patches", type=int, default=3)
    parser.add_argument("--padding-pixels", type=int, default=16)
    parser.add_argument("--parent-min-nodes", type=int, default=32)
    parser.add_argument("--child-min-nodes", type=int, default=4)
    parser.add_argument("--max-nodes", type=int, default=2500)
    parser.add_argument("--area-limit", type=int, default=0)
    parser.add_argument("--detector-remove-edge", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed-radius-pixels", type=int, default=1)
    parser.add_argument("--attention-distance-sigma", type=float, default=4.0)
    parser.add_argument("--target-sigma", type=float, default=1.5)
    parser.add_argument("--target-radius", type=int, default=5)
    return parser


def robust_norm(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    lo = float(np.nanpercentile(image, 1))
    hi = float(np.nanpercentile(image, 99))
    return np.clip((image - lo) / max(hi - lo, 1.0e-6), 0.0, 1.0)


def make_detector_config(args: argparse.Namespace, detection: float, analysis: float, reference: str) -> DnbCandidateDetectorConfig:
    return DnbCandidateDetectorConfig(
        detection_threshold=float(detection),
        analysis_threshold=float(analysis),
        threshold_reference=str(reference),
        smooth_sigma=0.0,
        lifetime_limit=0.0,
        lifetime_limit_fraction=1.001,
        area_limit=int(args.area_limit),
        min_nodes=int(args.child_min_nodes),
        max_nodes=int(args.max_nodes),
        remove_edge=bool(args.detector_remove_edge),
        drop_nested=False,
    )


def mask_from_clusters(shape: tuple[int, int], clusters: list[Any]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for cluster in clusters:
        rc = np.asarray(cluster.global_rc, dtype=np.int64)
        if rc.size:
            mask[rc[:, 0], rc[:, 1]] = True
    return mask


def masks_from_patches(shape: tuple[int, int], patches: list[DensityPatch]) -> dict[str, np.ndarray]:
    parent = np.zeros(shape, dtype=bool)
    child = np.zeros(shape, dtype=bool)
    crop = np.zeros(shape, dtype=bool)
    for patch in patches:
        r0, r1, c0, c1 = patch.crop_rc
        crop[r0 : r1 + 1, c0 : c1 + 1] |= patch.valid_mask.astype(bool)
        parent[r0 : r1 + 1, c0 : c1 + 1] |= patch.parent_mask.astype(bool)
        child[r0 : r1 + 1, c0 : c1 + 1] |= patch.child_union_mask.astype(bool)
    return {"parent": parent, "child": child, "crop": crop}


def summarize_patches(
    *,
    scene: SceneRaster,
    gt_count_map: np.ndarray,
    parent_store: Any,
    child_store: Any,
    patches: list[DensityPatch],
    top_n: int,
    case: ThresholdCase,
    valid_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    top_patches = patches[: max(int(top_n), 0)]
    scene_pixels = int(np.asarray(valid_mask, dtype=bool).sum()) if valid_mask is not None else int(scene.height * scene.width)
    scene_gt = float(gt_count_map.sum())
    all_masks = masks_from_patches(scene.shape, patches)
    top_masks = masks_from_patches(scene.shape, top_patches)
    parent_candidate_mask = mask_from_clusters(scene.shape, list(parent_store.clusters))
    child_candidate_mask = mask_from_clusters(scene.shape, list(child_store.clusters))

    def gt_mass(mask: np.ndarray) -> float:
        return float(gt_count_map[mask].sum())

    def ratio(value: float) -> float:
        return float(value / max(scene_gt, 1.0e-6))

    raw_sums = np.array([patch.raw_count_sum for patch in patches], dtype=np.float32) if patches else np.array([], dtype=np.float32)
    target_sums = np.array([patch.target_sum for patch in patches], dtype=np.float32) if patches else np.array([], dtype=np.float32)
    child_counts = np.array([len(patch.child_ids) for patch in patches], dtype=np.int64) if patches else np.array([], dtype=np.int64)
    parent_pixels = np.array([patch.roi_pixels for patch in patches], dtype=np.int64) if patches else np.array([], dtype=np.int64)
    crop_pixels = np.array([patch.valid_pixels for patch in patches], dtype=np.int64) if patches else np.array([], dtype=np.int64)

    parent_candidate_gt = gt_mass(parent_candidate_mask)
    child_candidate_gt = gt_mass(child_candidate_mask)
    all_parent_gt = gt_mass(all_masks["parent"])
    all_crop_gt = gt_mass(all_masks["crop"])
    top_parent_gt = gt_mass(top_masks["parent"])
    top_crop_gt = gt_mass(top_masks["crop"])
    return {
        "case": case.name,
        "mode": "dual" if case.is_dual else "single",
        "parent_reference": case.parent_reference,
        "parent_detection": float(case.parent_detection),
        "parent_analysis": float(case.parent_analysis),
        "child_reference": case.effective_child_reference,
        "child_detection": float(case.child_detection if case.child_detection is not None else case.parent_detection),
        "child_analysis": float(case.child_analysis if case.child_analysis is not None else case.parent_analysis),
        "scene_key": scene.key,
        "scene_gt_sum": scene_gt,
        "parent_candidate_count": int(len(parent_store.clusters)),
        "child_candidate_count": int(len(child_store.clusters)),
        "parent_candidate_pixels": int(parent_candidate_mask.sum()),
        "child_candidate_pixels": int(child_candidate_mask.sum()),
        "parent_candidate_gt_unique": parent_candidate_gt,
        "parent_candidate_gt_recall": ratio(parent_candidate_gt),
        "child_candidate_gt_unique": child_candidate_gt,
        "child_candidate_gt_recall": ratio(child_candidate_gt),
        "patch_count": int(len(patches)),
        "top_n": int(top_n),
        "top_patch_count": int(len(top_patches)),
        "patches_with_gt": int((raw_sums > 0).sum()) if raw_sums.size else 0,
        "patches_with_children": int((child_counts > 0).sum()) if child_counts.size else 0,
        "child_count_total": int(child_counts.sum()) if child_counts.size else 0,
        "child_count_median": float(np.median(child_counts)) if child_counts.size else 0.0,
        "child_count_max": int(child_counts.max()) if child_counts.size else 0,
        "parent_pixels_median": float(np.median(parent_pixels)) if parent_pixels.size else 0.0,
        "parent_pixels_max": int(parent_pixels.max()) if parent_pixels.size else 0,
        "crop_pixels_median": float(np.median(crop_pixels)) if crop_pixels.size else 0.0,
        "crop_pixels_max": int(crop_pixels.max()) if crop_pixels.size else 0,
        "all_parent_area_unique_ratio": float(all_masks["parent"].sum() / max(scene_pixels, 1)),
        "all_crop_area_unique_ratio": float(all_masks["crop"].sum() / max(scene_pixels, 1)),
        "top_parent_area_unique_ratio": float(top_masks["parent"].sum() / max(scene_pixels, 1)),
        "top_crop_area_unique_ratio": float(top_masks["crop"].sum() / max(scene_pixels, 1)),
        "all_parent_gt_unique": all_parent_gt,
        "all_parent_gt_recall": ratio(all_parent_gt),
        "all_crop_gt_unique": all_crop_gt,
        "all_crop_gt_recall": ratio(all_crop_gt),
        "top_parent_gt_unique": top_parent_gt,
        "top_parent_gt_recall": ratio(top_parent_gt),
        "top_crop_gt_unique": top_crop_gt,
        "top_crop_gt_recall": ratio(top_crop_gt),
        "raw_count_sum_over_patches": float(raw_sums.sum()) if raw_sums.size else 0.0,
        "target_density_sum_over_patches": float(target_sums.sum()) if target_sums.size else 0.0,
        "duplicate_factor_all_crops": float(raw_sums.sum() / max(scene_gt, 1.0e-6)) if raw_sums.size else 0.0,
        "max_abs_raw_target_diff": float(np.max(np.abs(raw_sums - target_sums))) if raw_sums.size else 0.0,
    }


def overlay_array(scene: SceneRaster, gt_count_map: np.ndarray, patches: list[DensityPatch], *, top_n: int) -> np.ndarray:
    base = robust_norm(scene.image)
    rgb = np.repeat(base[..., None], 3, axis=2).astype(np.float32)
    masks = masks_from_patches(scene.shape, patches[: max(int(top_n), 0)])
    parent = masks["parent"]
    child = masks["child"]
    crop = masks["crop"]
    rgb[crop, 0] = np.maximum(rgb[crop, 0], 0.45)
    rgb[crop, 1] = np.maximum(rgb[crop, 1], 0.45)
    rgb[parent, 1] = 1.0
    rgb[parent, 0] *= 0.35
    rgb[parent, 2] *= 0.35
    rgb[child, 1] = 0.9
    rgb[child, 2] = 1.0
    gt_rc = np.argwhere(gt_count_map > 0)
    if gt_rc.size:
        rgb[gt_rc[:, 0], gt_rc[:, 1], 0] = 1.0
        rgb[gt_rc[:, 0], gt_rc[:, 1], 1] = 0.0
        rgb[gt_rc[:, 0], gt_rc[:, 1], 2] = 0.0
    return np.clip(rgb, 0.0, 1.0)


def save_overlay(scene: SceneRaster, gt_count_map: np.ndarray, patches: list[DensityPatch], metrics: dict[str, Any], output_path: Path) -> None:
    image = overlay_array(scene, gt_count_map, patches, top_n=int(metrics["top_n"]))
    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    ax.imshow(image)
    ax.set_title(
        (
            f"{metrics['case']} | top{metrics['top_n']} crop GT recall={metrics['top_crop_gt_recall']:.3f} | "
            f"parent area={metrics['top_parent_area_unique_ratio']:.3f} | crop area={metrics['top_crop_area_unique_ratio']:.3f}"
        )
    )
    ax.axis("off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_summary_grid(scene: SceneRaster, gt_count_map: np.ndarray, case_rows: list[tuple[ThresholdCase, list[DensityPatch], dict[str, Any]]], output_path: Path) -> None:
    cols = 3
    rows = int(np.ceil(len(case_rows) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.8 * cols, 3.7 * rows), constrained_layout=True)
    axes_array = np.asarray(axes).reshape(-1)
    for ax, (case, patches, metrics) in zip(axes_array, case_rows):
        ax.imshow(overlay_array(scene, gt_count_map, patches, top_n=int(metrics["top_n"])))
        ax.set_title(
            f"{case.name}\nrecall={metrics['top_crop_gt_recall']:.2f}, area={metrics['top_crop_area_unique_ratio']:.2f}, patches={metrics['patch_count']}",
            fontsize=9,
        )
        ax.axis("off")
    for ax in axes_array[len(case_rows) :]:
        ax.axis("off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_metrics(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "threshold_sweep_metrics.json"
    csv_path = output_dir / "threshold_sweep_metrics.csv"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = [parse_case(item) for item in args.case] if args.case else default_cases()
    sea_mask_metadata: dict[str, Any] = {"enabled": False}
    valid_sea_mask: np.ndarray | None = None
    if bool(args.kr_eez_mask):
        mask_result = apply_kr_sea_mask(
            args.scene_tif,
            step3_dir=args.kr_eez_step3_dir,
            output_dir=args.kr_eez_mask_output_dir,
            crop_to_bounds=bool(args.kr_eez_crop_to_bounds),
            segment_policy=str(args.kr_eez_segment_policy),
            write_masked_tif=bool(args.kr_eez_write_masked_tif),
            all_touched=bool(args.kr_eez_all_touched),
        )
        scene = mask_result.scene
        valid_sea_mask = mask_result.valid_mask
        sea_mask_metadata = mask_result.metadata
    else:
        scene = SceneRaster.load(args.scene_tif)
    resolver = GroundTruthResolver(args.metadata_csv, args.ships_db, args.gt_geojson.parent)
    gt_path = resolver.resolve_geojson(scene, args.gt_geojson)
    gt_count_map = resolver.rasterize_counts(scene, resolver.load_points(gt_path))

    patch_config = DensityPatchConfig(
        padding_pixels=int(args.padding_pixels),
        size_divisor=16,
        max_patches=None,
        sort_by="node_count",
        parent_min_nodes=int(args.parent_min_nodes),
        child_min_nodes=int(args.child_min_nodes),
        seed_radius_pixels=int(args.seed_radius_pixels),
        attention_distance_sigma=float(args.attention_distance_sigma),
    )
    target_config = DensityTargetConfig(
        sigma_pixels=float(args.target_sigma),
        radius_pixels=int(args.target_radius),
        require_source_in_roi=False,
        renormalize_after_roi_mask=False,
    )

    rows: list[dict[str, Any]] = []
    case_rows: list[tuple[ThresholdCase, list[DensityPatch], dict[str, Any]]] = []
    for case in cases:
        parent_store = DnbCandidateDetector(
            make_detector_config(args, case.parent_detection, case.parent_analysis, case.parent_reference)
        ).build_store(scene, gt_count_map, valid_mask=valid_sea_mask)
        if case.is_dual:
            child_store = DnbCandidateDetector(
                make_detector_config(
                    args,
                    float(case.child_detection),
                    float(case.child_analysis),
                    case.effective_child_reference,
                )
            ).build_store(scene, gt_count_map, valid_mask=valid_sea_mask)
        else:
            child_store = parent_store
        patches = build_density_patches(
            scene,
            gt_count_map,
            parent_store,
            child_cluster_store=child_store,
            valid_mask=valid_sea_mask,
            patch_config=patch_config,
            target_config=target_config,
        )
        metrics = summarize_patches(
            scene=scene,
            gt_count_map=gt_count_map,
            parent_store=parent_store,
            child_store=child_store,
            patches=patches,
            top_n=int(args.top_n),
            case=case,
            valid_mask=valid_sea_mask,
        )
        metrics["sea_mask_enabled"] = bool(sea_mask_metadata.get("enabled", False))
        metrics["sea_mask_valid_pixels"] = int(sea_mask_metadata.get("selected_valid_pixels", scene.height * scene.width))
        rows.append(metrics)
        case_rows.append((case, patches, metrics))

        case_dir = output_dir / case.name
        save_overlay(scene, gt_count_map, patches, metrics, case_dir / f"{case.name}_top{int(args.top_n)}_overlay.png")
        save_density_patch_previews(
            patches,
            scene_key=f"{scene.key}_{case.name}",
            output_dir=case_dir / "patch_previews",
            limit=int(args.preview_patches),
        )
        print(json.dumps(metrics, ensure_ascii=False))

    write_metrics(rows, output_dir)
    save_summary_grid(scene, gt_count_map, case_rows, output_dir / "threshold_sweep_overlay_grid.png")
    print(json.dumps({"output_dir": str(output_dir), "case_count": len(rows)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
