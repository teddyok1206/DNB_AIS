from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from affine import Affine
from rasterio.transform import array_bounds

from .dnb_candidate_detector import DnbCandidateDetector, DnbCandidateDetectorConfig
from .dnb_gat_pipeline import DruidCluster, DruidClusterStore, SceneRaster


@dataclass(frozen=True)
class PHDownsampleConfig:
    factor: int = 1
    reducer: str = "max"


@dataclass(frozen=True)
class PHAnchorStoreResult:
    store: DruidClusterStore
    metadata: dict[str, Any]


def _domain_mask(shape: tuple[int, int], valid_mask: np.ndarray | None) -> np.ndarray:
    if valid_mask is None:
        return np.ones(shape, dtype=bool)
    domain = np.asarray(valid_mask, dtype=bool)
    if tuple(domain.shape) != tuple(shape):
        raise ValueError(f"valid_mask shape mismatch: {domain.shape} != {shape}")
    return domain


def _pad_to_factor(arr: np.ndarray, factor: int, *, fill_value: float | bool) -> np.ndarray:
    height, width = arr.shape
    out_h = int(np.ceil(height / factor) * factor)
    out_w = int(np.ceil(width / factor) * factor)
    if out_h == height and out_w == width:
        return arr
    padded = np.full((out_h, out_w), fill_value, dtype=arr.dtype)
    padded[:height, :width] = arr
    return padded


def downsample_scene_for_ph(
    scene: SceneRaster,
    *,
    valid_mask: np.ndarray | None = None,
    config: PHDownsampleConfig | None = None,
) -> tuple[SceneRaster, np.ndarray, dict[str, Any]]:
    config = config or PHDownsampleConfig()
    factor = max(int(config.factor), 1)
    reducer = str(config.reducer).lower()
    if factor <= 1:
        domain = _domain_mask(scene.shape, valid_mask)
        return scene, domain, {"enabled": False, "factor": 1, "reducer": reducer}
    if reducer not in {"max", "mean"}:
        raise ValueError("PH downsample reducer must be 'max' or 'mean'")

    domain = _domain_mask(scene.shape, valid_mask)
    image = np.asarray(scene.image, dtype=np.float32)
    padded_domain = _pad_to_factor(domain.astype(bool, copy=False), factor, fill_value=False)
    out_h = padded_domain.shape[0] // factor
    out_w = padded_domain.shape[1] // factor
    valid_blocks = padded_domain.reshape(out_h, factor, out_w, factor)
    low_valid = valid_blocks.any(axis=(1, 3))

    if reducer == "max":
        masked = np.where(domain, image, -np.inf).astype(np.float32, copy=False)
        padded = _pad_to_factor(masked, factor, fill_value=-np.inf)
        low_image = padded.reshape(out_h, factor, out_w, factor).max(axis=(1, 3)).astype(np.float32, copy=False)
        low_image[~np.isfinite(low_image)] = 0.0
    else:
        masked = np.where(domain, image, 0.0).astype(np.float32, copy=False)
        padded = _pad_to_factor(masked, factor, fill_value=0.0)
        sums = padded.reshape(out_h, factor, out_w, factor).sum(axis=(1, 3))
        counts = valid_blocks.sum(axis=(1, 3)).astype(np.float32)
        low_image = np.divide(sums, np.maximum(counts, 1.0), out=np.zeros_like(sums, dtype=np.float32), where=counts > 0)
    low_image[~low_valid] = 0.0

    transform = scene.transform * Affine.scale(factor, factor)
    bounds = array_bounds(int(out_h), int(out_w), transform)
    low_scene = SceneRaster(
        path=Path(scene.path),
        image=low_image.astype(np.float32, copy=False),
        transform=transform,
        crs=scene.crs,
        bounds=bounds,
        height=int(out_h),
        width=int(out_w),
    )
    metadata = {
        "enabled": True,
        "factor": int(factor),
        "reducer": reducer,
        "source_shape": [int(scene.height), int(scene.width)],
        "downsampled_shape": [int(out_h), int(out_w)],
        "source_valid_pixels": int(domain.sum()),
        "downsampled_valid_pixels": int(low_valid.sum()),
    }
    return low_scene, low_valid.astype(bool, copy=False), metadata


def _expand_low_rc_to_full(
    low_rc: np.ndarray,
    *,
    factor: int,
    target_shape: tuple[int, int],
    target_valid_mask: np.ndarray,
) -> np.ndarray:
    height, width = target_shape
    coords: list[np.ndarray] = []
    for low_r, low_c in np.asarray(low_rc, dtype=np.int64).tolist():
        r0 = int(low_r) * int(factor)
        r1 = min(r0 + int(factor), height)
        c0 = int(low_c) * int(factor)
        c1 = min(c0 + int(factor), width)
        if r0 >= height or c0 >= width:
            continue
        block = target_valid_mask[r0:r1, c0:c1]
        if not block.any():
            continue
        rr, cc = np.where(block)
        coords.append(np.column_stack([rr + r0, cc + c0]).astype(np.int32, copy=False))
    if not coords:
        return np.zeros((0, 2), dtype=np.int32)
    return np.concatenate(coords, axis=0).astype(np.int32, copy=False)


def _bbox_contour(rmin: int, rmax: int, cmin: int, cmax: int) -> np.ndarray:
    return np.array(
        [
            [int(rmin), int(cmin)],
            [int(rmin), int(cmax)],
            [int(rmax), int(cmax)],
            [int(rmax), int(cmin)],
        ],
        dtype=np.int32,
    )


def upsample_cluster_store(
    low_store: DruidClusterStore,
    *,
    target_scene: SceneRaster,
    gt_count_map: np.ndarray,
    target_valid_mask: np.ndarray | None = None,
    factor: int,
) -> DruidClusterStore:
    factor = max(int(factor), 1)
    if factor <= 1:
        return low_store
    target_domain = _domain_mask(target_scene.shape, target_valid_mask)
    gt = np.asarray(gt_count_map, dtype=np.float32)
    if tuple(gt.shape) != tuple(target_scene.shape):
        raise ValueError(f"gt_count_map shape mismatch: {gt.shape} != {target_scene.shape}")

    rows: list[dict[str, Any]] = []
    clusters: list[DruidCluster] = []
    for low_cluster in low_store.clusters:
        full_rc = _expand_low_rc_to_full(
            low_cluster.global_rc,
            factor=factor,
            target_shape=target_scene.shape,
            target_valid_mask=target_domain,
        )
        if full_rc.size == 0:
            continue
        rmin = int(full_rc[:, 0].min())
        rmax = int(full_rc[:, 0].max())
        cmin = int(full_rc[:, 1].min())
        cmax = int(full_rc[:, 1].max())
        local_rc = full_rc - np.array([rmin, cmin], dtype=np.int32)
        local_mask = np.zeros((rmax - rmin + 1, cmax - cmin + 1), dtype=np.uint8)
        local_mask[local_rc[:, 0], local_rc[:, 1]] = 1
        seed_r = min(max(int(low_cluster.seed_rc[0]) * factor + factor // 2, 0), target_scene.height - 1)
        seed_c = min(max(int(low_cluster.seed_rc[1]) * factor + factor // 2, 0), target_scene.width - 1)
        if not target_domain[seed_r, seed_c]:
            seed_r, seed_c = [int(v) for v in full_rc[0].tolist()]
        cluster = DruidCluster(
            cluster_id=int(low_cluster.cluster_id),
            lifetime=float(low_cluster.lifetime),
            birth=float(low_cluster.birth),
            death=float(low_cluster.death),
            contour_rc=_bbox_contour(rmin, rmax, cmin, cmax),
            bbox_rc=(rmin, rmax, cmin, cmax),
            seed_rc=(seed_r, seed_c),
            patch_image=target_scene.image[rmin : rmax + 1, cmin : cmax + 1],
            patch_gt=gt[rmin : rmax + 1, cmin : cmax + 1],
            mask=local_mask,
            local_rc=local_rc.astype(np.int32, copy=False),
            global_rc=full_rc.astype(np.int32, copy=False),
        )
        cluster.coords_set = set(map(tuple, cluster.global_rc.tolist()))
        clusters.append(cluster)
        rows.append(
            {
                "ID": int(cluster.cluster_id),
                "Birth": float(cluster.birth),
                "Death": float(cluster.death),
                "x1": int(seed_r),
                "y1": int(seed_c),
                "x2": int(seed_r),
                "y2": int(seed_c),
                "lifetime": float(cluster.lifetime),
                "lifetimeFrac": float(getattr(low_cluster, "lifetime", 0.0)),
                "area": int(cluster.node_count),
                "edge_flag": 0,
                "bbox1": int(rmin),
                "bbox2": int(cmin),
                "bbox3": int(rmax),
                "bbox4": int(cmax),
                "node_count": int(cluster.node_count),
                "gt_sum": float(cluster.gt_sum),
                "contour": cluster.contour_rc,
                "source_downsample_factor": int(factor),
            }
        )
    return DruidClusterStore(scene=target_scene, catalogue=pd.DataFrame(rows), clusters=clusters)


def build_ph_anchor_store(
    scene: SceneRaster,
    gt_count_map: np.ndarray,
    detector_config: DnbCandidateDetectorConfig,
    *,
    valid_mask: np.ndarray | None = None,
    downsample_config: PHDownsampleConfig | None = None,
) -> PHAnchorStoreResult:
    downsample_config = downsample_config or PHDownsampleConfig()
    factor = max(int(downsample_config.factor), 1)
    if factor <= 1:
        store = DnbCandidateDetector(detector_config).build_store(scene, gt_count_map, valid_mask=valid_mask)
        return PHAnchorStoreResult(store=store, metadata={"enabled": False, "factor": 1, "reducer": downsample_config.reducer})

    low_scene, low_valid, metadata = downsample_scene_for_ph(scene, valid_mask=valid_mask, config=downsample_config)
    low_gt = np.zeros(low_scene.shape, dtype=np.float32)
    low_store = DnbCandidateDetector(detector_config).build_store(low_scene, low_gt, valid_mask=low_valid)
    full_store = upsample_cluster_store(
        low_store,
        target_scene=scene,
        gt_count_map=gt_count_map,
        target_valid_mask=valid_mask,
        factor=factor,
    )
    metadata.update(
        {
            "downsampled_candidate_count": int(len(low_store.clusters)),
            "upsampled_candidate_count": int(len(full_store.clusters)),
            "upsampled_total_nodes": int(sum(cluster.node_count for cluster in full_store.clusters)),
        }
    )
    return PHAnchorStoreResult(store=full_store, metadata=metadata)
