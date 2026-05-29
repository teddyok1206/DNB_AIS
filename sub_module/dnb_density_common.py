from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .dnb_gat_pipeline import DruidCluster, DruidClusterStore, SceneRaster


@dataclass(frozen=True)
class DensityTargetConfig:
    kernel: str = "gaussian"
    sigma_pixels: float = 1.5
    radius_pixels: int = 5
    per_ship_mass: float = 1.0
    renormalize_after_roi_mask: bool = True
    require_source_in_roi: bool = True
    eps: float = 1.0e-8


@dataclass(frozen=True)
class DensityPatchConfig:
    padding_pixels: int = 16
    size_divisor: int = 16
    max_patches: int | None = None
    min_roi_pixels: int = 1
    sort_by: str = "lifetime"


@dataclass
class DensityPatch:
    cluster_id: int
    lifetime: float
    bbox_rc: tuple[int, int, int, int]
    crop_rc: tuple[int, int, int, int]
    image: np.ndarray
    roi_mask: np.ndarray
    target_density: np.ndarray
    raw_count: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return self.image.shape

    @property
    def roi_pixels(self) -> int:
        return int(self.roi_mask.sum())

    @property
    def raw_count_sum(self) -> float:
        return float(self.raw_count.sum())

    @property
    def target_sum(self) -> float:
        return float(self.target_density.sum())


def _round_up(value: int, divisor: int) -> int:
    divisor = max(int(divisor), 1)
    return int(((int(value) + divisor - 1) // divisor) * divisor)


def _crop_bounds(
    bbox_rc: tuple[int, int, int, int],
    shape: tuple[int, int],
    *,
    padding_pixels: int,
) -> tuple[int, int, int, int]:
    rmin, rmax, cmin, cmax = [int(v) for v in bbox_rc]
    pad = max(int(padding_pixels), 0)
    height, width = int(shape[0]), int(shape[1])
    return (
        max(rmin - pad, 0),
        min(rmax + pad, height - 1),
        max(cmin - pad, 0),
        min(cmax + pad, width - 1),
    )


def _roi_mask_for_cluster(cluster: DruidCluster, crop_rc: tuple[int, int, int, int]) -> np.ndarray:
    r0, r1, c0, c1 = [int(v) for v in crop_rc]
    roi = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=np.float32)
    rc = np.asarray(cluster.global_rc, dtype=np.int64)
    if rc.size == 0:
        return roi
    rows = rc[:, 0] - r0
    cols = rc[:, 1] - c0
    valid = (rows >= 0) & (rows < roi.shape[0]) & (cols >= 0) & (cols < roi.shape[1])
    roi[rows[valid], cols[valid]] = 1.0
    return roi


def _gaussian_window(radius: int, sigma: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    radius = max(int(radius), 0)
    sigma = max(float(sigma), 1.0e-6)
    offsets = np.arange(-radius, radius + 1, dtype=np.float32)
    yy, xx = np.meshgrid(offsets, offsets, indexing="ij")
    weights = np.exp(-0.5 * ((yy / sigma) ** 2 + (xx / sigma) ** 2)).astype(np.float32)
    return yy.astype(np.int32), xx.astype(np.int32), weights


def make_sum_preserving_density_target(
    raw_count: np.ndarray,
    roi_mask: np.ndarray,
    config: DensityTargetConfig,
) -> np.ndarray:
    raw = np.asarray(raw_count, dtype=np.float32)
    roi = (np.asarray(roi_mask, dtype=np.float32) > 0).astype(np.float32)
    if raw.shape != roi.shape:
        raise ValueError(f"raw_count and roi_mask shape mismatch: {raw.shape} != {roi.shape}")

    target = np.zeros_like(raw, dtype=np.float32)
    positive_rc = np.argwhere(raw > 0)
    if positive_rc.size == 0:
        return target

    kernel_name = str(config.kernel).lower()
    if kernel_name != "gaussian":
        raise ValueError("Only gaussian density kernel is implemented for now")

    dy, dx, kernel = _gaussian_window(int(config.radius_pixels), float(config.sigma_pixels))
    height, width = raw.shape
    for src_r, src_c in positive_rc.tolist():
        if bool(config.require_source_in_roi) and roi[int(src_r), int(src_c)] <= 0:
            continue
        mass = float(raw[int(src_r), int(src_c)]) * float(config.per_ship_mass)
        rr = int(src_r) + dy
        cc = int(src_c) + dx
        valid = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
        if not np.any(valid):
            continue
        rr_valid = rr[valid]
        cc_valid = cc[valid]
        weights = kernel[valid].astype(np.float32, copy=True)
        if bool(config.renormalize_after_roi_mask):
            weights *= roi[rr_valid, cc_valid]
        denom = float(weights.sum())
        if denom <= float(config.eps):
            continue
        target[rr_valid, cc_valid] += np.float32(mass) * (weights / denom)
    return target


def build_density_patches(
    scene: SceneRaster,
    gt_count_map: np.ndarray,
    cluster_store: DruidClusterStore,
    *,
    patch_config: DensityPatchConfig | None = None,
    target_config: DensityTargetConfig | None = None,
) -> list[DensityPatch]:
    patch_config = patch_config or DensityPatchConfig()
    target_config = target_config or DensityTargetConfig()
    gt = np.asarray(gt_count_map, dtype=np.float32)
    if gt.shape != scene.shape:
        raise ValueError(f"gt_count_map shape mismatch: {gt.shape} != {scene.shape}")

    clusters = list(cluster_store.clusters)
    if patch_config.sort_by == "lifetime":
        clusters = sorted(clusters, key=lambda item: item.lifetime, reverse=True)
    elif patch_config.sort_by == "cluster_id":
        clusters = sorted(clusters, key=lambda item: item.cluster_id)
    else:
        raise ValueError("sort_by must be 'lifetime' or 'cluster_id'")
    if patch_config.max_patches is not None:
        clusters = clusters[: int(patch_config.max_patches)]

    patches: list[DensityPatch] = []
    for cluster in clusters:
        crop_rc = _crop_bounds(cluster.bbox_rc, scene.shape, padding_pixels=int(patch_config.padding_pixels))
        r0, r1, c0, c1 = crop_rc
        image_crop = np.asarray(scene.image[r0 : r1 + 1, c0 : c1 + 1], dtype=np.float32)
        raw_crop = np.asarray(gt[r0 : r1 + 1, c0 : c1 + 1], dtype=np.float32)
        roi = _roi_mask_for_cluster(cluster, crop_rc)
        if int(roi.sum()) < int(patch_config.min_roi_pixels):
            continue
        target = make_sum_preserving_density_target(raw_crop, roi, target_config)
        patches.append(
            DensityPatch(
                cluster_id=int(cluster.cluster_id),
                lifetime=float(cluster.lifetime),
                bbox_rc=tuple(int(v) for v in cluster.bbox_rc),
                crop_rc=tuple(int(v) for v in crop_rc),
                image=image_crop,
                roi_mask=roi.astype(np.float32, copy=False),
                target_density=target.astype(np.float32, copy=False),
                raw_count=raw_crop.astype(np.float32, copy=False),
            )
        )
    return patches


class DensityPatchDataset(Dataset):
    def __init__(self, patches: Iterable[DensityPatch]) -> None:
        self.patches = list(patches)

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, index: int) -> DensityPatch:
        return self.patches[int(index)]


def density_patch_collate(
    batch: list[DensityPatch],
    *,
    size_divisor: int = 16,
) -> dict[str, Any]:
    if not batch:
        raise ValueError("empty density patch batch")
    max_h = _round_up(max(p.image.shape[0] for p in batch), int(size_divisor))
    max_w = _round_up(max(p.image.shape[1] for p in batch), int(size_divisor))
    batch_size = len(batch)

    x = torch.zeros((batch_size, 2, max_h, max_w), dtype=torch.float32)
    target = torch.zeros((batch_size, 1, max_h, max_w), dtype=torch.float32)
    roi_mask = torch.zeros((batch_size, 1, max_h, max_w), dtype=torch.float32)
    valid_mask = torch.zeros((batch_size, 1, max_h, max_w), dtype=torch.float32)

    metadata: list[dict[str, Any]] = []
    for idx, patch in enumerate(batch):
        h, w = patch.image.shape
        image = torch.from_numpy(np.asarray(patch.image, dtype=np.float32))
        roi = torch.from_numpy(np.asarray(patch.roi_mask, dtype=np.float32))
        y = torch.from_numpy(np.asarray(patch.target_density, dtype=np.float32))
        x[idx, 0, :h, :w] = image
        x[idx, 1, :h, :w] = roi
        target[idx, 0, :h, :w] = y
        roi_mask[idx, 0, :h, :w] = roi
        valid_mask[idx, 0, :h, :w] = 1.0
        metadata.append(
            {
                "cluster_id": int(patch.cluster_id),
                "lifetime": float(patch.lifetime),
                "shape": [int(h), int(w)],
                "bbox_rc": list(patch.bbox_rc),
                "crop_rc": list(patch.crop_rc),
                "roi_pixels": int(patch.roi_pixels),
                "raw_count_sum": float(patch.raw_count_sum),
                "target_sum": float(patch.target_sum),
            }
        )

    return {"x": x, "target": target, "roi_mask": roi_mask, "valid_mask": valid_mask, "metadata": metadata}


def move_density_batch_to_device(batch: dict[str, Any], device: torch.device | str) -> dict[str, Any]:
    device = torch.device(device)
    moved = dict(batch)
    for key in ["x", "target", "roi_mask", "valid_mask"]:
        moved[key] = batch[key].to(device)
    return moved


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = torch.clamp(mask.sum(), min=1.0)
    return (((pred - target) ** 2) * mask).sum() / denom


def masked_poisson_nll_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    element = F.poisson_nll_loss(pred, target, log_input=False, full=False, eps=1.0e-8, reduction="none")
    denom = torch.clamp(mask.sum(), min=1.0)
    return (element * mask).sum() / denom


def summarize_density_patches(patches: Iterable[DensityPatch]) -> dict[str, Any]:
    patch_list = list(patches)
    shapes = np.array([patch.shape for patch in patch_list], dtype=np.int64) if patch_list else np.zeros((0, 2), dtype=np.int64)
    roi_pixels = np.array([patch.roi_pixels for patch in patch_list], dtype=np.int64) if patch_list else np.array([], dtype=np.int64)
    raw_sums = np.array([patch.raw_count_sum for patch in patch_list], dtype=np.float32) if patch_list else np.array([], dtype=np.float32)
    target_sums = np.array([patch.target_sum for patch in patch_list], dtype=np.float32) if patch_list else np.array([], dtype=np.float32)
    return {
        "patch_count": int(len(patch_list)),
        "height_min": int(shapes[:, 0].min()) if shapes.size else 0,
        "height_max": int(shapes[:, 0].max()) if shapes.size else 0,
        "width_min": int(shapes[:, 1].min()) if shapes.size else 0,
        "width_max": int(shapes[:, 1].max()) if shapes.size else 0,
        "roi_pixels_total": int(roi_pixels.sum()) if roi_pixels.size else 0,
        "roi_pixels_median": int(np.median(roi_pixels)) if roi_pixels.size else 0,
        "patches_with_raw_gt": int((raw_sums > 0).sum()) if raw_sums.size else 0,
        "patches_with_target_gt": int((target_sums > 0).sum()) if target_sums.size else 0,
        "raw_count_sum": float(raw_sums.sum()) if raw_sums.size else 0.0,
        "target_density_sum": float(target_sums.sum()) if target_sums.size else 0.0,
    }
