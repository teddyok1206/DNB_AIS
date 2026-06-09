from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .dnb_pipeline_core import PHCluster, PHClusterStore, SceneRaster

DEFAULT_INPUT_CHANNELS = (
    "brightness",
    "ph_persistence_map",
    "ph_seed_map",
)


def _channel_array_for_patch(patch: "DensityPatch", channel_name: str) -> np.ndarray:
    normalized = str(channel_name).strip().lower().replace("-", "_")
    if normalized in {"brightness", "encoded_brightness", "arctan_brightness"}:
        return patch.image
    if normalized in {"ph_persistence_map", "persistence_map"}:
        return patch.persistence_map
    if normalized in {"ph_seed_map", "seed_map"}:
        return patch.seed_map
    supported = ", ".join(DEFAULT_INPUT_CHANNELS)
    raise ValueError(f"Unsupported density input channel: {channel_name}. Active channels are: {supported}")


@dataclass(frozen=True)
class DensityTargetConfig:
    kernel: str = "gaussian"
    sigma_pixels: float = 1.5
    radius_pixels: int = 5
    per_ship_mass: float = 1.0
    renormalize_after_roi_mask: bool = False
    require_source_in_roi: bool = False
    eps: float = 1.0e-8


@dataclass(frozen=True)
class DensityPatchConfig:
    padding_pixels: int = 16
    size_divisor: int = 16
    max_patches: int | None = None
    min_roi_pixels: int = 1
    sort_by: str = "node_count"
    parent_min_nodes: int = 32
    parent_max_nodes: int | None = None
    child_min_nodes: int = 4
    child_max_nodes: int | None = None
    max_children: int | None = None
    seed_radius_pixels: int = 1


@dataclass
class DensityPatch:
    cluster_id: int
    lifetime: float
    bbox_rc: tuple[int, int, int, int]
    crop_rc: tuple[int, int, int, int]
    image: np.ndarray
    seed_map: np.ndarray
    persistence_map: np.ndarray
    valid_mask: np.ndarray
    target_density: np.ndarray
    raw_count: np.ndarray
    parent_pixel_count: int = 0
    child_pixel_count: int = 0
    child_ids: list[int] = field(default_factory=list)
    partition_id: int | None = None
    partition_kind: str = "ph_proposal"
    anchor_cluster_id: int | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return self.image.shape

    @property
    def roi_pixels(self) -> int:
        return int(self.parent_pixel_count)

    @property
    def child_pixels(self) -> int:
        return int(self.child_pixel_count)

    @property
    def raw_count_sum(self) -> float:
        return float(self.raw_count.sum())

    @property
    def target_sum(self) -> float:
        return float(self.target_density.sum())

    @property
    def valid_pixels(self) -> int:
        return int((self.valid_mask > 0).sum())


def compact_density_patch(patch: DensityPatch) -> DensityPatch:
    """Drop retired dense arrays that can survive in older pickle caches."""

    if not hasattr(patch, "parent_pixel_count"):
        parent_mask = getattr(patch, "parent_mask", None)
        patch.parent_pixel_count = int(np.asarray(parent_mask, dtype=np.float32).sum()) if parent_mask is not None else 0
    if not hasattr(patch, "child_pixel_count"):
        child_union_mask = getattr(patch, "child_union_mask", None)
        patch.child_pixel_count = int(np.asarray(child_union_mask, dtype=np.float32).sum()) if child_union_mask is not None else 0
    for retired_attr in ("parent_mask", "child_union_mask", "soft_attention", "loss_weight"):
        if hasattr(patch, retired_attr):
            delattr(patch, retired_attr)
    return patch


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


def _cluster_coords_set(cluster: PHCluster) -> set[tuple[int, int]]:
    coords = getattr(cluster, "coords_set", None)
    if coords is not None:
        return set(coords)
    return set(map(tuple, np.asarray(cluster.global_rc, dtype=np.int64).tolist()))


def _mask_for_cluster(cluster: PHCluster, crop_rc: tuple[int, int, int, int]) -> np.ndarray:
    r0, r1, c0, c1 = [int(v) for v in crop_rc]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=np.float32)
    rc = np.asarray(cluster.global_rc, dtype=np.int64)
    if rc.size == 0:
        return mask
    rows = rc[:, 0] - r0
    cols = rc[:, 1] - c0
    valid = (rows >= 0) & (rows < mask.shape[0]) & (cols >= 0) & (cols < mask.shape[1])
    mask[rows[valid], cols[valid]] = 1.0
    return mask


def _gaussian_window(radius: int, sigma: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    radius = max(int(radius), 0)
    sigma = max(float(sigma), 1.0e-6)
    offsets = np.arange(-radius, radius + 1, dtype=np.float32)
    yy, xx = np.meshgrid(offsets, offsets, indexing="ij")
    weights = np.exp(-0.5 * ((yy / sigma) ** 2 + (xx / sigma) ** 2)).astype(np.float32)
    return yy.astype(np.int32), xx.astype(np.int32), weights


def make_sum_preserving_density_target(
    raw_count: np.ndarray,
    roi_mask: np.ndarray | None,
    config: DensityTargetConfig,
    *,
    domain_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Create a crop-level density target.

    By default this does not censor GT by PH ROI. Kernels are always clipped to the
    crop/domain boundary and renormalized there, preserving each ship's mass
    inside the valid sea-domain pixels. The legacy ROI-censoring behavior is
    still available through config flags.
    """

    raw = np.asarray(raw_count, dtype=np.float32)
    if roi_mask is None:
        roi = np.ones_like(raw, dtype=np.float32)
    else:
        roi = (np.asarray(roi_mask, dtype=np.float32) > 0).astype(np.float32)
    if raw.shape != roi.shape:
        raise ValueError(f"raw_count and roi_mask shape mismatch: {raw.shape} != {roi.shape}")
    if domain_mask is None:
        domain = np.ones_like(raw, dtype=np.float32)
    else:
        domain = (np.asarray(domain_mask, dtype=np.float32) > 0).astype(np.float32)
    if raw.shape != domain.shape:
        raise ValueError(f"raw_count and domain_mask shape mismatch: {raw.shape} != {domain.shape}")

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
        if domain[int(src_r), int(src_c)] <= 0:
            continue
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
        weights *= domain[rr_valid, cc_valid]
        if bool(config.renormalize_after_roi_mask):
            weights *= roi[rr_valid, cc_valid]
        denom = float(weights.sum())
        if denom <= float(config.eps):
            continue
        target[rr_valid, cc_valid] += np.float32(mass) * (weights / denom)
    return target


def _cluster_passes_size(cluster: PHCluster, min_nodes: int, max_nodes: int | None) -> bool:
    node_count = int(cluster.node_count)
    if node_count < int(min_nodes):
        return False
    if max_nodes is not None and node_count > int(max_nodes):
        return False
    return True


def _sort_clusters(clusters: list[PHCluster], sort_by: str) -> list[PHCluster]:
    if sort_by == "lifetime":
        return sorted(clusters, key=lambda item: item.lifetime, reverse=True)
    if sort_by == "cluster_id":
        return sorted(clusters, key=lambda item: item.cluster_id)
    if sort_by == "node_count":
        return sorted(clusters, key=lambda item: item.node_count, reverse=True)
    raise ValueError("sort_by must be 'lifetime', 'cluster_id', or 'node_count'")


def _select_parent_clusters(clusters: list[PHCluster], patch_config: DensityPatchConfig) -> list[PHCluster]:
    candidates = [
        cluster
        for cluster in clusters
        if _cluster_passes_size(cluster, int(patch_config.parent_min_nodes), patch_config.parent_max_nodes)
    ]
    # Select outer components first; contained smaller components become hierarchy children.
    candidates = sorted(candidates, key=lambda item: (item.node_count, item.lifetime), reverse=True)
    selected: list[PHCluster] = []
    selected_sets: list[set[tuple[int, int]]] = []
    for cluster in candidates:
        coords = _cluster_coords_set(cluster)
        if not coords:
            continue
        if any(coords.issubset(existing) for existing in selected_sets):
            continue
        selected.append(cluster)
        selected_sets.append(coords)

    selected = _sort_clusters(selected, patch_config.sort_by)
    if patch_config.max_patches is not None:
        selected = selected[: int(patch_config.max_patches)]
    return selected


def _children_for_parent(
    parent: PHCluster,
    clusters: list[PHCluster],
    patch_config: DensityPatchConfig,
) -> list[PHCluster]:
    parent_coords = _cluster_coords_set(parent)
    children: list[PHCluster] = []
    for child in clusters:
        if int(child.cluster_id) == int(parent.cluster_id):
            continue
        if not _cluster_passes_size(child, int(patch_config.child_min_nodes), patch_config.child_max_nodes):
            continue
        if int(child.node_count) >= int(parent.node_count):
            continue
        child_coords = _cluster_coords_set(child)
        if child_coords and child_coords.issubset(parent_coords):
            children.append(child)
    children = sorted(children, key=lambda item: item.lifetime, reverse=True)
    if patch_config.max_children is not None:
        children = children[: int(patch_config.max_children)]
    return children


def _seed_map_for_clusters(
    clusters: list[PHCluster],
    crop_rc: tuple[int, int, int, int],
    shape: tuple[int, int],
    *,
    radius_pixels: int,
) -> np.ndarray:
    seed_map = np.zeros(shape, dtype=np.float32)
    r0, _, c0, _ = [int(v) for v in crop_rc]
    radius = max(int(radius_pixels), 0)
    for cluster in clusters:
        seed_r, seed_c = [int(v) for v in cluster.seed_rc]
        local_r = seed_r - r0
        local_c = seed_c - c0
        if local_r < 0 or local_c < 0 or local_r >= shape[0] or local_c >= shape[1]:
            continue
        if radius <= 0:
            seed_map[local_r, local_c] = 1.0
            continue
        rr0 = max(local_r - radius, 0)
        rr1 = min(local_r + radius, shape[0] - 1)
        cc0 = max(local_c - radius, 0)
        cc1 = min(local_c + radius, shape[1] - 1)
        yy, xx = np.ogrid[rr0 : rr1 + 1, cc0 : cc1 + 1]
        disk = (yy - local_r) ** 2 + (xx - local_c) ** 2 <= radius**2
        seed_map[rr0 : rr1 + 1, cc0 : cc1 + 1][disk] = 1.0
    return seed_map


def _persistence_map_for_clusters(
    clusters: list[PHCluster],
    crop_rc: tuple[int, int, int, int],
    shape: tuple[int, int],
) -> np.ndarray:
    persistence = np.zeros(shape, dtype=np.float32)
    if not clusters:
        return persistence
    max_lifetime = max(float(cluster.lifetime) for cluster in clusters)
    max_lifetime = max(max_lifetime, 1.0e-6)
    for cluster in clusters:
        mask = _mask_for_cluster(cluster, crop_rc)
        score = np.float32(np.clip(float(cluster.lifetime) / max_lifetime, 0.0, 1.0))
        persistence = np.maximum(persistence, mask * score)
    return persistence.astype(np.float32, copy=False)


def build_density_patches(
    scene: SceneRaster,
    gt_count_map: np.ndarray,
    cluster_store: PHClusterStore,
    *,
    child_cluster_store: PHClusterStore | None = None,
    valid_mask: np.ndarray | None = None,
    patch_config: DensityPatchConfig | None = None,
    target_config: DensityTargetConfig | None = None,
) -> list[DensityPatch]:
    patch_config = patch_config or DensityPatchConfig()
    target_config = target_config or DensityTargetConfig()
    gt = np.asarray(gt_count_map, dtype=np.float32)
    if gt.shape != scene.shape:
        raise ValueError(f"gt_count_map shape mismatch: {gt.shape} != {scene.shape}")
    if valid_mask is None:
        domain = np.ones(scene.shape, dtype=np.float32)
    else:
        domain = (np.asarray(valid_mask, dtype=np.float32) > 0).astype(np.float32)
        if domain.shape != scene.shape:
            raise ValueError(f"valid_mask shape mismatch: {domain.shape} != {scene.shape}")

    parent_clusters = list(cluster_store.clusters)
    child_clusters = list(child_cluster_store.clusters) if child_cluster_store is not None else parent_clusters
    parents = _select_parent_clusters(parent_clusters, patch_config)

    patches: list[DensityPatch] = []
    for parent in parents:
        crop_rc = _crop_bounds(parent.bbox_rc, scene.shape, padding_pixels=int(patch_config.padding_pixels))
        r0, r1, c0, c1 = crop_rc
        valid_crop = np.asarray(domain[r0 : r1 + 1, c0 : c1 + 1], dtype=np.float32)
        image_crop = np.asarray(scene.image[r0 : r1 + 1, c0 : c1 + 1], dtype=np.float32) * valid_crop
        raw_crop = np.asarray(gt[r0 : r1 + 1, c0 : c1 + 1], dtype=np.float32) * valid_crop
        parent_mask = _mask_for_cluster(parent, crop_rc) * valid_crop
        parent_pixels = int(parent_mask.sum())
        if parent_pixels < int(patch_config.min_roi_pixels):
            continue

        children = _children_for_parent(parent, child_clusters, patch_config)
        child_masks = [_mask_for_cluster(child, crop_rc) * valid_crop for child in children]
        if child_masks:
            child_union_mask = np.maximum.reduce(child_masks).astype(np.float32, copy=False)
        else:
            child_union_mask = np.zeros_like(parent_mask, dtype=np.float32)
        child_pixels = int(child_union_mask.sum())

        seed_sources = children if children else [parent]
        persistence_sources = children if children else [parent]
        seed_map = _seed_map_for_clusters(
            seed_sources,
            crop_rc,
            image_crop.shape,
            radius_pixels=int(patch_config.seed_radius_pixels),
        ) * valid_crop
        persistence_map = _persistence_map_for_clusters(persistence_sources, crop_rc, image_crop.shape) * valid_crop
        target = make_sum_preserving_density_target(raw_crop, parent_mask, target_config, domain_mask=valid_crop)
        patches.append(
            DensityPatch(
                cluster_id=int(parent.cluster_id),
                lifetime=float(parent.lifetime),
                bbox_rc=tuple(int(v) for v in parent.bbox_rc),
                crop_rc=tuple(int(v) for v in crop_rc),
                image=image_crop,
                seed_map=seed_map.astype(np.float32, copy=False),
                persistence_map=persistence_map.astype(np.float32, copy=False),
                valid_mask=valid_crop.astype(np.float32, copy=False),
                target_density=target.astype(np.float32, copy=False),
                raw_count=raw_crop.astype(np.float32, copy=False),
                parent_pixel_count=parent_pixels,
                child_pixel_count=child_pixels,
                child_ids=[int(child.cluster_id) for child in children],
            )
        )
    return patches


class DensityPatchDataset(Dataset):
    def __init__(self, patches: Iterable[DensityPatch]) -> None:
        self.patches = [compact_density_patch(patch) for patch in patches]

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, index: int) -> DensityPatch:
        return self.patches[int(index)]


def density_patch_collate(
    batch: list[DensityPatch],
    *,
    size_divisor: int = 16,
    input_channels: Iterable[str] | None = None,
) -> dict[str, Any]:
    if not batch:
        raise ValueError("empty density patch batch")
    channel_names = tuple(input_channels) if input_channels is not None else DEFAULT_INPUT_CHANNELS
    if not channel_names:
        raise ValueError("input_channels must contain at least one channel")
    max_h = _round_up(max(p.image.shape[0] for p in batch), int(size_divisor))
    max_w = _round_up(max(p.image.shape[1] for p in batch), int(size_divisor))
    batch_size = len(batch)

    x = torch.zeros((batch_size, len(channel_names), max_h, max_w), dtype=torch.float32)
    target = torch.zeros((batch_size, 1, max_h, max_w), dtype=torch.float32)
    valid_mask = torch.zeros((batch_size, 1, max_h, max_w), dtype=torch.float32)
    raw_count = torch.zeros((batch_size, 1, max_h, max_w), dtype=torch.float32)

    metadata: list[dict[str, Any]] = []
    for idx, patch in enumerate(batch):
        patch = compact_density_patch(patch)
        h, w = patch.image.shape
        valid = torch.from_numpy(np.asarray(patch.valid_mask, dtype=np.float32))
        y = torch.from_numpy(np.asarray(patch.target_density, dtype=np.float32))
        raw = torch.from_numpy(np.asarray(patch.raw_count, dtype=np.float32))

        for channel_idx, channel_name in enumerate(channel_names):
            channel_arr = _channel_array_for_patch(patch, str(channel_name))
            if tuple(channel_arr.shape) != tuple(patch.shape):
                raise ValueError(f"channel {channel_name} shape mismatch: {channel_arr.shape} != {patch.shape}")
            x[idx, channel_idx, :h, :w] = torch.from_numpy(np.asarray(channel_arr, dtype=np.float32))
        target[idx, 0, :h, :w] = y
        valid_mask[idx, 0, :h, :w] = valid
        raw_count[idx, 0, :h, :w] = raw
        metadata.append(
            {
                "cluster_id": int(patch.cluster_id),
                "lifetime": float(patch.lifetime),
                "shape": [int(h), int(w)],
                "bbox_rc": list(patch.bbox_rc),
                "crop_rc": list(patch.crop_rc),
                "parent_pixels": int(patch.roi_pixels),
                "child_pixels": int(patch.child_pixels),
                "child_count": int(len(patch.child_ids)),
                "child_ids": [int(child_id) for child_id in patch.child_ids],
                "partition_id": None if patch.partition_id is None else int(patch.partition_id),
                "partition_kind": str(patch.partition_kind),
                "anchor_cluster_id": None if patch.anchor_cluster_id is None else int(patch.anchor_cluster_id),
                "raw_count_sum": float(patch.raw_count_sum),
                "target_sum": float(patch.target_sum),
                "valid_pixels": int(patch.valid_pixels),
                "input_channels": [str(name) for name in channel_names],
            }
        )

    return {
        "x": x,
        "input_channels": [str(name) for name in channel_names],
        "target": target,
        "valid_mask": valid_mask,
        "raw_count": raw_count,
        "metadata": metadata,
    }


def move_density_batch_to_device(batch: dict[str, Any], device: torch.device | str) -> dict[str, Any]:
    device = torch.device(device)
    moved = dict(batch)
    tensor_keys = [
        "x",
        "target",
        "valid_mask",
        "raw_count",
    ]
    for key in tensor_keys:
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
    parent_pixels = np.array([patch.roi_pixels for patch in patch_list], dtype=np.int64) if patch_list else np.array([], dtype=np.int64)
    child_pixels = np.array([patch.child_pixels for patch in patch_list], dtype=np.int64) if patch_list else np.array([], dtype=np.int64)
    child_counts = np.array([len(patch.child_ids) for patch in patch_list], dtype=np.int64) if patch_list else np.array([], dtype=np.int64)
    raw_sums = np.array([patch.raw_count_sum for patch in patch_list], dtype=np.float32) if patch_list else np.array([], dtype=np.float32)
    target_sums = np.array([patch.target_sum for patch in patch_list], dtype=np.float32) if patch_list else np.array([], dtype=np.float32)
    valid_pixels = np.array([patch.valid_pixels for patch in patch_list], dtype=np.int64) if patch_list else np.array([], dtype=np.int64)
    partition_kinds = [str(patch.partition_kind) for patch in patch_list]
    return {
        "patch_count": int(len(patch_list)),
        "height_min": int(shapes[:, 0].min()) if shapes.size else 0,
        "height_max": int(shapes[:, 0].max()) if shapes.size else 0,
        "width_min": int(shapes[:, 1].min()) if shapes.size else 0,
        "width_max": int(shapes[:, 1].max()) if shapes.size else 0,
        "parent_pixels_total": int(parent_pixels.sum()) if parent_pixels.size else 0,
        "parent_pixels_median": int(np.median(parent_pixels)) if parent_pixels.size else 0,
        "child_pixels_total": int(child_pixels.sum()) if child_pixels.size else 0,
        "child_count_total": int(child_counts.sum()) if child_counts.size else 0,
        "child_count_median": int(np.median(child_counts)) if child_counts.size else 0,
        "patches_with_raw_gt": int((raw_sums > 0).sum()) if raw_sums.size else 0,
        "patches_with_target_gt": int((target_sums > 0).sum()) if target_sums.size else 0,
        "raw_count_sum": float(raw_sums.sum()) if raw_sums.size else 0.0,
        "target_density_sum": float(target_sums.sum()) if target_sums.size else 0.0,
        "valid_pixels_total": int(valid_pixels.sum()) if valid_pixels.size else 0,
        "valid_pixels_median": int(np.median(valid_pixels)) if valid_pixels.size else 0,
        "ph_anchor_patch_count": int(sum(kind == "ph_anchor" for kind in partition_kinds)),
        "ph_child_patch_count": int(sum(kind == "ph_child" for kind in partition_kinds)),
        "fallback_grid_patch_count": int(sum(kind == "fallback_grid" for kind in partition_kinds)),
    }
