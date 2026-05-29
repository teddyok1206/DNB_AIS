from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable

import numpy as np

from .dnb_density_common import (
    DensityPatch,
    DensityPatchConfig,
    DensityTargetConfig,
    _cluster_passes_size,
    _crop_bounds,
    _loss_weight_from_attention,
    _mask_for_cluster,
    _persistence_map_for_clusters,
    _select_parent_clusters,
    _seed_map_for_clusters,
    _soft_attention_from_masks,
    make_sum_preserving_density_target,
)
from .dnb_gat_pipeline import DruidCluster, DruidClusterStore, SceneRaster


@dataclass(frozen=True)
class ScenePartitionConfig:
    enabled: bool = True
    fallback_tile_pixels: int = 96
    halo_pixels: int = 16
    anchor_padding_pixels: int = 16
    min_owner_pixels: int = 1
    min_fallback_owner_pixels: int = 1


@dataclass(frozen=True)
class ScenePartition:
    partition_id: int
    kind: str
    core_rc: tuple[int, int, int, int]
    crop_rc: tuple[int, int, int, int]
    owner_mask: np.ndarray
    anchor_cluster_id: int | None = None
    anchor_lifetime: float = 0.0

    @property
    def owner_pixels(self) -> int:
        return int((self.owner_mask > 0).sum())


def _domain_mask(shape: tuple[int, int], valid_mask: np.ndarray | None) -> np.ndarray:
    if valid_mask is None:
        return np.ones(shape, dtype=bool)
    domain = np.asarray(valid_mask, dtype=bool)
    if tuple(domain.shape) != tuple(shape):
        raise ValueError(f"valid_mask shape mismatch: {domain.shape} != {shape}")
    return domain


def _expand_rc(
    rc: tuple[int, int, int, int],
    shape: tuple[int, int],
    *,
    padding_pixels: int,
) -> tuple[int, int, int, int]:
    return _crop_bounds(rc, shape, padding_pixels=int(padding_pixels))


def _local_owner_mask(
    core_rc: tuple[int, int, int, int],
    crop_rc: tuple[int, int, int, int],
    owner_core: np.ndarray,
) -> np.ndarray:
    r0, r1, c0, c1 = [int(v) for v in crop_rc]
    cr0, cr1, cc0, cc1 = [int(v) for v in core_rc]
    owner = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=np.float32)
    owner[cr0 - r0 : cr1 - r0 + 1, cc0 - c0 : cc1 - c0 + 1] = owner_core.astype(np.float32, copy=False)
    return owner


def _make_partition(
    *,
    partition_id: int,
    kind: str,
    core_rc: tuple[int, int, int, int],
    owner_core: np.ndarray,
    scene_shape: tuple[int, int],
    halo_pixels: int,
    anchor_cluster_id: int | None = None,
    anchor_lifetime: float = 0.0,
) -> ScenePartition:
    crop_rc = _expand_rc(core_rc, scene_shape, padding_pixels=int(halo_pixels))
    owner_mask = _local_owner_mask(core_rc, crop_rc, owner_core)
    return ScenePartition(
        partition_id=int(partition_id),
        kind=str(kind),
        core_rc=tuple(int(v) for v in core_rc),
        crop_rc=tuple(int(v) for v in crop_rc),
        owner_mask=owner_mask,
        anchor_cluster_id=anchor_cluster_id,
        anchor_lifetime=float(anchor_lifetime),
    )


def build_scene_partitions(
    scene: SceneRaster,
    cluster_store: DruidClusterStore,
    *,
    valid_mask: np.ndarray | None = None,
    patch_config: DensityPatchConfig | None = None,
    partition_config: ScenePartitionConfig | None = None,
) -> list[ScenePartition]:
    """Build an exact-cover sea-domain partition using PH anchors plus grid fallback.

    PH anchors claim valid sea pixels inside their padded bounding boxes first.
    Remaining valid sea pixels are assigned to non-overlapping fallback grid tiles.
    Each partition stores an owner mask; halo pixels are context only.
    """

    patch_config = patch_config or DensityPatchConfig()
    partition_config = partition_config or ScenePartitionConfig()
    domain = _domain_mask(scene.shape, valid_mask)
    assigned = np.zeros(scene.shape, dtype=bool)
    partitions: list[ScenePartition] = []

    parent_config = replace(patch_config, max_patches=None)
    parents = _select_parent_clusters(list(cluster_store.clusters), parent_config)
    partition_id = 1
    for parent in parents:
        core_rc = _expand_rc(
            parent.bbox_rc,
            scene.shape,
            padding_pixels=int(partition_config.anchor_padding_pixels),
        )
        r0, r1, c0, c1 = core_rc
        owner_core = domain[r0 : r1 + 1, c0 : c1 + 1] & ~assigned[r0 : r1 + 1, c0 : c1 + 1]
        if int(owner_core.sum()) < int(partition_config.min_owner_pixels):
            continue
        partition = _make_partition(
            partition_id=partition_id,
            kind="ph_anchor",
            core_rc=core_rc,
            owner_core=owner_core,
            scene_shape=scene.shape,
            halo_pixels=int(partition_config.halo_pixels),
            anchor_cluster_id=int(parent.cluster_id),
            anchor_lifetime=float(parent.lifetime),
        )
        partitions.append(partition)
        assigned[r0 : r1 + 1, c0 : c1 + 1] |= owner_core
        partition_id += 1

    tile = max(int(partition_config.fallback_tile_pixels), 1)
    height, width = scene.shape
    for r0 in range(0, height, tile):
        r1 = min(r0 + tile - 1, height - 1)
        for c0 in range(0, width, tile):
            c1 = min(c0 + tile - 1, width - 1)
            owner_core = domain[r0 : r1 + 1, c0 : c1 + 1] & ~assigned[r0 : r1 + 1, c0 : c1 + 1]
            if int(owner_core.sum()) < int(partition_config.min_fallback_owner_pixels):
                continue
            partition = _make_partition(
                partition_id=partition_id,
                kind="fallback_grid",
                core_rc=(r0, r1, c0, c1),
                owner_core=owner_core,
                scene_shape=scene.shape,
                halo_pixels=int(partition_config.halo_pixels),
            )
            partitions.append(partition)
            assigned[r0 : r1 + 1, c0 : c1 + 1] |= owner_core
            partition_id += 1

    return partitions


def _bbox_intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ar1, ac0, ac1 = [int(v) for v in a]
    br0, br1, bc0, bc1 = [int(v) for v in b]
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _clusters_for_crop(clusters: Iterable[DruidCluster], crop_rc: tuple[int, int, int, int]) -> list[DruidCluster]:
    return [cluster for cluster in clusters if _bbox_intersects(cluster.bbox_rc, crop_rc)]


def _union_mask_for_clusters(clusters: list[DruidCluster], crop_rc: tuple[int, int, int, int], shape: tuple[int, int]) -> np.ndarray:
    if not clusters:
        return np.zeros(shape, dtype=np.float32)
    masks = [_mask_for_cluster(cluster, crop_rc) for cluster in clusters]
    return np.maximum.reduce(masks).astype(np.float32, copy=False)


def _soft_attention_or_zero(
    parent_mask: np.ndarray,
    child_union_mask: np.ndarray,
    seed_map: np.ndarray,
    persistence_map: np.ndarray,
    *,
    distance_sigma: float,
) -> np.ndarray:
    hard = np.maximum(np.maximum(parent_mask, child_union_mask), seed_map)
    if not bool(np.any(hard > 0)):
        return np.zeros_like(parent_mask, dtype=np.float32)
    return _soft_attention_from_masks(
        parent_mask,
        child_union_mask,
        seed_map,
        persistence_map,
        distance_sigma=float(distance_sigma),
    )


def build_partitioned_density_patches(
    scene: SceneRaster,
    gt_count_map: np.ndarray,
    cluster_store: DruidClusterStore,
    *,
    child_cluster_store: DruidClusterStore | None = None,
    valid_mask: np.ndarray | None = None,
    patch_config: DensityPatchConfig | None = None,
    target_config: DensityTargetConfig | None = None,
    partition_config: ScenePartitionConfig | None = None,
) -> tuple[list[DensityPatch], list[ScenePartition], dict[str, Any]]:
    patch_config = patch_config or DensityPatchConfig()
    target_config = target_config or DensityTargetConfig()
    partition_config = partition_config or ScenePartitionConfig()

    gt = np.asarray(gt_count_map, dtype=np.float32)
    if tuple(gt.shape) != tuple(scene.shape):
        raise ValueError(f"gt_count_map shape mismatch: {gt.shape} != {scene.shape}")
    domain = _domain_mask(scene.shape, valid_mask)

    parent_config = replace(patch_config, max_patches=None)
    selected_parents = _select_parent_clusters(list(cluster_store.clusters), parent_config)
    child_source = list(child_cluster_store.clusters) if child_cluster_store is not None else list(cluster_store.clusters)
    child_clusters = [
        cluster
        for cluster in child_source
        if _cluster_passes_size(cluster, int(patch_config.child_min_nodes), patch_config.child_max_nodes)
    ]
    partitions = build_scene_partitions(
        scene,
        cluster_store,
        valid_mask=domain,
        patch_config=parent_config,
        partition_config=partition_config,
    )

    patches: list[DensityPatch] = []
    coverage = np.zeros(scene.shape, dtype=np.int16)
    for partition in partitions:
        r0, r1, c0, c1 = [int(v) for v in partition.crop_rc]
        crop_shape = (r1 - r0 + 1, c1 - c0 + 1)
        sea_crop = domain[r0 : r1 + 1, c0 : c1 + 1].astype(np.float32, copy=False)
        owner_mask = np.asarray(partition.owner_mask, dtype=np.float32) * sea_crop
        if int(owner_mask.sum()) <= 0:
            continue

        parent_in_crop = _clusters_for_crop(selected_parents, partition.crop_rc)
        child_in_crop = _clusters_for_crop(child_clusters, partition.crop_rc)
        ph_sources = child_in_crop if child_in_crop else parent_in_crop

        image_crop = np.asarray(scene.image[r0 : r1 + 1, c0 : c1 + 1], dtype=np.float32) * sea_crop
        raw_crop = np.asarray(gt[r0 : r1 + 1, c0 : c1 + 1], dtype=np.float32) * owner_mask
        parent_mask = _union_mask_for_clusters(parent_in_crop, partition.crop_rc, crop_shape) * sea_crop
        child_union_mask = _union_mask_for_clusters(child_in_crop, partition.crop_rc, crop_shape) * sea_crop
        seed_map = _seed_map_for_clusters(
            ph_sources,
            partition.crop_rc,
            crop_shape,
            radius_pixels=int(patch_config.seed_radius_pixels),
        ) * sea_crop
        persistence_map = _persistence_map_for_clusters(ph_sources, partition.crop_rc, crop_shape) * sea_crop
        soft_attention = _soft_attention_or_zero(
            parent_mask,
            child_union_mask,
            seed_map,
            persistence_map,
            distance_sigma=float(patch_config.attention_distance_sigma),
        ) * sea_crop
        loss_weight = _loss_weight_from_attention(
            soft_attention,
            base_weight=float(patch_config.attention_base_weight),
            ph_weight=float(patch_config.attention_ph_weight),
        ) * owner_mask
        target = make_sum_preserving_density_target(raw_crop, parent_mask, target_config, domain_mask=owner_mask)

        patches.append(
            DensityPatch(
                cluster_id=int(partition.partition_id),
                lifetime=float(partition.anchor_lifetime),
                bbox_rc=tuple(int(v) for v in partition.core_rc),
                crop_rc=tuple(int(v) for v in partition.crop_rc),
                image=image_crop.astype(np.float32, copy=False),
                parent_mask=parent_mask.astype(np.float32, copy=False),
                child_union_mask=child_union_mask.astype(np.float32, copy=False),
                seed_map=seed_map.astype(np.float32, copy=False),
                persistence_map=persistence_map.astype(np.float32, copy=False),
                soft_attention=soft_attention.astype(np.float32, copy=False),
                loss_weight=loss_weight.astype(np.float32, copy=False),
                valid_mask=owner_mask.astype(np.float32, copy=False),
                target_density=target.astype(np.float32, copy=False),
                raw_count=raw_crop.astype(np.float32, copy=False),
                child_ids=[int(cluster.cluster_id) for cluster in child_in_crop],
                partition_id=int(partition.partition_id),
                partition_kind=str(partition.kind),
                anchor_cluster_id=partition.anchor_cluster_id,
            )
        )
        owner_global = owner_mask > 0
        coverage[r0 : r1 + 1, c0 : c1 + 1] += owner_global.astype(np.int16)

    valid_pixels = int(domain.sum())
    covered_once = int(((coverage == 1) & domain).sum())
    missed = int((domain & (coverage == 0)).sum())
    overlap = int((domain & (coverage > 1)).sum())
    raw_sums = np.array([patch.raw_count_sum for patch in patches], dtype=np.float32) if patches else np.array([], dtype=np.float32)
    target_sums = np.array([patch.target_sum for patch in patches], dtype=np.float32) if patches else np.array([], dtype=np.float32)
    summary = {
        "partition_count": int(len(partitions)),
        "patch_count": int(len(patches)),
        "ph_anchor_count": int(sum(1 for item in partitions if item.kind == "ph_anchor")),
        "fallback_grid_count": int(sum(1 for item in partitions if item.kind == "fallback_grid")),
        "valid_sea_pixels": valid_pixels,
        "covered_once_pixels": covered_once,
        "missed_valid_pixels": missed,
        "overlap_valid_pixels": overlap,
        "coverage_ratio": float(covered_once / max(valid_pixels, 1)),
        "raw_count_sum": float(raw_sums.sum()) if raw_sums.size else 0.0,
        "target_density_sum": float(target_sums.sum()) if target_sums.size else 0.0,
        "max_abs_raw_target_diff": float(np.max(np.abs(raw_sums - target_sums))) if raw_sums.size else 0.0,
        "fallback_tile_pixels": int(partition_config.fallback_tile_pixels),
        "halo_pixels": int(partition_config.halo_pixels),
        "anchor_padding_pixels": int(partition_config.anchor_padding_pixels),
    }
    return patches, partitions, summary
