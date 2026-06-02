from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .dnb_candidate_detector import DnbCandidateDetector, DnbCandidateDetectorConfig
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
from .dnb_pipeline_core import DruidCluster, DruidClusterStore, SceneRaster


@dataclass(frozen=True)
class ScenePartitionConfig:
    enabled: bool = True
    fallback_tile_pixels: int = 96
    halo_pixels: int = 16
    anchor_padding_pixels: int = 16
    min_owner_pixels: int = 1
    min_fallback_owner_pixels: int = 1
    hierarchical_ph_enabled: bool = False
    hierarchical_large_min_pixels: int = 65536
    hierarchical_large_min_height: int = 384
    hierarchical_large_min_width: int = 384
    hierarchical_child_anchor_padding_pixels: int = 8
    hierarchical_child_detection_threshold: float = 0.5
    hierarchical_child_analysis_threshold: float = 0.25
    hierarchical_child_threshold_reference: str = "median"
    hierarchical_child_smooth_sigma: float = 0.0
    hierarchical_child_lifetime_limit: float = 0.0
    hierarchical_child_lifetime_limit_fraction: float = 1.0005
    hierarchical_child_area_limit: int = 0
    hierarchical_child_min_nodes: int = 3
    hierarchical_child_max_nodes: int = 2048
    hierarchical_child_max_candidates_per_parent: int = 64
    hierarchical_keep_large_parent: bool = False


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


def _is_large_parent(
    core_rc: tuple[int, int, int, int],
    owner_core: np.ndarray,
    partition_config: ScenePartitionConfig,
) -> bool:
    r0, r1, c0, c1 = [int(v) for v in core_rc]
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)
    owner_pixels = int(np.asarray(owner_core, dtype=bool).sum())
    checks: list[bool] = []
    if int(partition_config.hierarchical_large_min_pixels) > 0:
        checks.append(owner_pixels >= int(partition_config.hierarchical_large_min_pixels))
    if int(partition_config.hierarchical_large_min_height) > 0:
        checks.append(height >= int(partition_config.hierarchical_large_min_height))
    if int(partition_config.hierarchical_large_min_width) > 0:
        checks.append(width >= int(partition_config.hierarchical_large_min_width))
    return bool(any(checks))


def _child_detector_config(partition_config: ScenePartitionConfig) -> DnbCandidateDetectorConfig:
    max_candidates = int(partition_config.hierarchical_child_max_candidates_per_parent)
    return DnbCandidateDetectorConfig(
        detection_threshold=float(partition_config.hierarchical_child_detection_threshold),
        analysis_threshold=float(partition_config.hierarchical_child_analysis_threshold),
        threshold_reference=str(partition_config.hierarchical_child_threshold_reference),
        smooth_sigma=float(partition_config.hierarchical_child_smooth_sigma),
        lifetime_limit=float(partition_config.hierarchical_child_lifetime_limit),
        lifetime_limit_fraction=float(partition_config.hierarchical_child_lifetime_limit_fraction),
        area_limit=int(partition_config.hierarchical_child_area_limit),
        min_nodes=int(partition_config.hierarchical_child_min_nodes),
        max_nodes=int(partition_config.hierarchical_child_max_nodes),
        max_candidates=max_candidates if max_candidates > 0 else None,
        connectivity=1,
        remove_edge=False,
        drop_nested=False,
    )


def _offset_cluster_to_global(
    cluster: DruidCluster,
    *,
    cluster_id: int,
    scene: SceneRaster,
    gt_count_map: np.ndarray,
    row_offset: int,
    col_offset: int,
) -> DruidCluster:
    global_rc = np.asarray(cluster.global_rc, dtype=np.int32) + np.array([int(row_offset), int(col_offset)], dtype=np.int32)
    if global_rc.size == 0:
        raise ValueError("cannot offset empty child cluster")
    rmin = int(global_rc[:, 0].min())
    rmax = int(global_rc[:, 0].max())
    cmin = int(global_rc[:, 1].min())
    cmax = int(global_rc[:, 1].max())
    local_rc = global_rc - np.array([rmin, cmin], dtype=np.int32)
    local_mask = np.zeros((rmax - rmin + 1, cmax - cmin + 1), dtype=np.uint8)
    local_mask[local_rc[:, 0], local_rc[:, 1]] = 1
    contour = np.asarray(cluster.contour_rc, dtype=np.int32) + np.array([int(row_offset), int(col_offset)], dtype=np.int32)
    seed_rc = (int(cluster.seed_rc[0]) + int(row_offset), int(cluster.seed_rc[1]) + int(col_offset))
    out = DruidCluster(
        cluster_id=int(cluster_id),
        lifetime=float(cluster.lifetime),
        birth=float(cluster.birth),
        death=float(cluster.death),
        contour_rc=contour.astype(np.int32, copy=False),
        bbox_rc=(rmin, rmax, cmin, cmax),
        seed_rc=seed_rc,
        patch_image=scene.image[rmin : rmax + 1, cmin : cmax + 1],
        patch_gt=gt_count_map[rmin : rmax + 1, cmin : cmax + 1],
        mask=local_mask,
        local_rc=local_rc.astype(np.int32, copy=False),
        global_rc=global_rc.astype(np.int32, copy=False),
    )
    out.coords_set = set(map(tuple, out.global_rc.tolist()))
    return out


def _cluster_catalogue_rows(clusters: list[DruidCluster]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cluster in clusters:
        rmin, rmax, cmin, cmax = [int(v) for v in cluster.bbox_rc]
        rows.append(
            {
                "ID": int(cluster.cluster_id),
                "Birth": float(cluster.birth),
                "Death": float(cluster.death),
                "x1": int(cluster.seed_rc[0]),
                "y1": int(cluster.seed_rc[1]),
                "x2": int(cluster.seed_rc[0]),
                "y2": int(cluster.seed_rc[1]),
                "lifetime": float(cluster.lifetime),
                "lifetimeFrac": float(cluster.birth / max(cluster.death, 1.0e-8)),
                "area": int(cluster.node_count),
                "edge_flag": 0,
                "bbox1": rmin,
                "bbox2": cmin,
                "bbox3": rmax,
                "bbox4": cmax,
                "node_count": int(cluster.node_count),
                "gt_sum": float(cluster.gt_sum),
                "contour": cluster.contour_rc,
                "source": "hierarchical_ph_child",
            }
        )
    return rows


def build_hierarchical_child_store(
    scene: SceneRaster,
    gt_count_map: np.ndarray,
    cluster_store: DruidClusterStore,
    *,
    valid_mask: np.ndarray | None = None,
    patch_config: DensityPatchConfig | None = None,
    partition_config: ScenePartitionConfig | None = None,
) -> tuple[DruidClusterStore, dict[str, Any]]:
    patch_config = patch_config or DensityPatchConfig()
    partition_config = partition_config or ScenePartitionConfig()
    empty = DruidClusterStore(scene=scene, catalogue=pd.DataFrame(), clusters=[])
    if not bool(partition_config.hierarchical_ph_enabled):
        return empty, {"enabled": False}

    gt = np.asarray(gt_count_map, dtype=np.float32)
    if tuple(gt.shape) != tuple(scene.shape):
        raise ValueError(f"gt_count_map shape mismatch: {gt.shape} != {scene.shape}")
    domain = _domain_mask(scene.shape, valid_mask)
    parent_config = replace(patch_config, max_patches=None)
    parents = _select_parent_clusters(list(cluster_store.clusters), parent_config)
    detector = DnbCandidateDetector(_child_detector_config(partition_config))

    next_id = max([int(cluster.cluster_id) for cluster in cluster_store.clusters] + [0]) + 1
    child_clusters: list[DruidCluster] = []
    large_parent_count = 0
    large_parent_with_children = 0

    for parent in parents:
        core_rc = _expand_rc(
            parent.bbox_rc,
            scene.shape,
            padding_pixels=int(partition_config.anchor_padding_pixels),
        )
        r0, r1, c0, c1 = [int(v) for v in core_rc]
        owner_core = domain[r0 : r1 + 1, c0 : c1 + 1]
        if not _is_large_parent(core_rc, owner_core, partition_config):
            continue
        large_parent_count += 1

        local_image = np.asarray(scene.image[r0 : r1 + 1, c0 : c1 + 1], dtype=np.float32)
        local_gt = gt[r0 : r1 + 1, c0 : c1 + 1]
        local_scene = SceneRaster(
            path=scene.path,
            image=local_image,
            transform=scene.transform,
            crs=scene.crs,
            bounds=scene.bounds,
            height=int(local_image.shape[0]),
            width=int(local_image.shape[1]),
        )
        local_store = detector.build_store(local_scene, local_gt, valid_mask=owner_core)
        local_children = sorted(local_store.clusters, key=lambda item: (item.lifetime, item.node_count), reverse=True)
        max_children = int(partition_config.hierarchical_child_max_candidates_per_parent)
        if max_children > 0:
            local_children = local_children[:max_children]
        if local_children:
            large_parent_with_children += 1
        for local_child in local_children:
            child_clusters.append(
                _offset_cluster_to_global(
                    local_child,
                    cluster_id=next_id,
                    scene=scene,
                    gt_count_map=gt,
                    row_offset=r0,
                    col_offset=c0,
                )
            )
            next_id += 1

    catalogue = pd.DataFrame(_cluster_catalogue_rows(child_clusters))
    metadata = {
        "enabled": True,
        "large_parent_count": int(large_parent_count),
        "large_parent_with_children": int(large_parent_with_children),
        "child_cluster_count": int(len(child_clusters)),
        "child_detection_threshold": float(partition_config.hierarchical_child_detection_threshold),
        "child_analysis_threshold": float(partition_config.hierarchical_child_analysis_threshold),
        "child_lifetime_limit_fraction": float(partition_config.hierarchical_child_lifetime_limit_fraction),
    }
    return DruidClusterStore(scene=scene, catalogue=catalogue, clusters=child_clusters), metadata


def _clusters_inside_core(
    clusters: Iterable[DruidCluster],
    core_rc: tuple[int, int, int, int],
) -> list[DruidCluster]:
    r0, r1, c0, c1 = [int(v) for v in core_rc]
    inside: list[DruidCluster] = []
    for cluster in clusters:
        cr0, cr1, cc0, cc1 = [int(v) for v in cluster.bbox_rc]
        if cr0 >= r0 and cr1 <= r1 and cc0 >= c0 and cc1 <= c1:
            inside.append(cluster)
    return sorted(inside, key=lambda item: (item.lifetime, item.node_count), reverse=True)


def build_scene_partitions(
    scene: SceneRaster,
    cluster_store: DruidClusterStore,
    *,
    child_clusters: Iterable[DruidCluster] | None = None,
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
    hierarchical_children = list(child_clusters) if child_clusters is not None else []

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
        is_large = bool(partition_config.hierarchical_ph_enabled) and _is_large_parent(core_rc, owner_core, partition_config)
        if is_large:
            child_padding = int(partition_config.hierarchical_child_anchor_padding_pixels)
            for child in _clusters_inside_core(hierarchical_children, core_rc):
                child_core = _expand_rc(
                    child.bbox_rc,
                    scene.shape,
                    padding_pixels=child_padding,
                )
                cr0, cr1, cc0, cc1 = [int(v) for v in child_core]
                child_core = (
                    max(cr0, r0),
                    min(cr1, r1),
                    max(cc0, c0),
                    min(cc1, c1),
                )
                cr0, cr1, cc0, cc1 = [int(v) for v in child_core]
                child_owner_core = domain[cr0 : cr1 + 1, cc0 : cc1 + 1] & ~assigned[cr0 : cr1 + 1, cc0 : cc1 + 1]
                if int(child_owner_core.sum()) < int(partition_config.min_owner_pixels):
                    continue
                partition = _make_partition(
                    partition_id=partition_id,
                    kind="ph_child",
                    core_rc=child_core,
                    owner_core=child_owner_core,
                    scene_shape=scene.shape,
                    halo_pixels=int(partition_config.halo_pixels),
                    anchor_cluster_id=int(child.cluster_id),
                    anchor_lifetime=float(child.lifetime),
                )
                partitions.append(partition)
                assigned[cr0 : cr1 + 1, cc0 : cc1 + 1] |= child_owner_core
                partition_id += 1
            if not bool(partition_config.hierarchical_keep_large_parent):
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
    hierarchical_store, hierarchical_summary = build_hierarchical_child_store(
        scene,
        gt,
        cluster_store,
        valid_mask=domain,
        patch_config=parent_config,
        partition_config=partition_config,
    )
    child_source = list(child_cluster_store.clusters) if child_cluster_store is not None else list(cluster_store.clusters)
    child_source = child_source + list(hierarchical_store.clusters)
    child_clusters = [
        cluster
        for cluster in child_source
        if _cluster_passes_size(cluster, int(patch_config.child_min_nodes), patch_config.child_max_nodes)
    ]
    partitions = build_scene_partitions(
        scene,
        cluster_store,
        child_clusters=hierarchical_store.clusters,
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
        "ph_child_count": int(sum(1 for item in partitions if item.kind == "ph_child")),
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
        "hierarchical_ph": hierarchical_summary,
    }
    return patches, partitions, summary
