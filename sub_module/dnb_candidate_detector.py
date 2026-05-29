from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, label

from .dnb_gat_pipeline import DruidCluster, DruidClusterStore, SceneRaster


@dataclass
class DnbCandidateDetectorConfig:
    """DRUID-inspired candidate extraction for DNB ship scenes.

    The output intentionally reuses DruidClusterStore so the downstream graph/model
    pipeline can consume either backend without changing its contract.
    """

    backend: str = "cripser"
    detection_threshold: float = 1.0
    analysis_threshold: float = 1.0
    threshold_reference: str = "zero"  # "zero" matches DRUID; "median" is often useful for DNB scenes.
    smooth_sigma: float = 0.0
    lifetime_limit: float = 0.0
    lifetime_limit_fraction: float = 1.001
    area_limit: int = 4
    min_nodes: int = 16
    max_nodes: int = 4096
    max_candidates: int | None = None
    connectivity: int = 1
    remove_edge: bool = True
    drop_nested: bool = True


@dataclass(frozen=True)
class RobustBackground:
    center: float
    sigma: float
    detection_floor: float
    analysis_floor: float


def robust_background(
    image: np.ndarray,
    *,
    detection_threshold: float,
    analysis_threshold: float,
    threshold_reference: str = "zero",
) -> RobustBackground:
    finite = np.asarray(image, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("image has no finite pixels")

    center = float(np.nanmedian(finite))
    mad = float(np.nanmedian(np.abs(finite - center)))
    sigma = 1.4826 * mad
    if not np.isfinite(sigma) or sigma <= 0.0:
        sigma = float(np.nanstd(finite))
    if not np.isfinite(sigma) or sigma <= 0.0:
        sigma = 1.0e-6

    reference = str(threshold_reference).lower()
    if reference == "zero":
        base = 0.0
    elif reference == "median":
        base = center
    else:
        raise ValueError("threshold_reference must be 'zero' or 'median'")

    return RobustBackground(
        center=center,
        sigma=float(sigma),
        detection_floor=float(base + float(detection_threshold) * sigma),
        analysis_floor=float(base + float(analysis_threshold) * sigma),
    )


def _connectivity_structure(connectivity: int) -> np.ndarray:
    if int(connectivity) == 1:
        return np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    if int(connectivity) == 2:
        return np.ones((3, 3), dtype=np.uint8)
    raise ValueError("connectivity must be 1 or 2")


def _component_mask(
    image: np.ndarray,
    *,
    seed_rc: tuple[int, int],
    birth: float,
    death: float,
    connectivity: int,
) -> np.ndarray | None:
    mask = np.logical_and(image <= float(birth), image > float(death))
    seed_r, seed_c = int(seed_rc[0]), int(seed_rc[1])
    if seed_r < 0 or seed_c < 0 or seed_r >= mask.shape[0] or seed_c >= mask.shape[1]:
        return None
    if not mask[seed_r, seed_c]:
        return None

    labeled, _ = label(mask, structure=_connectivity_structure(connectivity))
    component_id = int(labeled[seed_r, seed_c])
    if component_id == 0:
        return None
    return labeled == component_id


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    rmin = int(coords[:, 0].min())
    rmax = int(coords[:, 0].max())
    cmin = int(coords[:, 1].min())
    cmax = int(coords[:, 1].max())
    return rmin, rmax, cmin, cmax


def _touches_edge(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())


def _contour_from_local_mask(local_mask: np.ndarray, rmin: int, cmin: int) -> np.ndarray | None:
    contours, _ = cv2.findContours(local_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour_xy = max(contours, key=cv2.contourArea).reshape(-1, 2)
    if contour_xy.shape[0] < 3:
        return None
    return np.column_stack([contour_xy[:, 1] + int(rmin), contour_xy[:, 0] + int(cmin)]).astype(np.int32)


def _drop_nested(clusters: list[DruidCluster]) -> list[DruidCluster]:
    kept: list[DruidCluster] = []
    for cluster in sorted(clusters, key=lambda item: item.lifetime, reverse=True):
        coords = set(map(tuple, cluster.global_rc.tolist()))
        if any(coords and coords.issubset(existing.coords_set) for existing in kept):
            continue
        cluster.coords_set = coords
        kept.append(cluster)
    return kept


class DnbCandidateDetector:
    def __init__(self, config: DnbCandidateDetectorConfig | None = None) -> None:
        self.config = config or DnbCandidateDetectorConfig()

    def build_store(self, scene: SceneRaster, gt_count_map: np.ndarray | None = None) -> DruidClusterStore:
        if gt_count_map is None:
            gt_count_map = np.zeros(scene.shape, dtype=np.float32)
        if tuple(gt_count_map.shape) != tuple(scene.shape):
            raise ValueError(f"gt_count_map shape mismatch: {gt_count_map.shape} != {scene.shape}")

        smooth = self._smooth(scene.image)
        background = robust_background(
            smooth,
            detection_threshold=self.config.detection_threshold,
            analysis_threshold=self.config.analysis_threshold,
            threshold_reference=self.config.threshold_reference,
        )
        candidates = self._find_candidates(smooth, background)
        catalogue, clusters = self._build_clusters(scene, smooth, gt_count_map.astype(np.float32, copy=False), candidates)
        if self.config.drop_nested:
            clusters = _drop_nested(clusters)
            kept_ids = {int(cluster.cluster_id) for cluster in clusters}
            catalogue = catalogue[catalogue["ID"].isin(kept_ids)].copy().reset_index(drop=True)
        return DruidClusterStore(scene=scene, catalogue=catalogue, clusters=clusters)

    def _smooth(self, image: np.ndarray) -> np.ndarray:
        arr = np.asarray(image, dtype=np.float32)
        if float(self.config.smooth_sigma) <= 0.0:
            return arr
        return gaussian_filter(arr, sigma=float(self.config.smooth_sigma)).astype(np.float32, copy=False)

    def _find_candidates(self, image: np.ndarray, background: RobustBackground) -> pd.DataFrame:
        backend = str(self.config.backend).lower()
        if backend != "cripser":
            raise ValueError("Only backend='cripser' is implemented for now")

        try:
            import cripser
        except ImportError as exc:
            raise ImportError("cripser is required for DnbCandidateDetector backend='cripser'") from exc

        ph = cripser.computePH(-np.asarray(image, dtype=np.float64), maxdim=0)
        if ph is None or len(ph) == 0:
            return self._empty_catalogue()

        frame = pd.DataFrame(
            ph,
            columns=["dim", "Birth", "Death", "x1", "y1", "z1", "x2", "y2", "z2"],
        )
        frame = frame.loc[frame["dim"] == 0, ["Birth", "Death", "x1", "y1", "x2", "y2"]].copy()
        frame["Birth"] = -frame["Birth"].astype(float)
        frame["Death"] = -frame["Death"].astype(float)
        frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["Birth", "Death", "x1", "y1"])
        if frame.empty:
            return self._empty_catalogue()

        frame["Death"] = np.maximum(frame["Death"].to_numpy(dtype=float), float(background.analysis_floor))
        frame["lifetime"] = frame["Birth"] - frame["Death"]
        eps = max(float(background.analysis_floor), 1.0e-12)
        frame["lifetimeFrac"] = frame["Birth"] / np.maximum(frame["Death"], eps)
        frame = frame.loc[
            (frame["Birth"] > float(background.detection_floor))
            & (frame["lifetime"] > float(self.config.lifetime_limit))
            & (frame["lifetimeFrac"] > float(self.config.lifetime_limit_fraction))
        ].copy()
        if frame.empty:
            return self._empty_catalogue()

        frame["x1"] = frame["x1"].round().astype(int)
        frame["y1"] = frame["y1"].round().astype(int)
        frame["x2"] = frame["x2"].round().astype(int)
        frame["y2"] = frame["y2"].round().astype(int)
        frame = frame.sort_values("lifetime", ascending=False).reset_index(drop=True)
        if self.config.max_candidates is not None:
            frame = frame.head(int(self.config.max_candidates)).copy()
        frame["ID"] = np.arange(1, len(frame) + 1, dtype=int)
        frame["bg_rms"] = float(background.sigma)
        frame["mean_bg"] = float(background.center)
        frame["detection_floor"] = float(background.detection_floor)
        frame["analysis_floor"] = float(background.analysis_floor)
        return frame

    @staticmethod
    def _empty_catalogue() -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "ID",
                "Birth",
                "Death",
                "x1",
                "y1",
                "x2",
                "y2",
                "lifetime",
                "lifetimeFrac",
                "area",
                "edge_flag",
                "bbox1",
                "bbox2",
                "bbox3",
                "bbox4",
                "contour",
                "bg_rms",
                "mean_bg",
                "detection_floor",
                "analysis_floor",
            ]
        )

    def _build_clusters(
        self,
        scene: SceneRaster,
        smooth: np.ndarray,
        gt_count_map: np.ndarray,
        candidates: pd.DataFrame,
    ) -> tuple[pd.DataFrame, list[DruidCluster]]:
        if candidates.empty:
            return self._empty_catalogue(), []

        rows: list[dict[str, Any]] = []
        clusters: list[DruidCluster] = []
        for row in candidates.itertuples(index=False):
            component = _component_mask(
                smooth,
                seed_rc=(int(row.x1), int(row.y1)),
                birth=float(row.Birth),
                death=float(row.Death),
                connectivity=int(self.config.connectivity),
            )
            if component is None:
                continue
            area = int(component.sum())
            if area <= int(self.config.area_limit):
                continue
            edge_flag = _touches_edge(component)
            if bool(self.config.remove_edge) and edge_flag:
                continue
            bbox = _mask_bbox(component)
            if bbox is None:
                continue
            rmin, rmax, cmin, cmax = bbox
            local_mask = component[rmin : rmax + 1, cmin : cmax + 1].astype(np.uint8)
            local_rc = np.argwhere(local_mask == 1)
            node_count = int(local_rc.shape[0])
            if node_count < int(self.config.min_nodes) or node_count > int(self.config.max_nodes):
                continue
            contour = _contour_from_local_mask(local_mask, rmin, cmin)
            if contour is None:
                continue

            patch = scene.image[rmin : rmax + 1, cmin : cmax + 1]
            gt_patch = gt_count_map[rmin : rmax + 1, cmin : cmax + 1]
            global_rc = local_rc + np.array([rmin, cmin], dtype=np.int32)
            cluster_id = int(row.ID)
            cluster = DruidCluster(
                cluster_id=cluster_id,
                lifetime=float(row.lifetime),
                birth=float(row.Birth),
                death=float(row.Death),
                contour_rc=contour,
                bbox_rc=(rmin, rmax, cmin, cmax),
                seed_rc=(int(row.x1), int(row.y1)),
                patch_image=patch,
                patch_gt=gt_patch,
                mask=local_mask,
                local_rc=local_rc.astype(np.int32),
                global_rc=global_rc.astype(np.int32),
            )
            cluster.coords_set = set(map(tuple, cluster.global_rc.tolist()))
            clusters.append(cluster)

            rows.append(
                {
                    "ID": cluster_id,
                    "Birth": float(row.Birth),
                    "Death": float(row.Death),
                    "x1": int(row.x1),
                    "y1": int(row.y1),
                    "x2": int(row.x2),
                    "y2": int(row.y2),
                    "lifetime": float(row.lifetime),
                    "lifetimeFrac": float(row.lifetimeFrac),
                    "area": area,
                    "edge_flag": int(edge_flag),
                    "bbox1": int(rmin),
                    "bbox2": int(cmin),
                    "bbox3": int(rmax),
                    "bbox4": int(cmax),
                    "node_count": node_count,
                    "gt_sum": float(cluster.gt_sum),
                    "contour": contour,
                    "bg_rms": float(row.bg_rms),
                    "mean_bg": float(row.mean_bg),
                    "detection_floor": float(row.detection_floor),
                    "analysis_floor": float(row.analysis_floor),
                }
            )

        catalogue = pd.DataFrame(rows)
        if not catalogue.empty:
            catalogue = catalogue.sort_values("lifetime", ascending=False).reset_index(drop=True)
        return catalogue, clusters


def candidate_store_summary(store: DruidClusterStore) -> dict[str, Any]:
    clusters = list(store.clusters)
    node_counts = np.array([cluster.node_count for cluster in clusters], dtype=np.int64)
    gt_sums = np.array([cluster.gt_sum for cluster in clusters], dtype=np.float32)
    return {
        "candidate_count": int(len(clusters)),
        "catalogue_count": int(len(store.catalogue)),
        "total_nodes": int(node_counts.sum()) if node_counts.size else 0,
        "node_count_min": int(node_counts.min()) if node_counts.size else 0,
        "node_count_median": int(np.median(node_counts)) if node_counts.size else 0,
        "node_count_max": int(node_counts.max()) if node_counts.size else 0,
        "clusters_with_gt": int((gt_sums > 0).sum()) if gt_sums.size else 0,
        "gt_mass_in_candidates": float(gt_sums.sum()) if gt_sums.size else 0.0,
    }
