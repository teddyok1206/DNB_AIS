from __future__ import annotations

import re
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol
from shapely.geometry import shape


SCENE_KEY_PATTERN = re.compile(r"(A\d{7}_\d{4}_\d{3})")


def extract_scene_key(path_like: str | Path) -> str:
    match = SCENE_KEY_PATTERN.search(Path(path_like).name)
    if match is None:
        raise ValueError(f"Could not extract scene key from: {path_like}")
    return match.group(1)


@dataclass
class SceneRaster:
    path: Path
    image: np.ndarray
    transform: Any
    crs: Any
    bounds: Any
    height: int
    width: int

    @classmethod
    def load(cls, path: str | Path) -> "SceneRaster":
        path = Path(path).resolve()
        with rasterio.open(path) as src:
            image = src.read(1).astype(np.float32, copy=False)
            return cls(
                path=path,
                image=image,
                transform=src.transform,
                crs=src.crs,
                bounds=src.bounds,
                height=src.height,
                width=src.width,
            )

    @property
    def key(self) -> str:
        return extract_scene_key(self.path)

    @property
    def shape(self) -> tuple[int, int]:
        return self.image.shape


@dataclass
class PHCluster:
    """PH candidate component shared by the U-Net density pipeline."""

    cluster_id: int
    lifetime: float
    birth: float
    death: float
    contour_rc: np.ndarray
    bbox_rc: tuple[int, int, int, int]
    seed_rc: tuple[int, int]
    patch_image: np.ndarray
    patch_gt: np.ndarray
    mask: np.ndarray
    local_rc: np.ndarray
    global_rc: np.ndarray
    coords_set: set[tuple[int, int]] = field(default_factory=set)

    @property
    def weight(self) -> float:
        return max(self.lifetime, 1.0e-6)

    @property
    def node_count(self) -> int:
        return int(self.local_rc.shape[0])

    @property
    def gt_sum(self) -> float:
        return float(self.patch_gt[self.mask > 0].sum())

    @property
    def bbox_height(self) -> int:
        return int(self.bbox_rc[1] - self.bbox_rc[0] + 1)

    @property
    def bbox_width(self) -> int:
        return int(self.bbox_rc[3] - self.bbox_rc[2] + 1)


class GroundTruthResolver:
    def __init__(self, default_geojson_dir: str | Path) -> None:
        self.default_geojson_dir = Path(default_geojson_dir).resolve()

    def resolve_geojson(self, scene: SceneRaster, requested_path: str | Path | None = None) -> Path:
        if requested_path is not None:
            requested = Path(requested_path).expanduser()
            if requested.exists():
                return requested.resolve()

        default_path = self.default_geojson_dir / f"{scene.key}.geojson"
        if default_path.exists():
            return default_path.resolve()

        requested_msg = "none" if requested_path is None else str(Path(requested_path).expanduser())
        raise FileNotFoundError(
            f"GeoJSON ground truth not found for {scene.key}. "
            f"requested_path={requested_msg}; default_path={default_path}. "
            "DB-backed GeoJSON generation has been removed; run the bbox generation step first."
        )

    def load_points(self, geojson_path: str | Path) -> list[dict[str, Any]]:
        obj = json.loads(Path(geojson_path).read_text())
        points: list[dict[str, Any]] = []
        for feature in obj.get("features", []):
            geometry = shape(feature["geometry"])
            props = feature.get("properties", {})
            if "Lon" in props and "Lat" in props:
                lon = float(props["Lon"])
                lat = float(props["Lat"])
            else:
                centroid = geometry.centroid
                lon = float(centroid.x)
                lat = float(centroid.y)
            points.append(
                {
                    "lon": lon,
                    "lat": lat,
                    "count": 1,
                    "mmsi": props.get("MMSI"),
                    "itpl": props.get("ITPL"),
                }
            )
        return points

    def rasterize_counts(self, scene: SceneRaster, points: Iterable[dict[str, Any]]) -> np.ndarray:
        count_map = np.zeros(scene.shape, dtype=np.float32)
        for point in points:
            try:
                row, col = rowcol(scene.transform, float(point["lon"]), float(point["lat"]))
            except Exception:
                continue
            if 0 <= row < scene.height and 0 <= col < scene.width:
                count_map[row, col] += float(point.get("count", 1.0))
        return count_map


class PHClusterStore:
    """Container for PH candidate components used by density target construction."""

    def __init__(self, scene: SceneRaster, catalogue: pd.DataFrame, clusters: list[PHCluster]) -> None:
        self.scene = scene
        self.catalogue = catalogue.reset_index(drop=True)
        self.clusters = clusters

    def summary_frame(self) -> pd.DataFrame:
        rows = []
        for cluster in self.clusters:
            rows.append(
                {
                    "cluster_id": cluster.cluster_id,
                    "lifetime": cluster.lifetime,
                    "node_count": cluster.node_count,
                    "bbox_height": cluster.bbox_height,
                    "bbox_width": cluster.bbox_width,
                    "gt_sum": cluster.gt_sum,
                    "seed_row": cluster.seed_rc[0],
                    "seed_col": cluster.seed_rc[1],
                }
            )
        return pd.DataFrame(rows).sort_values("lifetime", ascending=False).reset_index(drop=True)

    def patch_size_suggestions(self) -> dict[str, int]:
        summary = self.summary_frame()
        if summary.empty:
            return {"recommended_min_nodes": 0, "recommended_median_nodes": 0, "recommended_max_nodes": 0}

        return {
            "recommended_min_nodes": int(summary["node_count"].quantile(0.25)),
            "recommended_median_nodes": int(summary["node_count"].quantile(0.50)),
            "recommended_max_nodes": int(summary["node_count"].quantile(0.90)),
        }
