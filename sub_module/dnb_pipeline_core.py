from __future__ import annotations

import json
import math
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol
from shapely.geometry import Polygon, shape


SCENE_KEY_PATTERN = re.compile(r"(A\d{7}_\d{4}_\d{3})")
YEAR = 2025


def extract_scene_key(path_like: str | Path) -> str:
    match = SCENE_KEY_PATTERN.search(Path(path_like).name)
    if match is None:
        raise ValueError(f"Could not extract scene key from: {path_like}")
    return match.group(1)


def scene_key_to_center_dt(scene_key: str, metadata_csv: str | Path) -> str:
    metadata = pd.read_csv(metadata_csv)
    tif_name = f"{scene_key}.tif"
    row = metadata.loc[metadata["tif_name"] == tif_name]
    if row.empty:
        raise FileNotFoundError(f"{tif_name} not found in metadata: {metadata_csv}")

    row = row.iloc[0]
    duration = timedelta(seconds=float(row["duration [sec]"]))
    doy = int(scene_key.split("_")[0][-3:])
    start_date = date(YEAR, 1, 1)
    date_str = (start_date + timedelta(days=doy - 1)).isoformat()
    start_dt = datetime.combine(date.fromisoformat(date_str), time.fromisoformat(str(row["scan_start_HHMMSS"])))
    center_dt = start_dt + duration / 2
    return center_dt.replace(microsecond=0).strftime("%Y-%m-%d %H:%M:%S")


def _ship_bbox_geometry(lon: float, lat: float, cog: float, ship_m: tuple[float, float]) -> Polygon:
    ship_length_m, ship_width_m = ship_m
    half_length = ship_length_m / 2.0
    half_width = ship_width_m / 2.0

    corners_fr = [
        (+half_length, +half_width),
        (+half_length, -half_width),
        (-half_length, -half_width),
        (-half_length, +half_width),
        (+half_length, +half_width),
    ]

    theta = math.radians(float(cog) % 360.0)
    forward_e = math.sin(theta)
    forward_n = math.cos(theta)
    right_e = math.sin(theta + math.pi / 2.0)
    right_n = math.cos(theta + math.pi / 2.0)

    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = max(meters_per_deg_lat * math.cos(math.radians(lat)), 1.0e-9)

    ring = []
    for df, dr in corners_fr:
        east_m = df * forward_e + dr * right_e
        north_m = df * forward_n + dr * right_n
        dlon = east_m / meters_per_deg_lon
        dlat = north_m / meters_per_deg_lat
        ring.append((lon + dlon, lat + dlat))

    return Polygon(ring)


def _lookup_ship_static(cur: sqlite3.Cursor, mmsi: int) -> tuple[str, str, tuple[int, int, int, int]]:
    row = cur.execute(
        """
        SELECT VesselName, VesselType, DimA, DimB, DimC, DimD
        FROM ships_static
        WHERE MMSI = ?
        """,
        (int(mmsi),),
    ).fetchone()
    if row is None:
        return "", "", (0, 0, 0, 0)

    vessel_name, vessel_type, dim_a, dim_b, dim_c, dim_d = row
    return (
        str(vessel_name or ""),
        str(vessel_type or ""),
        (
            int(dim_a or 0),
            int(dim_b or 0),
            int(dim_c or 0),
            int(dim_d or 0),
        ),
    )


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
class DruidCluster:
    """PH candidate component shared by the U-Net density pipeline.

    The historical name is retained for compatibility with existing PH/DRUID-inspired
    code, but this container is not tied to the deleted GAT model path.
    """

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
    def __init__(
        self,
        metadata_csv: str | Path,
        ships_db_path: str | Path,
        default_geojson_dir: str | Path,
    ) -> None:
        self.metadata_csv = Path(metadata_csv).resolve()
        self.ships_db_path = Path(ships_db_path).resolve()
        self.default_geojson_dir = Path(default_geojson_dir).resolve()

    def resolve_geojson(self, scene: SceneRaster, requested_path: str | Path | None = None) -> Path:
        if requested_path is not None and Path(requested_path).exists():
            return Path(requested_path).resolve()

        default_path = self.default_geojson_dir / f"{scene.key}.geojson"
        if default_path.exists():
            return default_path

        default_path.parent.mkdir(parents=True, exist_ok=True)
        self._create_geojson_from_db(scene, default_path)
        return default_path

    def _create_geojson_from_db(self, scene: SceneRaster, output_path: Path) -> None:
        center_dt = scene_key_to_center_dt(scene.key, self.metadata_csv)
        conn = sqlite3.connect(self.ships_db_path)
        cur = conn.cursor()
        rows = cur.execute(
            """
            SELECT id, MMSI, Date, Time, Lon, Lat, SOG, COG, ITPL, IMG_DT
            FROM ships_dynamic
            WHERE IMG_DT IS NOT NULL AND IMG_DT = ? AND ITPL IS NOT NULL AND ITPL >= 0
            """,
            (center_dt,),
        ).fetchall()

        features: list[dict[str, Any]] = []
        for row in rows:
            _, mmsi, date_str, time_str, lon, lat, sog, cog, itpl, img_dt = row
            vessel_name, vessel_type, dims = _lookup_ship_static(cur, int(mmsi))
            if 0 in dims:
                ship_m = (100.0, 20.0)
            else:
                dim_a, dim_b, dim_c, dim_d = dims
                ship_m = (float(dim_a + dim_b), float(dim_c + dim_d))

            polygon = _ship_bbox_geometry(float(lon), float(lat), float(cog), ship_m)
            features.append(
                {
                    "type": "Feature",
                    "properties": {
                        "MMSI": int(mmsi),
                        "VesselName": vessel_name,
                        "VesselType": vessel_type,
                        "Date": date_str,
                        "Time": time_str,
                        "IMG_DT": img_dt,
                        "Lon": float(lon),
                        "Lat": float(lat),
                        "SOG": float(sog),
                        "COG": float(cog),
                        "ITPL": int(itpl),
                        "ship_m": [float(ship_m[0]), float(ship_m[1])],
                    },
                    "geometry": polygon.__geo_interface__,
                }
            )

        conn.close()
        payload = {
            "type": "FeatureCollection",
            "properties": {"bbox_count": len(features), "generated_by": "sub_module.dnb_pipeline_core"},
            "features": features,
        }
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

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


class DruidClusterStore:
    """Container for PH candidate components used by density target construction."""

    def __init__(self, scene: SceneRaster, catalogue: pd.DataFrame, clusters: list[DruidCluster]) -> None:
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
