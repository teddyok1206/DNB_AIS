from __future__ import annotations

import json
import math
import os
import re
import shutil
import sqlite3
import sys
import warnings
from collections import deque
from dataclasses import asdict, dataclass, field, replace
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn.functional as F
from astropy.io import fits
from matplotlib import cm
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from rasterio.transform import rowcol
from scipy.ndimage import label
from shapely.geometry import Polygon, shape
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn.pool import radius_graph
from torch_geometric.utils import to_undirected


SCENE_KEY_PATTERN = re.compile(r"(A\d{7}_\d{4}_\d{3})")
YEAR = 2025


def extract_scene_key(path_like: str | Path) -> str:
    match = SCENE_KEY_PATTERN.search(Path(path_like).name)
    if match is None:
        raise ValueError(f"Could not extract scene key from: {path_like}")
    return match.group(1)


def resolve_device(preferred: str = "mps") -> torch.device:
    preferred = preferred.lower()
    if preferred == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_druid_on_path(druid_root: str | Path) -> None:
    druid_root = str(Path(druid_root).resolve())
    if druid_root not in sys.path:
        sys.path.insert(0, druid_root)


def load_druid_sf(druid_root: str | Path):
    druid_root = Path(druid_root).resolve()
    source_dir = druid_root / "DRUID"
    main_path = source_dir / "main.py"
    src_dir = source_dir / "src"
    if not main_path.exists():
        raise FileNotFoundError(f"DRUID main.py not found: {main_path}")
    if not src_dir.exists():
        raise FileNotFoundError(f"DRUID src directory not found: {src_dir}")

    runtime_root = Path("/tmp/codex_druid_runtime")
    runtime_pkg = runtime_root / "codex_druid_runtime"
    runtime_src = runtime_pkg / "src"
    runtime_root.mkdir(parents=True, exist_ok=True)
    runtime_pkg.mkdir(parents=True, exist_ok=True)

    shutil.copy2(main_path, runtime_pkg / "main.py")
    shutil.copytree(
        src_dir,
        runtime_src,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"),
    )
    (runtime_pkg / "__init__.py").write_text("from .main import sf\n", encoding="utf-8")
    (runtime_src / "__init__.py").write_text("", encoding="utf-8")

    if str(runtime_root) not in sys.path:
        sys.path.insert(0, str(runtime_root))

    package_name = "codex_druid_runtime"
    stale_modules = [name for name in sys.modules if name == package_name or name.startswith(f"{package_name}.")]
    for module_name in stale_modules:
        del sys.modules[module_name]

    from codex_druid_runtime import sf

    return sf


def druid_debug(message: str) -> None:
    if os.environ.get("CODEX_DRUID_DEBUG", "0") == "1":
        print(f"[DRUID_DEBUG] {message}", flush=True)


def build_fits_header(scene: "SceneRaster") -> fits.Header:
    header = fits.Header()
    header["INSTRUME"] = "DNB_TIF"
    header["NAXIS1"] = scene.width
    header["NAXIS2"] = scene.height
    header["PSF_FWHM"] = 3.5
    header["EFFGAIN"] = 1.0
    header["EXPTIME"] = 1.0
    header["EFFRON"] = 5.0
    header["CD1_1"] = scene.transform.a
    header["CD1_2"] = scene.transform.b
    header["CD2_1"] = scene.transform.d
    header["CD2_2"] = scene.transform.e
    return header


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
    meters_per_deg_lon = max(meters_per_deg_lat * math.cos(math.radians(lat)), 1e-9)

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
class DruidConfig:
    detection_threshold: float = 1.0
    analysis_threshold: float = 1.0
    background_mode: str = "SEX"
    lifetime_limit_fraction: float = 1.001
    area_limit: int = 4
    cutup: bool = False
    cutup_size: int = 512
    cutup_buffer: int = 64
    smooth_sigma: float = 0.0
    max_catalogue_clusters: int | None = None
    min_nodes: int = 32
    max_nodes: int = 4096
    remove_edge: bool = True


@dataclass
class GraphConfig:
    radius_pixels: float = 2.0
    normalize_coordinates: bool = True
    make_undirected: bool = True
    gt_smoothing_hop_weights: tuple[float, ...] | None = (1.0, 0.6, 0.2)
    gt_smoothing_preserve_mass: bool = True


@dataclass
class TrainingConfig:
    hidden_channels: int = 48
    heads: int = 4
    num_layers: int = 3
    dropout: float = 0.1
    output_activation: str = "softplus"
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-4
    epochs: int = 4
    batch_size: int = 4
    loss_name: str = "poisson_nll"
    positive_weight: float = 0.0
    count_weight_alpha: float = 0.0
    count_sum_lambda: float = 0.0
    target_scale: float = 1.0
    target_field: str = "y_edge_decay"


@dataclass
class DruidCluster:
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
            WHERE IMG_DT = ? AND ITPL > 0
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
            "properties": {"bbox_count": len(features), "generated_by": "sub_module.dnb_gat_pipeline"},
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


def _drop_nested_clusters(clusters: list[DruidCluster]) -> list[DruidCluster]:
    kept: list[DruidCluster] = []
    for cluster in sorted(clusters, key=lambda item: item.lifetime, reverse=True):
        nested = False
        for existing in kept:
            if (
                cluster.bbox_rc[0] >= existing.bbox_rc[0]
                and cluster.bbox_rc[1] <= existing.bbox_rc[1]
                and cluster.bbox_rc[2] >= existing.bbox_rc[2]
                and cluster.bbox_rc[3] <= existing.bbox_rc[3]
                and cluster.coords_set.issubset(existing.coords_set)
            ):
                nested = True
                break
        if not nested:
            kept.append(cluster)
    return kept


class DruidClusterStore:
    def __init__(self, scene: SceneRaster, catalogue: pd.DataFrame, clusters: list[DruidCluster]) -> None:
        self.scene = scene
        self.catalogue = catalogue.reset_index(drop=True)
        self.clusters = clusters

    @classmethod
    def from_scene(
        cls,
        scene: SceneRaster,
        gt_count_map: np.ndarray,
        druid_root: str | Path,
        config: DruidConfig,
    ) -> "DruidClusterStore":
        druid_debug("running DRUID catalogue")
        catalogue = cls._run_druid(scene, druid_root, config)
        druid_debug(f"catalogue rows={len(catalogue)}")
        druid_debug("building clusters")
        clusters = cls._build_clusters(scene, gt_count_map, catalogue, config)
        druid_debug(f"clusters built={len(clusters)}")
        druid_debug("dropping nested clusters")
        clusters = _drop_nested_clusters(clusters)
        druid_debug(f"clusters after nested drop={len(clusters)}")
        filtered_ids = {cluster.cluster_id for cluster in clusters}
        catalogue = catalogue[catalogue["ID"].isin(filtered_ids)].copy().reset_index(drop=True)
        return cls(scene=scene, catalogue=catalogue, clusters=clusters)

    @staticmethod
    def _run_druid(scene: SceneRaster, druid_root: str | Path, config: DruidConfig) -> pd.DataFrame:
        ensure_druid_on_path(druid_root)
        druid_debug("loading sf")
        sf = load_druid_sf(druid_root)
        druid_debug("sf loaded")

        warnings.filterwarnings("ignore")
        druid_debug("initializing finder")
        finder = sf(
            image=scene.image,
            header=build_fits_header(scene),
            mode="optical",
            cutup=config.cutup,
            cutup_size=config.cutup_size,
            cutup_buff=config.cutup_buffer,
            smooth_sigma=config.smooth_sigma,
            output=False,
            GPU=False,
            area_limit=config.area_limit,
            remove_edge=config.remove_edge,
        )
        druid_debug("finder initialized")
        druid_debug("setting background")
        finder.set_background(
            detection_threshold=config.detection_threshold,
            analysis_threshold=config.analysis_threshold,
            mode=config.background_mode,
        )
        druid_debug("background set")
        druid_debug("running phsf")
        finder.phsf(lifetime_limit=0, lifetime_limit_fraction=config.lifetime_limit_fraction)
        druid_debug("phsf complete")

        base_catalogue = finder.catalogue.copy()
        if "lifetime" not in base_catalogue.columns:
            raise RuntimeError("DRUID catalogue does not contain lifetime values.")

        base_catalogue = base_catalogue.sort_values("lifetime", ascending=False).reset_index(drop=True)
        if config.max_catalogue_clusters is not None:
            base_catalogue = base_catalogue.head(config.max_catalogue_clusters).copy()

        druid_debug("attaching contours")
        contour_catalogue = DruidClusterStore._attach_contours(
            np.asarray(finder.image_smooth, dtype=np.float32),
            base_catalogue,
        )
        druid_debug("contours attached")
        return contour_catalogue.sort_values("lifetime", ascending=False).reset_index(drop=True)

    @staticmethod
    def _attach_contours(membership_image: np.ndarray, catalogue: pd.DataFrame) -> pd.DataFrame:
        contour_rows: list[dict[str, Any]] = []
        for row in catalogue.itertuples(index=False):
            contour = DruidClusterStore._contour_from_catalogue_row(membership_image, row)
            if contour is None or contour.shape[0] < 3:
                continue
            row_dict = dict(row._asdict())
            row_dict["contour"] = contour
            contour_rows.append(row_dict)
        return pd.DataFrame(contour_rows)

    @staticmethod
    def _contour_from_catalogue_row(membership_image: np.ndarray, row: Any) -> np.ndarray | None:
        bbox1 = int(max(row.bbox1 - 1, 0))
        bbox2 = int(max(row.bbox2 - 1, 0))
        bbox3 = int(min(row.bbox3 + 1, membership_image.shape[0] - 1))
        bbox4 = int(min(row.bbox4 + 1, membership_image.shape[1] - 1))
        cropped = membership_image[bbox1 : bbox3 + 1, bbox2 : bbox4 + 1]
        if cropped.size == 0:
            return None

        seed_row = int(row.x1) - bbox1
        seed_col = int(row.y1) - bbox2
        if not (0 <= seed_row < cropped.shape[0] and 0 <= seed_col < cropped.shape[1]):
            return None

        level_mask = np.logical_and(cropped <= float(row.Birth), cropped > float(row.Death))
        labeled, _ = label(level_mask.astype(np.uint8))
        label_at_seed = int(labeled[seed_row, seed_col])
        if label_at_seed == 0:
            return None

        component_mask = (labeled == label_at_seed).astype(np.uint8)
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None

        contour_xy = max(contours, key=cv2.contourArea).reshape(-1, 2)
        contour_rc = np.column_stack(
            [
                contour_xy[:, 1] + bbox1,
                contour_xy[:, 0] + bbox2,
            ]
        ).astype(np.int32)
        return contour_rc

    @staticmethod
    def _build_clusters(
        scene: SceneRaster,
        gt_count_map: np.ndarray,
        catalogue: pd.DataFrame,
        config: DruidConfig,
    ) -> list[DruidCluster]:
        clusters: list[DruidCluster] = []
        total_rows = len(catalogue)
        for idx, row in enumerate(catalogue.itertuples(index=False), start=1):
            if idx == 1 or idx % 50 == 0 or idx == total_rows:
                druid_debug(f"cluster build progress {idx}/{total_rows}")
            contour = np.asarray(row.contour, dtype=np.int32)
            if contour.ndim != 2 or contour.shape[0] < 3:
                continue

            row_coords = contour[:, 0]
            col_coords = contour[:, 1]
            rmin = int(np.clip(row_coords.min(), 0, scene.height - 1))
            rmax = int(np.clip(row_coords.max(), 0, scene.height - 1))
            cmin = int(np.clip(col_coords.min(), 0, scene.width - 1))
            cmax = int(np.clip(col_coords.max(), 0, scene.width - 1))
            if rmax < rmin or cmax < cmin:
                continue

            patch = scene.image[rmin : rmax + 1, cmin : cmax + 1]
            gt_patch = gt_count_map[rmin : rmax + 1, cmin : cmax + 1]

            local_contour = contour - np.array([rmin, cmin], dtype=np.int32)
            contour_xy = local_contour[:, [1, 0]].reshape(-1, 1, 2).astype(np.int32)
            mask = np.zeros(patch.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [contour_xy], 1)

            local_rc = np.argwhere(mask == 1)
            if local_rc.shape[0] < config.min_nodes or local_rc.shape[0] > config.max_nodes:
                continue

            global_rc = local_rc + np.array([rmin, cmin], dtype=np.int32)
            clusters.append(
                DruidCluster(
                    cluster_id=int(row.ID),
                    lifetime=float(row.lifetime),
                    birth=float(row.Birth),
                    death=float(row.Death),
                    contour_rc=contour,
                    bbox_rc=(rmin, rmax, cmin, cmax),
                    seed_rc=(int(row.x1), int(row.y1)),
                    patch_image=(patch * mask).astype(np.float32),
                    patch_gt=(gt_patch * mask).astype(np.float32),
                    mask=mask,
                    local_rc=local_rc.astype(np.int32),
                    global_rc=global_rc.astype(np.int32),
                    coords_set={tuple(coord) for coord in global_rc.tolist()},
                )
            )
        return clusters

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


def area_limit_sweep(
    scene: SceneRaster,
    gt_count_map: np.ndarray,
    druid_root: str | Path,
    area_limits: Iterable[int],
    base_config: DruidConfig,
    *,
    min_nodes_override: int = 1,
    max_nodes_override: int | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    gt_nonzero = int((gt_count_map > 0).sum())

    for area_limit in area_limits:
        config = replace(
            base_config,
            area_limit=int(area_limit),
            max_catalogue_clusters=None,
            min_nodes=int(min_nodes_override),
            max_nodes=int(max_nodes_override if max_nodes_override is not None else base_config.max_nodes),
        )
        catalogue = DruidClusterStore._run_druid(scene, druid_root, config)
        clusters = _drop_nested_clusters(DruidClusterStore._build_clusters(scene, gt_count_map, catalogue, config))

        covered = np.zeros(scene.shape, dtype=np.uint8)
        for cluster in clusters:
            covered[cluster.global_rc[:, 0], cluster.global_rc[:, 1]] = 1

        node_counts = np.array([cluster.node_count for cluster in clusters], dtype=np.int32)
        clusters_with_gt = int(sum(cluster.gt_sum > 0 for cluster in clusters))
        unique_gt_pixels_covered = int(((gt_count_map > 0) & (covered > 0)).sum())

        if node_counts.size == 0:
            node_stats = {
                "node_count_min": 0,
                "node_count_p25": 0,
                "node_count_median": 0,
                "node_count_p75": 0,
                "node_count_p90": 0,
                "node_count_max": 0,
                "clusters_lt8": 0,
                "clusters_lt16": 0,
                "clusters_lt32": 0,
            }
        else:
            node_stats = {
                "node_count_min": int(node_counts.min()),
                "node_count_p25": int(np.quantile(node_counts, 0.25)),
                "node_count_median": int(np.quantile(node_counts, 0.50)),
                "node_count_p75": int(np.quantile(node_counts, 0.75)),
                "node_count_p90": int(np.quantile(node_counts, 0.90)),
                "node_count_max": int(node_counts.max()),
                "clusters_lt8": int((node_counts < 8).sum()),
                "clusters_lt16": int((node_counts < 16).sum()),
                "clusters_lt32": int((node_counts < 32).sum()),
            }

        rows.append(
            {
                "area_limit": int(area_limit),
                "min_nodes_used": int(config.min_nodes),
                "druid_catalogue_count": int(len(catalogue)),
                "final_cluster_count": int(len(clusters)),
                "total_nodes": int(node_counts.sum()) if node_counts.size else 0,
                "clusters_with_gt": clusters_with_gt,
                "gt_pixels_covered": unique_gt_pixels_covered,
                "gt_pixel_coverage_ratio": float(unique_gt_pixels_covered / max(gt_nonzero, 1)),
                **node_stats,
            }
        )

    return pd.DataFrame(rows).sort_values("area_limit").reset_index(drop=True)


def graph_receptive_field_sweep(
    scene: SceneRaster,
    gt_count_map: np.ndarray,
    clusters: Iterable[DruidCluster],
    radius_values: Iterable[float],
    layer_values: Iterable[int],
    base_training_config: TrainingConfig,
    device: torch.device,
    *,
    base_graph_config: GraphConfig | None = None,
    seed: int = 1,
) -> pd.DataFrame:
    cluster_list = list(clusters)
    if not cluster_list:
        raise ValueError("No clusters supplied for graph receptive field sweep.")

    if base_graph_config is None:
        base_graph_config = GraphConfig()

    gt_scene_positive = int((gt_count_map > 0).sum())
    scene_positive_mask = gt_count_map > 0
    rows: list[dict[str, Any]] = []

    for radius in radius_values:
        graph_config = replace(base_graph_config, radius_pixels=float(radius))
        graphs = GraphBuilder(graph_config).build(cluster_list)
        total_nodes = int(sum(int(graph.num_nodes) for graph in graphs))
        total_edges = int(sum(int(graph.edge_index.shape[1]) for graph in graphs))
        all_y = torch.cat([graph.y for graph in graphs]).cpu().numpy()
        positive_mask = all_y > 0
        positive_count = int(positive_mask.sum())

        for num_layers in layer_values:
            np.random.seed(seed)
            torch.manual_seed(seed)

            training_config = replace(base_training_config, num_layers=int(num_layers))
            model = GATDensityRegressor(
                in_channels=3,
                hidden_channels=training_config.hidden_channels,
                heads=training_config.heads,
                num_layers=training_config.num_layers,
                dropout=training_config.dropout,
                output_activation=training_config.output_activation,
            )

            history = train_gat(model, graphs, device, training_config)
            predictions = predict_graphs(model, graphs, device)
            all_pred = np.concatenate(
                [
                    predictions[int(graph.cluster_id.detach().cpu().numpy().reshape(-1)[0])]
                    for graph in graphs
                ]
            )

            gt_node_pred_mean = float(all_pred[positive_mask].mean()) if positive_count else 0.0
            bg_node_pred_mean = float(all_pred[~positive_mask].mean()) if (~positive_mask).any() else 0.0
            gt_bg_ratio = float(gt_node_pred_mean / max(bg_node_pred_mean, 1.0e-6))

            graph_topk_hits = 0
            if positive_count > 0:
                top_idx = np.argpartition(all_pred, -positive_count)[-positive_count:]
                graph_topk_hits = int((all_y[top_idx] > 0).sum())

            assembler = SceneAssembler(scene)
            for cluster in cluster_list:
                assembler.accumulate(cluster, predictions[cluster.cluster_id])
            heatmap = assembler.finalize()

            scene_topk_hits = 0
            if gt_scene_positive > 0 and np.any(heatmap > 0):
                flat_heatmap = heatmap.reshape(-1)
                top_scene_idx = np.argpartition(flat_heatmap, -gt_scene_positive)[-gt_scene_positive:]
                scene_topk_hits = int(scene_positive_mask.reshape(-1)[top_scene_idx].sum())

            heatmap_gt_mean = float(heatmap[scene_positive_mask].mean()) if gt_scene_positive else 0.0
            heatmap_bg_mean = float(heatmap[~scene_positive_mask].mean()) if (~scene_positive_mask).any() else 0.0

            rows.append(
                {
                    "radius_pixels": float(radius),
                    "num_layers": int(num_layers),
                    "graph_count": int(len(graphs)),
                    "total_nodes": total_nodes,
                    "total_edges": total_edges,
                    "mean_edges_per_node": float(total_edges / max(total_nodes, 1)),
                    "positive_graph_nodes": positive_count,
                    "loss_name": training_config.loss_name,
                    "final_train_loss": float(history["train_loss"].iloc[-1]),
                    "graph_topk_hits": graph_topk_hits,
                    "graph_topk_hit_rate": float(graph_topk_hits / max(positive_count, 1)),
                    "gt_node_pred_mean": gt_node_pred_mean,
                    "bg_node_pred_mean": bg_node_pred_mean,
                    "gt_bg_ratio": gt_bg_ratio,
                    "pred_sum_graph_nodes": float(all_pred.sum()),
                    "gt_sum_graph_nodes": float(all_y.sum()),
                    "pred_sum_ratio": float(all_pred.sum() / max(float(all_y.sum()), 1.0e-6)),
                    "scene_topk_hits": scene_topk_hits,
                    "scene_topk_hit_rate": float(scene_topk_hits / max(gt_scene_positive, 1)),
                    "heatmap_nonzero_pixels": int((heatmap > 0).sum()),
                    "heatmap_max": float(heatmap.max()),
                    "heatmap_mean": float(heatmap.mean()),
                    "heatmap_sum": float(heatmap.sum()),
                    "heatmap_gt_mean": heatmap_gt_mean,
                    "heatmap_bg_mean": heatmap_bg_mean,
                }
            )

    return (
        pd.DataFrame(rows)
        .sort_values(["scene_topk_hit_rate", "graph_topk_hit_rate", "gt_bg_ratio"], ascending=False)
        .reset_index(drop=True)
    )


def loss_weighting_sweep(
    scene: SceneRaster,
    gt_count_map: np.ndarray,
    clusters: Iterable[DruidCluster],
    graph_config: GraphConfig,
    base_training_config: TrainingConfig,
    device: torch.device,
    *,
    seed: int = 1,
) -> pd.DataFrame:
    cluster_list = list(clusters)
    if not cluster_list:
        raise ValueError("No clusters supplied for loss weighting sweep.")

    graphs = GraphBuilder(graph_config).build(cluster_list)
    all_y = torch.cat([graph.y for graph in graphs]).cpu().numpy()
    positive_mask = all_y > 0
    positive_count = int(positive_mask.sum())
    gt_scene_positive = int((gt_count_map > 0).sum())
    scene_positive_mask = gt_count_map > 0

    variants = [
        (
            "baseline_pos12",
            replace(
                base_training_config,
                positive_weight=12.0,
                count_weight_alpha=0.0,
                count_sum_lambda=0.0,
                target_scale=1.0,
            ),
        ),
        (
            "count_weight_yx20",
            replace(
                base_training_config,
                positive_weight=12.0,
                count_weight_alpha=20.0,
                count_sum_lambda=0.0,
                target_scale=1.0,
            ),
        ),
        (
            "count_weight_yx20_sum0.01",
            replace(
                base_training_config,
                positive_weight=12.0,
                count_weight_alpha=20.0,
                count_sum_lambda=0.01,
                target_scale=1.0,
            ),
        ),
        (
            "target_scale_x6_ref",
            replace(
                base_training_config,
                positive_weight=12.0,
                count_weight_alpha=0.0,
                count_sum_lambda=0.0,
                target_scale=6.0,
            ),
        ),
    ]

    rows: list[dict[str, Any]] = []
    for variant_name, training_config in variants:
        np.random.seed(seed)
        torch.manual_seed(seed)

        model = GATDensityRegressor(
            in_channels=3,
            hidden_channels=training_config.hidden_channels,
            heads=training_config.heads,
            num_layers=training_config.num_layers,
            dropout=training_config.dropout,
            output_activation=training_config.output_activation,
        )
        history = train_gat(model, graphs, device, training_config)
        predictions = predict_graphs(model, graphs, device)
        all_pred = np.concatenate(
            [
                predictions[int(graph.cluster_id.detach().cpu().numpy().reshape(-1)[0])]
                for graph in graphs
            ]
        )

        gt_node_pred_mean = float(all_pred[positive_mask].mean()) if positive_count else 0.0
        bg_node_pred_mean = float(all_pred[~positive_mask].mean()) if (~positive_mask).any() else 0.0
        gt_bg_ratio = float(gt_node_pred_mean / max(bg_node_pred_mean, 1.0e-6))

        graph_topk_hits = 0
        if positive_count > 0:
            top_idx = np.argpartition(all_pred, -positive_count)[-positive_count:]
            graph_topk_hits = int((all_y[top_idx] > 0).sum())

        assembler = SceneAssembler(scene)
        for cluster in cluster_list:
            assembler.accumulate(cluster, predictions[cluster.cluster_id])
        heatmap = assembler.finalize()

        scene_topk_hits = 0
        if gt_scene_positive > 0 and np.any(heatmap > 0):
            flat_heatmap = heatmap.reshape(-1)
            top_scene_idx = np.argpartition(flat_heatmap, -gt_scene_positive)[-gt_scene_positive:]
            scene_topk_hits = int(scene_positive_mask.reshape(-1)[top_scene_idx].sum())

        heatmap_gt_mean = float(heatmap[scene_positive_mask].mean()) if gt_scene_positive else 0.0
        heatmap_bg_mean = float(heatmap[~scene_positive_mask].mean()) if (~scene_positive_mask).any() else 0.0

        rows.append(
            {
                "variant": variant_name,
                "positive_weight": float(training_config.positive_weight),
                "count_weight_alpha": float(training_config.count_weight_alpha),
                "count_sum_lambda": float(training_config.count_sum_lambda),
                "target_scale": float(training_config.target_scale),
                "loss_name": training_config.loss_name,
                "graph_count": int(len(graphs)),
                "total_nodes": int(sum(int(graph.num_nodes) for graph in graphs)),
                "positive_graph_nodes": positive_count,
                "final_train_loss": float(history["train_loss"].iloc[-1]),
                "graph_topk_hits": graph_topk_hits,
                "graph_topk_hit_rate": float(graph_topk_hits / max(positive_count, 1)),
                "scene_topk_hits": scene_topk_hits,
                "scene_topk_hit_rate": float(scene_topk_hits / max(gt_scene_positive, 1)),
                "gt_node_pred_mean": gt_node_pred_mean,
                "bg_node_pred_mean": bg_node_pred_mean,
                "gt_bg_ratio": gt_bg_ratio,
                "pred_sum_graph_nodes": float(all_pred.sum()),
                "gt_sum_graph_nodes": float(all_y.sum()),
                "pred_sum_ratio": float(all_pred.sum() / max(float(all_y.sum()), 1.0e-6)),
                "heatmap_nonzero_pixels": int((heatmap > 0).sum()),
                "heatmap_max": float(heatmap.max()),
                "heatmap_mean": float(heatmap.mean()),
                "heatmap_sum": float(heatmap.sum()),
                "heatmap_gt_mean": heatmap_gt_mean,
                "heatmap_bg_mean": heatmap_bg_mean,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["scene_topk_hit_rate", "graph_topk_hit_rate", "gt_bg_ratio"], ascending=False)
        .reset_index(drop=True)
    )


def positive_weight_sweep(
    scene: SceneRaster,
    gt_count_map: np.ndarray,
    clusters: Iterable[DruidCluster],
    graph_config: GraphConfig,
    base_training_config: TrainingConfig,
    device: torch.device,
    *,
    positive_weights: Iterable[float],
    seed: int = 1,
) -> pd.DataFrame:
    cluster_list = list(clusters)
    if not cluster_list:
        raise ValueError("No clusters supplied for positive_weight sweep.")

    graphs = GraphBuilder(graph_config).build(cluster_list)
    all_y = torch.cat([graph.y for graph in graphs]).cpu().numpy()
    positive_mask = all_y > 0
    positive_count = int(positive_mask.sum())
    gt_scene_positive = int((gt_count_map > 0).sum())
    scene_positive_mask = gt_count_map > 0

    rows: list[dict[str, Any]] = []
    for positive_weight in positive_weights:
        np.random.seed(seed)
        torch.manual_seed(seed)

        training_config = replace(base_training_config, positive_weight=float(positive_weight))
        model = GATDensityRegressor(
            in_channels=3,
            hidden_channels=training_config.hidden_channels,
            heads=training_config.heads,
            num_layers=training_config.num_layers,
            dropout=training_config.dropout,
            output_activation=training_config.output_activation,
        )
        history = train_gat(model, graphs, device, training_config)
        predictions = predict_graphs(model, graphs, device)
        all_pred = np.concatenate(
            [
                predictions[int(graph.cluster_id.detach().cpu().numpy().reshape(-1)[0])]
                for graph in graphs
            ]
        )

        gt_node_pred_mean = float(all_pred[positive_mask].mean()) if positive_count else 0.0
        bg_node_pred_mean = float(all_pred[~positive_mask].mean()) if (~positive_mask).any() else 0.0
        gt_bg_ratio = float(gt_node_pred_mean / max(bg_node_pred_mean, 1.0e-6))

        graph_topk_hits = 0
        if positive_count > 0:
            top_idx = np.argpartition(all_pred, -positive_count)[-positive_count:]
            graph_topk_hits = int((all_y[top_idx] > 0).sum())

        assembler = SceneAssembler(scene)
        for cluster in cluster_list:
            assembler.accumulate(cluster, predictions[cluster.cluster_id])
        heatmap = assembler.finalize()

        scene_topk_hits = 0
        if gt_scene_positive > 0 and np.any(heatmap > 0):
            flat_heatmap = heatmap.reshape(-1)
            top_scene_idx = np.argpartition(flat_heatmap, -gt_scene_positive)[-gt_scene_positive:]
            scene_topk_hits = int(scene_positive_mask.reshape(-1)[top_scene_idx].sum())

        heatmap_gt_mean = float(heatmap[scene_positive_mask].mean()) if gt_scene_positive else 0.0
        heatmap_bg_mean = float(heatmap[~scene_positive_mask].mean()) if (~scene_positive_mask).any() else 0.0

        rows.append(
            {
                "positive_weight": float(positive_weight),
                "count_weight_alpha": float(training_config.count_weight_alpha),
                "loss_name": training_config.loss_name,
                "output_activation": training_config.output_activation,
                "graph_count": int(len(graphs)),
                "total_nodes": int(sum(int(graph.num_nodes) for graph in graphs)),
                "positive_graph_nodes": positive_count,
                "final_train_loss": float(history["train_loss"].iloc[-1]),
                "graph_topk_hits": graph_topk_hits,
                "graph_topk_hit_rate": float(graph_topk_hits / max(positive_count, 1)),
                "scene_topk_hits": scene_topk_hits,
                "scene_topk_hit_rate": float(scene_topk_hits / max(gt_scene_positive, 1)),
                "gt_node_pred_mean": gt_node_pred_mean,
                "bg_node_pred_mean": bg_node_pred_mean,
                "gt_bg_ratio": gt_bg_ratio,
                "pred_sum_graph_nodes": float(all_pred.sum()),
                "gt_sum_graph_nodes": float(all_y.sum()),
                "pred_sum_ratio": float(all_pred.sum() / max(float(all_y.sum()), 1.0e-6)),
                "heatmap_nonzero_pixels": int((heatmap > 0).sum()),
                "heatmap_max": float(heatmap.max()),
                "heatmap_mean": float(heatmap.mean()),
                "heatmap_sum": float(heatmap.sum()),
                "heatmap_gt_mean": heatmap_gt_mean,
                "heatmap_bg_mean": heatmap_bg_mean,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["scene_topk_hit_rate", "graph_topk_hit_rate", "gt_bg_ratio"], ascending=False)
        .reset_index(drop=True)
    )


def weighting_grid_sweep(
    scene: SceneRaster,
    gt_count_map: np.ndarray,
    clusters: Iterable[DruidCluster],
    graph_config: GraphConfig,
    base_training_config: TrainingConfig,
    device: torch.device,
    *,
    positive_weights: Iterable[float],
    count_weight_alphas: Iterable[float],
    seed: int = 1,
) -> pd.DataFrame:
    cluster_list = list(clusters)
    if not cluster_list:
        raise ValueError("No clusters supplied for weighting grid sweep.")

    graphs = GraphBuilder(graph_config).build(cluster_list)
    all_y = torch.cat([graph.y for graph in graphs]).cpu().numpy()
    positive_mask = all_y > 0
    positive_count = int(positive_mask.sum())
    gt_scene_positive = int((gt_count_map > 0).sum())
    scene_positive_mask = gt_count_map > 0

    rows: list[dict[str, Any]] = []
    for count_weight_alpha in count_weight_alphas:
        for positive_weight in positive_weights:
            np.random.seed(seed)
            torch.manual_seed(seed)

            training_config = replace(
                base_training_config,
                positive_weight=float(positive_weight),
                count_weight_alpha=float(count_weight_alpha),
            )
            model = GATDensityRegressor(
                in_channels=3,
                hidden_channels=training_config.hidden_channels,
                heads=training_config.heads,
                num_layers=training_config.num_layers,
                dropout=training_config.dropout,
                output_activation=training_config.output_activation,
            )
            history = train_gat(model, graphs, device, training_config)
            predictions = predict_graphs(model, graphs, device)
            all_pred = np.concatenate(
                [
                    predictions[int(graph.cluster_id.detach().cpu().numpy().reshape(-1)[0])]
                    for graph in graphs
                ]
            )

            gt_node_pred_mean = float(all_pred[positive_mask].mean()) if positive_count else 0.0
            bg_node_pred_mean = float(all_pred[~positive_mask].mean()) if (~positive_mask).any() else 0.0
            gt_bg_ratio = float(gt_node_pred_mean / max(bg_node_pred_mean, 1.0e-6))

            graph_topk_hits = 0
            if positive_count > 0:
                top_idx = np.argpartition(all_pred, -positive_count)[-positive_count:]
                graph_topk_hits = int((all_y[top_idx] > 0).sum())

            assembler = SceneAssembler(scene)
            for cluster in cluster_list:
                assembler.accumulate(cluster, predictions[cluster.cluster_id])
            heatmap = assembler.finalize()

            scene_topk_hits = 0
            if gt_scene_positive > 0 and np.any(heatmap > 0):
                flat_heatmap = heatmap.reshape(-1)
                top_scene_idx = np.argpartition(flat_heatmap, -gt_scene_positive)[-gt_scene_positive:]
                scene_topk_hits = int(scene_positive_mask.reshape(-1)[top_scene_idx].sum())

            heatmap_gt_mean = float(heatmap[scene_positive_mask].mean()) if gt_scene_positive else 0.0
            heatmap_bg_mean = float(heatmap[~scene_positive_mask].mean()) if (~scene_positive_mask).any() else 0.0

            rows.append(
                {
                    "positive_weight": float(positive_weight),
                    "count_weight_alpha": float(count_weight_alpha),
                    "loss_name": training_config.loss_name,
                    "output_activation": training_config.output_activation,
                    "graph_count": int(len(graphs)),
                    "total_nodes": int(sum(int(graph.num_nodes) for graph in graphs)),
                    "positive_graph_nodes": positive_count,
                    "final_train_loss": float(history["train_loss"].iloc[-1]),
                    "graph_topk_hits": graph_topk_hits,
                    "graph_topk_hit_rate": float(graph_topk_hits / max(positive_count, 1)),
                    "scene_topk_hits": scene_topk_hits,
                    "scene_topk_hit_rate": float(scene_topk_hits / max(gt_scene_positive, 1)),
                    "gt_node_pred_mean": gt_node_pred_mean,
                    "bg_node_pred_mean": bg_node_pred_mean,
                    "gt_bg_ratio": gt_bg_ratio,
                    "pred_sum_graph_nodes": float(all_pred.sum()),
                    "gt_sum_graph_nodes": float(all_y.sum()),
                    "pred_sum_ratio": float(all_pred.sum() / max(float(all_y.sum()), 1.0e-6)),
                    "heatmap_nonzero_pixels": int((heatmap > 0).sum()),
                    "heatmap_max": float(heatmap.max()),
                    "heatmap_mean": float(heatmap.mean()),
                    "heatmap_sum": float(heatmap.sum()),
                    "heatmap_gt_mean": heatmap_gt_mean,
                    "heatmap_bg_mean": heatmap_bg_mean,
                }
            )

    return (
        pd.DataFrame(rows)
        .sort_values(["scene_topk_hit_rate", "graph_topk_hit_rate", "gt_bg_ratio"], ascending=False)
        .reset_index(drop=True)
    )


def choose_representative_cluster(clusters: Iterable[DruidCluster]) -> DruidCluster:
    cluster_list = list(clusters)
    if not cluster_list:
        raise ValueError("No clusters supplied for graph visualization.")
    return max(cluster_list, key=lambda cluster: (cluster.gt_sum, cluster.lifetime, cluster.node_count))


def choose_overfit_cluster(clusters: Iterable[DruidCluster]) -> DruidCluster:
    cluster_list = [cluster for cluster in clusters if cluster.gt_sum > 0]
    if not cluster_list:
        raise ValueError("No positive-GT clusters supplied for overfit troubleshooting.")

    single_ship_candidates = [cluster for cluster in cluster_list if cluster.gt_sum <= 1.0 + 1.0e-6]
    if single_ship_candidates:
        return max(single_ship_candidates, key=lambda cluster: (cluster.lifetime, cluster.node_count))

    min_gt_sum = min(cluster.gt_sum for cluster in cluster_list)
    min_sum_candidates = [cluster for cluster in cluster_list if abs(cluster.gt_sum - min_gt_sum) < 1.0e-6]
    return max(min_sum_candidates, key=lambda cluster: (cluster.lifetime, cluster.node_count))


def visualize_graph_cluster(
    cluster: DruidCluster,
    graph: Data,
    *,
    pred_values: np.ndarray | None = None,
    max_edges_to_draw: int = 6000,
) -> plt.Figure:
    pos = graph.pos.detach().cpu().numpy()
    edge_index = graph.edge_index.detach().cpu().numpy()
    brightness = graph.x[:, 0].detach().cpu().numpy()
    gt_values = graph.y.detach().cpu().numpy()
    gt_edge_decay = graph.y_edge_decay.detach().cpu().numpy() if hasattr(graph, "y_edge_decay") else None

    segments = np.stack([pos[edge_index[0]], pos[edge_index[1]]], axis=1) if edge_index.size else np.empty((0, 2, 2))
    if segments.shape[0] > max_edges_to_draw:
        stride = max(int(math.ceil(segments.shape[0] / max_edges_to_draw)), 1)
        segments = segments[::stride]

    num_panels = 5 if gt_edge_decay is not None else 4
    fig, axes = plt.subplots(1, num_panels, figsize=(5 * num_panels, 5))

    axes[0].imshow(cluster.patch_image, cmap="cividis")
    axes[0].contour(cluster.mask.astype(float), levels=[0.5], colors="cyan", linewidths=1.0)
    positive_local = cluster.local_rc[gt_values > 0]
    if positive_local.size:
        axes[0].scatter(
            positive_local[:, 1],
            positive_local[:, 0],
            c=gt_values[gt_values > 0],
            cmap="inferno",
            s=48,
            edgecolors="white",
            linewidths=0.5,
        )
    axes[0].set_title(
        f"Cluster {cluster.cluster_id} patch\nGT sum={cluster.gt_sum:.1f}, nodes={cluster.node_count}"
    )
    axes[0].set_axis_off()

    axes[1].imshow(cluster.mask.astype(float), cmap="bone")
    if segments.size:
        axes[1].add_collection(LineCollection(segments, colors=(0.15, 0.45, 0.95, 0.12), linewidths=0.35))
    axes[1].scatter(pos[:, 0], pos[:, 1], c=brightness, cmap="viridis", s=14)
    axes[1].set_title("Graph nodes and edges")
    axes[1].set_xlim(-0.5, cluster.bbox_width - 0.5)
    axes[1].set_ylim(cluster.bbox_height - 0.5, -0.5)
    axes[1].set_aspect("equal")

    gt_scatter = axes[2].scatter(
        pos[:, 0],
        pos[:, 1],
        c=gt_values,
        cmap="inferno",
        s=16,
        vmin=0.0,
        vmax=max(float(gt_values.max()), 1.0),
    )
    axes[2].set_title("Node GT ship counts")
    axes[2].set_xlim(-0.5, cluster.bbox_width - 0.5)
    axes[2].set_ylim(cluster.bbox_height - 0.5, -0.5)
    axes[2].set_aspect("equal")
    fig.colorbar(gt_scatter, ax=axes[2], fraction=0.046, pad=0.04)

    pred_axis_idx = 3
    if gt_edge_decay is not None:
        decay_scatter = axes[3].scatter(
            pos[:, 0],
            pos[:, 1],
            c=gt_edge_decay,
            cmap="plasma",
            s=16,
            vmin=0.0,
            vmax=max(float(np.max(gt_edge_decay)), 1.0e-6),
        )
        axes[3].set_title(f"Node GT edge-decay\nsum={float(np.sum(gt_edge_decay)):.2f}")
        axes[3].set_xlim(-0.5, cluster.bbox_width - 0.5)
        axes[3].set_ylim(cluster.bbox_height - 0.5, -0.5)
        axes[3].set_aspect("equal")
        fig.colorbar(decay_scatter, ax=axes[3], fraction=0.046, pad=0.04)
        pred_axis_idx = 4

    pred_panel = pred_values if pred_values is not None else brightness
    pred_label = "Node predictions" if pred_values is not None else "Node brightness"
    pred_scatter = axes[pred_axis_idx].scatter(
        pos[:, 0],
        pos[:, 1],
        c=pred_panel,
        cmap="magma",
        s=16,
        vmin=0.0,
        vmax=max(float(np.max(pred_panel)), 1.0e-6),
    )
    axes[pred_axis_idx].set_title(pred_label)
    axes[pred_axis_idx].set_xlim(-0.5, cluster.bbox_width - 0.5)
    axes[pred_axis_idx].set_ylim(cluster.bbox_height - 0.5, -0.5)
    axes[pred_axis_idx].set_aspect("equal")
    fig.colorbar(pred_scatter, ax=axes[pred_axis_idx], fraction=0.046, pad=0.04)

    for ax in axes[1:]:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    return fig


def edge_decay_smoothed_target(
    raw_target: np.ndarray,
    edge_index: torch.Tensor,
    num_nodes: int,
    hop_weights: tuple[float, ...],
    *,
    preserve_mass: bool = True,
) -> torch.Tensor:
    if len(hop_weights) == 0:
        raise ValueError("hop_weights must contain at least one value.")
    if any(float(weight) < 0.0 for weight in hop_weights):
        raise ValueError("hop_weights must be non-negative.")

    smoothed = np.zeros(int(num_nodes), dtype=np.float32)
    positive_nodes = np.flatnonzero(raw_target > 0)
    if positive_nodes.size == 0:
        return torch.from_numpy(smoothed)

    adjacency: list[set[int]] = [set() for _ in range(int(num_nodes))]
    edges_np = edge_index.detach().cpu().numpy()
    for src, dst in edges_np.T:
        adjacency[int(src)].add(int(dst))

    max_hop = len(hop_weights) - 1
    for source in positive_nodes.tolist():
        base_value = float(raw_target[source])
        distances: dict[int, int] = {int(source): 0}
        queue: deque[int] = deque([int(source)])

        while queue:
            node = queue.popleft()
            current_hop = distances[node]
            if current_hop >= max_hop:
                continue
            for neighbor in adjacency[node]:
                if neighbor in distances:
                    continue
                next_hop = current_hop + 1
                distances[neighbor] = next_hop
                queue.append(neighbor)

        raw_kernel = {int(node): float(hop_weights[hop]) for node, hop in distances.items()}
        if preserve_mass:
            kernel_total = float(sum(raw_kernel.values()))
            if kernel_total <= 0.0:
                raw_kernel = {int(source): 1.0}
                kernel_total = 1.0
            scale = base_value / kernel_total
            for node, kernel_value in raw_kernel.items():
                smoothed[node] += np.float32(kernel_value * scale)
        else:
            for node, kernel_value in raw_kernel.items():
                smoothed[node] += np.float32(base_value * kernel_value)

    return torch.from_numpy(smoothed)


class GraphBuilder:
    def __init__(self, config: GraphConfig) -> None:
        self.config = config

    def cluster_to_data(self, cluster: DruidCluster) -> Data:
        local_rc = cluster.local_rc.astype(np.float32)
        brightness = cluster.patch_image[cluster.local_rc[:, 0], cluster.local_rc[:, 1]].astype(np.float32)
        gt_values = cluster.patch_gt[cluster.local_rc[:, 0], cluster.local_rc[:, 1]].astype(np.float32)

        x_coords = local_rc[:, 1].copy()
        y_coords = local_rc[:, 0].copy()
        pos = np.column_stack([x_coords, y_coords]).astype(np.float32)
        if self.config.normalize_coordinates:
            width_scale = max(cluster.bbox_width - 1, 1)
            height_scale = max(cluster.bbox_height - 1, 1)
            x_coords = x_coords / float(width_scale)
            y_coords = y_coords / float(height_scale)

        features = np.column_stack([brightness, x_coords, y_coords]).astype(np.float32)
        pos_tensor = torch.from_numpy(pos)
        edge_index = radius_graph(pos_tensor, r=self.config.radius_pixels, loop=False)
        if self.config.make_undirected:
            edge_index = to_undirected(edge_index, num_nodes=int(pos_tensor.shape[0]))

        y_tensor = torch.from_numpy(gt_values)

        data = Data(
            x=torch.from_numpy(features),
            edge_index=edge_index,
            y=y_tensor,
            pos=pos_tensor,
        )
        data.y_point = y_tensor.clone()
        if self.config.gt_smoothing_hop_weights is not None:
            data.y_edge_decay = edge_decay_smoothed_target(
                raw_target=gt_values,
                edge_index=edge_index,
                num_nodes=int(pos_tensor.shape[0]),
                hop_weights=self.config.gt_smoothing_hop_weights,
                preserve_mass=bool(self.config.gt_smoothing_preserve_mass),
            )
        data.cluster_id = torch.tensor([cluster.cluster_id], dtype=torch.long)
        data.global_rc = torch.from_numpy(cluster.global_rc.astype(np.int64))
        data.cluster_weight = torch.tensor([cluster.weight], dtype=torch.float32)
        return data

    def build(self, clusters: Iterable[DruidCluster]) -> list[Data]:
        return [self.cluster_to_data(cluster) for cluster in clusters]


class GATDensityRegressor(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 48,
        heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
        output_activation: str = "softplus",
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden_channels = int(hidden_channels)
        self.heads = int(heads)
        self.num_layers = int(num_layers)
        self.dropout = dropout
        self.output_activation = str(output_activation).lower()
        self.convs = nn.ModuleList()
        self.convs.append(GATv2Conv(in_channels, hidden_channels, heads=heads, concat=False, dropout=dropout))
        for _ in range(max(num_layers - 2, 0)):
            self.convs.append(GATv2Conv(hidden_channels, hidden_channels, heads=heads, concat=False, dropout=dropout))
        self.convs.append(GATv2Conv(hidden_channels, hidden_channels, heads=heads, concat=False, dropout=dropout))
        self.head = nn.Linear(hidden_channels, 1)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.constant_(self.head.bias, 0.1)

    def architecture_dict(self) -> dict[str, Any]:
        return {
            "in_channels": self.in_channels,
            "hidden_channels": self.hidden_channels,
            "heads": self.heads,
            "num_layers": self.num_layers,
            "dropout": float(self.dropout),
            "output_activation": self.output_activation,
        }

    def forward(self, data: Data) -> torch.Tensor:
        x = data.x
        edge_index = data.edge_index
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.head(x).squeeze(-1)
        if self.output_activation == "relu":
            return F.relu(x)
        if self.output_activation == "softplus":
            return F.softplus(x)
        raise ValueError(f"Unsupported output activation: {self.output_activation}")


def count_model_parameters(model: nn.Module, *, trainable_only: bool = False) -> int:
    if trainable_only:
        return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))
    return int(sum(parameter.numel() for parameter in model.parameters()))


def save_model_checkpoint(
    output_path: str | Path,
    model: nn.Module,
    graph_config: GraphConfig,
    training_config: TrainingConfig,
    *,
    scene: SceneRaster | None = None,
    metadata: dict[str, Any] | None = None,
    save_summary_json: bool = True,
) -> dict[str, Any]:
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    architecture = model.architecture_dict() if hasattr(model, "architecture_dict") else {}
    parameter_count = count_model_parameters(model)
    trainable_parameter_count = count_model_parameters(model, trainable_only=True)
    bundle = {
        "model_class": type(model).__name__,
        "architecture": architecture,
        "graph_config": asdict(graph_config),
        "training_config": asdict(training_config),
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "metadata": metadata or {},
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
    }
    if scene is not None:
        bundle["scene"] = {
            "scene_key": scene.key,
            "scene_path": str(scene.path),
            "shape": [int(scene.height), int(scene.width)],
            "crs": str(scene.crs),
        }

    torch.save(bundle, output_path)
    file_size_bytes = int(output_path.stat().st_size)

    summary = {
        "checkpoint_path": str(output_path),
        "summary_json_path": "",
        "model_class": bundle["model_class"],
        "architecture": architecture,
        "graph_config": bundle["graph_config"],
        "training_config": bundle["training_config"],
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "file_size_bytes": file_size_bytes,
        "file_size_mb": float(file_size_bytes / (1024 ** 2)),
        "saved_at": bundle["saved_at"],
        "metadata": bundle["metadata"],
    }

    if save_summary_json:
        summary_path = output_path.with_suffix(".json")
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        summary["summary_json_path"] = str(summary_path)

    return summary


def load_model_checkpoint(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint_path = Path(checkpoint_path).resolve()
    bundle = torch.load(checkpoint_path, map_location=map_location)
    if bundle.get("model_class") != "GATDensityRegressor":
        raise ValueError(f"Unsupported model class in checkpoint: {bundle.get('model_class')}")

    architecture = bundle.get("architecture", {})
    if "output_activation" not in architecture:
        architecture["output_activation"] = "relu"
    model = GATDensityRegressor(**architecture)
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    if isinstance(map_location, torch.device):
        model.to(map_location)
    elif str(map_location) not in {"cpu", "meta"}:
        model.to(torch.device(str(map_location)))
    return model, bundle


def train_gat(
    model: nn.Module,
    graphs: list[Data],
    device: torch.device,
    config: TrainingConfig,
) -> pd.DataFrame:
    if not graphs:
        raise ValueError("No graphs supplied for training.")

    loader = DataLoader(graphs, batch_size=config.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    history = []
    loss_name = str(config.loss_name).lower()
    target_field = str(config.target_field)

    model.to(device)
    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_nodes = 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch)
            if not hasattr(batch, target_field):
                raise ValueError(f"Batch does not contain target field: {target_field}")
            target_tensor = getattr(batch, target_field)
            target = target_tensor * float(config.target_scale)
            weights = torch.ones_like(target)
            if config.positive_weight != 0.0:
                weights = weights + (target > 0).float() * config.positive_weight
            if config.count_weight_alpha != 0.0:
                weights = weights + target * float(config.count_weight_alpha)

            if loss_name == "mse":
                element_loss = (pred - target) ** 2
            elif loss_name == "poisson_nll":
                element_loss = F.poisson_nll_loss(
                    pred,
                    target,
                    log_input=False,
                    full=False,
                    eps=1.0e-8,
                    reduction="none",
                )
            else:
                raise ValueError(f"Unsupported loss_name: {config.loss_name}")

            loss = (element_loss * weights).mean()

            if config.count_sum_lambda > 0.0:
                batch_index = batch.batch
                num_graphs = int(batch_index.max().item()) + 1 if batch_index.numel() else 0
                pred_sums = torch.zeros(num_graphs, device=device, dtype=pred.dtype)
                target_sums = torch.zeros(num_graphs, device=device, dtype=pred.dtype)
                pred_sums.index_add_(0, batch_index, pred)
                target_sums.index_add_(0, batch_index, target)
                loss = loss + float(config.count_sum_lambda) * ((pred_sums - target_sums) ** 2).mean()
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.detach().cpu()) * int(batch.num_nodes)
            epoch_nodes += int(batch.num_nodes)

        history.append(
            {
                "epoch": epoch,
                "loss_name": loss_name,
                "train_loss": epoch_loss / max(epoch_nodes, 1),
                "num_graphs": len(graphs),
                "num_nodes": epoch_nodes,
                "target_field": target_field,
            }
        )

    return pd.DataFrame(history)


def predict_graphs(model: nn.Module, graphs: list[Data], device: torch.device) -> dict[int, np.ndarray]:
    predictions: dict[int, np.ndarray] = {}
    model.eval()
    model.to(device)
    with torch.no_grad():
        for graph in graphs:
            graph = graph.to(device)
            pred = model(graph).detach().cpu().numpy().astype(np.float32)
            cluster_id = int(graph.cluster_id.detach().cpu().numpy().reshape(-1)[0])
            predictions[cluster_id] = pred
    return predictions


def single_graph_overfit_sweep(
    graph: Data,
    configs: Iterable[tuple[str, TrainingConfig]],
    device: torch.device,
    *,
    seed: int = 1,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, pd.DataFrame]]:
    rows: list[dict[str, Any]] = []
    predictions: dict[str, np.ndarray] = {}
    histories: dict[str, pd.DataFrame] = {}

    raw_target = graph.y_point.detach().cpu().numpy() if hasattr(graph, "y_point") else graph.y.detach().cpu().numpy()
    raw_positive_mask = raw_target > 0

    for name, training_config in configs:
        np.random.seed(seed)
        torch.manual_seed(seed)

        model = GATDensityRegressor(
            in_channels=int(graph.x.shape[1]),
            hidden_channels=training_config.hidden_channels,
            heads=training_config.heads,
            num_layers=training_config.num_layers,
            dropout=training_config.dropout,
            output_activation=training_config.output_activation,
        )
        history = train_gat(model, [graph.clone()], device, training_config)
        model.eval()
        with torch.no_grad():
            pred = model(graph.clone().to(device)).detach().cpu().numpy().astype(np.float32)

        target_tensor = getattr(graph, training_config.target_field)
        selected_target = target_tensor.detach().cpu().numpy()
        selected_positive_mask = selected_target > 0

        rows.append(
            {
                "experiment": name,
                "target_field": training_config.target_field,
                "loss_name": training_config.loss_name,
                "positive_weight": float(training_config.positive_weight),
                "count_weight_alpha": float(training_config.count_weight_alpha),
                "epochs": int(training_config.epochs),
                "lr": float(training_config.lr),
                "weight_decay": float(training_config.weight_decay),
                "dropout": float(training_config.dropout),
                "raw_positive_count": int(raw_positive_mask.sum()),
                "selected_positive_count": int(selected_positive_mask.sum()),
                "selected_target_max": float(selected_target.max()) if selected_target.size else 0.0,
                "pred_max": float(pred.max()) if pred.size else 0.0,
                "pred_sum": float(pred.sum()),
                "pred_on_raw_positive_mean": float(pred[raw_positive_mask].mean()) if raw_positive_mask.any() else 0.0,
                "pred_on_raw_positive_max": float(pred[raw_positive_mask].max()) if raw_positive_mask.any() else 0.0,
                "pred_on_selected_positive_mean": float(pred[selected_positive_mask].mean()) if selected_positive_mask.any() else 0.0,
                "pred_on_selected_positive_max": float(pred[selected_positive_mask].max()) if selected_positive_mask.any() else 0.0,
                "final_train_loss": float(history["train_loss"].iloc[-1]),
                "reached_peak_0_9": bool(float(pred.max()) >= 0.9) if pred.size else False,
            }
        )
        predictions[name] = pred
        histories[name] = history

    result = pd.DataFrame(rows).sort_values(["pred_max", "pred_on_raw_positive_max"], ascending=False).reset_index(drop=True)
    return result, predictions, histories


class SceneAssembler:
    def __init__(self, scene: SceneRaster) -> None:
        self.scene = scene
        self.weighted_sum = np.zeros(scene.shape, dtype=np.float32)
        self.weight_sum = np.zeros(scene.shape, dtype=np.float32)

    def accumulate(self, cluster: DruidCluster, pred_values: np.ndarray) -> None:
        rows = cluster.global_rc[:, 0]
        cols = cluster.global_rc[:, 1]
        weight = np.float32(cluster.weight)
        self.weighted_sum[rows, cols] += pred_values.astype(np.float32) * weight
        self.weight_sum[rows, cols] += weight

    def finalize(self) -> np.ndarray:
        return np.divide(
            self.weighted_sum,
            self.weight_sum,
            out=np.zeros_like(self.weighted_sum),
            where=self.weight_sum > 0,
        )

    def save_geotiff(self, output_path: str | Path, array: np.ndarray) -> Path:
        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=self.scene.height,
            width=self.scene.width,
            count=1,
            dtype="float32",
            crs=self.scene.crs,
            transform=self.scene.transform,
        ) as dst:
            dst.write(array.astype(np.float32), 1)
        return output_path


def make_overlay_rgb(scene_image: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    gray = np.clip(scene_image.astype(np.float32), 0.0, 1.0)
    gray_rgb = np.dstack([gray, gray, gray])

    if np.any(heatmap > 0):
        vmax = float(np.quantile(heatmap[heatmap > 0], 0.98))
        vmax = max(vmax, float(heatmap.max()))
    else:
        vmax = 1.0

    heat_norm = Normalize(vmin=0.0, vmax=max(vmax, 1.0e-6))
    heat_rgb = cm.get_cmap("magma")(heat_norm(heatmap))[..., :3]
    alpha = (heatmap > 0).astype(np.float32)[..., None] * 0.55
    overlay = gray_rgb * (1.0 - alpha) + heat_rgb.astype(np.float32) * alpha
    return np.clip(overlay, 0.0, 1.0)
