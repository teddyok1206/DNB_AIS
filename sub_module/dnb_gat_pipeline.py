from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import sys
import warnings
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn.functional as F
from astropy.io import fits
from matplotlib import cm
from matplotlib.colors import Normalize
from rasterio.transform import rowcol
from scipy.ndimage import label
from shapely.geometry import Polygon, shape
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn.pool import radius_graph


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
    max_catalogue_clusters: int | None = 32
    min_nodes: int = 32
    max_nodes: int = 4096
    remove_edge: bool = True


@dataclass
class GraphConfig:
    radius_pixels: float = 2.0
    normalize_coordinates: bool = True


@dataclass
class TrainingConfig:
    hidden_channels: int = 48
    heads: int = 4
    num_layers: int = 3
    dropout: float = 0.1
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-4
    epochs: int = 4
    batch_size: int = 4
    positive_weight: float = 8.0


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
        catalogue = cls._run_druid(scene, druid_root, config)
        clusters = cls._build_clusters(scene, gt_count_map, catalogue, config)
        clusters = _drop_nested_clusters(clusters)
        filtered_ids = {cluster.cluster_id for cluster in clusters}
        catalogue = catalogue[catalogue["ID"].isin(filtered_ids)].copy().reset_index(drop=True)
        return cls(scene=scene, catalogue=catalogue, clusters=clusters)

    @staticmethod
    def _run_druid(scene: SceneRaster, druid_root: str | Path, config: DruidConfig) -> pd.DataFrame:
        ensure_druid_on_path(druid_root)
        from DRUID import sf

        warnings.filterwarnings("ignore")
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
        finder.set_background(
            detection_threshold=config.detection_threshold,
            analysis_threshold=config.analysis_threshold,
            mode=config.background_mode,
        )
        finder.phsf(lifetime_limit=0, lifetime_limit_fraction=config.lifetime_limit_fraction)

        base_catalogue = finder.catalogue.copy()
        if "lifetime" not in base_catalogue.columns:
            raise RuntimeError("DRUID catalogue does not contain lifetime values.")

        base_catalogue = base_catalogue.sort_values("lifetime", ascending=False).reset_index(drop=True)
        if config.max_catalogue_clusters is not None:
            base_catalogue = base_catalogue.head(config.max_catalogue_clusters).copy()

        contour_catalogue = DruidClusterStore._attach_contours(scene.image, base_catalogue)
        return contour_catalogue.sort_values("lifetime", ascending=False).reset_index(drop=True)

    @staticmethod
    def _attach_contours(scene_image: np.ndarray, catalogue: pd.DataFrame) -> pd.DataFrame:
        contour_rows: list[dict[str, Any]] = []
        for row in catalogue.itertuples(index=False):
            contour = DruidClusterStore._contour_from_catalogue_row(scene_image, row)
            if contour is None or contour.shape[0] < 3:
                continue
            row_dict = dict(row._asdict())
            row_dict["contour"] = contour
            contour_rows.append(row_dict)
        return pd.DataFrame(contour_rows)

    @staticmethod
    def _contour_from_catalogue_row(scene_image: np.ndarray, row: Any) -> np.ndarray | None:
        bbox1 = int(max(row.bbox1 - 1, 0))
        bbox2 = int(max(row.bbox2 - 1, 0))
        bbox3 = int(min(row.bbox3 + 1, scene_image.shape[0] - 1))
        bbox4 = int(min(row.bbox4 + 1, scene_image.shape[1] - 1))
        cropped = scene_image[bbox1 : bbox3 + 1, bbox2 : bbox4 + 1]
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
        for row in catalogue.itertuples(index=False):
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

        data = Data(
            x=torch.from_numpy(features),
            edge_index=edge_index,
            y=torch.from_numpy(gt_values),
            pos=pos_tensor,
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
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(GATv2Conv(in_channels, hidden_channels, heads=heads, concat=False, dropout=dropout))
        for _ in range(max(num_layers - 2, 0)):
            self.convs.append(GATv2Conv(hidden_channels, hidden_channels, heads=heads, concat=False, dropout=dropout))
        self.convs.append(GATv2Conv(hidden_channels, hidden_channels, heads=heads, concat=False, dropout=dropout))
        self.head = nn.Linear(hidden_channels, 1)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.constant_(self.head.bias, 0.1)

    def forward(self, data: Data) -> torch.Tensor:
        x = data.x
        edge_index = data.edge_index
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.head(x).squeeze(-1)
        return F.relu(x)


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

    model.to(device)
    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_nodes = 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch)
            weights = torch.ones_like(batch.y) + (batch.y > 0).float() * config.positive_weight
            loss = (((pred - batch.y) ** 2) * weights).mean()
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.detach().cpu()) * int(batch.num_nodes)
            epoch_nodes += int(batch.num_nodes)

        history.append(
            {
                "epoch": epoch,
                "train_mse": epoch_loss / max(epoch_nodes, 1),
                "num_graphs": len(graphs),
                "num_nodes": epoch_nodes,
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
