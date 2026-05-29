from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.transform import array_bounds
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
from scipy.ndimage import label as ndi_label
from shapely.geometry import mapping
from shapely.ops import unary_union
from shapely.validation import make_valid

from .dnb_gat_pipeline import SceneRaster


@dataclass(frozen=True)
class KRSeaMaskResult:
    scene: SceneRaster
    valid_mask: np.ndarray
    metadata: dict[str, Any]


def _union_geometries(frame: gpd.GeoDataFrame):
    geometries = frame.geometry.dropna()
    if geometries.empty:
        raise ValueError("KR sea geometry file contains no valid geometries")
    geometries = geometries.apply(make_valid)
    union_all = getattr(geometries, "union_all", None)
    if callable(union_all):
        return union_all()
    return geometries.unary_union


def load_kr_sea_geometry(step3_dir: str | Path, target_crs: Any | None = None):
    """Load Korean EEZ plus 12nm sea-domain geometry.

    The source shapefiles are kept in EPSG:4326 in STEP 3. The returned geometry is
    reprojected to target_crs when a scene raster uses a different CRS.
    """

    step3 = Path(step3_dir).expanduser().resolve()
    sources = [
        step3 / "eez_12nm" / "eez_12nm.shp",
        step3 / "eez" / "eez.shp",
    ]
    frames: list[gpd.GeoDataFrame] = []
    for source in sources:
        if not source.exists():
            raise FileNotFoundError(f"KR sea shapefile not found: {source}")
        frame = gpd.read_file(source)
        if frame.crs is None:
            frame = frame.set_crs("EPSG:4326")
        else:
            frame = frame.to_crs("EPSG:4326")
        if target_crs is not None:
            frame = frame.to_crs(target_crs)
        frames.append(frame)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="invalid value encountered in unary_union",
            category=RuntimeWarning,
        )
        return make_valid(unary_union([_union_geometries(frame) for frame in frames]))


def _segment_stats(valid_mask: np.ndarray) -> tuple[int, list[int]]:
    labeled, segment_count = ndi_label(valid_mask.astype(np.uint8))
    if segment_count <= 0:
        return 0, []
    areas = np.bincount(labeled.ravel())[1:]
    return int(segment_count), [int(v) for v in np.sort(areas)[::-1][:5]]


def _largest_segment_crop(
    data: np.ndarray,
    valid_mask: np.ndarray,
    transform: Any,
) -> tuple[np.ndarray, np.ndarray, Any]:
    labeled, segment_count = ndi_label(valid_mask.astype(np.uint8))
    if segment_count <= 0:
        return data, valid_mask, transform
    areas = np.bincount(labeled.ravel())[1:]
    largest_id = int(np.argmax(areas)) + 1
    largest_mask = labeled == largest_id
    rows, cols = np.where(largest_mask)
    rmin, rmax = int(rows.min()), int(rows.max())
    cmin, cmax = int(cols.min()), int(cols.max())
    cropped_data = data[rmin : rmax + 1, cmin : cmax + 1].copy()
    cropped_valid = largest_mask[rmin : rmax + 1, cmin : cmax + 1]
    cropped_data[~cropped_valid] = 0
    cropped_transform = window_transform(
        Window(col_off=cmin, row_off=rmin, width=cmax - cmin + 1, height=rmax - rmin + 1),
        transform,
    )
    return cropped_data, cropped_valid, cropped_transform


def apply_kr_sea_mask(
    scene_tif: str | Path,
    *,
    step3_dir: str | Path,
    output_dir: str | Path | None = None,
    crop_to_bounds: bool = True,
    segment_policy: str = "single_scene",
    write_masked_tif: bool = False,
    all_touched: bool = False,
) -> KRSeaMaskResult:
    """Mask a DNB scene to Korea EEZ + 12nm sea-domain pixels.

    valid_mask is geometry-derived, not brightness-derived. That matters because a
    genuinely dark sea pixel is still a valid training/evaluation pixel.
    """

    if segment_policy not in {"single_scene", "largest_segment"}:
        raise ValueError("segment_policy must be one of {'single_scene', 'largest_segment'}")

    scene_path = Path(scene_tif).expanduser().resolve()
    with rasterio.open(scene_path) as src:
        kr_geometry = load_kr_sea_geometry(step3_dir, target_crs=src.crs)
        masked_bands, masked_transform = rio_mask(
            src,
            [mapping(kr_geometry)],
            crop=bool(crop_to_bounds),
            filled=False,
            all_touched=bool(all_touched),
        )
        masked = masked_bands[0]
        valid_mask = ~np.ma.getmaskarray(masked)
        data = np.asarray(masked.filled(0), dtype=np.float32)
        data[~valid_mask] = 0

        initial_segment_count, initial_top5 = _segment_stats(valid_mask)
        if segment_policy == "largest_segment":
            data, valid_mask, masked_transform = _largest_segment_crop(data, valid_mask, masked_transform)

        selected_segment_count, selected_top5 = _segment_stats(valid_mask)
        selected_valid_pixels = int(valid_mask.sum())
        selected_positive_pixels = int((data != 0).sum())
        suffix = "kr_eez_12nm_masked" if segment_policy == "single_scene" else "kr_eez_12nm_largest_segment"

        output_path: Path | None = None
        if bool(write_masked_tif):
            out_dir = Path(output_dir).expanduser().resolve() if output_dir is not None else scene_path.parent
            out_dir.mkdir(parents=True, exist_ok=True)
            output_path = out_dir / f"{scene_path.stem}_{suffix}.tif"
            write_meta = src.meta.copy()
            write_meta.update(
                {
                    "driver": "GTiff",
                    "height": int(data.shape[0]),
                    "width": int(data.shape[1]),
                    "transform": masked_transform,
                    "count": 1,
                    "nodata": 0,
                }
            )
            with rasterio.open(output_path, "w", **write_meta) as dst:
                dst.write(data, 1)

        height, width = int(data.shape[0]), int(data.shape[1])
        bounds = array_bounds(height, width, masked_transform)
        scene = SceneRaster(
            path=(output_path.resolve() if output_path is not None else scene_path),
            image=data.astype(np.float32, copy=False),
            transform=masked_transform,
            crs=src.crs,
            bounds=bounds,
            height=height,
            width=width,
        )

    metadata: dict[str, Any] = {
        "enabled": True,
        "source": "kr_eez_plus_12nm",
        "input_path": str(scene_path),
        "output_path": str(output_path.resolve()) if output_path is not None else None,
        "crop_to_bounds": bool(crop_to_bounds),
        "segment_policy": segment_policy,
        "all_touched": bool(all_touched),
        "initial_segment_count": int(initial_segment_count),
        "initial_segment_top5_areas": initial_top5,
        "selected_segment_count": int(selected_segment_count),
        "selected_segment_top5_areas": selected_top5,
        "selected_valid_pixels": int(selected_valid_pixels),
        "selected_positive_pixels": int(selected_positive_pixels),
        "height": int(height),
        "width": int(width),
    }
    return KRSeaMaskResult(scene=scene, valid_mask=valid_mask.astype(bool, copy=False), metadata=metadata)
