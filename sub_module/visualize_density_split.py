from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.transform import rowcol
from rasterio.warp import transform as warp_transform
from shapely.geometry import shape

from .dnb_project_paths import DENSITY_OUTPUT_ROOT


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create quick visual checks for a density scene split.")
    parser.add_argument("--scene-split-csv", type=Path, default=DENSITY_OUTPUT_ROOT / "splits" / "density_smoke_split_10_3_2" / "scene_split.csv")
    parser.add_argument("--output-dir", type=Path, default=DENSITY_OUTPUT_ROOT / "splits" / "density_smoke_split_10_3_2" / "visuals")
    parser.add_argument("--max-thumb-size", type=int, default=900)
    parser.add_argument("--point-size", type=float, default=1.5)
    parser.add_argument("--point-alpha", type=float, default=0.55)
    parser.add_argument("--contact-sheet-cols", type=int, default=4)
    parser.add_argument("--contact-sheet-dpi", type=int, default=150)
    parser.add_argument("--max-scenes-per-split", type=int, default=0)
    return parser


def read_scene_split(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def robust_norm(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    lo = float(np.nanpercentile(finite, 1))
    hi = float(np.nanpercentile(finite, 99.7))
    return np.clip((arr - lo) / max(hi - lo, 1.0e-6), 0.0, 1.0)


def read_thumbnail(src: rasterio.io.DatasetReader, max_size: int) -> tuple[np.ndarray, float, float]:
    scale = min(float(max_size) / max(int(src.height), int(src.width)), 1.0)
    out_h = max(int(round(int(src.height) * scale)), 1)
    out_w = max(int(round(int(src.width) * scale)), 1)
    data = src.read(1, out_shape=(out_h, out_w), masked=True)
    image = np.asarray(data.filled(0), dtype=np.float32)
    mask = np.ma.getmaskarray(data)
    image[mask] = 0.0
    return robust_norm(image), float(out_h) / float(src.height), float(out_w) / float(src.width)


def load_gt_lonlat(geojson_path: Path) -> tuple[np.ndarray, np.ndarray]:
    obj = json.loads(geojson_path.read_text(encoding="utf-8"))
    lons: list[float] = []
    lats: list[float] = []
    for feature in obj.get("features", []):
        props = feature.get("properties", {})
        if "Lon" in props and "Lat" in props:
            lon = float(props["Lon"])
            lat = float(props["Lat"])
        else:
            centroid = shape(feature["geometry"]).centroid
            lon = float(centroid.x)
            lat = float(centroid.y)
        lons.append(lon)
        lats.append(lat)
    return np.asarray(lons, dtype=np.float64), np.asarray(lats, dtype=np.float64)


def lonlat_to_thumbnail_rc(
    src: rasterio.io.DatasetReader,
    lons: np.ndarray,
    lats: np.ndarray,
    *,
    row_scale: float,
    col_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    xs = lons
    ys = lats
    if src.crs is not None and src.crs.to_epsg() != 4326:
        xs_list, ys_list = warp_transform("EPSG:4326", src.crs, lons.tolist(), lats.tolist())
        xs = np.asarray(xs_list, dtype=np.float64)
        ys = np.asarray(ys_list, dtype=np.float64)

    rows, cols = rowcol(src.transform, xs, ys)
    rows_arr = np.asarray(rows, dtype=np.float64)
    cols_arr = np.asarray(cols, dtype=np.float64)
    valid = (rows_arr >= 0) & (rows_arr < src.height) & (cols_arr >= 0) & (cols_arr < src.width)
    return rows_arr[valid] * float(row_scale), cols_arr[valid] * float(col_scale)


def save_scene_quicklook(record: dict[str, str], output_dir: Path, max_thumb_size: int, point_size: float, point_alpha: float) -> dict[str, Any]:
    split = record["split"]
    scene_key = record["scene_key"]
    tif_path = Path(record["tif_path"])
    geojson_path = Path(record["geojson_path"])
    split_dir = output_dir / "scenes" / split
    split_dir.mkdir(parents=True, exist_ok=True)
    out_path = split_dir / f"{scene_key}_quicklook.png"

    with rasterio.open(tif_path) as src:
        image, row_scale, col_scale = read_thumbnail(src, max_thumb_size)
        lons, lats = load_gt_lonlat(geojson_path)
        gt_rows, gt_cols = lonlat_to_thumbnail_rc(src, lons, lats, row_scale=row_scale, col_scale=col_scale)

        fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
        ax.imshow(image, cmap="magma")
        if gt_rows.size:
            ax.scatter(gt_cols, gt_rows, s=float(point_size), c="#00e5ff", alpha=float(point_alpha), linewidths=0)
        ax.set_title(f"{split} | {scene_key} | GT all={len(lons)} | GT in scene={gt_rows.size}")
        ax.axis("off")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

    return {
        "split": split,
        "scene_key": scene_key,
        "quicklook_path": str(out_path),
        "gt_feature_count": int(len(lons)),
        "gt_points_in_scene": int(gt_rows.size),
    }


def save_contact_sheet(records: list[dict[str, Any]], split: str, output_dir: Path, cols: int, dpi: int) -> str | None:
    split_records = [record for record in records if record["split"] == split]
    if not split_records:
        return None
    cols = max(int(cols), 1)
    rows = int(math.ceil(len(split_records) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), constrained_layout=True)
    axes_arr = np.asarray(axes).reshape(-1)
    for ax in axes_arr:
        ax.axis("off")
    for ax, record in zip(axes_arr, split_records):
        image = plt.imread(record["quicklook_path"])
        ax.imshow(image)
        ax.set_title(record["scene_key"], fontsize=9)
        ax.axis("off")
    path = output_dir / f"contact_sheet_{split}.png"
    fig.savefig(path, dpi=int(dpi))
    plt.close(fig)
    return str(path)


def save_day_timeline(records: list[dict[str, str]], output_dir: Path) -> str:
    split_colors = {"train": "#2ca25f", "val": "#3182bd", "test": "#de2d26"}
    by_day: dict[str, dict[str, int]] = {}
    for record in records:
        day = record["day_key"]
        split = record["split"]
        by_day.setdefault(day, {"train": 0, "val": 0, "test": 0})
        by_day[day][split] += 1
    days = sorted(by_day)
    fig, ax = plt.subplots(figsize=(max(12, len(days) * 0.25), 4), constrained_layout=True)
    labels_seen: set[str] = set()
    for idx, day in enumerate(days):
        bottom = 0
        for split in ("train", "val", "test"):
            value = by_day[day].get(split, 0)
            if value:
                label = split if split not in labels_seen else None
                labels_seen.add(split)
                ax.bar(idx, value, bottom=bottom, color=split_colors[split], label=label)
                bottom += value
    ax.set_xticks(range(len(days)))
    ax.set_xticklabels(days, rotation=90, fontsize=7)
    ax.set_ylabel("scene count")
    ax.set_title("Smoke split by day")
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc="upper right")
    path = output_dir / "day_split_timeline.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)


def main() -> int:
    args = build_arg_parser().parse_args()
    scene_split_path = args.scene_split_csv.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records = read_scene_split(scene_split_path)
    if int(args.max_scenes_per_split) > 0:
        limited: list[dict[str, str]] = []
        for split in ("train", "val", "test"):
            limited.extend([record for record in records if record["split"] == split][: int(args.max_scenes_per_split)])
        records = limited

    quicklooks = [
        save_scene_quicklook(
            record,
            output_dir,
            int(args.max_thumb_size),
            float(args.point_size),
            float(args.point_alpha),
        )
        for record in records
    ]
    contact_sheets = {
        split: save_contact_sheet(quicklooks, split, output_dir, int(args.contact_sheet_cols), int(args.contact_sheet_dpi))
        for split in ("train", "val", "test")
    }
    timeline_path = save_day_timeline(records, output_dir)
    summary = {
        "scene_split_csv": str(scene_split_path),
        "quicklook_count": int(len(quicklooks)),
        "quicklooks": quicklooks,
        "contact_sheets": contact_sheets,
        "timeline_path": timeline_path,
    }
    summary_path = output_dir / "visual_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary_path": str(summary_path), "quicklook_count": len(quicklooks), "timeline_path": timeline_path}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
