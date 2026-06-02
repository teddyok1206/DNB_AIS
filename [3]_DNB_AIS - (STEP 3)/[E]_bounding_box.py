from __future__ import annotations

import os
import sys
import argparse

import json
import math
import pandas as pd

from pyproj import Geod
from dataclasses import dataclass
from typing import Union, Optional
from datetime import date, time, datetime, timedelta

import sqlite3

# typedef
NumLike = Union[str, float, int]

def parse_bool_flag(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}

# Global Variables
YEAR = 2025
ROOT = "./bboxes_JPSS-2/"
DB_PATH = os.path.join("/Users/jungtaeuk/ships/ships.db")
TXT_PATH = os.path.join(ROOT, "Bboxes_completed.txt")
ITPL_TXT_PATH = "/Users/jungtaeuk/ships/ITPL_completed_JPSS-2.txt"
METADATA_PATH = "./metadata_JPSS-2.csv"
TIF_THRES = -1 # 몇 개의 영상 처리할 건지 (-1 : 제한 해제)
MMSI_THRES = -1 # 하나의 영상에 대하여 몇 개의 선박 처리할 건지 (-1 : 제한 해제)
BBOX_COEF = 1 # Bounding Box를 실제 선박 크기 대비 몇 배 키울 것인지
WGS84_GEOD = Geod(ellps="WGS84")
REBUILD_BBOXES = os.environ.get("REBUILD_BBOXES", "0") == "1"

# Helper Functions
def mapped_date()->dict: # 위성 영상의 날짜 (001 ~ 365) -> datetime 객체로 변환
    start_date = date(YEAR, 1, 1)
    doy_to_date = {doy: (start_date + timedelta(days=doy - 1)).isoformat() for doy in range(1, 366)}
    return doy_to_date

def str_to_datetime(date_str : str, time_str : str)->datetime:
    return datetime.combine(date.fromisoformat(date_str), time.fromisoformat(time_str))

def parse_key(fname :str)->tuple[int, int]:
    base = fname.replace(".tif", "")
    date, hhmm, _ = base.split("_")
    return (int(date[1:]), int(hhmm))

def mark_Bboxes_completed(tif_name: str, path: str):
    needs_separator = os.path.exists(path) and os.path.getsize(path) > 0
    if needs_separator:
        with open(path, "rb") as f:
            f.seek(-1, os.SEEK_END)
            needs_separator = f.read(1) != b"\n"

    with open(path, "a", encoding="utf-8") as f:
        if needs_separator:
            f.write("\n")
        f.write(tif_name + "\n")

def get_marked_TIF_set(path: str) -> set[str]:
    if not os.path.exists(path):
        return set()

    with open(path, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())

def get_TIF_todo_list(df)->list[str]:
    done_tifs = set() if REBUILD_BBOXES else get_marked_TIF_set(TXT_PATH)
    tif_name_list = list(
        (set(df["tif_name"].dropna().tolist()) & get_marked_TIF_set(ITPL_TXT_PATH))
        - done_tifs
    )
    tif_name_list.sort(key = parse_key)
    return tif_name_list

def get_cols_from_DB(cur : sqlite3.Connection.cursor)->list:
    cols = [
                row[1] for row in cur.execute("PRAGMA table_info(ships_dynamic)")
            ]

    if not cols:
        raise RuntimeError("No columns left after excluding Date/Time. Check schema.")
    return cols

def bearing_from_EN(east_m: float, north_m: float) -> float:
    return (math.degrees(math.atan2(east_m, north_m)) + 360.0) % 360.0

def get_ship_static(cur, MMSI: int)->tuple[str, str, tuple[int, int, int, int]]:
    cur.execute("""
        SELECT VesselName, VesselType, DimA, DimB, DimC, DimD
        FROM ships_static
        WHERE MMSI = ?;
    """, (int(MMSI),))

    row = cur.fetchone()
    if row is None:
        return ("", "", (0, 0, 0, 0))

    VesselName, VesselType, DimA, DimB, DimC, DimD = row
    return (
        str(VesselName) if VesselName is not None else "",
        str(VesselType) if VesselType is not None else "",
        (
        int(DimA) if DimA is not None else 0,
        int(DimB) if DimB is not None else 0,
        int(DimC) if DimC is not None else 0,
        int(DimD) if DimD is not None else 0,
        )
    )

def normalize_static(
    VesselName,
    VesselType,
    DimA,
    DimB,
    DimC,
    DimD,
) -> tuple[str, str, tuple[int, int, int, int]]:
    return (
        str(VesselName) if VesselName is not None else "",
        str(VesselType) if VesselType is not None else "",
        (
            int(DimA) if DimA is not None else 0,
            int(DimB) if DimB is not None else 0,
            int(DimC) if DimC is not None else 0,
            int(DimD) if DimD is not None else 0,
        )
    )

def _to_float(x: NumLike, name: str) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a float-like value (got {x!r}).")

def _wrap_lon(lon: float) -> float:
    """Wrap longitude to [-180, 180)."""
    return ((lon + 180.0) % 360.0) - 180.0

def _clamp_lat(lat: float) -> float:
    """Clamp latitude to [-90, 90]."""
    return max(-90.0, min(90.0, lat))

def filter_by_tif_name(tif_name : str)->bool:
    _, HHMM, _ = tif_name.split("_")
    return ((int(HHMM) < 1550) or (int(HHMM) > 1810))

def get_tif_center_dt(mapped_date_dict : dict, df : pd.DataFrame, tif_name : str)->tuple[str, str]: # tif 관측 평균 시각(시작 시각 + 3분) 전후 DELTA분 기간에 포함되는 선박들의 MMSI 집합
    row = df.loc[df["tif_name"] == tif_name, ["scan_start_HHMMSS", "duration [sec]"]].iloc[0]
    duration = timedelta(seconds=float(row["duration [sec]"]))
    parsed = tif_name.split("_")
    tif_date_str = mapped_date_dict[int(parsed[0][-3:])]
    tif_time_str = row["scan_start_HHMMSS"]
    tif_center_dt = str_to_datetime(tif_date_str, tif_time_str) + (duration / 2)
    tif_center_dt_date_str = tif_center_dt.date().isoformat()
    tif_center_dt_time_str = tif_center_dt.time().replace(microsecond=0).isoformat()
    tif_center_dt_str = tif_center_dt.replace(microsecond=0).strftime("%Y-%m-%d %H:%M:%S")

    return (tif_center_dt_date_str, tif_center_dt_time_str, tif_center_dt_str)

# Class Definition
@dataclass(frozen=True)
class BBoxDegrees:
    """
    EPSG:4326 bounding box size in degrees.
    half_width_deg  : half size in longitude degrees
    half_height_deg : half size in latitude degrees
    """
    half_width_deg: float = 0.01
    half_height_deg: float = 0.01

def make_bbox_feature(lon: NumLike, lat: NumLike, *, properties: Optional[dict] = None, label: Optional[str] = None) -> dict:
    lon_f = _wrap_lon(_to_float(lon, "lon"))
    lat_f = _clamp_lat(_to_float(lat, "lat"))

    props = dict(properties) if properties else {}

    if label is not None:
        props["label"] = str(label)

    # 수학 계산 시작
    ship_m = props["ship_m"]
    ship_length_m = float(ship_m[0])
    ship_width_m = float(ship_m[1])
    hl = ship_length_m / 2.0
    hw = ship_width_m / 2.0

    corners_fr = [
        (+hl, +hw),
        (+hl, -hw),
        (-hl, -hw),
        (-hl, +hw),
        (+hl, +hw),
    ]

    COG = float(props["COG"]) % 360.0
    theta = math.radians(COG)
    f_e = math.sin(theta)
    f_n = math.cos(theta)
    r_e = math.sin(theta + math.pi/2)
    r_n = math.cos(theta + math.pi/2)

    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = meters_per_deg_lat * math.cos(math.radians(lat))

    ring = []
    for df, dr in corners_fr:
        east_m = df * f_e + dr * r_e
        north_m = df * f_n + dr * r_n
        dlon = east_m  / meters_per_deg_lon
        dlat = north_m / meters_per_deg_lat
        ring.append([_wrap_lon(lon_f + dlon), _clamp_lat(lat_f + dlat)])

    return {
            "type": "Feature",
            "properties": props,
            "geometry": {"type": "Polygon", "coordinates": [ring]},
           }

def make_feature_collection(features: list[dict]) -> dict:
    return {"type": "FeatureCollection", "properties": {
            "bbox_count": len(features)
          },"features": features}

def save_geojson(path: str, geojson_obj: dict) -> None:
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(geojson_obj, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)

def prepare_bbox_rows(conn: sqlite3.Connection, tif_to_img_dt: dict[str, str]) -> None:
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS bbox_targets;")
    cur.execute("DROP TABLE IF EXISTS bbox_rows;")
    cur.execute("""
        CREATE TEMP TABLE bbox_targets (
            tif_name TEXT PRIMARY KEY,
            IMG_DT TEXT NOT NULL
        );
    """)
    cur.executemany(
        "INSERT INTO bbox_targets (tif_name, IMG_DT) VALUES (?, ?);",
        list(tif_to_img_dt.items())
    )
    cur.execute("CREATE INDEX bbox_targets_img_dt ON bbox_targets(IMG_DT);")

    print("[Preparing bbox rows from DB...]")
    cur.execute("""
        CREATE TEMP TABLE bbox_rows AS
        SELECT
            t.tif_name,
            d.id,
            d.MMSI,
            d.Date,
            d.Time,
            d.Lon,
            d.Lat,
            d.SOG,
            d.COG,
            d.ITPL,
            d.IMG_DT,
            s.VesselName,
            s.VesselType,
            s.DimA,
            s.DimB,
            s.DimC,
            s.DimD
        FROM ships_dynamic AS d NOT INDEXED
        JOIN bbox_targets AS t ON d.IMG_DT = t.IMG_DT
        LEFT JOIN ships_static AS s ON s.MMSI = d.MMSI;
    """)
    cur.execute("CREATE INDEX bbox_rows_tif_id ON bbox_rows(tif_name, id);")
    row_count = cur.execute("SELECT COUNT(*) FROM bbox_rows;").fetchone()[0]
    print(f"[Prepared] {row_count} DB rows for {len(tif_to_img_dt)} tif files")

def main(TIF_THRES : int, MMSI_THRES : int)->None:
    os.makedirs(ROOT, exist_ok=True)
    if REBUILD_BBOXES:
        open(TXT_PATH, "w", encoding="utf-8").close()
        print("[REBUILD_BBOXES=1] cleared Bboxes_completed.txt; existing geojson files will be overwritten.")

    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=30)
    conn.execute("PRAGMA busy_timeout=30000;")
    conn.execute("PRAGMA temp_store=FILE;")

    df = pd.read_csv(METADATA_PATH)
    tif_name_list = [tif_name for tif_name in get_TIF_todo_list(df) if not filter_by_tif_name(tif_name)]
    if (TIF_THRES == -1):
        TIF_THRES = len(tif_name_list)
    tif_name_list = tif_name_list[:TIF_THRES]
    print(f"[TODO] {len(tif_name_list)} bbox files")
    mapped_date_dict = mapped_date()
    tif_to_img_dt = {
        tif_name: get_tif_center_dt(mapped_date_dict, df, tif_name)[2]
        for tif_name in tif_name_list
    }

    if tif_to_img_dt:
        prepare_bbox_rows(conn, tif_to_img_dt)

    for i, tif_name in enumerate(tif_name_list):
        try:
            print("="*80+f"\n[START] processing {tif_name}")
            tif_center_dt_date_str, tif_center_dt_time_str, tif_center_dt_str = get_tif_center_dt(mapped_date_dict, df, tif_name)

            cur = conn.cursor()
            sql = """
            SELECT
                id, MMSI, Date, Time, Lon, Lat, SOG, COG, ITPL, IMG_DT,
                VesselName, VesselType, DimA, DimB, DimC, DimD
            FROM bbox_rows
            WHERE tif_name = ?
            ORDER BY id;
            """

            cur.execute(sql, (tif_name,))
            rows = cur.fetchall()
            mmsi_thres = len(rows) if MMSI_THRES == -1 else MMSI_THRES

            targets = []
            for j, row in enumerate(rows):
                if (j >= mmsi_thres):
                    break
                (
                    row_id, MMSI, Date, Time, Lon, Lat, SOG, COG, ITPL, IMG_DT,
                    VesselName, VesselType, DimA, DimB, DimC, DimD
                ) = row
                VesselName, VesselType, Dim = normalize_static(VesselName, VesselType, DimA, DimB, DimC, DimD)
                if 0 in Dim:
                    # default value
                    ship_length_m = 100
                    ship_width_m = 20

                else:
                    DimA, DimB, DimC, DimD = Dim
                    ship_length_m = DimA + DimB
                    ship_width_m = DimC + DimD

                ship_m = (BBOX_COEF * ship_length_m, BBOX_COEF * ship_width_m)
                Label = f"* Time: {Time} * IMG_DT: {IMG_DT}\n\n* MMSI: {MMSI}\n* Name: {VesselName}\n* Type: {VesselType}\n\n* cord[EPSG 4326]: ({Lon:.3f}, {Lat:.3f})\n* SOG[knots]: {SOG:.3f}\n* COG[DEG]: {COG:.3f}\n* ITPL: {ITPL}"
                target = (MMSI, VesselName, VesselType, Date, Time, Lon, Lat, SOG, COG, Label, ITPL, IMG_DT, ship_m)
                targets.append(target)

            features: list[dict] = []

            for _, (MMSI, VesselName, VesselType, Date, Time, Lon, Lat, SOG, COG, Label, ITPL, IMG_DT, ship_m) in enumerate(targets, start=1):
                feat = make_bbox_feature(
                    Lon,
                    Lat,
                    properties={"MMSI": MMSI, "VesselName": VesselName, "VesselType": VesselType,
                                "Date": Date, "Time": Time, "IMG_DT": IMG_DT,
                                "SOG": SOG, "COG": COG, "ITPL": ITPL, "ship_m": ship_m},
                    label=Label,
                )
                features.append(feat)

            geojson = make_feature_collection(features)
            out_path = os.path.join(ROOT, f"{tif_name.split(".")[0]}.geojson")
            print(f"[Saving geojson at {out_path}]")
            save_geojson(out_path, geojson)
            print("[Saved]")
            mark_Bboxes_completed(tif_name, TXT_PATH)
            print(f"[COMPLETE] processing {tif_name}\n"+"="*80)
        except Exception as e:
            print(f"[Exception] || loc : main() || message : {e}")

    conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--REBUILD_BBOXES", default=None)
    args = parser.parse_args()
    if args.REBUILD_BBOXES is not None:
        REBUILD_BBOXES = parse_bool_flag(args.REBUILD_BBOXES)

    main(TIF_THRES, MMSI_THRES)
