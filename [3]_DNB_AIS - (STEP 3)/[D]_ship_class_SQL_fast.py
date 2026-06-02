from __future__ import annotations

import sys
import os

from dataclasses import dataclass
from collections import defaultdict

from netCDF4 import Dataset
import pandas as pd
import numpy as np
from scipy.interpolate import splrep, splev
from scipy.spatial import cKDTree
from pyproj import Transformer
from osgeo import gdal

from datetime import date, time, datetime, timedelta
from astropy.time import Time
from astropy.time import TimeDelta
import astropy.units as u

import sqlite3


# Global Variables
YEAR = 2025
TYPE = "JPSS-2" # "S-NPP" "JPSS-1" "JPSS-2"
ROOT_HDD = f"/Volumes/SAMSUNG/{TYPE}_VIIRS/"
DNB_dict = {"S-NPP" : ["VNP02DNB", "VNP03DNB"],
            "JPSS-1" : ["VJ102DNB", "VJ103DNB"],
            "JPSS-2" : ["VJ202DNB", "VJ203DNB"]}
DNB_02 = DNB_dict[TYPE][0]
DNB_03 = DNB_dict[TYPE][1]
ROOT = '/Users/jungtaeuk/ships/'
DB_PATH = os.path.join(ROOT, "ships.db")
TXT_PATH = ROOT + f"ITPL_completed_{TYPE}.txt"
METADATA_PATH = f"./metadata_{TYPE}.csv"
SNAPSHOT_PATH = os.path.join(ROOT, "ships.snapshot.db")
DELTA = timedelta(minutes=30) # 촬영 평균 시각 전후 +- 몇 분 검색하는지
TIF_THRES = -1 # 몇 개의 영상 처리할 건지 (-1 : 제한 해제)
MMSI_THRES = -1 # 하나의 영상에 대하여 몇 개의 선박 처리할 건지 (-1 : 제한 해제)
ITPL_DEPTH = 10
PIXEL_SIZE = 0.003 # [DEG]
LEAPSECOND = TimeDelta(10, format="sec")
N_WORKERS = 6
KD_TREE_WORKERS = -1
t0 = Time("1993-01-01 00:00:00", scale="tai")

dtypes = {
        "Date": "str",
        "Time": "str",
        "Lon": np.float64,
        "Lat": np.float64,
        "SOG": np.float64,
        "COG": np.float64,
        "ITPL" : np.int32 # 보간 데이터 : 1, 관측 데이터 : 0 (default)
        }
dyn_dict_frame = {
        "Date": None,
        "Time": None,
        "Lon": np.nan,
        "Lat": np.nan,
        "SOG": np.nan,
        "COG": np.nan,
        "ITPL" : 0, # 관측 데이터
        }

# Class Definition
class Ship:
    def __init__(self, MMSI : str, conn : sqlite3.Connection):
        try:
            self.MMSI = int(MMSI)
            self.conn = conn
            self.statics = {
                "VesselName" : None,
                "VesselType" : None,
                "Dim" : None,
            }
            self.conn.execute(
                "INSERT OR IGNORE INTO ships_static (MMSI) VALUES (?);",
                (self.MMSI,)
            )
        except Exception as e:
            print(f"[Exception] || loc : __init(self, MMSI : int) || message : {e}")
            sys.exit(1)

    def fill_static(self, statics : dict)->None:
        self.statics = statics
        try:
            dim = self.statics.get("Dim")
            dimA = dimB = dimC = dimD = None
            if isinstance(dim, tuple) and len(dim) == 4:
                dimA, dimB, dimC, dimD = map(int, dim)

            self.conn.execute("""
                UPDATE ships_static
                SET VesselName = ?,
                    VesselType = ?,
                    DimA = ?,
                    DimB = ?,
                    DimC = ?,
                    DimD = ?
                WHERE MMSI = ?;
            """, (
                self.statics.get("VesselName"),
                self.statics.get("VesselType"),
                dimA, dimB, dimC, dimD,
                self.MMSI
            ))
        except Exception as e:
            print(f"[Exception] || loc : save_statics(self, statics : dict) || message : {e}")
            sys.exit(1)

    # row_dict의 구조는 dtypes와 일치
    def fill_dynamic(self, row_dict : dict)->None:
        try:
            self.conn.execute("""
                INSERT INTO ships_dynamic
                (MMSI, Date, Time, Lon, Lat, SOG, COG, ITPL)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                self.MMSI,
                row_dict.get("Date"),
                row_dict.get("Time"),
                row_dict.get("Lon"),
                row_dict.get("Lat"),
                row_dict.get("SOG"),
                row_dict.get("COG"),
                row_dict.get("ITPL")
            ))
        except Exception as e:
            print(f"[Exception] || loc : fill_dynamic(self, row_dict : dict) || message : {e}")
            sys.exit(1)

# util functions
def filter_by_tif_name(tif_name : str)->bool:
    _, HHMM, _ = tif_name.split("_")
    return ((int(HHMM) < 1550) or (int(HHMM) > 1810))

def _mapped_date()->dict: # 위성 영상의 날짜 (001 ~ 365) -> datetime 객체로 변환
    start_date = date(YEAR, 1, 1)
    doy_to_date = {doy: (start_date + timedelta(days=doy - 1)).isoformat() for doy in range(1, 366)}
    return doy_to_date

def _str_to_datetime(date_str : str, time_str : str)->datetime:
    return datetime.combine(date.fromisoformat(date_str), time.fromisoformat(time_str))

def _COG_to_unit_vector(cog_deg : float) -> tuple[float, float]:
    rad = np.deg2rad(cog_deg)
    return np.cos(rad), np.sin(rad)

def _unit_vector_to_COG(vx: float, vy: float) -> float:
    deg = np.rad2deg(np.arctan2(vy, vx))
    return deg % 360

def parse_key(fname :str)->tuple[int, int]:
    base = fname.replace(".tif", "")
    date, hhmm, _ = base.split("_")
    return (int(date[1:]), int(hhmm))

def GetScanInfo(dnb03_path:str):
    print("  [open]", os.path.basename(dnb03_path), flush=True)
    with Dataset(dnb03_path, 'r') as dnb03:
        latitude = dnb03['/geolocation_data/latitude'][:]
        longitude = dnb03['/geolocation_data/longitude'][:]
        sstime = dnb03['/scan_line_attributes/scan_start_time'][:]
        setime = dnb03['/scan_line_attributes/scan_end_time'][:]
        return latitude, longitude, sstime, setime

@dataclass
class PreparedLinearSeries:
    t0: datetime
    x: np.ndarray
    lon: np.ndarray
    lat: np.ndarray
    sog: np.ndarray
    cog_x: np.ndarray
    cog_y: np.ndarray

@dataclass
class SeriesItem:
    times: list[datetime]
    lons: list[float]
    lats: list[float]
    sogs: list[float]
    cogs: list[float]
    time_set: set[datetime]
    prepared: PreparedLinearSeries | None = None
    prepare_error: Exception | None = None
    prepare_attempted: bool = False

@dataclass
class GeoIndex:
    shape: tuple[int, int]
    flat_lon: np.ndarray
    flat_lat: np.ndarray
    valid_flat_idx: np.ndarray
    tree: cKDTree
    sstimes: np.ndarray
    setimes: np.ndarray

    @property
    def n_cols(self) -> int:
        return self.shape[1]

    def nearest_flat_indices(self, lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
        points = np.column_stack((lons, lats))
        try:
            distances, tree_pos = self.tree.query(points, k=1, eps=0.0, workers=KD_TREE_WORKERS)
        except TypeError:
            distances, tree_pos = self.tree.query(points, k=1, eps=0.0)

        chosen = np.empty(len(points), dtype=np.int64)
        for i, (lon, lat, dist, pos) in enumerate(zip(lons, lats, distances, tree_pos)):
            radius = np.nextafter(float(dist), np.inf)
            cand_pos = self.tree.query_ball_point((float(lon), float(lat)), r=radius)
            if not cand_pos:
                cand_pos = [int(pos)]

            cand_flat = self.valid_flat_idx[np.asarray(cand_pos, dtype=np.int64)]
            dist2 = (self.flat_lon[cand_flat] - float(lon)) ** 2 + (self.flat_lat[cand_flat] - float(lat)) ** 2
            min_dist2 = dist2.min()
            # Preserve np.argmin tie behavior: first minimum in C-order flattened array.
            chosen[i] = cand_flat[dist2 == min_dist2].min()
        return chosen

    def exact_times_from_flat_indices(self, flat_indices: np.ndarray) -> np.ndarray:
        rows = flat_indices // self.n_cols
        cols = flat_indices % self.n_cols
        line_idx = rows // (self.shape[0] // 201)
        itpl_time = self.sstimes[line_idx] + (cols / self.n_cols) * (self.setimes[line_idx] - self.sstimes[line_idx])
        return (t0 + TimeDelta(np.asarray(itpl_time, dtype=np.float64) * u.s)).utc.to_datetime()

def make_dnb_03_dict() -> dict:
    def name_handler(ncname : str) -> str:
        tup = ncname.split(".")
        return "_".join(str(tup[i]) for i in (1, 2, 3)) + ".tif"

    dnb_03_list = [ncname for ncname in os.listdir(f'{ROOT_HDD}{DNB_03}/') if (not ncname.startswith('.') and len(ncname) > 20)]

    return {name_handler(ncname) : f"{ROOT_HDD}{DNB_03}/{ncname}" for ncname in dnb_03_list}

# main functions
def search_MMSI_set(mapped_date_dict : dict, df : pd.DataFrame, tif_name : str, conn : sqlite3.Connection)->set: # tif 관측 평균 시각(시작 시각 + 3분) 전후 DELTA분 기간에 포함되는 선박들의 MMSI 집합
    row = df.loc[df["tif_name"] == tif_name, ["scan_start_HHMMSS", "duration [sec]"]].iloc[0]
    duration = timedelta(seconds=float(row["duration [sec]"]))
    parsed = tif_name.split("_")
    tif_date_str = mapped_date_dict[int(parsed[0][-3:])]
    tif_time_str = row["scan_start_HHMMSS"]
    tif_center_dt = _str_to_datetime(tif_date_str, tif_time_str) + (duration / 2)

    start_dt = (tif_center_dt - DELTA) #.strftime("%Y-%m-%d %H:%M:%S")
    end_dt   = (tif_center_dt + DELTA) #.strftime("%Y-%m-%d %H:%M:%S")

    start_dt_date = start_dt.date().isoformat()
    start_dt_time = start_dt.time().isoformat()
    end_dt_date = end_dt.date().isoformat()
    end_dt_time = end_dt.time().isoformat()

    rows = conn.execute("""
    SELECT DISTINCT MMSI
    FROM ships_dynamic
    WHERE ITPL = 0
      AND (
            Date > ?
         OR (Date = ? AND Time >= ?)
      )
      AND (
            Date < ?
         OR (Date = ? AND Time <= ?)
      )
""", (start_dt_date, start_dt_date, start_dt_time,
      end_dt_date, end_dt_date, end_dt_time)).fetchall()

    return {int(MMSI) for (MMSI,) in rows}, tif_date_str, tif_center_dt

def load_whole_day(MMSI: int, date: str, conn: sqlite3.Connection): # day 자리에 search_MMSI_set()[1]
    rows = conn.execute("""
        SELECT Date, Time, Lon, Lat, SOG, COG
        FROM ships_dynamic
        WHERE MMSI = ? AND Date = ? AND ITPL = 0
        ORDER BY Time
    """, (MMSI, date)).fetchall()

    Dates, Times, Lons, Lats, SOGs, COGs = zip(*rows)
    datetimes = [datetime.fromisoformat(f"{d} {t}") for d, t in zip(Dates, Times)]
    Lons  = list(Lons)
    Lats  = list(Lats)
    SOGs  = list(SOGs)
    COGs  = list(COGs)
    return datetimes, Lons, Lats, SOGs, COGs

def exact_raw_record(
    MMSI: int,
    times: list[datetime],
    Lons: list[float],
    Lats: list[float],
    SOGs: list[float],
    COGs: list[float],
    t_query: datetime,
) -> dict | None:
    exact_dt = t_query.replace(microsecond=0)
    exact_idx = None
    for i, t in enumerate(times):
        if t == exact_dt:
            exact_idx = i

    if exact_idx is None:
        return None

    return {
        "MMSI": int(MMSI),
        "Date": exact_dt.date().isoformat(),
        "Time": exact_dt.time().isoformat(),
        "Lon": float(Lons[exact_idx]),
        "Lat": float(Lats[exact_idx]),
        "SOG": float(SOGs[exact_idx]),
        "COG": float(COGs[exact_idx]),
        "ITPL": 0,
    }

def build_geo_index(geoloc_lon, geoloc_lat, sstimes, setimes) -> GeoIndex:
    lon_data = np.asarray(np.ma.getdata(geoloc_lon), dtype=np.float64).ravel(order="C")
    lat_data = np.asarray(np.ma.getdata(geoloc_lat), dtype=np.float64).ravel(order="C")

    lon_mask = np.ma.getmaskarray(geoloc_lon).ravel(order="C")
    lat_mask = np.ma.getmaskarray(geoloc_lat).ravel(order="C")
    valid = ~(lon_mask | lat_mask) & np.isfinite(lon_data) & np.isfinite(lat_data)
    valid_flat_idx = np.flatnonzero(valid).astype(np.int64, copy=False)
    if valid_flat_idx.size == 0:
        raise ValueError("No valid geolocation points")

    points = np.column_stack((lon_data[valid_flat_idx], lat_data[valid_flat_idx]))
    tree = cKDTree(points, compact_nodes=True, balanced_tree=True, copy_data=False)
    return GeoIndex(
        shape=geoloc_lon.shape,
        flat_lon=lon_data,
        flat_lat=lat_data,
        valid_flat_idx=valid_flat_idx,
        tree=tree,
        sstimes=np.asarray(sstimes, dtype=np.float64),
        setimes=np.asarray(setimes, dtype=np.float64),
    )

def load_day_series_cache(date_str: str, MMSI_list: list[int], conn: sqlite3.Connection) -> dict[int, SeriesItem]:
    grouped = defaultdict(lambda: ([], [], [], [], []))
    chunk_size = 900
    for start in range(0, len(MMSI_list), chunk_size):
        chunk = MMSI_list[start:start + chunk_size]
        if not chunk:
            continue

        placeholders = ",".join("?" for _ in chunk)
        rows = conn.execute(f"""
            SELECT MMSI, Date, Time, Lon, Lat, SOG, COG
            FROM ships_dynamic
            WHERE ITPL = 0
              AND Date = ?
              AND MMSI IN ({placeholders})
            ORDER BY MMSI, Time, id
        """, (date_str, *chunk)).fetchall()

        for MMSI, Date_, Time_, Lon, Lat, SOG, COG in rows:
            times, lons, lats, sogs, cogs = grouped[int(MMSI)]
            times.append(datetime.fromisoformat(f"{Date_} {Time_}"))
            lons.append(float(Lon))
            lats.append(float(Lat))
            sogs.append(float(SOG))
            cogs.append(float(COG))

    cache: dict[int, SeriesItem] = {}
    for MMSI in MMSI_list:
        times, lons, lats, sogs, cogs = grouped[int(MMSI)]
        cache[int(MMSI)] = SeriesItem(
            times=times,
            lons=lons,
            lats=lats,
            sogs=sogs,
            cogs=cogs,
            time_set=set(times),
        )
    return cache

def _prepare_linear_series(item: SeriesItem) -> PreparedLinearSeries:
    if item.prepare_attempted:
        if item.prepare_error is not None:
            raise item.prepare_error
        return item.prepared

    item.prepare_attempted = True
    try:
        if len(item.times) < 2:
            raise ValueError("need at least 2 points")

        idx = np.argsort(item.times)
        t_sorted = [item.times[i] for i in idx]
        lon_sorted = np.asarray([item.lons[i] for i in idx], dtype=np.float64)
        lat_sorted = np.asarray([item.lats[i] for i in idx], dtype=np.float64)
        sog_sorted = np.asarray([item.sogs[i] for i in idx], dtype=np.float64)
        cog_x_sorted = np.empty(len(idx), dtype=np.float64)
        cog_y_sorted = np.empty(len(idx), dtype=np.float64)
        for out_i, src_i in enumerate(idx):
            cog_x_sorted[out_i], cog_y_sorted[out_i] = _COG_to_unit_vector(item.cogs[src_i])

        t_dedup: list[datetime] = []
        lon_dedup: list[float] = []
        lat_dedup: list[float] = []
        sog_dedup: list[float] = []
        cog_x_dedup: list[float] = []
        cog_y_dedup: list[float] = []

        for i, t in enumerate(t_sorted):
            if not t_dedup or t != t_dedup[-1]:
                t_dedup.append(t)
                lon_dedup.append(float(lon_sorted[i]))
                lat_dedup.append(float(lat_sorted[i]))
                sog_dedup.append(float(sog_sorted[i]))
                cog_x_dedup.append(float(cog_x_sorted[i]))
                cog_y_dedup.append(float(cog_y_sorted[i]))
            else:
                lon_dedup[-1] = float(lon_sorted[i])
                lat_dedup[-1] = float(lat_sorted[i])
                sog_dedup[-1] = float(sog_sorted[i])
                cog_x_dedup[-1] = float(cog_x_sorted[i])
                cog_y_dedup[-1] = float(cog_y_sorted[i])

        if len(t_dedup) < 2:
            raise ValueError("need at least 2 unique time points")

        series_t0 = t_dedup[0]
        x = np.asarray([(t - series_t0).total_seconds() for t in t_dedup], dtype=np.float64)
        item.prepared = PreparedLinearSeries(
            t0=series_t0,
            x=x,
            lon=np.asarray(lon_dedup, dtype=np.float64),
            lat=np.asarray(lat_dedup, dtype=np.float64),
            sog=np.asarray(sog_dedup, dtype=np.float64),
            cog_x=np.asarray(cog_x_dedup, dtype=np.float64),
            cog_y=np.asarray(cog_y_dedup, dtype=np.float64),
        )
        return item.prepared
    except Exception as e:
        item.prepare_error = e
        raise

def interpolate_series_one(MMSI: int, item: SeriesItem, t_query: datetime, ITPL: int) -> dict | None:
    if t_query.replace(microsecond=0) in item.time_set:
        record = exact_raw_record(MMSI, item.times, item.lons, item.lats, item.sogs, item.cogs, t_query)
        if record is not None and ITPL == 1:
            record["ITPL"] = 1
            record["CENTER_EXACT"] = True
        return record

    if len(item.times) < 2:
        return None

    try:
        prepared = _prepare_linear_series(item)
        xq = np.asarray([(t_query - prepared.t0).total_seconds()], dtype=np.float64)
        xq = np.clip(xq, float(prepared.x[0]), float(prepared.x[-1]))

        lon = float(np.interp(xq, prepared.x, prepared.lon)[0])
        lat = float(np.interp(xq, prepared.x, prepared.lat)[0])
        sog = float(np.interp(xq, prepared.x, prepared.sog)[0])
        vx = float(np.interp(xq, prepared.x, prepared.cog_x)[0])
        vy = float(np.interp(xq, prepared.x, prepared.cog_y)[0])
        COG = _unit_vector_to_COG(vx, vy)
        return {
            "MMSI": int(MMSI),
            "Date": t_query.date().isoformat(),
            "Time": t_query.time().replace(microsecond=0).isoformat(),
            "Lon": lon,
            "Lat": lat,
            "SOG": sog,
            "COG": float(COG),
            "ITPL": ITPL,
            "CENTER_EXACT": False,
        }
    except Exception as e:
        print(f"[Exception] || loc : interpolate_MMSI_to_df(args...)) || message : {e}")
        return None

def interpolate_many_prepared(
    ordered_mmsi: list[int],
    series_cache: dict[int, SeriesItem],
    t_query: datetime,
    ITPL: int,
) -> pd.DataFrame:
    records = []
    for MMSI in ordered_mmsi:
        record = interpolate_series_one(MMSI, series_cache[MMSI], t_query, ITPL)
        if record is not None:
            records.append(record)
    return pd.DataFrame.from_records(records)

def refine_df_ITPL_fast(
    df_ITPL: pd.DataFrame,
    series_cache: dict[int, SeriesItem],
    geo_index: GeoIndex,
    tif_center_dt_str: str,
) -> pd.DataFrame:
    if df_ITPL.empty:
        return df_ITPL

    MMSIs = df_ITPL["MMSI"].to_numpy(dtype=np.int64)
    lons = df_ITPL["Lon"].to_numpy(dtype=np.float64).copy()
    lats = df_ITPL["Lat"].to_numpy(dtype=np.float64).copy()
    sogs = df_ITPL["SOG"].to_numpy(dtype=np.float64).copy()
    cogs = df_ITPL["COG"].to_numpy(dtype=np.float64).copy()
    original_itpls = df_ITPL["ITPL"].to_numpy(dtype=np.int32)
    itpls = np.where(original_itpls == 0, 0, 2).astype(np.int32, copy=False)
    times = df_ITPL["Time"].to_numpy(dtype=object).copy()
    active = original_itpls != 0
    center_exact = (
        df_ITPL["CENTER_EXACT"].fillna(False).to_numpy(dtype=bool)
        if "CENTER_EXACT" in df_ITPL.columns
        else np.zeros(len(df_ITPL), dtype=bool)
    )
    valid = np.ones(len(df_ITPL), dtype=bool)

    for ITPL in range(2, ITPL_DEPTH):
        active_idx = np.flatnonzero(active & (itpls == ITPL))
        if active_idx.size == 0:
            continue

        nearest_flat = geo_index.nearest_flat_indices(lons[active_idx], lats[active_idx])
        exact_times = geo_index.exact_times_from_flat_indices(nearest_flat)

        for row_idx, exact_time in zip(active_idx, exact_times):
            times[row_idx] = exact_time.time().replace(microsecond=0).isoformat()
            old_lon = lons[row_idx]
            old_lat = lats[row_idx]
            record = interpolate_series_one(int(MMSIs[row_idx]), series_cache[int(MMSIs[row_idx])], exact_time, ITPL)
            if record is None:
                # A center_dt exact row is only an initial estimate. If it cannot be
                # resolved at the actual scan-line time, it must not be saved.
                if center_exact[row_idx]:
                    valid[row_idx] = False
                active[row_idx] = False
                continue

            new_lon = float(record["Lon"])
            new_lat = float(record["Lat"])
            lons[row_idx] = new_lon
            lats[row_idx] = new_lat
            sogs[row_idx] = float(record["SOG"])
            cogs[row_idx] = float(record["COG"])
            if int(record["ITPL"]) == 0:
                itpls[row_idx] = 0
                active[row_idx] = False
                continue

            if abs(new_lon - old_lon) < PIXEL_SIZE * 1e-10 and abs(new_lat - old_lat) < PIXEL_SIZE * 1e-10:
                active[row_idx] = False
            else:
                itpls[row_idx] = ITPL + 1

    df_ITPL = df_ITPL.loc[valid].copy()
    df_ITPL["Time"] = times[valid]
    df_ITPL["Lon"] = lons[valid]
    df_ITPL["Lat"] = lats[valid]
    df_ITPL["ITPL"] = itpls[valid]
    df_ITPL["IMG_DT"] = tif_center_dt_str
    df_ITPL["SOG"] = sogs[valid]
    df_ITPL["COG"] = cogs[valid]
    return df_ITPL

def _build_tck(mode : str, datetimes : list[datetime], values : list[float], k : int = 3, s : float = 0.0) -> tuple:
    if len(datetimes) != len(values):
        raise ValueError("times and values must have same length")
    if len(datetimes) < 2:
        raise ValueError("need at least 2 points")

    # 정렬
    idx = np.argsort(datetimes)
    t_sorted = [datetimes[i] for i in idx]
    y_sorted = np.array([values[i] for i in idx], dtype=np.float64)

    # 같은 시간 중복 제거(마지막 값 유지)
    t_dedup: list[datetime] = []
    y_dedup: list[float] = []
    for t, y in zip(t_sorted, y_sorted):
        if not t_dedup or t != t_dedup[-1]:
            t_dedup.append(t)
            y_dedup.append(float(y))
        else:
            y_dedup[-1] = float(y)

    if len(t_dedup) < 2:
        raise ValueError("need at least 2 unique time points")

    t0 = t_dedup[0]
    x = np.array([(t - t0).total_seconds() for t in t_dedup], dtype=np.float64)

    if mode == "linear":
        tck = (x, np.array(y_dedup, dtype=np.float64))
        return (t0, tck)
    elif mode == "spline":
        # k는 최대 len(x)-1
        k_eff = min(k, len(x) - 1)
        tck = splrep(x, np.array(y_dedup, dtype=np.float64), k=k_eff, s=s)
        return (t0, tck)
    else:
        raise ValueError("Unknown Mode for _build_tck()")

def _eval_tck_spline(
    t0: datetime,
    tck: tuple,
    t_queries: list[datetime],
    clamp: bool = True,
) -> np.ndarray:
    # splrep가 만든 knot 범위는 x축에서 tck[0] (knots)로 추정 가능
    knots = tck[0]
    x_min, x_max = float(knots[0]), float(knots[-1])

    xq = np.array([(t - t0).total_seconds() for t in t_queries], dtype=np.float64)

    if clamp:
        xq = np.clip(xq, x_min, x_max)

    yq = splev(xq, tck)
    return np.asarray(yq, dtype=np.float64)

def _eval_tck_linear(
    t0: datetime,
    tck: tuple,
    t_queries: list[datetime],
    clamp: bool = True,
) -> np.ndarray:
    if not isinstance(tck, tuple) or len(tck) != 2:
        raise ValueError("For _eval_linear, tck must be a tuple (x, y) in seconds-since-t0.")

    x, y = tck

    if x.ndim != 1 or y.ndim != 1 or x.size != y.size:
        raise ValueError("x and y must be 1D arrays of the same length.")

    if x.size == 0:
        return np.full(len(t_queries), np.nan, dtype=np.float64)

    x_min, x_max = float(x[0]), float(x[-1])

    xq = np.array([(t - t0).total_seconds() for t in t_queries], dtype=np.float64)
    if clamp:
        xq = np.clip(xq, x_min, x_max)

    # Degenerate case: only one sample point -> constant
    if x.size == 1:
        return np.full_like(xq, float(y[0]), dtype=np.float64)

    # Piecewise-linear interpolation using adjacent points
    # np.interp clamps automatically outside [x0, xN] to y endpoints; we already clamped xq if clamp=True.
    yq = np.interp(xq, x, y)

    return np.asarray(yq, dtype=np.float64)

def _eval_tck(
    mode: str,
    t0: datetime,
    tck: tuple,
    t_queries: list[datetime],
    clamp: bool = True,
) -> np.ndarray:
    if mode == "linear":
        return _eval_tck_linear(t0, tck, t_queries, clamp)
    elif mode == "spline":
        return _eval_tck_spline(t0, tck, t_queries, clamp)
    else:
        raise ValueError("Unknown Mode for _eval_tck()")

def interpolate_MMSI_to_df(
    MMSI_to_series: dict[int, tuple[list[datetime], list[float], list[float], list[float], list[float]]],
    t_queries: list[datetime],
    ITPL_MODE : str,
    ITPL : int,
    k: int = 3,
    s: float = 0.0,
    clamp: bool = True,
) -> pd.DataFrame:
    if not t_queries:
        raise ValueError("t_queries is empty")

    records = []
    exact = 0 # DEBUG
    # debug_len = dict()
    for MMSI, (times, Lons, Lats, SOGs, COGs) in MMSI_to_series.items():
        # if len(times) in debug_len.keys():
        #     debug_len[len(times)] += 1
        # else:
        #     debug_len[len(times)] = 0
        # continue
        # ITPL=1(center_dt) exact는 scan-line 보정의 초기 위치로만 사용한다.
        # 최종 ITPL=0은 ITPL>=2에서 실제 scan-line 촬영 시각과 match될 때만 확정한다.
        if t_queries[0].replace(microsecond=0) in set(times):
            exact += 1 # DEBUG
            record = exact_raw_record(MMSI, times, Lons, Lats, SOGs, COGs, t_queries[0])
            if record is not None:
                if ITPL == 1:
                    record["ITPL"] = 1
                    record["CENTER_EXACT"] = True
                records.append(record)
            continue

        # 최소 점 수 체크
        if len(times) < 2:
            continue

        try:
            lon_t0, lon_tck = _build_tck(ITPL_MODE, times, Lons, k=k, s=s)
            lat_t0, lat_tck = _build_tck(ITPL_MODE, times, Lats, k=k, s=s)
            SOG_t0, SOG_tck = _build_tck(ITPL_MODE, times, SOGs, k=k, s=s)

            COG_vx = []
            COG_vy = []
            for COG in COGs:
                vx, vy = _COG_to_unit_vector(COG)
                COG_vx.append(vx)
                COG_vy.append(vy)
            COGx_t0, COGx_tck = _build_tck(ITPL_MODE, times, COG_vx, k=k, s=s)
            COGy_t0, COGy_tck = _build_tck(ITPL_MODE, times, COG_vy, k=k, s=s)

            if len(set([lon_t0, lat_t0, SOG_t0, COGx_t0, COGy_t0])) == 1: # 이론상 모든 시각은 동일해야 함
                lon_vals = _eval_tck(ITPL_MODE, lon_t0, lon_tck, t_queries, clamp=clamp)
                lat_vals = _eval_tck(ITPL_MODE, lat_t0, lat_tck, t_queries, clamp=clamp)
                SOG_vals = _eval_tck(ITPL_MODE, SOG_t0, SOG_tck, t_queries, clamp=clamp)
                COGx_vals = _eval_tck(ITPL_MODE, COGx_t0, COGx_tck, t_queries, clamp=clamp)
                COGy_vals = _eval_tck(ITPL_MODE, COGy_t0, COGy_tck, t_queries, clamp=clamp)

                for tq, lon, lat, SOG, vx, vy in zip(t_queries, lon_vals, lat_vals, SOG_vals, COGx_vals, COGy_vals):
                    date_str = tq.date().isoformat()
                    time_str = tq.time().replace(microsecond=0).isoformat() # 통일성을 위해 마이크로초부터는 제거하고 저장
                    COG = _unit_vector_to_COG(float(vx), float(vy))
                    records.append({
                        "MMSI": int(MMSI),
                        "Date": date_str,
                        "Time": time_str,
                        "Lon": float(lon),
                        "Lat": float(lat),
                        "SOG": float(SOG),
                        "COG": float(COG),
                        "ITPL": ITPL, # 보간 횟수
                        "CENTER_EXACT": False,
                    })
            else:
                raise(ValueError(f"lon_t0 : {lon_t0} || lat_t0 : {lat_t0} || SOG_t0 : {SOG_t0} || COGx_t0 : {COGx_t0} || COGy_t0 : {COGy_t0}"))

        except Exception as e:
            # 이 MMSI는 보간 실패(점 부족/중복 시간 등) → 에러 메시지 출력 후 스킵
            print(f"[Exception] || loc : interpolate_MMSI_to_df(args...)) || message : {e}")
            continue
    # print(f"{exact} exact time-matched MMSIs for center dt {t_queries[0].replace(microsecond=0)}")
    # return debug_len
    return pd.DataFrame.from_records(records)

def save_df_to_DB(df_ITPL: pd.DataFrame, conn: sqlite3.Connection):
    cols = ["MMSI", "Date", "Time", "Lon", "Lat", "SOG", "COG", "ITPL", "IMG_DT"]
    rows = df_ITPL[cols].itertuples(index=False, name=None)

    conn.executemany("""
        INSERT INTO ships_dynamic (MMSI, Date, Time, Lon, Lat, SOG, COG, ITPL, IMG_DT)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
    """, list(rows))

    conn.commit()

# DB helper functions
def init_db(conn : sqlite3.Connection) -> None:
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS ships_static (
        MMSI INTEGER PRIMARY KEY,
        VesselName TEXT,
        VesselType TEXT,
        DimA INTEGER,
        DimB INTEGER,
        DimC INTEGER,
        DimD INTEGER
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS ships_dynamic (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        MMSI INTEGER NOT NULL,
        Date TEXT,
        Time TEXT,
        Lon REAL,
        Lat REAL,
        SOG REAL,
        COG REAL,
        ITPL INTEGER NOT NULL DEFAULT 0,
        TIF_NAME TEXT,
        FOREIGN KEY(MMSI) REFERENCES ships_static(MMSI)
    );
    """)

    cur.execute("PRAGMA table_info(ships_dynamic);")
    existing_cols = {row[1] for row in cur.fetchall()}

    if "IMG_DT" not in existing_cols:
        cur.execute("ALTER TABLE ships_dynamic ADD COLUMN IMG_DT TEXT;")

    cur.execute("CREATE INDEX IF NOT EXISTS idx_ais_MMSI ON ships_dynamic(MMSI);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_ais_MMSI_dt ON ships_dynamic(MMSI, Date, Time);")

    conn.commit()

def prep_db(conn : sqlite3.Connection) -> None:
    init_db(conn)

def backup(conn: sqlite3.Connection) -> None:
    tmp_path = SNAPSHOT_PATH + ".tmp"

    with sqlite3.connect(tmp_path) as dst:
        conn.backup(dst)
        dst.commit()

    os.replace(tmp_path, SNAPSHOT_PATH)

def mark_ITPL_completed(tif_name: str, path: str):
    with open(path, "a", encoding="utf-8") as f:
        f.write(tif_name + "\n")

def get_marked_TIF_set(path: str)->set:
    if not os.path.exists(path):
        return set()

    with open(path, "r", encoding="utf-8") as f:
        return set([line.strip() for line in f if line.strip()])

# DB maker functions
def find_already_exist(conn : sqlite3.Connection)->dict:
    try:
        cur = conn.cursor()
        cur.execute("SELECT MMSI FROM ships_static;")
        rows = cur.fetchall()
        return {int(MMSI): True for (MMSI,) in rows}
    except Exception as e:
        print(f"[Exception] || loc : find_already_exist() || message : {e}")
        conn.close()
        sys.exit(1)

def process_one_AIS_csv(AIS_csv_path : str, curr_Ship_dict : dict, curr_already_exist : set, conn : sqlite3.Connection)->dict: # (key, val) : (MMSI, Ship 객체)
    df = pd.read_csv(AIS_csv_path, low_memory=False)

    Ship_dict = curr_Ship_dict
    MMSI_set = set(df["MMSI"])
    exist_MMSI_set = set(curr_already_exist.keys())
    curr_already_exist.update({MMSI : False for MMSI in (MMSI_set - exist_MMSI_set)})
    already_exist = curr_already_exist

    for row in df.itertuples(index=False):
        MMSI = row.MMSI
        dyn_dict = dyn_dict_frame.copy()
        row_dict = row._asdict()
        for key in row_dict.keys():
            if (key in dyn_dict.keys()) and (key != "ITPL"):
                dyn_dict[key] = row_dict[key]
        if already_exist[MMSI] == False:
            # create new Ship object
            ship = Ship(MMSI, conn)
            statics = {
                "VesselName": getattr(row, "VesselName", None),
                "VesselType": getattr(row, "VesselType", None),
                "Dim": (getattr(row, "DimA", 0), getattr(row, "DimB", 0),
                        getattr(row, "DimC", 0), getattr(row, "DimD", 0)),
            }
            ship.fill_static(statics)
            ship.fill_dynamic(dyn_dict)
            Ship_dict[MMSI] = ship
            already_exist[MMSI] = True
        else:
            # update existing Ship object
            ship = Ship_dict.get(MMSI)
            if ship is None:
                ship = Ship(MMSI, conn)
                Ship_dict[MMSI] = ship
            ship.fill_dynamic(dyn_dict)
    return Ship_dict, already_exist

def make_all_ship_csv(AIS_folder_path : str, conn : sqlite3.Connection) -> dict:
    prep_db(conn)

    AIS_csv_list = sorted([name for name in os.listdir(AIS_folder_path) if not name.startswith('.')])
    Ship_dict = dict()
    already_exist = find_already_exist(conn)

    try:
        for AIS_csv_name in AIS_csv_list:
            AIS_csv_path = os.path.join(AIS_folder_path, AIS_csv_name)
            print("="*80+f"\n[START] processing {AIS_csv_path}")
            print("[Main Step...]")
            Ship_dict, already_exist = process_one_AIS_csv(AIS_csv_path, Ship_dict, already_exist, conn)
            print("[Saving to DB...]")
            conn.commit()
            print("[Making a snapshot...]")
            backup(conn)
            print(f"[COMPLETE] processing {AIS_csv_path}\n"+"="*80)
            # # [DEBUGGING]
            # if input("press y or n") != "y":
            #     break
    finally:
        conn.commit()
        conn.close()

# fine interpolation by multi processing
def split_ranges(n: int, k: int):
    base = n // k
    rem  = n % k
    ranges = []
    start = 0
    for i in range(k):
        size = base + (1 if i < rem else 0)
        end = start + size
        ranges.append((start, end))
        start = end
    return ranges

def worker_range(args):
    (start, end, df_chunk, db_path, tif_date, tif_center_dt_str, geoloc_lon, geoloc_lat, sstimes, setimes, ITPL_DEPTH) = args

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    results = []  # (idx, new_lon, new_lat, final_itpl, exact_time_str)

    for row in df_chunk.itertuples():
        idx = row.Index
        MMSI = int(row.MMSI)
        new_lon = float(row.Lon)
        new_lat = float(row.Lat)
        SOG = float(row.SOG)
        COG = float(row.COG)
        center_exact = bool(getattr(row, "CENTER_EXACT", False))
        skip_record = False

        ITPL = 2
        old_lon, old_lat = new_lon, new_lat

        day_df = load_whole_day(MMSI, tif_date, conn)

        while ITPL < ITPL_DEPTH:
            # print(f"{MMSI} : old (lon, lat) = ({old_lon}, {old_lat})")
            exact_time = find_exact_time(new_lon, new_lat, geoloc_lon, geoloc_lat, sstimes, setimes)

            df_fine = interpolate_MMSI_to_df(
                {MMSI: day_df},
                [exact_time],
                "linear",
                ITPL, k=3, s=0.0, clamp=True
            )
            if df_fine.empty:
                if center_exact:
                    skip_record = True
                break

            new_lon = float(df_fine.loc[0, "Lon"])
            new_lat = float(df_fine.loc[0, "Lat"])
            SOG = df_fine.loc[0, "SOG"]
            COG = df_fine.loc[0, "COG"]
            if int(df_fine.loc[0, "ITPL"]) == 0:
                ITPL = 0
                break
            # print(f"{MMSI} : new (lon, lat) = ({new_lon}, {new_lat})")
            lon_diff = new_lon - old_lon
            lat_diff = new_lat - old_lat
            # print(f"[MMSI : {MMSI}] ITPL : {ITPL}, diff : ({lon_diff},{lat_diff})")
            if abs(lon_diff) < PIXEL_SIZE * 1e-10 and abs(lat_diff) < PIXEL_SIZE * 1e-10:
                break

            old_lon, old_lat = new_lon, new_lat
            ITPL += 1
        if skip_record:
            results.append((idx, None, None, None, None, None, None, None))
            continue
        exact_time_str = exact_time.time().replace(microsecond=0).isoformat()
        results.append((idx, exact_time_str, new_lon, new_lat, ITPL, tif_center_dt_str, SOG, COG))

    conn.close()
    return results

def find_exact_time(Lon, Lat, geoloc_lon, geoloc_lat, sstimes, setimes):
    dist2 = (geoloc_lon - Lon)**2 + (geoloc_lat - Lat)**2
    idx = np.argmin(dist2)
    shape = geoloc_lon.shape
    r, c = np.unravel_index(idx, shape)
    line_idx = r // (shape[0] // 201) # 0 ~ 200
    sstime = sstimes[line_idx]
    setime = setimes[line_idx]
    itpl_time = sstime + (c / shape[1]) * (setime - sstime) # linear time interpolation
    return (t0 + TimeDelta(itpl_time * u.s)).utc.to_datetime()

def ITPL(TIF_THRES, MMSI_THRES, tif_name_list : list, mapped_date_dict : dict, df : pd.DataFrame,conn : sqlite3.Connection):

    dnb_03_dict = make_dnb_03_dict()

    # main loop
    for i, tif_name in enumerate(tif_name_list):
        if filter_by_tif_name(tif_name): # 외삽이 아닌 보간이므로, AIS 데이터 시간대 범위를 벗어나면 제외
            continue
        try:
            if (i >= TIF_THRES):
                # print(f"REACHED to TIF_THRES({TIF_THRES})") # [DEBUGGING]
                break
            print("="*80+f"\n[START] processing {tif_name}")

            geoloc_lat, geoloc_lon, sstimes, setimes = GetScanInfo(dnb_03_dict[tif_name])

            MMSI_set, tif_date, tif_center_dt = search_MMSI_set(mapped_date_dict, df, tif_name, conn)

            ordered_mmsi = []
            if (MMSI_THRES == -1):
                MMSI_THRES = len(MMSI_set)

            for j, MMSI in enumerate(MMSI_set):
                if (j >= MMSI_THRES):
                    # print(f"REACHED to MMSI_THRES({MMSI_THRES})") # [DEBUGGING]
                    break
                ordered_mmsi.append(int(MMSI))

            series_cache = load_day_series_cache(tif_date, ordered_mmsi, conn)

            # Interpolation - (1) AIS 데이터 기반 단순 선형 보간
            t_queries = [tif_center_dt]
            df_ITPL = interpolate_many_prepared(ordered_mmsi, series_cache, t_queries[0], 1)
            # for key in df_ITPL:
            #     print("# of AIS data point :", key, "# of such record :", df_ITPL[key])
            # continue

            # Interpolation - (2) scan line별 관측 시간 차이 보정
            print(f"len of df_ITPL: {len(df_ITPL)}")
            print(df_ITPL)

            tif_center_dt_str = tif_center_dt.replace(microsecond=0).strftime("%Y-%m-%d %H:%M:%S")
            if len(df_ITPL) > 0:
                geo_index = build_geo_index(geoloc_lon, geoloc_lat, sstimes, setimes)
                df_ITPL = refine_df_ITPL_fast(df_ITPL, series_cache, geo_index, tif_center_dt_str)

            if len(df_ITPL) > 0:
                print("[Saving to DB...]")
                save_df_to_DB(df_ITPL, conn)
                print("[Saved]")
            else:
                print("[Nothing to Save...]")

            mark_ITPL_completed(tif_name, TXT_PATH)
            if (i % 10 == 0):
                conn.execute("PRAGMA wal_checkpoint(PASSIVE);")
                # [DEBUGGING]
                pd.set_option("display.max_rows", None)
                pd.set_option("display.max_columns", None)
                pd.set_option("display.expand_frame_repr", False)
                print(df_ITPL)
            print(f"[COMPLETE] processing (ITPL) {tif_name}\n"+"="*80)
            # 보완 필요

        except Exception as e:
            print(f"[Exception] || loc : ITPL() || message : {e}")

def main(TIF_THRES, MMSI_THRES):
    # prepare - (1)
    os.makedirs(ROOT, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")

    conn.execute("""
    CREATE INDEX IF NOT EXISTS idx_MMSI_Date_ITPL_Time
    ON ships_dynamic(MMSI, Date, ITPL, Time);
    """)
    conn.commit()

    # make DB
    # make_all_ship_csv("./AIS", conn) # DB가 없어졌을 경우 등에만 실행해서 생성

    # prepare - (2)
    df = pd.read_csv(METADATA_PATH)
    tif_name_list = list(set(df["tif_name"].dropna().tolist()) - get_marked_TIF_set(TXT_PATH))
    tif_name_list.sort(key = parse_key)
    # print(tif_name_list) # [DEBUGGING]
    if (TIF_THRES == -1):
        TIF_THRES = len(tif_name_list)
    mapped_date_dict = _mapped_date()

    ITPL(TIF_THRES, MMSI_THRES, tif_name_list, mapped_date_dict, df, conn)

    conn.close()
if __name__ == "__main__":
    main(TIF_THRES, MMSI_THRES)
