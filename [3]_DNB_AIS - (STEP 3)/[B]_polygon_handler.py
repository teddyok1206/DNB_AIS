import os
import sys
import rasterio
import numpy as np
from rasterio.features import shapes
from shapely.geometry import shape
from shapely.geometry import LineString
from shapely.geometry import MultiPolygon
from shapely.ops import unary_union
from shapely.prepared import prep
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd

TYPE = "JPSS-2" # "S-NPP" "JPSS-1" "JPSS-2"

META_PATH = f"./metadata_{TYPE}.csv"
ROOT = f"/Volumes/SAMSUNG/{TYPE}_VIIRS"
SRC_DIR = ROOT + "/GeoTIFF"
DST_O = SRC_DIR + "/OVERLAP_O"
DST_X = SRC_DIR + "/OVERLAP_X"

EEZ_PATH = "./eez_12nm"
EEZ_12nm_PATH = "./eez"

AUTO_SAVE_INTERVAL = 5

def import_eez_12nm_and_eez():
    eez_12nm = gpd.read_file(EEZ_PATH + "/eez_12nm.shp").to_crs("EPSG:4326")
    eez = gpd.read_file(EEZ_12nm_PATH + "/eez.shp").to_crs("EPSG:4326")

    ## [VISUALIZING]
    # ax = eez_12nm.plot()
    # plt.show()

    # 각각을 하나의 geometry로
    geom_12nm = eez_12nm.union_all()
    geom_eez = eez.union_all()

    # 합집합 (영해 + EEZ)
    geom_union = unary_union([geom_12nm, geom_eez])

    gdf_union = gpd.GeoDataFrame(
        geometry=[geom_union],
        crs="EPSG:4326"
    )

    return gdf_union

def make_tif_list()->list:
    def parse_key(fname:str)->tuple[int, int]:
        base = fname.replace(".tif", "")
        date, hhmm, _ = base.split("_")
        return (int(date[1:]), int(hhmm))

    tif_list = [name for name in os.listdir(ROOT + '/GeoTIFF') if (not name.startswith('.')) and (len(name) > 10)]

    # Sort by Year, Day, Time (each corresponds to prefixes)
    tif_list.sort(key = parse_key)
    return tif_list

def tif_valid_footprint(tif_path:str):
    with rasterio.open(tif_path) as src:
        arr = src.read(1, masked=False)
        transform = src.transform
        crs = src.crs
        mask = arr != 0

        # shapes: mask==True 영역을 폴리곤으로 추출
        geoms = []
        for geom, val in shapes(mask.astype(np.uint8), mask=mask, transform=transform):
            if val == 1:
                geoms.append(shape(geom))

        if not geoms:
            return None, crs

        return unary_union(geoms), crs # unary_union(geoms) -> is being used as footprint

def visualize_plt(footprint, crs:str):
    fig, ax = plt.subplots(figsize=(6,6))

    if footprint.geom_type == "Polygon":
        x, y = footprint.exterior.xy
        ax.plot(x, y, color="red", linewidth=2)
    elif footprint.geom_type == "MultiPolygon":
        for poly in footprint.geoms:
            x, y = poly.exterior.xy
            ax.plot(x, y, color="red", linewidth=2)

    ax.set_title("Valid pixel footprint (raw CRS)")
    ax.set_aspect("equal")
    plt.show()

def export_as_geojson(footprint, output_path:str)->bool:
    try:
        if isinstance(footprint[0], MultiPolygon):
            final_footprint = max(footprint[0].geoms, key=lambda g: g.area)
        else:
            final_footprint = footprint[0]
        boundary_line = LineString(final_footprint.exterior.coords)
        gdf = gpd.GeoDataFrame(geometry=[boundary_line], crs="EPSG:4326")
        gdf.to_file(output_path, driver="GeoJSON")
        print("[CONVERT SUCCESS]")
        return True
    except Exception as e:
        print(f"[export_as_geojson] Exception [{e}] has occured in processing " + output_path)
        return False

def lonlat_minmax(df:pd.DataFrame)->None:
    print("Pre-processing data...")
    df = df[df["KR_Sea_overlap"] > 0.0] # filtering
    #df = df[df["tif_name"] != "A2025249_1548_002.tif"] -> 반대편으로 넘어간 유일한 데이터
    idx_lon_min_all = df.index[df["lon_min"] == df["lon_min"].min()]
    idx_lon_max_all = df.index[df["lon_max"] == df["lon_max"].max()]
    idx_lat_min_all = df.index[df["lat_min"] == df["lat_min"].min()]
    idx_lat_max_all = df.index[df["lat_max"] == df["lat_max"].max()]
    tif_lon_min_all = df.loc[idx_lon_min_all, "tif_name"].tolist()
    tif_lon_max_all = df.loc[idx_lon_max_all, "tif_name"].tolist()
    tif_lat_min_all = df.loc[idx_lat_min_all, "tif_name"].tolist()
    tif_lat_max_all = df.loc[idx_lat_max_all, "tif_name"].tolist()
    result_dict = {"lon_min" : zip(idx_lon_min_all, tif_lon_min_all), "lon_max" : zip(idx_lon_max_all, tif_lon_max_all),
                   "lat_min" : zip(idx_lat_min_all, tif_lat_min_all), "lat_max" : zip(idx_lat_max_all, tif_lat_max_all)}
    print("Pre-processing has been completed")
    for key in result_dict.keys():
        print("="*80)
        print("-*-*-"*5 + f"ITERATION for : [{key}]" + 5*"-*-*-")
        pairs = list(result_dict[key])
        path = f"./layers_{TYPE}/{key}/"
        time_dict = dict()
        for pair in pairs:
            print(f"-*-*- sub-ITERATION for : [{pair[1]}] -*-*-")
            footprint = tif_valid_footprint(SRC_DIR + '/' + pair[1])
            complete_path = path+f"{key}_{pair[1]}"
            os.makedirs(os.path.dirname(complete_path), exist_ok=True)
            if export_as_geojson(footprint, complete_path):
                time_dict[pair[1]] = df.loc[pair[0], "scan_start_HHMMSS"]
            else:
                time_dict[pair[1]] = "Failed to export as geojson."
        print("="*40)
        txt_path = path + f"scan_times_{key}.txt"
        print(f"saving to " + txt_path)
        with open(txt_path, "w", encoding="utf-8") as f:
            for name, time in time_dict.items():
                f.write(f"{name}\t{time}\n")
        print("saved.")
    print("="*80)

def check_raw_tif_name(raw_name:str, tif_name:str)->bool:
    _, raw_pfx_1, raw_pfx_2, _, _, _ = raw_name.split('.')
    tif_pfx_1, tif_pfx_2, _ = tif_name.split('_')
    return (raw_pfx_1 == tif_pfx_1) and (raw_pfx_2 == tif_pfx_2)

def move_to_dir(df):
    os.makedirs(DST_O, exist_ok=True)
    os.makedirs(DST_X, exist_ok=True)

    set_O = set(df.loc[df["KR_Sea_overlap"] > 0.0, "tif_name"].tolist())
    print(f"Length of set_O : {len(set_O)}")

    for fname in os.listdir(SRC_DIR):
        src_path = os.path.join(SRC_DIR, fname)
        if not os.path.isfile(src_path):
            continue
        if fname in set_O:
            shutil.move(src_path, os.path.join(DST_O, fname))
        else:
            shutil.move(src_path, os.path.join(DST_X, fname))

def main():
    errors = dict()
    df = pd.read_csv(META_PATH)
    # df["KR_Sea_overlap"] = None
    required_cols = ["raw_name", "tif_name", "KR_Sea_overlap", "lon_min", "lon_max", "lat_min", "lat_max"]
    for c in required_cols:
        if c not in df.columns:
            df[c] = pd.NA

    KR_sea = import_eez_12nm_and_eez()
    g2 = KR_sea.to_crs("EPSG:5179").geometry.iloc[0]
    g2_prep = prep(g2) # acceleration
    g2_area = g2.area

    tif_to_idx = {}
    dup_tifs = set()
    for idx, name in df["tif_name"].items():
        if pd.isna(name):
            continue
        name = str(name)
        if name in tif_to_idx:
            dup_tifs.add(name)
        else:
            tif_to_idx[name] = idx

    if dup_tifs:
        raise ValueError(f"Duplicate tif_name in CSV (must be unique): {sorted(list(dup_tifs))[:10]} ...")

    # [TESTING]
    #tif_list = tif_list[:200]
    tif_list = make_tif_list()
    processed = 0

    for tif_name in tif_list:
        idx = tif_to_idx.get(tif_name)
        if idx is None:
            errors[tif_name] = "tif_name not found in CSV"
            continue

        already = (
            pd.notna(df.at[idx, "KR_Sea_overlap"]) and
            pd.notna(df.at[idx, "lon_min"]) and
            pd.notna(df.at[idx, "lon_max"]) and
            pd.notna(df.at[idx, "lat_min"]) and
            pd.notna(df.at[idx, "lat_max"])
        )

        if already:
            print(f"Overlap data for [{tif_name}] is already calculated (row={idx})")
            continue

        raw_name = df.loc[idx, "raw_name"]
        if not check_raw_tif_name(raw_name, tif_name):
            continue

        if (idx % AUTO_SAVE_INTERVAL == 1): # auto-saving
            print("="*80)
            print("[Auto-Saving...]")
            df.to_csv(META_PATH, index=False)
            print("="*80)
        print("="*80)
        print(f"-*-*- ITERATION COUNT : [{idx}] -*-*-")

        print(tif_name, f"(row={idx})")

        try:
            footprint, crs = tif_valid_footprint(SRC_DIR + '/' + tif_name)
            if footprint is None:
                errors[tif_name] = "No valid footprint (all pixels might be 0)"
                continue

            lon_min, lat_min, lon_max, lat_max = footprint.bounds

            g1 = gpd.GeoSeries([footprint], crs="EPSG:4326").to_crs("EPSG:5179").iloc[0]

            if g2_prep.intersects(g1):
                inter = g2.intersection(g1)
                pct = 100.0 * inter.area / g2_area
            else:
                pct = 0.0

            print(f"Overlap(%) of KR Sea within footprint: {pct:.2f}%")

            print("Saving datas to df...")
            df.loc[idx, "tif_name"] = tif_name
            df.loc[idx, "KR_Sea_overlap"] = pct
            df.loc[idx, "lon_min"] = lon_min
            df.loc[idx, "lon_max"] = lon_max
            df.loc[idx, "lat_min"] = lat_min
            df.loc[idx, "lat_max"] = lat_max
            print("completed.")

            processed += 1

        except Exception as e:
            errors[tif_name] = f"Exception: {e}"

    print("="*80)
    df.to_csv(META_PATH, index=False)
    print(f"df saved. processed_new={processed}, total_tifs_seen={len(tif_list)}")
    print("=" * 40)
    print(f"errors({len(errors)}):")
    for k, v in list(errors.items())[:30]:
        print(f"- {k}: {v}")
    print("=" * 40)
    return df

if __name__ == "__main__":
    result_df = main()
    print('[main() completed. lonlat_minmax() started.]')
    lonlat_minmax(result_df)
    # print('[lonlat_minmax(result_df) completed. move_to_dir() started.]')
    # move_to_dir(result_df)
    # print('[move_to_dir(result_df)] completed.')
    # print("ALL FUNCTIONS COMPLETED!")