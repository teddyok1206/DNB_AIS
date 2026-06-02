import numpy as np
import pandas as pd
from netCDF4 import Dataset
from osgeo import gdal
from scipy.interpolate import griddata
from pyproj import Transformer
from scipy.spatial import cKDTree  # Added for distance calculation
from astropy.time import Time
from astropy.time import TimeDelta
import astropy.units as u
from shapely.geometry import Polygon
import os
import sys

TYPE = "JPSS-2" # "S-NPP" "JPSS-1" "JPSS-2"
GEOTIFF_PATH = "/GeoTIFF/" # "/GeoTIFF/"
PIXEL_SIZE = 0.003 # degree

# Spec of VIIRS
L_MIN = 3.0e-9
L_MAX = 2.0e-2

ROOT = f"/Volumes/SAMSUNG/{TYPE}_VIIRS"
DNB_dict = {"S-NPP" : ["VNP02DNB", "VNP03DNB"],
            "JPSS-1" : ["VJ102DNB", "VJ103DNB"],
            "JPSS-2" : ["VJ202DNB", "VJ203DNB"]}
DNB_02 = DNB_dict[TYPE][0]
DNB_03 = DNB_dict[TYPE][1]

# TAI93 epoch
t0 = Time("1993-01-01 00:00:00", scale="tai")
errors = dict()

# # TEST_2
# def log_sigmoid(x):
#     #x = np.maximum(x, 1e-30)  # log 안정화
#     return 1.0 / (1.0 + np.exp(-2.0 * (np.log10(x) + 9.0)))


class FileName:
    def __init__(self, name):
        self.raw_name = name
        tup = tuple(name.split('.'))[:4] # ignore processing timestamp
        if tup[0] == DNB_02:
            self.file_type = 2 # 2 for dnb_02
        elif tup[0] == DNB_03:
            self.file_type = 3 # 3 for dnb_03
        else:
            self.file_type = 0 # 0 for ERROR
        self.prefix_1 = tup[1] # ex) A2024041 | YYYYDDD (DDD : # of DAY)
        self.prefix_2 = tup[2] # ex) 1512 | HHMM
        self.prefix_3 = tup[3] # ex) 002

    def SetScanInfo(self, args):
        self.scan_start_time = args[0] * u.s # [TAI93 sec]
        self.scan_end_time = args[1] * u.s # [TAI93 sec]
        self.area = args[2]

        self.duration = self.scan_end_time - self.scan_start_time # [sec]

        self.scan_start_HHMM = ((t0 + self.scan_start_time).utc).strftime("%H:%M") # HHMM
        self.scan_start_HHMMSS = ((t0 + self.scan_start_time).utc).strftime("%H:%M:%S") # HHMMSS

        # # [DEBUGGING]
        # if self.scan_start_HHMM != self.prefix_2:
        #     print(f"[{self.raw_name}]\n : prefix_2 is [{self.prefix_2}]\n : scan_start_HHMM is [{self.scan_start_HHMM}]")

    def IsPair(self, other):
        cond_0 = (self.file_type != other.file_type)
        cond_1 = (self.prefix_1 == other.prefix_1)
        cond_2 = (self.prefix_2 == other.prefix_2)
        cond_3 = (self.prefix_3 == other.prefix_3)
        return (cond_0 and cond_1 and cond_2 and cond_3)

    def PrintName(self):
        print("=" * 80)
        print("name : " + self.raw_name)
        print("type :", self.file_type)
        print("pfx1 : " + self.prefix_1)
        print("pfx2 : " + self.prefix_2)
        print("pfx3 : " + self.prefix_3)

    def to_dict(self):
        return {
            "raw_name": self.raw_name,
            "file_type": self.file_type,
            "prefix_1": self.prefix_1,
            "prefix_2": self.prefix_2,
            "prefix_3": self.prefix_3,
            "scan_start_time [TAI 93]": self.scan_start_time.to_value(u.s) if hasattr(self, "scan_start_time") else None,
            "scan_end_time [TAI 93]": self.scan_end_time.to_value(u.s) if hasattr(self, "scan_end_time") else None,
            "duration [sec]": self.duration.to_value(u.s) if hasattr(self, "duration") else None,
            "scan_start_HHMM": getattr(self, "scan_start_HHMM", None),
            "scan_start_HHMMSS": getattr(self, "scan_start_HHMMSS", None),
            "area_wkt": Polygon(self.area).wkt if hasattr(self, "area") else None,}

def DEBUG_print_all_list(dnb_02_list, dnb_03_list):
    for inst in dnb_02_list:
        inst.PrintName()
    for inst in dnb_03_list:
        inst.PrintName()
    print("=" * 80)

def filter_by_tif_name(tif_name : str)->bool:
    _, HHMM, _ = tif_name.split("_")
    return ((int(HHMM) < 1550) or (int(HHMM) > 1810))

def GetScanInfo(VJ103dnb_path:str)->tuple[float, float, list]:
    print("  [open]", os.path.basename(VJ103dnb_path), flush=True)
    with Dataset(VJ103dnb_path, 'r') as VJ103:
        print("  [vars]", flush=True)
        s = VJ103['/scan_line_attributes/scan_start_time']
        e = VJ103['/scan_line_attributes/scan_end_time']

        print("  [attrs]", flush=True)
        lon = list(VJ103.getncattr("GRingPointLongitude"))
        lat = list(VJ103.getncattr("GRingPointLatitude"))

        print("  [index]", flush=True)
        return float(s[0]), float(e[-1]), list(zip(lon, lat))

def check_non_pair(dnb_02_list, dnb_03_list):
    only_in_dnb_02 = list(set([(x.prefix_1, x.prefix_2, x.prefix_3) for x in dnb_02_list]) - set([(x.prefix_1, x.prefix_2, x.prefix_3) for x in dnb_03_list]))
    only_in_dnb_03 = list(set([(x.prefix_1, x.prefix_2, x.prefix_3) for x in dnb_03_list]) - set([(x.prefix_1, x.prefix_2, x.prefix_3) for x in dnb_02_list]))
    if (len(only_in_dnb_02) == 0 and len(only_in_dnb_03) == 0):
        print("[check_non_pair] Two lists are pair-wise.")
        print(f"[check_non_pair] dnb_02_list length : {len(dnb_02_list)}")
        print(f"[check_non_pair] dnb_03_list length : {len(dnb_02_list)}")
        return 0, 0

    with open("OnlyIn.txt", "w") as f:
        only_in_dnb_02.sort()
        only_in_dnb_03.sort()
        f.write("Only in dnb_02_list:\n")
        for item in only_in_dnb_02:
            f.write(f"{item[0]}.{item[1]}.{item[2]}\n")

        f.write("\nOnly in dnb_03_list:\n")
        for item in only_in_dnb_03:
            f.write(f"{item[0]}.{item[1]}.{item[2]}\n")

    print("=" * 80)
    print(f"dnb_02_list length : {len(dnb_02_list)}")
    print(f"dnb_03_list length : {len(dnb_03_list)}")
    print(f"Can_Process length : {len(dnb_02_list) - len(only_in_dnb_02)}")
    print(f"dnb_02_only length : {len(only_in_dnb_02)}")
    print(f"dnb_03_only length : {len(only_in_dnb_03)}")
    return only_in_dnb_02, only_in_dnb_03

def done_already():
    tif_list = [name for name in os.listdir(ROOT + GEOTIFF_PATH) if (not name.startswith('.') and len(name) > 15)]
    return tif_list

def make_dnb_list():

    # To handle error cases manually
    # ls1 = ["VJ102DNB.A2025324.1542.002.2025324230124.nc", "VJ102DNB.A2025324.1548.002.2025325002030.nc", "VJ102DNB.A2025356.1730.002.2025357002436.nc"]
    # ls2 = ["VJ103DNB.A2025324.1542.002.2025324222834.nc", "VJ103DNB.A2025324.1548.002.2025324223029.nc", "VJ103DNB.A2025356.1730.002.2025356235009.nc"]
    dnb_02_list = [FileName(name) for name in os.listdir(ROOT + f'/{DNB_02}/') if (not name.startswith('.') and len(name) > 20)] # and (name in ls1)]
    dnb_03_list = [FileName(name) for name in os.listdir(ROOT + f'/{DNB_03}/') if (not name.startswith('.') and len(name) > 20)] # and (name in ls2)]

    # Sort by Year, Day, Time (each corresponds to prefixes)
    dnb_02_list.sort(key = lambda x: (int(x.prefix_1[1:]), int(x.prefix_2), int(x.prefix_3)))
    dnb_03_list.sort(key = lambda x: (int(x.prefix_1[1:]), int(x.prefix_2), int(x.prefix_3)))

    # Check non-pair elements
    only_in_dnb_02, only_in_dnb_03 = check_non_pair(dnb_02_list, dnb_03_list)

    # [TESTING]
    # DEBUG_print_all_list(dnb_02_list, dnb_03_list)

    # [DEBUGGING]
    # if (len(dnb_02_list) != len(dnb_03_list)):
    #     print("[ERROR] Different # of elements in each folder... exit()")
    #     sys.exit()

    return dnb_02_list, dnb_03_list, only_in_dnb_02, only_in_dnb_03

def create_img_and_metadata(VJ102dnb_path, d3, VJ103dnb_path, output_tif_path):
    """
    Extract and resample VIIRS DNB data to a regular grid and save as a GeoTIFF.
    Ensures areas outside original data coverage are set to 0.
    """
    # Step 1: Read data from VJ102DNB file
    print(f"[Step 1] Reading DNB observations from {DNB_02} file...")
    with Dataset(VJ102dnb_path, 'r') as VJ102:
        dnb_observations = VJ102['/observation_data/DNB_observations'][:]
    print("[Step 1] completed.")

    # Step 2: Read geolocation data from VJ103DNB file
    print(f"[Step 2] Reading geolocation data from {DNB_03} file...")
    with Dataset(VJ103dnb_path, 'r') as VJ103:
        latitude = VJ103['/geolocation_data/latitude'][:]
        longitude = VJ103['/geolocation_data/longitude'][:]
        print("[Step 2.5] SetScanInfo started.")
        s = VJ103['/scan_line_attributes/scan_start_time']
        e = VJ103['/scan_line_attributes/scan_end_time']
        lon = list(VJ103.getncattr("GRingPointLongitude"))
        lat = list(VJ103.getncattr("GRingPointLatitude"))
        info = tuple([float(s[0]), float(e[-1]), list(zip(lon, lat))])
        d3.SetScanInfo(info)
        print("[Step 2.5] SetScanInfo completed.")
    print("[Step 2] completed.")

    # Step 3: Handle invalid data (replace with NaN)
    print("[Step 3] Handling invalid data...")
    dnb_observations = np.ma.filled(dnb_observations, np.nan)
    latitude = np.ma.filled(latitude, np.nan)
    longitude = np.ma.filled(longitude, np.nan)
    print("[Step 3] completed.")

    # Step 4: Flatten and filter valid data points
    print("[Step 4] Flattening arrays for interpolation...")
    rows, cols = dnb_observations.shape
    lats = latitude.flatten()
    lons = longitude.flatten()
    values = dnb_observations.flatten()

    mask = ~np.isnan(values) & ~np.isnan(lats) & ~np.isnan(lons)
    lats = lats[mask].astype(float)
    lons = lons[mask].astype(float)
    values = values[mask].astype(float)
    print("[Step 4] completed.")

    # We Don't need to Convert!
    # # Step 5: Convert to Web Mercator (EPSG:3857)
    # print("[Step 5] Converting coordinates to EPSG:3857...")
    # transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    # x_mercator, y_mercator = transformer.transform(lons, lats)
    # print("[Step 5] completed.")

    print("[Step 5] SKIP")

    # Step 6: Define output grid parameters
    print("[Step 6] Defining output grid...")
    lon_min, lon_max = np.min(lons), np.max(lons)
    lat_min, lat_max = np.min(lats), np.max(lats)

    pixel_size = PIXEL_SIZE
    lon_grid = np.arange(lon_min, lon_max, pixel_size)
    lat_grid = np.arange(lat_max, lat_min, -pixel_size)
    grid_lon, grid_lat = np.meshgrid(lon_grid, lat_grid)
    print("[Step 6] completed.")

    # Step 7: Perform interpolation with distance masking
    print("[Step 7] Performing interpolation with distance masking...")
    grid_brightness = griddata(
        (lons, lats), values, (grid_lon, grid_lat),
        method='linear', fill_value=np.nan
    )
    # bottleneck START
    print("[FLAG 1] : bottleneck START")
    # Create KDTree and calculate distances
    tree = cKDTree(np.column_stack((lons, lats)))
    print("[FLAG 2] : np.column_stack START")
    grid_points = np.column_stack((grid_lon.ravel(), grid_lat.ravel()))
    print("[FLAG 3] : tree.query START")
    distances, _ = tree.query(grid_points, k=1, workers=-1)
    print("[FLAG 4] : distances.reshape START")
    distance_grid = distances.reshape(grid_lon.shape)
    print("[FLAG 5] : bottleneck END")
    # bottleneck END

    # Apply distance threshold (2*pixel size)
    grid_brightness[distance_grid > 3*pixel_size] = np.nan
    print("[Step 7] completed.")

    # Step 8: Apply log scale and normalization
    print("[Step 8] Processing data values...")
    grid_brightness = np.nan_to_num(grid_brightness, nan=0.0)

    # TEST_4
    grid_brightness = (2/np.pi)*np.arctan(grid_brightness / 1e-9)

    # # TEST_3
    # grid_brightness = grid_brightness / (grid_brightness + 1e-7)

    # # TEST_2
    # temp = np.zeros_like(grid_brightness, dtype=float)
    # mask = grid_brightness > 0
    # temp[mask] = log_sigmoid(grid_brightness[mask])
    # grid_brightness = temp

    # grid_brightness = np.log1p(grid_brightness)

    # bright_cap = np.percentile(grid_brightness[grid_brightness > 0], 99)
    # grid_brightness = np.minimum(grid_brightness, bright_cap)

    # min_val = np.min(grid_brightness)
    # max_val = np.max(grid_brightness)
    # grid_brightness = ((grid_brightness - min_val) / (max_val - min_val)).astype(np.float32)
    print("[Step 8] completed.")

    # Step 9: Create GeoTIFF
    print("[Step 9] Writing GeoTIFF...")
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(output_tif_path, grid_brightness.shape[1],
                            grid_brightness.shape[0], 1, gdal.GDT_Float32)

    spatial_ref = gdal.osr.SpatialReference()
    spatial_ref.ImportFromEPSG(4326)
    dataset.SetGeoTransform((lon_min, pixel_size, 0, lat_max, 0, -pixel_size))
    dataset.SetProjection(spatial_ref.ExportToWkt())
    dataset.GetRasterBand(1).WriteArray(grid_brightness)
    dataset.FlushCache()
    dataset = None
    driver = None
    print(f"[END] Output saved to: {output_tif_path}")

def main():
    dnb_02_list, dnb_03_list, only_in_dnb_02, only_in_dnb_03 = make_dnb_list()
    done_tif_set = set(done_already())
    errors = dict()

    # To Handle error cases Manually
    # dnb_02_list = [FileName("VJ102DNB.A2025195.1606.002.2025196023633.nc"), FileName("VJ102DNB.A2025195.1742.002.2025196024906.nc")]
    # dnb_03_list = [FileName("VJ103DNB.A2025195.1606.002.2025196021225.nc"), FileName("VJ103DNB.A2025195.1742.002.2025196022429.nc")]

    if (only_in_dnb_02 == 0 and only_in_dnb_03 == 0): # start processing only if all files are pairwise
        for i in range(len(dnb_02_list)): # substitute len(dnb_02_list) with constant num when testing
            d2 = dnb_02_list[i]
            d3 = dnb_03_list[i]
            if d2.IsPair(d3):
                result_name = f"{d2.prefix_1}_{d2.prefix_2}_{d2.prefix_3}.tif"
                # if filter_by_tif_name(result_name):
                #     continue
                print("=" * 80)
                print(f"-*-*- ITERATION COUNT : [{i + 1}] -*-*-")
                try:
                    if result_name not in done_tif_set:
                        create_img_and_metadata(ROOT + f'/{DNB_02}/'+d2.raw_name, d3, ROOT + f'/{DNB_03}/' + d3.raw_name, ROOT + GEOTIFF_PATH + result_name)
                    else:
                        pass
                except Exception as e:
                    errors[result_name] = e
            len_error = len(errors.keys())
            if (len_error != 0):
                print("=" * 40)
                for key in errors.keys():
                    print(f"error occured while handling [{key}]\nerror message : [{errors[key]}]")
                print("=" * 40)
        print("=" * 80)

    # df = pd.DataFrame([f.to_dict() for f in dnb_03_list])
    # print("df created.")
    # print("Saving df...")
    # df.to_csv("./metadata_JPSS-2.csv", index = False)
    # print("df saved.")

if __name__ == "__main__":
    main()