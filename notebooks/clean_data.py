#import required modules
import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# define working directory
wd = r"C:\Users\sanne\integrated-modelling-in-hydrology\1_stream\data\cru_ts4.09.1901.2024.pre.dat.nc"

input_folder = os.path.join(wd, "Input")
output_folder = os.path.join(wd, "Output")

# Open the netCDF file
pre_data = xr.open_dataset(os.path.join(input_folder, "cru_ts4.09.1901.2024.pre.dat.nc" ))

subset = pre_data.sel(lon=slice(8.0, 16.5), lat=slice(48.0, 56.0))
subset.to_netcdf("precipitation_subset.dat.nc")

prec = pre_data.pre # Extract temperature data
prec_catchment = pre_data.pre.sel(lon=slice(8.0,16.5), lat=slice(48.0,56.0)) # Select catchment area
prec_catchment[0].plot()

prec = prec.values # Convert to numpy array

import geopandas as gpd

catchment = gpd.read_file("C:\Users\sanne\integrated-modelling-in-hydrology\1_stream\data\Elbe_catchment.gpkg")

# %%
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import shapely.geometry as geom
import numpy as np

nan_mask = prec_catchment.isnull().all(dim="time")

# Extract lat/lon arrays
lats = prec_catchment.lat.values
lons = prec_catchment.lon.values

# Maak een lijst met punten voor alle gridcellen
points = []
nan_values = []

for i, lat in enumerate(lats):
    for j, lon in enumerate(lons):
        points.append(geom.Point(lon, lat))
        nan_values.append(bool(nan_mask.values[i, j]))

# GeoDataFrame maken
gdf = gpd.GeoDataFrame({"is_nan": nan_values}, geometry=points, crs="EPSG:4326")

# Alleen NaN‑punten selecteren
nan_points = gdf[gdf.is_nan]

catchment = gpd.read_file(r"C:\Users\sanne\integrated-modelling-in-hydrology\1_stream\data\Elbe_catchment.gpkg")

# Reprojecteren naar lat/lon indien nodig
if catchment.crs != "EPSG:4326":
    catchment = catchment.to_crs("EPSG:4326")

fig, ax = plt.subplots(figsize=(8, 8))

# Stroomgebied
catchment.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=1)

# NaN‑punten
nan_points.plot(ax=ax, color="red", markersize=20, label="NaN cells")

plt.title("NaN grid cells within CRU precipitation data")
plt.legend()
plt.show()