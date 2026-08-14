#!/usr/bin/env python3

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt


# =========================
# Paths
# =========================
OBS_WEST = Path(
	"./Veals_etal_2026/CoCo/CoCoRaHS_2024_2025_with_FinalPrecip_inches_QCfiltered_West.csv"
)

OBS_EAST = Path(
	"./Veals_etal_2026/CoCo/CoCoRaHS_2024_2025_with_FinalPrecip_inches_QCfiltered_East.csv"
)

SNOWSAFETY_CSV = Path(
	"./SnowSafety_All.csv"
)

OUTPUT_PNG = Path(
	"./stations_us_map_counts.png"
)


# =========================
# Settings
# =========================
THRESH_MM = 0.0
INCH_TO_MM = 25.4

MAP_EXTENT = [-125, -66.5, 29, 50]

TILE_ZOOM = 5
TERRAIN_ALPHA = 0.55

COCO_MARKER_SIZE = 10
SNOWSAFETY_MARKER_SIZE = 12

DPI = 300
FIGSIZE = (13, 8)


# =========================
# Hard-coded SnowSafety lat/lon
# =========================
SNOWSAFETY_LATLON = {
	"MT-SS-BBL": (45.829109, -110.922451),
	"CA-SS-CSSL": (39.3255, -120.3678),
	"WY-SS-GTH": (43.79066, -110.94524),
	"ID-SS-HLY": (43.515967, -114.317825),
	"WY-SS-JHBA": (43.591, -110.734),
	"WY-SS-JHMM": (43.598, -110.848),
	"UT-SS-PVC": (40.344373, -111.606285),
	"UT-SS-CLN": (40.5763, -111.6383),
	"WA-SS-SNQ": (47.4249, -121.4138),
	"UT-SS-SPC": (40.7503, -111.6499),
	"WA-SS-STV": (47.7461, -121.0926),
	"CO-SS-TRD": (37.91278, -107.82181),
}


# =========================
# ESRI shaded relief tiles
# =========================
class EsriShadedRelief(cimgt.GoogleTiles):
	def _image_url(self, tile):
		x, y, z = tile
		return (
			"https://server.arcgisonline.com/ArcGIS/rest/services/"
			f"World_Shaded_Relief/MapServer/tile/{z}/{y}/{x}"
		)


def find_column(df, candidates, label):
	for col in candidates:
		if col in df.columns:
			print(f"Using {label} column: {col}")
			return col

	raise ValueError(
		f"Could not find {label} column.\n"
		f"Available columns are:\n{list(df.columns)}"
	)


def load_coco_station_data():
	df_w = pd.read_csv(OBS_WEST)
	df_e = pd.read_csv(OBS_EAST)

	obs = pd.concat([df_w, df_e], ignore_index=True)

	station_col = find_column(
		obs,
		["StationNumber", "Station", "StationID", "StationName"],
		"CoCo station",
	)

	precip_col = find_column(
		obs,
		["FinalPrecip", "Precipitation", "Precip", "precip"],
		"CoCo precip",
	)

	lat_col = find_column(
		obs,
		["Latitude", "Lat", "latitude"],
		"CoCo latitude",
	)

	lon_col = find_column(
		obs,
		["Longitude", "Lon", "longitude"],
		"CoCo longitude",
	)

	obs[precip_col] = pd.to_numeric(obs[precip_col], errors="coerce")
	obs = obs.dropna(subset=[station_col, precip_col, lat_col, lon_col]).copy()

	obs["precip_mm"] = obs[precip_col] * INCH_TO_MM
	obs["event"] = obs["precip_mm"] > THRESH_MM

	stations = (
		obs.groupby(station_col)
		.agg({
			lat_col: "first",
			lon_col: "first",
			"event": "sum",
		})
		.reset_index()
		.rename(columns={
			station_col: "StationNumber",
			lat_col: "Latitude",
			lon_col: "Longitude",
			"event": "event_count",
		})
	)

	stations["event_count"] = stations["event_count"].astype(int)

	return stations


def load_snowsafety_station_data():
	obs = pd.read_csv(SNOWSAFETY_CSV)

	station_col = find_column(
		obs,
		["Site", "StationNumber", "Station", "StationID", "StationName", "site"],
		"SnowSafety station",
	)

	precip_col = find_column(
		obs,
		[
			"24hrSWE_cm",
			"FinalPrecip",
			"FinalPrecip_mm",
			"Precip_mm",
			"precip_mm",
			"Precip",
			"precip",
			"Precipitation",
			"Precipitation_mm",
		],
		"SnowSafety precip",
	)

	obs[precip_col] = pd.to_numeric(obs[precip_col], errors="coerce")
	obs = obs.dropna(subset=[station_col, precip_col]).copy()

	if precip_col == "24hrSWE_cm" or precip_col.lower().endswith("_cm"):
		obs["precip_mm"] = obs[precip_col] * 10.0
	elif "inch" in precip_col.lower() or precip_col.lower().endswith("_in"):
		obs["precip_mm"] = obs[precip_col] * INCH_TO_MM
	else:
		obs["precip_mm"] = obs[precip_col]

	obs["event"] = obs["precip_mm"] > THRESH_MM

	stations = (
		obs.groupby(station_col)["event"]
		.sum()
		.astype(int)
		.rename("event_count")
		.reset_index()
		.rename(columns={station_col: "StationNumber"})
	)

	latitudes = []
	longitudes = []

	for station in stations["StationNumber"]:
		if station not in SNOWSAFETY_LATLON:
			raise ValueError(f"Missing hard-coded lat/lon for SnowSafety station: {station}")

		lat, lon = SNOWSAFETY_LATLON[station]
		latitudes.append(lat)
		longitudes.append(lon)

	stations["Latitude"] = latitudes
	stations["Longitude"] = longitudes

	return stations


def main():
	coco_stations = load_coco_station_data()
	snowsafety_stations = load_snowsafety_station_data()

	print(f"Loaded {len(coco_stations)} CoCo stations")
	print(f"Loaded {len(snowsafety_stations)} HighMountain stations")
	print(f"CoCo min/max count: {coco_stations['event_count'].min()} / {coco_stations['event_count'].max()}")
	print(f"SnowSafety min/max count: {snowsafety_stations['event_count'].min()} / {snowsafety_stations['event_count'].max()}")

	# =========================
	# Discrete bins
	# =========================
	bounds = [5, 11, 21, 31, 41, 51, 61, 71, 81, 9999]

	cmap = plt.get_cmap("plasma", len(bounds) - 1)
	norm = mcolors.BoundaryNorm(bounds, cmap.N)

	labels = [
		"5–10", "11–20", "21–30", "31–40",
		"41–50", "51–60", "61–70", "71–80", ">80"
	]

	# =========================
	# Map
	# =========================
	tiler = EsriShadedRelief()
	map_crs = tiler.crs
	data_crs = ccrs.PlateCarree()

	fig = plt.figure(figsize=FIGSIZE)
	ax = plt.axes(projection=map_crs)

	ax.set_extent(MAP_EXTENT, crs=data_crs)

	ax.add_image(tiler, TILE_ZOOM, alpha=TERRAIN_ALPHA, zorder=0)

	ax.add_feature(cfeature.OCEAN, facecolor="white", alpha=0.45, zorder=1)
	ax.add_feature(cfeature.LAKES, facecolor="white", alpha=0.40, zorder=2)
	ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=3)
	ax.add_feature(cfeature.BORDERS, linewidth=0.5, zorder=3)
	ax.add_feature(cfeature.STATES, linewidth=0.45, edgecolor="black", zorder=4)

	sc = ax.scatter(
		coco_stations["Longitude"],
		coco_stations["Latitude"],
		c=coco_stations["event_count"],
		cmap=cmap,
		norm=norm,
		s=COCO_MARKER_SIZE,
		marker="o",
		edgecolors="black",
		linewidths=0.25,
		transform=data_crs,
		zorder=5,
		label="CoCo",
	)

	ax.scatter(
		snowsafety_stations["Longitude"],
		snowsafety_stations["Latitude"],
		c=snowsafety_stations["event_count"],
		cmap=cmap,
		norm=norm,
		s=SNOWSAFETY_MARKER_SIZE,
		marker="D",
		edgecolors="black",
		linewidths=0.65,
		transform=data_crs,
		zorder=6,
		label="HighMountain",
	)

	# =========================
	# Gridlines with small labels
	# =========================
	gl = ax.gridlines(
		crs=data_crs,
		draw_labels=True,
		linewidth=0.3,
		color="gray",
		alpha=0.35,
		linestyle="--",
		zorder=7,
	)

	gl.top_labels = False
	gl.right_labels = False
	gl.xlabel_style = {"size": 6}
	gl.ylabel_style = {"size": 6}

	# =========================
	# Horizontal colorbar
	# =========================
	cbar = plt.colorbar(
		sc,
		ax=ax,
		orientation="horizontal",
		pad=0.05,
		shrink=0.8,
		aspect=35,
	)

	cbar.set_ticks([(bounds[i] + bounds[i + 1]) / 2 for i in range(len(bounds) - 1)])
	cbar.set_ticklabels(labels)
	cbar.set_label("Number of Observations with Precip > 0.0 mm")

	ax.legend(
		loc="lower left",
		fontsize=8,
		frameon=True,
		framealpha=0.85,
	)

	#ax.set_title("Station Wet-Day Counts (>0 mm)", fontsize=14)

	plt.subplots_adjust(bottom=0.14)
	plt.savefig(OUTPUT_PNG, dpi=DPI, bbox_inches="tight")
	plt.close()

	print(f"Saved map to: {OUTPUT_PNG}")


if __name__ == "__main__":
	main()
