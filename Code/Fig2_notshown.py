#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


# -----------------------------------------------------------------------------
# Version 14 change
# -----------------------------------------------------------------------------
# Snow Safety (SS/SSDS) observations are assigned to the LOCAL calendar date at
# the observation-ending time: CA-* and WA-* sites use America/Los_Angeles; all
# other SS sites use America/Denver. Each site contributes at most once per local
# date after duplicate date/site rows are collapsed.
#
# V13 behavior retained: On every SS/SSDS local date with precipitation measured at
# at least one site, the complete SS site roster is represented. Any site without a
# row on that date is inserted with 0.0 mm before the daily spatial mean is calculated.
# This zero-fill rule is applied consistently to observed and model daily means.
#
# NEW IN V14: CO-SS-TRD is excluded completely from SS and SSDS before the site roster,
# zero-fill grid, daily means, or running accumulations are built. SS/SSDS therefore
# use an 11-site roster. CoCo date handling and aggregation are unchanged.


# -----------------------------------------------------------------------------
# User settings
# -----------------------------------------------------------------------------
START_DATE = "2024-11-15"
END_DATE = "2025-04-01"

OUTPUT_DIR = Path("/uufs/chpc.utah.edu/common/home/steenburgh-group12/peter/Plots_Test6")
OUTPUT_FIG = OUTPUT_DIR / "CoCo_SS_running_accumulation_5panel_v15_20241115_20250401.png"

# Exact requested figure size. 6.5 / 9 = 0.7222, effectively preserving
# the requested 0.72:1 horizontal:vertical aspect ratio.
FIG_WIDTH = 6.5
FIG_HEIGHT = 9.0
FIGSIZE = (FIG_WIDTH, FIG_HEIGHT)

DPI = 200
LINEWIDTH = 1.0
OBS_LINEWIDTH = 1.7
YMAX_DEFAULT = 400.0

# Keep the same known-bad observation-ending dates excluded by the source scripts.
EXCLUDED_OBS_END_DATES = {
	"2024-11-07",
	"2025-01-10",
	"2025-02-13",
	"2025-02-14",
	"2025-04-14",
	"2025-04-15",
}

# Snow Safety sites excluded completely from SS and SSDS analysis.
SS_EXCLUDED_SITES = {"CO-SS-TRD"}

# Panel order from top to bottom.
PANEL_CONFIGS = [
{
		"title": "(a) HighMountain ",
		"ymax": 650.0,
		"model_files": {
			"AIFS": Path("./AIFS_vs_HighMountain.csv"),
			"GFS": Path("./GFS_vs_HighMountain.csv"),
			"HRRR": Path("./HRRR_vs_HighMountain.csv"),
			"IFS": Path("./IFS_vs_HighMountain.csv"),
			"Silurian": Path("./Silurian_vs_HighMountain.csv"),
			"Graph": Path("./Graph_vs_HighMountain.csv"),
		},
		"model_order": ["AIFS", "GFS", "HRRR", "IFS", "Silurian", "Graph"],
		"obs_end_column_by_model": {
			"AIFS": "obs_window_end_utc",
			"GFS": "ObsEndUTC",
			"HRRR": "ObsValidTimeUTC",
			"IFS": "obs_end",
			"Silurian": "obs_window_end_utc",
			"Graph": "obs_window_end_utc",
		},
		"use_ss_local_date": True,
		"zero_fill_missing_sites": True,
		"column_map": {
			"AIFS": {"obs_col": "observed_24h_cm", "fcst_col": "forecast_24h_cm", "units": "cm"},
			"GFS": {"obs_col": "Obs_mm", "fcst_col": "Forecast_mm", "units": "mm"},
			"HRRR": {"obs_col": "ObservedPrecip_cm", "fcst_col": "ForecastPrecip_cm", "units": "cm"},
			"IFS": {"obs_col": "obs_precip", "fcst_col": "forecast_precip", "units_col": "comparison_units"},
			"Silurian": {"obs_col": "observed_24h_cm", "fcst_col": "forecast_24h_cm", "units": "cm"},
			"Graph": {"obs_col": "observed_24h_cm", "fcst_col": "forecast_24h_cm", "units": "cm"},
		},
		"output_csv": OUTPUT_DIR / "SnowSafety_running_accumulation_5panel_v15_SS_11site_noTRD_V9plot_20241115_20250401.csv",
	},
{
		"title": "(b) HighMountain (PRISM-Downscaled)",
		"ymax": 650.0,
		"model_files": {
			"AIFS-d": Path("./AIFS_vs_HighMountainDS.csv"),
			"GFS-d": Path("./GFS_vs_HighMountainDS.csv"),
			"HRRR-d": Path("./HRRR_vs_HighMountainDS.csv"),
			"IFS-d": Path("./IFS_vs_HighMountainDS.csv"),
			"Silurian-d": Path("./Silurian_vs_HighMountainDS.csv"),
			"Graph-d": Path("./Graph_vs_HighMountainDS.csv"),
		},
		"model_order": ["AIFS-d", "GFS-d", "HRRR-d", "IFS-d", "Silurian-d", "Graph-d"],
		"obs_end_column_by_model": {
			"AIFS-d": "obs_window_end_utc",
			"GFS-d": "ObsEndUTC",
			"HRRR-d": "ObsValidTimeUTC",
			"IFS-d": "obs_end",
			"Silurian-d": "obs_window_end_utc",
			"Graph-d": "obs_window_end_utc",
		},
		"use_ss_local_date": True,
		"zero_fill_missing_sites": True,
		"column_map": {
			"AIFS-d": {"obs_col": "observed_24h_cm", "fcst_col": "forecast_24h_cm", "units": "cm"},
			"GFS-d": {"obs_col": "Obs_mm", "fcst_col": "Forecast_mm", "units": "mm"},
			"HRRR-d": {"obs_col": "ObservedPrecip_cm", "fcst_col": "ForecastPrecip_cm", "units": "cm"},
			"IFS-d": {"obs_col": "obs_precip", "fcst_col": "forecast_precip", "units_col": "comparison_units"},
			"Silurian-d": {"obs_col": "observed_24h_cm", "fcst_col": "forecast_24h_cm", "units": "cm"},
			"Graph-d": {"obs_col": "observed_24h_cm", "fcst_col": "forecast_24h_cm", "units": "cm"},
		},
		"output_csv": OUTPUT_DIR / "SnowSafetyDS_running_accumulation_5panel_v15_SS_11site_noTRD_V9plot_20241115_20250401.csv",
	},
{
		"title": "(c) CoCoWest",
		"ymax": 325.0,
		"apply_coco_zero_threshold": True,
		"model_files": {
			"AIFS": Path("./AIFS_vs_CoCoWest.csv"),
			"GFS": Path("./GFS_vs_CoCoWest.csv"),
			"HRRR": Path("./HRRR_vs_CoCoWest.csv"),
			"IFS": Path("./IFS_vs_CoCoWest.csv"),
			"Silurian": Path("./Silurian_vs_CoCoWest.csv"),
			"Graph": Path("./Graph_vs_CoCoWest.csv"),
		},
		"model_order": ["AIFS", "GFS", "HRRR", "IFS", "Silurian", "Graph"],
		"obs_end_column_by_model": {
			"AIFS": "obs_window_end_utc",
			"GFS": "ObsEndUTC",
			"HRRR": "ObsValidTimeUTC",
			"IFS": "obs_end_utc",
			"Silurian": "obs_window_end_utc",
			"Graph": "obs_window_end_utc",
		},
		"column_map": None,
		"output_csv": OUTPUT_DIR / "CoCoWest_running_accumulation_5panel_v15_SS_11site_noTRD_V9plot_20241115_20250401.csv",
	},
{
		"title": "(d) CoCoWest (PRISM-Downscaled)",
		"ymax": 325.0,
		"apply_coco_zero_threshold": True,
		"model_files": {
			"AIFS-d": Path("./AIFS_vs_CoCoWestDS.csv"),
			"GFS-d": Path("./GFS_vs_CoCoWestDS.csv"),
			"HRRR-d": Path("./HRRR_vs_CoCoWestDS.csv"),
			"IFS-d": Path("./IFS_vs_CoCoWestDS.csv"),
			"Silurian-d": Path("./Silurian_vs_CoCoWestDS.csv"),
			"Graph-d": Path("./Graph_vs_CoCoWestDS.csv"),
		},
		"model_order": ["AIFS-d", "GFS-d", "HRRR-d", "IFS-d", "Silurian-d", "Graph-d"],
		"obs_end_column_by_model": {
			"AIFS-d": "obs_window_end_utc",
			"GFS-d": "ObsEndUTC",
			"HRRR-d": "ObsValidTimeUTC",
			"IFS-d": "obs_end_utc",
			"Silurian-d": "obs_window_end_utc",
			"Graph-d": "obs_window_end_utc",
		},
		"column_map": None,
		"output_csv": OUTPUT_DIR / "CoCoWestDS_running_accumulation_5panel_v15_SS_11site_noTRD_V9plot_20241115_20250401.csv",
	},
{
		"title": "(e) CoCoEast",
		"ymax": 325.0,
		"apply_coco_zero_threshold": True,
		"model_files": {
			"AIFS": Path("./AIFS_vs_CoCoEast.csv"),
			"GFS": Path("./GFS_vs_CoCoEast.csv"),
			"HRRR": Path("./HRRR_vs_CoCoEast.csv"),
			"IFS": Path("./IFS_vs_CoCoEast.csv"),
			"Silurian": Path("./Silurian_vs_CoCoEast.csv"),
			"Graph": Path("./Graph_vs_CoCoEast.csv"),
		},
		"model_order": ["AIFS", "GFS", "HRRR", "IFS", "Silurian", "Graph"],
		"obs_end_column_by_model": {
			"AIFS": "obs_window_end_utc",
			"GFS": "ObsEndUTC",
			"HRRR": "ObsValidTimeUTC",
			"IFS": "obs_end_utc",
			"Silurian": "obs_window_end_utc",
			"Graph": "obs_window_end_utc",
		},
		"column_map": None,
		"output_csv": OUTPUT_DIR / "CoCoEast_running_accumulation_5panel_v15_SS_11site_noTRD_V9plot_20241115_20250401.csv",
	},
]

OBS_COLUMN_CANDIDATES = [
	"observed_24h_mm",
	"ObservedPrecip_mm",
	"Obs_mm",
	"obs_precip_mm",
	"obs_precip",
]

FORECAST_COLUMN_CANDIDATES = [
	"forecast_24h_mm",
	"ForecastPrecip_mm",
	"Forecast_mm",
	"forecast_precip_mm",
	"forecast_precip",
]

SITE_COLUMN_CANDIDATES = [
	"StationNumber",
	"station_number",
	"Station_Number",
	"station_id",
	"Station_ID",
	"station",
	"Station",
	"station_name",
	"StationName",
	"site_name",
	"SiteName",
	"Site_ID",
	"site_id",
	"site",
	"Site",
	"Location",
	"location",
	"stn_id",
	"STATION",
]

LAT_COLUMN_CANDIDATES = ["latitude", "Latitude", "lat", "Lat", "LAT"]
LON_COLUMN_CANDIDATES = ["longitude", "Longitude", "lon", "Lon", "LON"]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def ensure_parent_dir(path: Path) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)


def normalize_column_name(name: str) -> str:
	return "".join(character.lower() for character in str(name) if character.isalnum())


def find_column(
	df: pd.DataFrame,
	candidates: List[str],
	model_name: str,
	kind: str,
	required: bool = True,
) -> Optional[str]:
	for col in candidates:
		if col in df.columns:
			return col

	normalized_available = {}
	for col in df.columns:
		normalized_available.setdefault(normalize_column_name(col), col)

	for candidate in candidates:
		match = normalized_available.get(normalize_column_name(candidate))
		if match is not None:
			return match

	if not required:
		return None

	raise ValueError(
		f"{model_name}: could not find a {kind} column.\n"
		f"Tried: {candidates}\n"
		f"Available columns: {list(df.columns)}"
	)


def convert_to_mm(values: pd.Series, units: str) -> pd.Series:
	units = str(units).strip().lower()

	if units == "mm":
		return values
	if units == "cm":
		return values * 10.0
	if units in {"in", "inch", "inches"}:
		return values * 25.4

	raise ValueError(f"Unsupported precipitation units: {units!r}")


def get_units_for_model(df: pd.DataFrame, model_name: str, spec: Dict) -> str:
	if "units" in spec:
		return spec["units"]

	units_col = spec.get("units_col")
	if units_col is None:
		raise ValueError(f"{model_name}: no units or units_col defined in column map.")
	if units_col not in df.columns:
		raise ValueError(
			f"{model_name}: units column {units_col!r} not found. "
			f"Available columns: {list(df.columns)}"
		)

	unique_units = df[units_col].dropna().astype(str).str.strip().str.lower().unique()
	if len(unique_units) != 1:
		raise ValueError(
			f"{model_name}: expected one unique unit in {units_col!r}, "
			f"but found {unique_units}"
		)

	return unique_units[0]


def build_site_key(df: pd.DataFrame, model_name: str) -> pd.Series:
	site_col = find_column(
		df,
		SITE_COLUMN_CANDIDATES,
		model_name,
		"site identifier",
		required=False,
	)
	if site_col is not None:
		print(f"{model_name}: site identifier column = {site_col!r}")
		return df[site_col].astype("string").str.strip()

	lat_col = find_column(
		df,
		LAT_COLUMN_CANDIDATES,
		model_name,
		"latitude",
		required=False,
	)
	lon_col = find_column(
		df,
		LON_COLUMN_CANDIDATES,
		model_name,
		"longitude",
		required=False,
	)
	if lat_col is not None and lon_col is not None:
		lat = pd.to_numeric(df[lat_col], errors="coerce")
		lon = pd.to_numeric(df[lon_col], errors="coerce")
		print(
			f"{model_name}: no named site column found; using "
			f"{lat_col!r}/{lon_col!r} as the site identifier"
		)
		return lat.map(lambda value: f"{value:.5f}" if np.isfinite(value) else "nan") + "," + lon.map(
			lambda value: f"{value:.5f}" if np.isfinite(value) else "nan"
		)

	raise ValueError(
		f"{model_name}: could not determine a site identifier. Tried named site columns "
		f"{SITE_COLUMN_CANDIDATES} and latitude/longitude columns.\n"
		f"Available columns: {list(df.columns)}"
	)



def build_ss_local_date(obs_end_utc: pd.Series, site: pd.Series) -> pd.Series:
	"""Assign Snow Safety observations to the local calendar date at observation end.

	CA-* and WA-* sites use America/Los_Angeles. All other Snow Safety sites use
	America/Denver. The input timestamps remain UTC for forecast matching; this
	helper is used only to determine the nominal date used for daily averaging.
	"""
	local_dates = pd.Series(pd.NaT, index=obs_end_utc.index, dtype="datetime64[ns]")
	site_upper = site.astype("string").str.strip().str.upper()
	pacific = site_upper.str.startswith("CA-") | site_upper.str.startswith("WA-")
	mountain = ~pacific

	if pacific.any():
		local_dates.loc[pacific] = (
			obs_end_utc.loc[pacific]
			.dt.tz_convert("America/Los_Angeles")
			.dt.tz_localize(None)
			.dt.normalize()
		)
	if mountain.any():
		local_dates.loc[mountain] = (
			obs_end_utc.loc[mountain]
			.dt.tz_convert("America/Denver")
			.dt.tz_localize(None)
			.dt.normalize()
		)

	return local_dates

def load_model_rows(
	model_name: str,
	csv_path: Path,
	obs_end_column_by_model: Dict[str, str],
	column_map: Optional[Dict[str, Dict]],
	use_ss_local_date: bool = False,
) -> pd.DataFrame:
	if not csv_path.exists():
		raise FileNotFoundError(f"{model_name}: file not found: {csv_path}")

	df = pd.read_csv(csv_path)

	date_col = obs_end_column_by_model[model_name]
	if date_col not in df.columns:
		raise ValueError(
			f"{model_name}: expected date column {date_col!r} not found.\n"
			f"Available columns: {list(df.columns)}"
		)

	if column_map is None:
		obs_col = find_column(df, OBS_COLUMN_CANDIDATES, model_name, "observed precipitation")
		fcst_col = find_column(df, FORECAST_COLUMN_CANDIDATES, model_name, "forecast precipitation")
		units = "mm"
	else:
		spec = column_map[model_name]
		obs_col = spec["obs_col"]
		fcst_col = spec["fcst_col"]
		missing = [col for col in [obs_col, fcst_col] if col not in df.columns]
		if missing:
			raise ValueError(
				f"{model_name}: required column(s) missing: {missing}\n"
				f"File: {csv_path}\n"
				f"Available columns: {list(df.columns)}"
			)
		units = get_units_for_model(df, model_name, spec)

	observed = pd.to_numeric(df[obs_col], errors="coerce")
	forecast = pd.to_numeric(df[fcst_col], errors="coerce")
	observed = convert_to_mm(observed, units)
	forecast = convert_to_mm(forecast, units)

	obs_end_utc = pd.to_datetime(df[date_col], errors="coerce", utc=True)
	site = build_site_key(df, model_name)

	# For Snow Safety only, remove excluded sites before any local-date grouping,
	# duplicate collapsing, site-roster construction, zero filling, or averaging.
	if use_ss_local_date and SS_EXCLUDED_SITES:
		site_upper = site.astype("string").str.strip().str.upper()
		excluded_site_upper = {value.upper() for value in SS_EXCLUDED_SITES}
		keep_site = ~site_upper.isin(excluded_site_upper)
		n_removed = int((~keep_site).sum())
		if n_removed > 0:
			print(
				f"{model_name}: excluded {n_removed:,} rows from SS site(s) "
				f"{sorted(SS_EXCLUDED_SITES)}"
			)
	else:
		keep_site = pd.Series(True, index=df.index)

	# Apply the known-bad observation-ending-date exclusions using the original
	# UTC observation-ending calendar date, matching the source scripts.
	keep = keep_site.copy()
	if EXCLUDED_OBS_END_DATES:
		excluded_utc_dates = set(pd.to_datetime(sorted(EXCLUDED_OBS_END_DATES)).date)
		keep &= ~obs_end_utc.dt.date.isin(excluded_utc_dates)

	if use_ss_local_date:
		date = build_ss_local_date(obs_end_utc, site)
	else:
		date = obs_end_utc.dt.normalize().dt.tz_localize(None)

	out = pd.DataFrame({
		"date": date,
		"site": site,
		"observed_mm": observed,
		"forecast_mm": forecast,
	})
	out = out.loc[keep].copy()
	out = out.dropna(subset=["date", "site"])
	out = out[(out["site"] != "") & (out["site"] != "nan,nan")]

	start = pd.Timestamp(START_DATE)
	end = pd.Timestamp(END_DATE)
	out = out[out["date"].between(start, end, inclusive="both")]

	duplicate_mask = out.duplicated(subset=["date", "site"], keep=False)
	if duplicate_mask.any():
		duplicate_rows = int(duplicate_mask.sum())
		duplicate_groups = int(
			out.loc[duplicate_mask, ["date", "site"]]
			.drop_duplicates()
			.shape[0]
		)

		obs_spread = (
			out.loc[duplicate_mask]
			.groupby(["date", "site"])["observed_mm"]
			.agg(lambda values: values.max() - values.min())
		)
		conflicting_groups = int((obs_spread > 1.0e-6).sum())

		out = (
			out.groupby(["date", "site"], as_index=False, sort=False)
			.agg(
				observed_mm=("observed_mm", "mean"),
				forecast_mm=("forecast_mm", "mean"),
			)
		)

		print(
			f"{model_name}: collapsed {duplicate_rows:,} duplicate rows in "
			f"{duplicate_groups:,} date/site groups by averaging observed and "
			f"forecast values; {conflicting_groups:,} groups had differing observations"
		)

	date_method = "SS local calendar date" if use_ss_local_date else "UTC calendar date"
	print(
		f"{model_name}: obs={obs_col!r}, fcst={fcst_col!r}, input_units={units!r}; "
		f"date_method={date_method}; retained {len(out):,} rows, "
		f"{out['site'].nunique():,} sites, {out['date'].nunique():,} dates"
	)

	return out


def build_observed_daily_mean(model_rows: Dict[str, pd.DataFrame]) -> Tuple[pd.Series, pd.Series]:
	observation_frames = []

	for model_name, df in model_rows.items():
		part = df[["date", "site", "observed_mm"]].dropna(subset=["observed_mm"]).copy()
		part["source_model"] = model_name
		observation_frames.append(part)

	all_obs = pd.concat(observation_frames, ignore_index=True)

	spread = all_obs.groupby(["date", "site"])["observed_mm"].agg(
		minimum="min",
		maximum="max",
		count="count",
	)
	conflicts = spread[(spread["maximum"] - spread["minimum"]).abs() > 1.0e-6]
	if not conflicts.empty:
		print(
			f"Observed precipitation differed among model files for "
			f"{len(conflicts):,} site/date combinations; using the median value "
			"for each combination"
		)

	unique_obs = (
		all_obs.groupby(["date", "site"], as_index=False)["observed_mm"]
		.median()
	)

	daily_mean = unique_obs.groupby("date")["observed_mm"].mean()
	daily_count = unique_obs.groupby("date")["observed_mm"].count()

	return daily_mean, daily_count


def build_forecast_daily_mean(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
	valid = df.dropna(subset=["forecast_mm"])
	daily_mean = valid.groupby("date")["forecast_mm"].mean()
	daily_count = valid.groupby("date")["forecast_mm"].count()
	return daily_mean, daily_count


def build_ss_site_roster(model_rows: Dict[str, pd.DataFrame]) -> List[str]:
	"""Return the union of all Snow Safety site identifiers across model files."""
	sites = set()
	for df in model_rows.values():
		sites.update(df["site"].dropna().astype(str).str.strip().tolist())
	sites.discard("")
	return sorted(sites)


def build_ss_active_dates(model_rows: Dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
	"""Dates on which at least one SS site has a positive measured precipitation entry."""
	dates = set()
	for df in model_rows.values():
		valid = df.dropna(subset=["observed_mm"])
		valid = valid[valid["observed_mm"] > 0.0]
		dates.update(valid["date"].tolist())
	return pd.DatetimeIndex(sorted(dates))


def build_observed_daily_mean_zero_filled(
	model_rows: Dict[str, pd.DataFrame],
	site_roster: List[str],
	active_dates: pd.DatetimeIndex,
) -> Tuple[pd.Series, pd.Series]:
	"""Build SS observed daily means with missing site/date values filled as 0 mm."""
	observation_frames = []

	for model_name, df in model_rows.items():
		part = df[["date", "site", "observed_mm"]].dropna(subset=["observed_mm"]).copy()
		part["source_model"] = model_name
		observation_frames.append(part)

	all_obs = pd.concat(observation_frames, ignore_index=True)
	unique_obs = (
		all_obs.groupby(["date", "site"], as_index=False)["observed_mm"]
		.median()
	)

	full_index = pd.MultiIndex.from_product(
		[active_dates, site_roster],
		names=["date", "site"],
	)
	filled = (
		unique_obs.set_index(["date", "site"])["observed_mm"]
		.reindex(full_index)
		.fillna(0.0)
	)

	daily_mean = filled.groupby(level="date").mean()
	daily_count = filled.groupby(level="date").count()

	print(
		f"SS zero-fill observed: {len(site_roster)} sites in roster; "
		f"{len(active_dates)} wet local dates; every wet date uses all sites"
	)
	return daily_mean, daily_count


def build_forecast_daily_mean_zero_filled(
	df: pd.DataFrame,
	site_roster: List[str],
	active_dates: pd.DatetimeIndex,
	model_name: str,
) -> Tuple[pd.Series, pd.Series]:
	"""Build SS forecast daily means with absent site/date rows filled as 0 mm."""
	valid = df.dropna(subset=["forecast_mm"])[["date", "site", "forecast_mm"]].copy()
	valid = (
		valid.groupby(["date", "site"], as_index=False)["forecast_mm"]
		.mean()
	)

	full_index = pd.MultiIndex.from_product(
		[active_dates, site_roster],
		names=["date", "site"],
	)
	filled = (
		valid.set_index(["date", "site"])["forecast_mm"]
		.reindex(full_index)
		.fillna(0.0)
	)

	daily_mean = filled.groupby(level="date").mean()
	daily_count = filled.groupby(level="date").count()
	print(f"{model_name}: SS zero-fill forecast daily means use {len(site_roster)} sites")
	return daily_mean, daily_count


def make_output_table(
	model_rows: Dict[str, pd.DataFrame],
	model_order: List[str],
	zero_fill_missing_sites: bool = False,
) -> pd.DataFrame:
	all_dates = pd.date_range(START_DATE, END_DATE, freq="D")
	output = pd.DataFrame(index=all_dates)
	output.index.name = "date"

	if zero_fill_missing_sites:
		site_roster = build_ss_site_roster(model_rows)
		if len(site_roster) != 11:
			raise ValueError(
				f"Expected 11 Snow Safety sites after excluding TRD, but found "
				f"{len(site_roster)}: {site_roster}"
			)
		print(f"Snow Safety analysis roster confirmed: {len(site_roster)} sites; TRD excluded")
		active_dates = build_ss_active_dates(model_rows)
		observed_mean, observed_count = build_observed_daily_mean_zero_filled(
			model_rows,
			site_roster,
			active_dates,
		)
	else:
		observed_mean, observed_count = build_observed_daily_mean(model_rows)

	output["Observed_mean_mm"] = observed_mean.reindex(all_dates)
	output["Observed_N_sites"] = observed_count.reindex(all_dates).astype("Int64")

	for model_name in model_order:
		if zero_fill_missing_sites:
			forecast_mean, forecast_count = build_forecast_daily_mean_zero_filled(
				model_rows[model_name],
				site_roster,
				active_dates,
				model_name,
			)
		else:
			forecast_mean, forecast_count = build_forecast_daily_mean(model_rows[model_name])

		output[f"{model_name}_mean_mm"] = forecast_mean.reindex(all_dates)
		output[f"{model_name}_N_sites"] = forecast_count.reindex(all_dates).astype("Int64")

	output["Observed_accumulation_mm"] = output["Observed_mean_mm"].fillna(0.0).cumsum()
	for model_name in model_order:
		output[f"{model_name}_accumulation_mm"] = (
			output[f"{model_name}_mean_mm"].fillna(0.0).cumsum()
		)

	return output


def load_panel(config: Dict) -> pd.DataFrame:
	print()
	print("=" * 78)
	print(config["title"])
	print("=" * 78)

	model_rows: Dict[str, pd.DataFrame] = {}
	for model_name in config["model_order"]:
		model_rows[model_name] = load_model_rows(
			model_name,
			config["model_files"][model_name],
			config["obs_end_column_by_model"],
			config["column_map"],
			use_ss_local_date=config.get("use_ss_local_date", False),
		)

	output = make_output_table(
		model_rows,
		config["model_order"],
		zero_fill_missing_sites=config.get("zero_fill_missing_sites", False),
	)
	ensure_parent_dir(config["output_csv"])
	output.to_csv(config["output_csv"], float_format="%.4f")
	print(f"Wrote panel data to: {config['output_csv']}")

	return output


def plot_panel(ax, output: pd.DataFrame, config: Dict) -> None:
	ax.plot(
		output.index,
		output["Observed_accumulation_mm"],
		label="Observed",
		color="black",
		linewidth=OBS_LINEWIDTH,
		linestyle='--',
		zorder=10,
	)

	for model_name in config["model_order"]:
		ax.plot(
			output.index,
			output[f"{model_name}_accumulation_mm"],
			label=model_name,
			linewidth=LINEWIDTH,
			alpha=0.9,
		)

	ax.set_xlim(pd.Timestamp(START_DATE), pd.Timestamp(END_DATE))
	ax.set_ylim(bottom=0, top=config.get("ymax", YMAX_DEFAULT))
	ax.set_title(
		config["title"],
		loc="left",
		fontsize=9.0,
		fontweight="bold",
		pad=2.0,
	)
	ax.grid(True, alpha=0.3)

	ax.xaxis.set_major_locator(mdates.MonthLocator())
	ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
	ax.xaxis.set_minor_locator(mdates.WeekdayLocator(interval=1))

	ax.tick_params(axis="both", which="major", labelsize=7.5, pad=1.5)
	for tick_label in ax.get_xticklabels():
		tick_label.set_rotation(30)
		tick_label.set_ha("right")
		tick_label.set_fontweight("bold")
	for tick_label in ax.get_yticklabels():
		tick_label.set_fontweight("bold")

	legend = ax.legend(
		loc="upper left",
		ncol=4,
		frameon=True,
		fontsize=6.8,
	)
	for legend_text in legend.get_texts():
		legend_text.set_fontweight("bold")


def plot_five_panel(outputs: List[pd.DataFrame]) -> None:
	fig, axes = plt.subplots(
		5,
		1,
		figsize=FIGSIZE,
		sharex=True,
		sharey=False,
	)

	for ax, output, config in zip(axes, outputs, PANEL_CONFIGS):
		plot_panel(ax, output, config)

	# Keep all panel titles ABOVE their axes. Give the top title real headroom.
	# Put the y-axis description on the center panel only, outside the axes,
	# with enough label padding that it cannot overlap the numeric tick labels.
	axes[2].set_ylabel(
		"Running accumulation of daily mean precipitation (mm)",
		fontsize=7.8,
		fontweight="bold",
		labelpad=18.0,
	)
	axes[-1].set_xlabel(
		"Observation-ending date",
		fontsize=9.0,
		fontweight="bold",
		labelpad=3.0,
	)

	# These margins are intentionally modest. They are NOT relied on to remove
	# the visible side whitespace; the export is cropped to the actual rendered
	# artists below.
	fig.subplots_adjust(
		left=0.090,
		right=0.995,
		top=0.972,
		bottom=0.090,
		hspace=0.16,
	)

	ensure_parent_dir(OUTPUT_FIG)
	temp_fig = OUTPUT_FIG.with_name(OUTPUT_FIG.stem + "_tight_temp.png")

	# IMPORTANT: crop to the TRUE rendered content bounds. This is what removes
	# the left/right blank canvas that subplots_adjust() alone cannot remove.
	fig.savefig(
		temp_fig,
		dpi=DPI,
		bbox_inches="tight",
		pad_inches=0.01,
		facecolor="white",
	)
	plt.close(fig)

	# Normalize the tightly cropped result back to the requested exact physical
	# size: 6.5 x 9 inches at 200 dpi = 1300 x 1800 pixels. Because the crop is
	# derived from actual artists, this does not reintroduce blank side gutters.
	target_width_px = int(round(FIG_WIDTH * DPI))
	target_height_px = int(round(FIG_HEIGHT * DPI))

	with Image.open(temp_fig) as image:
		image = image.convert("RGB")
		image = image.resize(
			(target_width_px, target_height_px),
			resample=Image.Resampling.LANCZOS,
		)
		image.save(OUTPUT_FIG, dpi=(DPI, DPI))

	temp_fig.unlink(missing_ok=True)


def print_panel_summary(output: pd.DataFrame, config: Dict) -> None:
	print()
	print(config["title"])
	print(f"  {'Observed':12s}: {output['Observed_accumulation_mm'].iloc[-1]:.2f} mm")
	for model_name in config["model_order"]:
		print(
			f"  {model_name:12s}: "
			f"{output[f'{model_name}_accumulation_mm'].iloc[-1]:.2f} mm"
		)


def main() -> None:
	outputs = [load_panel(config) for config in PANEL_CONFIGS]
	plot_five_panel(outputs)

	print()
	print("Final running accumulations:")
	for output, config in zip(outputs, PANEL_CONFIGS):
		print_panel_summary(output, config)

	print()
	print(f"Wrote V9-format 5-panel figure to: {OUTPUT_FIG}")
	print(f"Figure size: {FIG_WIDTH:.2f} x {FIG_HEIGHT:.2f} inches")
	print(f"Horizontal:vertical aspect ratio: {FIG_WIDTH / FIG_HEIGHT:.2f}:1")
	print(f"Date range: {START_DATE} through {END_DATE}, inclusive")


if __name__ == "__main__":
	main()
