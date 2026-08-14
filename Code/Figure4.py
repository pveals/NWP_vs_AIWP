#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


MODEL_FILES = {
	"AIFS-d": Path("./AIFS_vs_HighMountainDS.csv"),
	"GFS-d": Path("./GFS_vs_HighMountainDS.csv"),
	"HRRR-d": Path("./HRRR_vs_HighMountainDS.csv"),
	"IFS-d": Path("./IFS_vs_HighMountainDS.csv"),
	"Silurian-d": Path("./Silurian_vs_HighMountainDS.csv"),
	"Graph-d": Path("./Graph_vs_HighMountainDS.csv"),
}

OUTPUT_FIG = Path("./Fig4.png")
FIGSIZE = (16, 16)
DPI = 200
MIN_OBS_MM = 2.7
BIN_SIZE_MM = 2.5
FIXED_AXIS_MAX_MM = 100.0

COLOR_SCALE_MIN = 1
COLOR_SCALE_MAX = 500
COLORMAP = "plasma"

PANEL_ORDER = ["AIFS-d", "GFS-d", "Graph-d", "HRRR-d", "IFS-d", "Silurian-d"]
PANEL_LABELS = ["a", "b", "c", "d", "e", "f"]


EXCLUDED_OBS_END_DATES = {
	"2024-11-07",
	"2025-01-10",
	"2025-02-13",
	"2025-02-14",
    "2025-04-14",
    "2025-04-15",
}

OBS_END_COLUMN_BY_MODEL = {
	"AIFS-d": "obs_window_end_utc",
	"GFS-d": "ObsEndUTC",
	"HRRR-d": "ObsValidTimeUTC",
	"IFS-d": "obs_end",
	"Silurian-d": "obs_window_end_utc",
	"Graph-d": "obs_window_end_utc",
}


def filter_excluded_obs_end_dates(df: pd.DataFrame, model_name: str, csv_path: Path) -> pd.DataFrame:
	if not EXCLUDED_OBS_END_DATES:
		return df

	date_col = OBS_END_COLUMN_BY_MODEL[model_name]

	if date_col not in df.columns:
		raise ValueError(
			f"{model_name}: expected observation-ending column {date_col!r} "
			f"was not found in {csv_path}.\n"
			f"Available columns: {list(df.columns)}"
		)

	obs_end_dates = pd.to_datetime(df[date_col], errors="coerce").dt.strftime("%Y-%m-%d")

	if obs_end_dates.notna().sum() == 0:
		raise ValueError(
			f"{model_name}: column {date_col!r} was found for observation-ending dates, "
			f"but none of its values could be parsed as dates.\n"
			f"File: {csv_path}"
		)

	excluded = obs_end_dates.isin(EXCLUDED_OBS_END_DATES)
	n_excluded = int(excluded.sum())

	print(
		f"{model_name}: excluded {n_excluded} forecast-observation pairs with "
		f"{date_col!r} in {sorted(EXCLUDED_OBS_END_DATES)}"
	)

	return df.loc[~excluded].copy()


COLUMN_MAP = {
	"AIFS-d": {
		"obs_col": "observed_24h_cm",
		"fcst_col": "forecast_24h_cm",
		"units": "cm",
	},
	"GFS-d": {
		"obs_col": "Obs_mm",
		"fcst_col": "Forecast_mm",
		"units": "mm",
	},
	"HRRR-d": {
		"obs_col": "ObservedPrecip_cm",
		"fcst_col": "ForecastPrecip_cm",
		"units": "cm",
	},
	"IFS-d": {
		"obs_col": "obs_precip",
		"fcst_col": "forecast_precip",
		"units_col": "comparison_units",
	},
	"Silurian-d": {
		"obs_col": "observed_24h_cm",
		"fcst_col": "forecast_24h_cm",
		"units": "cm",
	},
	"Graph-d": {
		"obs_col": "observed_24h_cm",
		"fcst_col": "forecast_24h_cm",
		"units": "cm",
	},
}


def ensure_parent_dir(path: Path) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)


def convert_to_mm(values: np.ndarray, units: str) -> np.ndarray:
	units = str(units).strip().lower()

	if units == "mm":
		return values
	if units == "cm":
		return values * 10.0
	if units in {"in", "inch", "inches"}:
		return values * 25.4

	raise ValueError(f"Unsupported precip units: {units!r}")


def get_units_for_model(df: pd.DataFrame, model_name: str) -> str:
	spec = COLUMN_MAP[model_name]

	if "units" in spec:
		return spec["units"]

	units_col = spec.get("units_col")
	if units_col is None:
		raise ValueError(f"{model_name}: no units or units_col defined in COLUMN_MAP.")

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


def load_model_data(model_name: str, csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
	if not csv_path.exists():
		raise FileNotFoundError(f"{model_name}: file not found: {csv_path}")

	df = pd.read_csv(csv_path)
	df = filter_excluded_obs_end_dates(df, model_name, csv_path)
	spec = COLUMN_MAP[model_name]

	obs_col = spec["obs_col"]
	fcst_col = spec["fcst_col"]

	missing = [c for c in [obs_col, fcst_col] if c not in df.columns]
	if missing:
		raise ValueError(
			f"{model_name}: required locked column(s) missing: {missing}\n"
			f"File: {csv_path}\n"
			f"Available columns: {list(df.columns)}"
		)

	units = get_units_for_model(df, model_name)

	obs_raw = pd.to_numeric(df[obs_col], errors="coerce").to_numpy(dtype=float)
	fcst_raw = pd.to_numeric(df[fcst_col], errors="coerce").to_numpy(dtype=float)

	obs = convert_to_mm(obs_raw, units)
	fcst = convert_to_mm(fcst_raw, units)

	valid = np.isfinite(obs) & np.isfinite(fcst)
	obs = obs[valid]
	fcst = fcst[valid]

	keep = obs > MIN_OBS_MM
	obs = obs[keep]
	fcst = fcst[keep]

	print(
		f"{model_name}: obs={obs_col!r}, fcst={fcst_col!r}, "
		f"input_units={units!r}, N={len(obs)}"
	)

	return obs, fcst


def compute_metrics(obs: np.ndarray, fcst: np.ndarray) -> Dict[str, float]:
	if len(obs) == 0:
		return {
			"N": 0,
			"MAE": np.nan,
			"RMSE": np.nan,
			#"Correlation": np.nan,
			"MPE": np.nan,
			"Bias": np.nan,
		}

	err = fcst - obs

	bias = np.mean(err)
	mae = np.mean(np.abs(err))
	rmse = np.sqrt(np.mean(err ** 2))
	mpe = np.mean((err / obs) * 100.0)

	#corr = np.nan
	#if len(obs) >= 2 and np.std(obs) > 0 and np.std(fcst) > 0:
	#	corr = np.corrcoef(obs, fcst)[0, 1]

	return {
		"N": len(obs),
		"Bias": bias,
		"MAE": mae,
		"RMSE": rmse,
		#"Correlation": corr,
		"MPE": mpe,
	}


def format_stats_text(stats: Dict[str, float]) -> str:
	#corr_txt = f"{stats['Correlation']:.3f}" if np.isfinite(stats["Correlation"]) else "nan"
	mpe_txt = f"{stats['MPE']:.1f}%" if np.isfinite(stats["MPE"]) else "nan"

	return (
		f"N = {int(stats['N'])}\n"
		f"MAE = {stats['MAE']:.2f} mm\n"
		f"RMSE = {stats['RMSE']:.2f} mm\n"
		f"MBE = {stats['Bias']:.1f} mm\n"
		f"MPE = {stats['MPE']:.1f} %"
	)


def main() -> None:
	ensure_parent_dir(OUTPUT_FIG)

	model_data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
	model_stats: Dict[str, Dict[str, float]] = {}

	for model_name in PANEL_ORDER:
		obs, fcst = load_model_data(model_name, MODEL_FILES[model_name])
		model_data[model_name] = (obs, fcst)
		model_stats[model_name] = compute_metrics(obs, fcst)

	axis_max = FIXED_AXIS_MAX_MM
	bins = np.arange(0, axis_max + BIN_SIZE_MM, BIN_SIZE_MM)

	fig, axes = plt.subplots(3, 2, figsize=FIGSIZE)
	axes = axes.ravel()

	for ax, panel_label, model_name in zip(axes, PANEL_LABELS, PANEL_ORDER):
		obs, fcst = model_data[model_name]
		stats = model_stats[model_name]

		h = ax.hist2d(
			obs,
			fcst,
			bins=[bins, bins],
			cmin=1,
			norm=LogNorm(vmin=COLOR_SCALE_MIN, vmax=COLOR_SCALE_MAX),
			cmap=COLORMAP,
		)

		ax.plot([0, axis_max], [0, axis_max], linestyle="--", linewidth=1)

		ax.set_xlim(0, axis_max)
		ax.set_ylim(0, axis_max)
		ax.grid(True, alpha=0.3)

		ax.set_title(f"({panel_label}) {model_name}", fontsize=16, fontweight="bold")
		# Only bottom row gets x-axis labels
		if panel_label in ["e", "f"]:
			ax.set_xlabel("Observed 24-h precipitation (mm)", fontsize=12, fontweight="bold")
		else:
			ax.set_xlabel("")

		# Only left column gets y-axis labels
		if panel_label in ["a", "c", "e"]:
			ax.set_ylabel("Forecast 24-h precipitation (mm)", fontsize=12, fontweight="bold")
		else:
			ax.set_ylabel("")
		ax.tick_params(axis="both", labelsize=12)
		plt.setp(ax.get_xticklabels(), fontweight="bold")
		plt.setp(ax.get_yticklabels(), fontweight="bold")
		ax.text(
			0.02,
			0.96,
			format_stats_text(stats),
			transform=ax.transAxes,
			va="top",
			ha="left",
			fontsize=12,
			fontweight="bold",
			linespacing=1.2,
			bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.85),
		)

		cbar = fig.colorbar(h[3], ax=ax)
		cbar.set_label("Count (log scale)", fontsize=12, fontweight="bold")
		cbar.set_ticks([1, 3, 10, 30, 100, 300, 500])
		cbar.ax.tick_params(labelsize=10)
		plt.setp(cbar.ax.get_yticklabels(), fontweight="bold")

	#fig.suptitle(
	#    f"Forecast vs CoCoRaHS West 24-h Precipitation 2D Histogram "
	#    f"(Obs > {MIN_OBS_MM:.1f} mm; bin = {BIN_SIZE_MM:.1f} mm)",
	#    fontsize=16,
	#)

	plt.tight_layout(rect=[0, 0, 1, 0.97])
	plt.savefig(OUTPUT_FIG, dpi=DPI, bbox_inches="tight")
	plt.close()

	print(f"Wrote 6-panel 2D histogram figure to: {OUTPUT_FIG}")
	print(f"Bin size: {BIN_SIZE_MM:.1f} mm")
	print(f"Shared axis max: {axis_max:.1f} mm")
	print(f"Color scale: LogNorm({COLOR_SCALE_MIN}, {COLOR_SCALE_MAX}), cmap={COLORMAP}")
	print()
	print("Panel statistics:")
	for model_name in PANEL_ORDER:
		s = model_stats[model_name]
		#corr_txt = f"{s['Correlation']:.3f}" if np.isfinite(s["Correlation"]) else "nan"
		mpe_txt = f"{s['MPE']:.1f}%" if np.isfinite(s["MPE"]) else "nan"
		print(
			f"  {model_name:17s} "
			f"N={int(s['N']):5d}  "
			f"Bias={s['Bias']:7.2f} mm  "
			f"MAE={s['MAE']:7.2f} mm  "
			f"RMSE={s['RMSE']:7.2f} mm  "
			f"MPE={mpe_txt:>8s}  "
			#f"r={corr_txt}"
		)


if __name__ == "__main__":
	main()
