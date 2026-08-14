#!/usr/bin/env python3
"""
Calculate and plot Equitable Threat Score (ETS) versus fixed precipitation
threshold for six aligned 24-hour precipitation forecast datasets.

All models are aligned using the common event key:
	station, observation-window start, observation-window end

The analysis includes all events with finite observed precipitation greater
than or equal to MIN_OBSERVED_PRECIP_MM. With the default setting of 0.0 mm,
zero-observation events are retained and can contribute false alarms and
correct negatives.

Saves:
	1) PNG figure
	2) CSV of contingency-table counts and ETS values
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# User settings
# -----------------------------------------------------------------------------

MIN_OBSERVED_PRECIP_MM = 0.0
EVENT_THRESHOLDS_MM = [1, 5, 10, 15, 20, 25, 30, 35]

MODEL_FILES = {
	"AIFS": Path("./AIFS_vs_CoCoEast.csv"),
	"GFS": Path("./GFS_vs_CoCoEast.csv"),
	"HRRR": Path("./HRRR_vs_CoCoEast.csv"),
	"IFS": Path("./IFS_vs_CoCoEast.csv"),
	"Silurian": Path("./Silurian_vs_CoCoEast.csv"),
	"Graph": Path("./Graph_vs_CoCoEast.csv"),
}

OUTPUT_FIG = Path(
	"/Figure11.png"
)
OUTPUT_STATS_CSV = Path(
	"./Figure11_stats_allevents.csv"
)

MODEL_ORDER = ["AIFS", "GFS", "HRRR", "IFS", "Silurian", "Graph"]
REFERENCE_MODEL = "Graph"

MODEL_CONFIG = {
	"AIFS": {
		"station": "Site",
		"obs_start": "obs_window_start_utc",
		"obs_end": "obs_window_end_utc",
		"observed": "observed_24h_mm",
		"forecast": "forecast_24h_mm",
	},
	"GFS": {
		"station": "Site",
		"obs_start": "ObsStartUTC",
		"obs_end": "ObsEndUTC",
		"observed": "Obs_mm",
		"forecast": "Forecast_mm",
	},
	"HRRR": {
		"station": "StationNumber",
		"obs_start": "ObsStartTimeUTC",
		"obs_end": "ObsValidTimeUTC",
		"observed": "ObservedPrecip_mm",
		"forecast": "ForecastPrecip_mm",
	},
	"IFS": {
		"station": "station_id",
		"obs_start": "obs_start_utc",
		"obs_end": "obs_end_utc",
		"observed": "obs_precip_mm",
		"forecast": "forecast_precip_mm",
	},
	"Silurian": {
		"station": "Site",
		"obs_start": "obs_window_start_utc",
		"obs_end": "obs_window_end_utc",
		"observed": "observed_24h_mm",
		"forecast": "forecast_24h_mm",
	},
	"Graph": {
		"station": "Site",
		"obs_start": "obs_window_start_utc",
		"obs_end": "obs_window_end_utc",
		"observed": "observed_24h_mm",
		"forecast": "forecast_24h_mm",
	},
}

MODEL_STYLES = {
	"AIFS": {"marker": "o"},
	"IFS": {"marker": "D"},
	"GFS": {"marker": "s"},
	"HRRR": {"marker": "^"},
	"Graph": {"marker": "X"},
	"Silurian": {"marker": "P"},
}


# -----------------------------------------------------------------------------
# Data loading, alignment, and filtering
# -----------------------------------------------------------------------------

def read_and_standardize_model(model_name):
	"""Read one model CSV and create standardized event and value columns."""
	path = MODEL_FILES[model_name]
	config = MODEL_CONFIG[model_name]

	if not path.exists():
		raise FileNotFoundError(f"Could not find {model_name} file: {path}")

	df = pd.read_csv(path)
	required_columns = [
		config["station"],
		config["obs_start"],
		config["obs_end"],
		config["observed"],
		config["forecast"],
	]
	missing_columns = [column for column in required_columns if column not in df.columns]
	if missing_columns:
		raise KeyError(f"{model_name} is missing required columns: {missing_columns}")

	standardized = pd.DataFrame({
		"station": df[config["station"]].astype(str).str.strip(),
		"obs_start": pd.to_datetime(df[config["obs_start"]], utc=True, errors="raise"),
		"obs_end": pd.to_datetime(df[config["obs_end"]], utc=True, errors="raise"),
		"observed_mm": pd.to_numeric(df[config["observed"]], errors="coerce"),
		"forecast_mm": pd.to_numeric(df[config["forecast"]], errors="coerce"),
	})

	key_columns = ["station", "obs_start", "obs_end"]
	duplicate_count = int(standardized.duplicated(key_columns).sum())
	if duplicate_count > 0:
		raise ValueError(
			f"{model_name} contains {duplicate_count:,} duplicate event keys."
		)

	standardized.set_index(key_columns, inplace=True)
	return standardized


def load_align_and_filter():
	"""Align all models to Graph and retain one identical >= minimum event set."""
	data = {
		model_name: read_and_standardize_model(model_name)
		for model_name in MODEL_ORDER
	}

	reference_index = data[REFERENCE_MODEL].index
	reference_key_set = set(reference_index)

	for model_name in MODEL_ORDER:
		model_key_set = set(data[model_name].index)
		missing_events = reference_key_set - model_key_set
		extra_events = model_key_set - reference_key_set

		if missing_events or extra_events:
			raise ValueError(
				f"{model_name} does not contain exactly the same events as "
				f"{REFERENCE_MODEL}. Missing: {len(missing_events):,}; "
				f"extra: {len(extra_events):,}."
			)

		data[model_name] = data[model_name].loc[reference_index].copy()

	reference_observed = data[REFERENCE_MODEL]["observed_mm"].to_numpy(dtype=float)
	common_mask = np.isfinite(reference_observed) & (
		reference_observed >= MIN_OBSERVED_PRECIP_MM
	)

	filtered = {}
	for model_name in MODEL_ORDER:
		model_df = data[model_name].loc[common_mask].copy()
		observed = model_df["observed_mm"].to_numpy(dtype=float)
		forecast = model_df["forecast_mm"].to_numpy(dtype=float)

		if not np.allclose(
			observed,
			reference_observed[common_mask],
			rtol=0.0,
			atol=1.0e-6,
			equal_nan=True,
		):
			raise ValueError(
				f"Observed precipitation in {model_name} does not match "
				f"{REFERENCE_MODEL} after alignment."
			)

		if not np.all(np.isfinite(forecast)):
			n_bad = int(np.sum(~np.isfinite(forecast)))
			raise ValueError(
				f"{model_name} has {n_bad:,} non-finite forecast values after filtering."
			)

		filtered[model_name] = {
			"observed": observed,
			"forecast": forecast,
		}

	return filtered


# -----------------------------------------------------------------------------
# ETS calculation
# -----------------------------------------------------------------------------

def contingency_and_ets(forecast, observed, threshold):
	"""Calculate ETS and all contingency-table components."""
	forecast_event = forecast >= threshold
	observed_event = observed >= threshold

	hits = int(np.sum(forecast_event & observed_event))
	misses = int(np.sum(~forecast_event & observed_event))
	false_alarms = int(np.sum(forecast_event & ~observed_event))
	correct_negatives = int(np.sum(~forecast_event & ~observed_event))

	n = hits + misses + false_alarms + correct_negatives
	hits_random = (
		((hits + misses) * (hits + false_alarms)) / n
		if n > 0
		else np.nan
	)

	denominator = hits + misses + false_alarms - hits_random
	ets = (
		(hits - hits_random) / denominator
		if np.isfinite(hits_random) and denominator != 0
		else np.nan
	)

	return {
		"threshold_mm": float(threshold),
		"n_pairs": n,
		"hits": hits,
		"misses": misses,
		"false_alarms": false_alarms,
		"correct_negatives": correct_negatives,
		"hits_random": hits_random,
		"ETS": ets,
	}


# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------

def main():
	OUTPUT_FIG.parent.mkdir(parents=True, exist_ok=True)
	OUTPUT_STATS_CSV.parent.mkdir(parents=True, exist_ok=True)

	filtered = load_align_and_filter()
	n_events = len(filtered[REFERENCE_MODEL]["observed"])

	print(
		f"Common aligned events with observed precipitation >= "
		f"{MIN_OBSERVED_PRECIP_MM:g} mm: {n_events:,}"
	)
	for model_name in MODEL_ORDER:
		print(f"{model_name:<10}: {len(filtered[model_name]['observed']):,}")

	all_stats = []
	fig, ax = plt.subplots(figsize=(10.5, 6.5))

	for model_name in MODEL_ORDER:
		forecast = filtered[model_name]["forecast"]
		observed = filtered[model_name]["observed"]

		model_stats = []
		for threshold in EVENT_THRESHOLDS_MM:
			stats = contingency_and_ets(forecast, observed, threshold)
			stats["model"] = model_name
			model_stats.append(stats)
			all_stats.append(stats)

		stats_df = pd.DataFrame(model_stats)
		style = MODEL_STYLES.get(model_name, {"marker": "o"})
		ax.plot(
			stats_df["threshold_mm"],
			stats_df["ETS"],
			marker=style["marker"],
			linewidth=2.4,
			markersize=8,
			label=model_name,
		)

	ax.axhline(0.0, color="0.45", linewidth=1.2, linestyle="--")
	ax.set_ylim(0, 0.7)
	ax.set_xlabel(
		"Precipitation event threshold (mm)",
		fontsize=13,
		fontweight="bold",
	)
	ax.set_ylabel("ETS", fontsize=13, fontweight="bold")
	ax.set_xticks(EVENT_THRESHOLDS_MM)
	ax.grid(True, color="0.82", linewidth=1.0)

	for spine in ax.spines.values():
		spine.set_linewidth(1.8)

	ax.tick_params(axis="both", labelsize=11, width=1.5, length=5)
	for label in ax.get_xticklabels():
		label.set_fontweight("bold")
	for label in ax.get_yticklabels():
		label.set_fontweight("bold")

	legend = ax.legend(
		loc="best",
		frameon=True,
		fontsize=10,
		title="Models",
		title_fontsize=11,
	)
	legend.get_title().set_fontweight("bold")
	for text in legend.get_texts():
		text.set_fontweight("bold")

	fig.tight_layout()
	fig.savefig(OUTPUT_FIG, dpi=300, bbox_inches="tight")
	plt.close(fig)

	stats_out = pd.DataFrame(all_stats)
	stats_out = stats_out[[
		"model",
		"threshold_mm",
		"n_pairs",
		"hits",
		"misses",
		"false_alarms",
		"correct_negatives",
		"hits_random",
		"ETS",
	]]
	stats_out.to_csv(OUTPUT_STATS_CSV, index=False)

	ets_table = stats_out.pivot(
		index="model",
		columns="threshold_mm",
		values="ETS",
	)
	ets_table = ets_table.reindex(index=MODEL_ORDER, columns=EVENT_THRESHOLDS_MM)
	ets_table.columns = [f"{threshold:g} mm" for threshold in ets_table.columns]

	print("\nETS by model and precipitation threshold:")
	print(ets_table.to_string(float_format=lambda value: f"{value:.4f}"))
	print(f"Saved figure to: {OUTPUT_FIG}")
	print(f"Saved stats to: {OUTPUT_STATS_CSV}")


if __name__ == "__main__":
	main()
