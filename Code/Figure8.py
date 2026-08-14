#!/usr/bin/env python3
"""
Plot equitable threat score (ETS) versus precipitation event threshold
for multiple 24-h precipitation forecast models.

Saves:
	1) PNG figure
	2) CSV of contingency-table counts and ETS values
"""

import matplotlib
matplotlib.use("Agg")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# User settings
# =========================
MODEL_FILES = {
	"AIFS-d": Path("./AIFS_vs_CoCoWestDS.csv"),
	"GFS-d": Path("./GFS_vs_CoCoWestDS.csv"),
	"HRRR-d": Path("./HRRR_vs_CoCoWestDS.csv"),
	"IFS-d": Path("./IFS_vs_CoCoWestDS.csv"),
	"Silurian": Path("./Silurian_vs_CoCoWest.csv"),
	"Graph-d": Path("./Graph_vs_CoCoWestDS.csv"),
}

EVENT_THRESHOLDS_MM = [1, 5, 10, 15, 20, 25, 30, 35]

# Minimum observed 24-h precipitation required for a pair to be included.
# Set to 0.0 to include all non-NaN pairs.
MIN_OBSERVED_PRECIP_MM = 0.0

OUTPUT_FIG = Path(
	"./Figure8.png"
)

OUTPUT_STATS_CSV = Path(
	"./Figure8_stats_allevents.csv"
)

COLUMN_OVERRIDES = {
	"AIFS-d": {"forecast": None, "observed": None},
	"GFS-d": {"forecast": None, "observed": None},
	"HRRR-d": {"forecast": None, "observed": None},
	"IFS-d": {"forecast": None, "observed": None},
	"Silurian": {"forecast": None, "observed": None},
	"Graph-d": {"forecast": None, "observed": None},
}

MODEL_STYLES = {
	"AIFS-d": {"marker": "o"},
	"GFS-d": {"marker": "s"},
	"HRRR-d": {"marker": "^"},
	"IFS-d": {"marker": "D"},
	"Silurian": {"marker": "P"},
	"Graph-d": {"marker": "X"},
}

FORECAST_CANDIDATES = [
	"forecast_24h_mm",
	"forecast_mm",
	"forecast_precip_mm",
	"forecasted_precip_mm",
	"fcst_24h_mm",
	"fcst_mm",
	"model_precip_mm",
	"ForecastPrecip_mm",
	"Forecast_mm",
	"Fcst_mm",
	"FCST_mm",
]

OBSERVED_CANDIDATES = [
	"observed_24h_mm",
	"observed_mm",
	"observed_precip_mm",
	"obs_24h_mm",
	"obs_mm",
	"ObsPrecip_mm",
	"ObservedPrecip_mm",
	"Observation_mm",
	"OBS_mm",
]


def find_precip_column(df, candidates, required_keywords, forbidden_keywords):
	cols = list(df.columns)

	for c in candidates:
		if c in cols:
			return c

	possible = []
	for c in cols:
		lc = c.lower()
		has_required = any(k in lc for k in required_keywords)
		has_forbidden = any(k in lc for k in forbidden_keywords)
		has_precip_hint = (
			"mm" in lc
			or "precip" in lc
			or "precipitation" in lc
			or "apcp" in lc
			or "tp" in lc
		)

		if has_required and not has_forbidden and has_precip_hint:
			possible.append(c)

	if len(possible) == 1:
		return possible[0]

	raise ValueError(
		"Could not uniquely auto-detect precip column. Please set COLUMN_OVERRIDES.\n"
		f"Possible columns: {possible}\n"
		f"All columns: {cols}"
	)


def detect_columns(model_name, df):
	override = COLUMN_OVERRIDES.get(model_name, {})
	forecast_col = override.get("forecast")
	observed_col = override.get("observed")

	if forecast_col is None:
		forecast_col = find_precip_column(
			df,
			FORECAST_CANDIDATES,
			required_keywords=["forecast", "fcst", "model"],
			forbidden_keywords=["obs", "observ"],
		)

	if observed_col is None:
		observed_col = find_precip_column(
			df,
			OBSERVED_CANDIDATES,
			required_keywords=["obs", "observ"],
			forbidden_keywords=["forecast", "fcst", "model"],
		)

	return forecast_col, observed_col


def read_model_pairs(model_name, path):
	df = pd.read_csv(path)
	forecast_col, observed_col = detect_columns(model_name, df)

	out = df[[forecast_col, observed_col]].copy()
	out.columns = ["forecast_mm", "observed_mm"]

	out["forecast_mm"] = pd.to_numeric(out["forecast_mm"], errors="coerce")
	out["observed_mm"] = pd.to_numeric(out["observed_mm"], errors="coerce")

	before = len(out)
	out = out.dropna()
	after_dropna = len(out)

	out = out[out["observed_mm"] >= MIN_OBSERVED_PRECIP_MM].copy()
	after_threshold = len(out)

	print(
		f"{model_name}: forecast_col='{forecast_col}', "
		f"observed_col='{observed_col}', n={after_threshold}, "
		f"dropped_non_numeric_or_nan={before - after_dropna}, "
		f"dropped_obs_below_{MIN_OBSERVED_PRECIP_MM:g}mm="
		f"{after_dropna - after_threshold}"
	)

	return out


def contingency_and_ets(forecast, observed, threshold):
	fcst_event = forecast >= threshold
	obs_event = observed >= threshold

	hits = int(np.sum(fcst_event & obs_event))
	misses = int(np.sum(~fcst_event & obs_event))
	false_alarms = int(np.sum(fcst_event & ~obs_event))
	correct_negatives = int(np.sum(~fcst_event & ~obs_event))

	n = hits + misses + false_alarms + correct_negatives

	hits_random = (
		((hits + misses) * (hits + false_alarms)) / n
		if n > 0
		else np.nan
	)

	denom = hits + misses + false_alarms - hits_random
	ets = (
		(hits - hits_random) / denom
		if np.isfinite(hits_random) and denom != 0
		else np.nan
	)

	return {
		"threshold_mm": threshold,
		"n_pairs": n,
		"hits": hits,
		"misses": misses,
		"false_alarms": false_alarms,
		"correct_negatives": correct_negatives,
		"hits_random": hits_random,
		"ETS": ets,
	}


def main():
	OUTPUT_FIG.parent.mkdir(parents=True, exist_ok=True)
	OUTPUT_STATS_CSV.parent.mkdir(parents=True, exist_ok=True)

	all_stats = []

	fig, ax = plt.subplots(figsize=(10.5, 6.5))

	for model_name, path in MODEL_FILES.items():
		if not path.exists():
			raise FileNotFoundError(f"Missing file for {model_name}: {path}")

		df = read_model_pairs(model_name, path)
		forecast = df["forecast_mm"].to_numpy()
		observed = df["observed_mm"].to_numpy()

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
	ax.set_ylim(0,0.7)

	ax.set_xlabel(
		"Precipitation event threshold (mm)",
		fontsize=13,
		fontweight="bold",
	)
	ax.set_ylabel(
		"ETS",
		fontsize=13,
		fontweight="bold",
	)

	#ax.set_title(
	#	"24-h Precipitation ETS by Event Threshold",
	#	fontsize=14,
	#	fontweight="bold",
	#)

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

	plt.tight_layout()
	plt.savefig(OUTPUT_FIG, dpi=300, bbox_inches="tight")
	plt.close(fig)

	stats_out = pd.DataFrame(all_stats)
	stats_out = stats_out[
		[
			"model",
			"threshold_mm",
			"n_pairs",
			"hits",
			"misses",
			"false_alarms",
			"correct_negatives",
			"hits_random",
			"ETS",
		]
	]
	stats_out.to_csv(OUTPUT_STATS_CSV, index=False)

	ets_table = stats_out.pivot(
		index="model",
		columns="threshold_mm",
		values="ETS",
	)
	ets_table = ets_table.reindex(index=MODEL_FILES.keys(), columns=EVENT_THRESHOLDS_MM)
	ets_table.columns = [f"{threshold:g} mm" for threshold in ets_table.columns]

	print("\nETS by model and precipitation threshold:")
	print(ets_table.to_string(float_format=lambda value: f"{value:.4f}"))

	print(f"\nSaved figure to: {OUTPUT_FIG}")
	print(f"Saved stats to: {OUTPUT_STATS_CSV}")


if __name__ == "__main__":
	main()