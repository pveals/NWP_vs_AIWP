#!/usr/bin/env python3
"""
Multi-model performance diagram for 24-h forecasted precip vs observed precip.
Saves PNG only; no display window.
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

OUTPUT_FIG = Path(
	"./Fig7.png"
)

OUTPUT_STATS_CSV = Path(
	"./Fig7_stats_allevents.csv"
)

# Only include forecast-observed pairs with observed precipitation >= this threshold.
# Set to None to disable this observed-precipitation filter.
MIN_OBSERVED_PRECIP_MM = 0.0

EVENT_THRESHOLDS_MM = None
OBS_PERCENTILES = [30, 70, 95, 98]
PERCENTILES_FROM_NONZERO_OBS = True
THRESHOLD_MODE = "pooled_obs"


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


def contingency_stats(forecast, observed, threshold):
	fcst_event = forecast >= threshold
	obs_event = observed >= threshold

	hits = np.sum(fcst_event & obs_event)
	misses = np.sum(~fcst_event & obs_event)
	false_alarms = np.sum(fcst_event & ~obs_event)
	correct_negatives = np.sum(~fcst_event & ~obs_event)

	pod = hits / (hits + misses) if (hits + misses) > 0 else np.nan
	success_ratio = hits / (hits + false_alarms) if (hits + false_alarms) > 0 else np.nan
	far = false_alarms / (hits + false_alarms) if (hits + false_alarms) > 0 else np.nan
	csi = hits / (hits + misses + false_alarms) if (hits + misses + false_alarms) > 0 else np.nan
	bias = (hits + false_alarms) / (hits + misses) if (hits + misses) > 0 else np.nan

	return {
		"hits": hits,
		"misses": misses,
		"false_alarms": false_alarms,
		"correct_negatives": correct_negatives,
		"POD": pod,
		"success_ratio": success_ratio,
		"FAR": far,
		"CSI": csi,
		"bias": bias,
	}


def format_bias_label(b):
	if b >= 1:
		return f"{b:.1f}"
	return f"{b:g}"


def add_performance_background(ax):
	sr = np.linspace(0.001, 1.0, 500)
	pod = np.linspace(0.001, 1.0, 500)
	SR, POD = np.meshgrid(sr, pod)

	CSI = 1.0 / ((1.0 / SR) + (1.0 / POD) - 1.0)
	csi_levels = np.arange(0.0, 1.01, 0.1)

	cf = ax.contourf(
		SR,
		POD,
		CSI,
		levels=csi_levels,
		cmap="Greys",
		alpha=0.65,
	)

	ax.contour(
		SR,
		POD,
		CSI,
		levels=csi_levels,
		colors="0.40",
		linewidths=0.9,
	)

	bias_values = [0.25, 0.5, 1.0, 2.0, 4.0]
	x = np.linspace(0.001, 1.0, 500)

	label_x_by_bias = {
		0.25: 0.62,
		0.5: 0.70,   # moved down-left
		1.0: 0.5,   # moved down-left
		2.0: 0.36,   # moved down-left
		4.0: 0.18,
	}

	base_break = 0.035

	for b in bias_values:
		y = b * x
		mask = y <= 1.0

		label_x = label_x_by_bias[b]
		label_y = b * label_x

		break_halfwidth = base_break / np.sqrt(1.0 + b**2)

		left_mask = (x <= label_x - break_halfwidth) & mask
		right_mask = (x >= label_x + break_halfwidth) & mask

		ax.plot(
			x[left_mask],
			y[left_mask],
			"--",
			color="0.35",
			linewidth=1.6,
		)

		ax.plot(
			x[right_mask],
			y[right_mask],
			"--",
			color="0.35",
			linewidth=1.6,
		)

		angle = np.degrees(np.arctan(b))

		ax.text(
			label_x,
			label_y,
			format_bias_label(b),
			fontsize=10,
			fontweight="bold",
			color="0.15",
			rotation=angle,
			rotation_mode="anchor",
			ha="center",
			va="center",
		)

	#ax.text(
	#	0.78,
	#	0.08,
	#	"CSI",
	#	color="0.15",
	#	fontsize=11,
	#	fontweight="bold",
	#)

	ax.text(
		0.45,
		0.50,
		"Frequency bias",
		color="0.15",
		fontsize=11,
		fontweight="bold",
		rotation=45,
		rotation_mode="anchor",
		ha="center",
		va="center",
	)

	return cf


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

	if MIN_OBSERVED_PRECIP_MM is not None:
		out = out[out["observed_mm"] >= float(MIN_OBSERVED_PRECIP_MM)].copy()

	after_obs_threshold = len(out)

	print(
		f"{model_name}: forecast_col='{forecast_col}', "
		f"observed_col='{observed_col}', n={after_obs_threshold}, "
		f"dropped_non_numeric_or_nan={before - after_dropna}, "
		f"dropped_obs_below_threshold={after_dropna - after_obs_threshold}, "
		f"min_observed_precip_mm={MIN_OBSERVED_PRECIP_MM}"
	)

	return out


def get_thresholds(model_data):
	if EVENT_THRESHOLDS_MM is not None:
		thresholds = np.asarray(EVENT_THRESHOLDS_MM, dtype=float)
		labels = [f"{t:.1f} mm" for t in thresholds]
		return thresholds, labels

	if THRESHOLD_MODE == "pooled_obs":
		all_obs = np.concatenate(
			[df["observed_mm"].to_numpy() for df in model_data.values()]
		)
		obs_for_pct = all_obs[all_obs > 0] if PERCENTILES_FROM_NONZERO_OBS else all_obs

		if len(obs_for_pct) == 0:
			raise ValueError(
				"No observed precip values > 0 found. "
				"Check observed-column detection or set COLUMN_OVERRIDES."
			)

		thresholds = np.percentile(obs_for_pct, OBS_PERCENTILES)
		labels = [f"{p}th" for p in OBS_PERCENTILES]
		return thresholds, labels

	if THRESHOLD_MODE == "per_model_obs":
		return None, None

	raise ValueError("THRESHOLD_MODE must be 'pooled_obs' or 'per_model_obs'.")


def get_label_offset(model_name, label):
	offset = (6, 5)

	if model_name == "AIFS-d":
		if label == "70th":
			offset = (-24, 3)
		elif label == "95th":
			offset = (5, -4)
		elif label == "98th":
			offset = (7, -2)

	if model_name == "IFS-d":
		if label == "98th":
			offset = (-22, -6)
		elif label == "95th":
			offset = (-19, -10)
		elif label == "70th":
			offset = (-20, 2)

	if model_name == "Graph-d":
		if label == "70th":
			offset = (-23, 0)
		elif label == "30th":
			offset = (-21, -8)
		elif label == "95th":
			offset = (-20, -10)
		elif label == "98th":
			offset = (-24, -5)

	if model_name == "GFS-d":
		if label == "30th":
			offset = (1, 8)
		elif label == "70th":
			offset = (0, -15)
		elif label == "98th":
			offset = (-12, 7)

	if model_name == "HRRR-d":
		if label == "95th":
			offset = (-18, 5)
		elif label == "98th":
			offset = (-21, 0)
		elif label == "70th":
			offset = (-24, 0)
		elif label == "30th":
			offset = (0, -10)

	if model_name == "Silurian-d":
		if label == "30th":
			offset = (7, 0)
		elif label == "70th":
			offset = (4, -5)
		elif label == "95th":
			offset = (1, -10)
		elif label == "98th":
			offset = (1, -10)

	return offset


def main():
	OUTPUT_FIG.parent.mkdir(parents=True, exist_ok=True)
	OUTPUT_STATS_CSV.parent.mkdir(parents=True, exist_ok=True)

	model_data = {}
	for model_name, path in MODEL_FILES.items():
		if not path.exists():
			raise FileNotFoundError(f"Missing file for {model_name}: {path}")
		model_data[model_name] = read_model_pairs(model_name, path)

	pooled_thresholds, pooled_labels = get_thresholds(model_data)

	fig, ax = plt.subplots(figsize=(10.5, 8.5))
	cf = add_performance_background(ax)

	all_stats = []

	for model_name, df in model_data.items():
		forecast = df["forecast_mm"].to_numpy()
		observed = df["observed_mm"].to_numpy()

		if THRESHOLD_MODE == "per_model_obs" and EVENT_THRESHOLDS_MM is None:
			obs_for_pct = observed[observed > 0] if PERCENTILES_FROM_NONZERO_OBS else observed

			if len(obs_for_pct) == 0:
				print(f"Skipping {model_name}: no valid observed precip values.")
				continue

			thresholds = np.percentile(obs_for_pct, OBS_PERCENTILES)
			threshold_labels = [f"{p}th" for p in OBS_PERCENTILES]
		else:
			thresholds = pooled_thresholds
			threshold_labels = pooled_labels

		model_stats = []
		for threshold, threshold_label in zip(thresholds, threshold_labels):
			s = contingency_stats(forecast, observed, threshold)
			s["model"] = model_name
			s["threshold_mm"] = threshold
			s["threshold_label"] = threshold_label
			s["n_pairs"] = len(df)
			s["min_observed_precip_mm"] = MIN_OBSERVED_PRECIP_MM
			model_stats.append(s)
			all_stats.append(s)

		stats_df = pd.DataFrame(model_stats)
		sr = stats_df["success_ratio"].to_numpy()
		pod = stats_df["POD"].to_numpy()

		style = MODEL_STYLES.get(model_name, {"marker": "o"})

		line, = ax.plot(
			sr,
			pod,
			marker=style["marker"],
			linewidth=2.4,
			markersize=7.5,
			label=model_name,
			zorder=10,
		)
		model_color = line.get_color()

		for i, label in enumerate(threshold_labels):
			if not (np.isfinite(sr[i]) and np.isfinite(pod[i])):
				continue

			# Omit only the GFS 95th-percentile label
			if model_name == "GFS-d" and label == "95th":
				continue

			ax.annotate(
				label,
				xy=(sr[i], pod[i]),
				xytext=get_label_offset(model_name, label),
				textcoords="offset points",
				fontsize=7,
				fontweight="bold",
				color=model_color,
				zorder=11,
			)




	cbar = plt.colorbar(
		cf,
		ax=ax,
		pad=0.02,
		ticks=np.arange(0.0, 1.01, 0.1),
	)
	cbar.set_label(
		"Critical Success Index (CSI)",
		fontsize=12,
		fontweight="bold",
	)
	cbar.ax.tick_params(labelsize=10, width=1.4)
	cbar.outline.set_linewidth(1.6)

	for label in cbar.ax.get_yticklabels():
		label.set_fontweight("bold")

	for spine in ax.spines.values():
		spine.set_linewidth(2.0)

	ax.set_xlim(0, 1)
	ax.set_ylim(0, 1)
	ax.set_aspect("equal", adjustable="box")

	ax.set_xlabel(
		"Success Ratio = Hits / (Hits + False Alarms)",
		fontsize=13,
		fontweight="bold",
	)
	ax.set_ylabel(
		"Probability of Detection = Hits / (Hits + Misses)",
		fontsize=13,
		fontweight="bold",
	)

	if EVENT_THRESHOLDS_MM is None:
		title_threshold_info = (
			f"Observed precip percentiles: {OBS_PERCENTILES}; "
			f"threshold mode: {THRESHOLD_MODE}"
		)
	else:
		title_threshold_info = f"Event thresholds: {EVENT_THRESHOLDS_MM} mm"

	if MIN_OBSERVED_PRECIP_MM is None:
		obs_filter_info = "Observed precip filter: none"
	else:
		obs_filter_info = f"Observed precip filter: obs >= {MIN_OBSERVED_PRECIP_MM:g} mm"

	#ax.set_title(
	#	"24-h Precipitation Performance Diagram\n"
	#	f"{title_threshold_info}\n"
	#	f"{obs_filter_info}",
	#	fontsize=13,
	#	fontweight="bold",
	#)

	ax.tick_params(axis="both", labelsize=11, width=1.6, length=5)

	for label in ax.get_xticklabels():
		label.set_fontweight("bold")

	for label in ax.get_yticklabels():
		label.set_fontweight("bold")

	ax.grid(True, color="0.82", linewidth=1.0)

	model_legend = ax.legend(
		loc="lower right",
		frameon=True,
		fontsize=10,
		title="Models",
		title_fontsize=11,
	)
	ax.add_artist(model_legend)

	if EVENT_THRESHOLDS_MM is None:
		threshold_handles = []
		threshold_legend_labels = []

		for p, t in zip(OBS_PERCENTILES, pooled_thresholds):
			threshold_handles.append(
				plt.Line2D([], [], linestyle="none")
			)
			threshold_legend_labels.append(f"{p}th percentile = {t:.1f} mm")

		threshold_legend = ax.legend(
			threshold_handles,
			threshold_legend_labels,
			loc="upper left",
			frameon=True,
			fontsize=14,
			title="Event Thresholds",
			title_fontsize=14,
			handlelength=0,
			handletextpad=0,
		)

		threshold_legend.get_title().set_fontweight("bold")
		for text in threshold_legend.get_texts():
			text.set_fontweight("bold")

	model_legend.get_title().set_fontweight("bold")
	for text in model_legend.get_texts():
		text.set_fontweight("bold")

	plt.tight_layout()
	plt.savefig(OUTPUT_FIG, dpi=300, bbox_inches="tight")
	plt.close(fig)

	stats_out = pd.DataFrame(all_stats)
	first_cols = [
		"model",
		"threshold_label",
		"threshold_mm",
		"n_pairs",
		"min_observed_precip_mm",
		"hits",
		"misses",
		"false_alarms",
		"correct_negatives",
		"POD",
		"success_ratio",
		"FAR",
		"CSI",
		"bias",
	]
	stats_out = stats_out[first_cols]
	stats_out.to_csv(OUTPUT_STATS_CSV, index=False)

	print("\nCSI by model and event threshold:")
	csi_table = stats_out.pivot(
		index="model",
		columns="threshold_label",
		values="CSI",
	)

	# Keep models and thresholds in the same order used in the plot.
	model_order = [m for m in MODEL_FILES if m in csi_table.index]
	threshold_order = [label for label in pooled_labels if label in csi_table.columns]
	csi_table = csi_table.reindex(index=model_order, columns=threshold_order)

	print(csi_table.to_string(float_format=lambda x: f"{x:.4f}"))

	print(f"\nSaved figure to: {OUTPUT_FIG}")
	print(f"Saved stats to: {OUTPUT_STATS_CSV}")


if __name__ == "__main__":
	main()
