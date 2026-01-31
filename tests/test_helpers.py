"""Test helper functions for backward compatibility."""

import logging
from pathlib import Path

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

from resource_estimator.validation import (
	validate_model_predictions,
	plot_actual_vs_predicted,
	plot_error_distribution,
	plot_error_vs_parameter,
	calculate_feature_importance,
	plot_feature_importance,
)

logger = logging.getLogger(__name__)


def generate_all_plots(data: pd.DataFrame, model: Ridge, poly: PolynomialFeatures, device_name: str, output_dir: Path):
	"""Generate all validation plots (test helper for backward compatibility).

	Args:
		data: Input data
		model: Trained model
		poly: Polynomial transformer
		device_name: Device name
		output_dir: Output directory
	"""
	output_dir.mkdir(parents=True, exist_ok=True)

	# Prepare features
	feature_cols = ["qubits", "depth", "batches", "kshots"]
	X = data[feature_cols]
	y = data["qpu_seconds"].values

	# Validate
	results = validate_model_predictions(model, poly, X, y)

	# Plot actual vs predicted
	device_slug = device_name.lower().replace(" ", "-")
	plot_actual_vs_predicted(y, results["y_pred"], device_name, output_dir / f"actual_vs_predicted-{device_slug}.png")

	# Plot error distribution
	plot_error_distribution(results["errors"], output_dir / "error_distribution.png")

	# Plot error vs parameters
	plot_error_vs_parameter(
		data["qubits"].values, results["errors"], "Number of Qubits", output_dir / "error_vs_qubits.png"
	)

	plot_error_vs_parameter(data["kshots"].values, results["errors"], "K-Shots", output_dir / "error_vs_kshots.png")

	# Plot feature importance
	importance = calculate_feature_importance(model, poly, feature_cols)
	plot_feature_importance(importance, output_dir / "feature_importance.png")

	logger.info(f"Generated all plots in {output_dir}")
	return results
