"""Validation and visualization module."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

logger = logging.getLogger(__name__)


def validate_model_predictions(model: Ridge, poly: PolynomialFeatures, X: pd.DataFrame, y: np.ndarray) -> dict:
	"""Validate model and calculate metrics.

	Args:
		model: Trained model
		poly: Polynomial transformer
		X: Features
		y: True values

	Returns:
		Dictionary of validation metrics
	"""
	X_poly = poly.transform(X)
	y_pred = model.predict(X_poly)

	errors = y_pred - y
	safe_divisor = np.maximum(1e-3, np.abs(y))
	mape = np.mean(np.abs(errors / safe_divisor)) * 100

	return {
		"r2_score": r2_score(y, y_pred),
		"rmse": np.sqrt(mean_squared_error(y, y_pred)),
		"mae": np.mean(np.abs(errors)),
		"mape": mape,
		"y_pred": y_pred,
		"errors": errors,
	}


def plot_actual_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray, device_name: str, output_path: Path):
	"""Plot actual vs predicted values.

	Args:
		y_true: Actual values
		y_pred: Predicted values
		device_name: Device name for title
		output_path: Output file path
	"""
	plt.figure(figsize=(10, 6))
	plt.scatter(y_true, y_pred, alpha=0.5, s=30)

	max_val = max(np.max(y_true), np.max(y_pred))
	plt.plot([0, max_val], [0, max_val], "k--", linewidth=2, label="Perfect prediction")

	plt.xlabel("Actual QPU Seconds", fontsize=12)
	plt.ylabel("Predicted QPU Seconds", fontsize=12)
	plt.title(f"Actual vs Predicted QPU Seconds ({device_name})", fontsize=14)
	plt.legend()
	plt.grid(alpha=0.3)
	plt.tight_layout()

	plt.savefig(output_path, dpi=150)
	plt.close()
	logger.info(f"Saved plot: {output_path}")


def plot_error_distribution(errors: np.ndarray, output_path: Path):
	"""Plot error distribution histogram.

	Args:
		errors: Prediction errors
		output_path: Output file path
	"""
	plt.figure(figsize=(10, 6))
	plt.hist(errors, bins=30, edgecolor="black", alpha=0.7)
	plt.axvline(x=0, color="r", linestyle="--", linewidth=2, label="Zero error")

	plt.xlabel("Prediction Error (seconds)", fontsize=12)
	plt.ylabel("Frequency", fontsize=12)
	plt.title("Prediction Error Distribution", fontsize=14)
	plt.legend()
	plt.grid(axis="y", alpha=0.3)
	plt.tight_layout()

	plt.savefig(output_path, dpi=150)
	plt.close()
	logger.info(f"Saved plot: {output_path}")


def plot_error_vs_parameter(param_values: np.ndarray, errors: np.ndarray, param_name: str, output_path: Path):
	"""Plot errors vs a parameter.

	Args:
		param_values: Parameter values
		errors: Prediction errors
		param_name: Parameter name
		output_path: Output file path
	"""
	plt.figure(figsize=(10, 6))
	plt.scatter(param_values, errors, alpha=0.5, s=30)
	plt.axhline(y=0, color="k", linestyle="--", linewidth=2)

	plt.xlabel(param_name, fontsize=12)
	plt.ylabel("Prediction Error (seconds)", fontsize=12)
	plt.title(f"Error vs {param_name}", fontsize=14)
	plt.grid(alpha=0.3)
	plt.tight_layout()

	plt.savefig(output_path, dpi=150)
	plt.close()
	logger.info(f"Saved plot: {output_path}")


def calculate_feature_importance(model: Ridge, poly: PolynomialFeatures, feature_names: list[str]) -> dict[str, float]:
	"""Calculate relative feature importance.

	Args:
		model: Trained model
		poly: Polynomial transformer
		feature_names: Feature names

	Returns:
		Dictionary of feature importance scores
	"""
	coefficients = model.coef_
	poly_names = poly.get_feature_names_out(feature_names)

	importance = {f: 0.0 for f in feature_names}

	for term, coef in zip(poly_names, coefficients):
		if term != "1":
			for feature in feature_names:
				if feature in term:
					importance[feature] += abs(coef)

	total = sum(importance.values())
	if total > 0:
		importance = {f: v / total for f, v in importance.items()}

	return importance


def plot_feature_importance(importance: dict[str, float], output_path: Path):
	"""Plot feature importance.

	Args:
		importance: Feature importance dictionary
		output_path: Output file path
	"""
	features = list(importance.keys())
	values = list(importance.values())
	indices = np.argsort(values)

	plt.figure(figsize=(10, 6))
	plt.barh(range(len(indices)), [values[i] for i in indices], align="center")
	plt.yticks(range(len(indices)), [features[i] for i in indices])
	plt.xlabel("Relative Importance", fontsize=12)
	plt.title("Feature Importance", fontsize=14)
	plt.tight_layout()

	plt.savefig(output_path, dpi=150)
	plt.close()
	logger.info(f"Saved plot: {output_path}")


def generate_all_plots(data: pd.DataFrame, model: Ridge, poly: PolynomialFeatures, device_name: str, output_dir: Path):
	"""Generate all validation plots.

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
