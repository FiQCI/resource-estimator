"""Plotting utilities for resource estimation models."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def plot_actual_vs_predicted_simple(
	y_true: np.ndarray, y_pred: np.ndarray, device_name: str, r2: float, output_path: Path
) -> None:
	"""Generate simple actual vs predicted plot for documentation.

	Args:
		y_true: Actual QPU seconds
		y_pred: Predicted QPU seconds
		device_name: Name of the device
		r2: R² score
		output_path: Where to save the plot
	"""
	plt.figure(figsize=(8, 6))
	plt.scatter(y_true, y_pred, alpha=0.6, edgecolors="k", linewidth=0.5)

	min_val = min(y_true.min(), y_pred.min())
	max_val = max(y_true.max(), y_pred.max())
	plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Prediction")

	plt.xlabel("Actual QPU Seconds", fontsize=12)
	plt.ylabel("Predicted QPU Seconds", fontsize=12)
	plt.title(device_name, fontsize=14, fontweight="bold")

	metrics_text = f"R² = {r2:.4f}"
	plt.text(
		0.05,
		0.95,
		metrics_text,
		transform=plt.gca().transAxes,
		fontsize=11,
		verticalalignment="top",
		bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
	)

	plt.legend()
	plt.grid(True, alpha=0.3)
	plt.tight_layout()

	plt.savefig(output_path, dpi=300, bbox_inches="tight")
	logger.info(f"Saved public plot: {output_path}")
	plt.close()


def plot_analytical_model_3panel(
	y_true: np.ndarray,
	y_pred: np.ndarray,
	batches: np.ndarray,
	efficiency: np.ndarray,
	r2: float,
	rmse: float,
	mae: float,
	output_path: Path,
) -> None:
	"""Generate 3-panel plot for analytical model (technical analysis).

	Args:
		y_true: Actual QPU seconds
		y_pred: Predicted QPU seconds
		batches: Batch sizes
		efficiency: Efficiency values per batch
		r2: R² score
		rmse: Root mean squared error
		mae: Mean absolute error
		output_path: Where to save the plot
	"""
	_ = plt.figure(figsize=(14, 5))

	# Plot 1: Actual vs Predicted
	plt.subplot(1, 3, 1)
	plt.scatter(y_true, y_pred, alpha=0.5, s=20)

	min_val = min(y_true.min(), y_pred.min())
	max_val = max(y_true.max(), y_pred.max())
	plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Prediction")

	plt.xlabel("Actual QPU Seconds")
	plt.ylabel("Predicted QPU Seconds")
	plt.title(f"Analytical Model: R² = {r2:.4f}")
	plt.legend()
	plt.grid(True, alpha=0.3)

	# Plot 2: Residuals
	plt.subplot(1, 3, 2)
	residuals = y_pred - y_true
	plt.scatter(y_pred, residuals, alpha=0.5, s=20)
	plt.axhline(y=0, color="r", linestyle="--", lw=2)
	plt.xlabel("Predicted QPU Seconds")
	plt.ylabel("Residuals (Predicted - Actual)")
	plt.title(f"Residual Plot (RMSE = {rmse:.2f}s)")
	plt.grid(True, alpha=0.3)

	# Plot 3: Efficiency Curve
	plt.subplot(1, 3, 3)
	unique_batches = np.unique(batches)
	avg_efficiency = [efficiency[batches == b].mean() for b in unique_batches]

	plt.plot(unique_batches, avg_efficiency, "o-", linewidth=2, markersize=6)
	plt.xlabel("Batch Size")
	plt.ylabel("Batching Efficiency η(B)")
	plt.title("Efficiency vs Batch Size")
	plt.grid(True, alpha=0.3)
	plt.ylim([0, 1.05])

	plt.tight_layout()
	plt.savefig(output_path, dpi=150, bbox_inches="tight")
	logger.info(f"Saved plot: {output_path}")
	plt.close()


def plot_polynomial_model(
	y_true: np.ndarray,
	y_pred: np.ndarray,
	device_name: str,
	test_r2: float,
	test_rmse: float,
	test_mae: float,
	degree: int,
	num_terms: int,
	output_path: Path,
) -> None:
	"""Generate plot for polynomial model.

	Args:
		y_true: Actual QPU seconds
		y_pred: Predicted QPU seconds
		device_name: Name of the device
		test_r2: R² score
		test_rmse: Root mean squared error
		test_mae: Mean absolute error
		degree: Polynomial degree
		num_terms: Number of terms in model
		output_path: Where to save the plot
	"""
	_ = plt.figure(figsize=(12, 5))

	# Plot 1: Actual vs Predicted
	plt.subplot(1, 2, 1)
	plt.scatter(y_true, y_pred, alpha=0.6, edgecolors="k", linewidth=0.5)

	min_val = min(y_true.min(), y_pred.min())
	max_val = max(y_true.max(), y_pred.max())
	plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Prediction")

	plt.xlabel("Actual QPU Seconds", fontsize=12)
	plt.ylabel("Predicted QPU Seconds", fontsize=12)
	plt.title(f"{device_name} - Polynomial Model (degree={degree})", fontsize=13, fontweight="bold")
	plt.legend()
	plt.grid(True, alpha=0.3)

	# Add metrics text box
	metrics_text = f"R² = {test_r2:.4f}\\n"
	metrics_text += f"RMSE = {test_rmse:.2f}s\\n"
	metrics_text += f"MAE = {test_mae:.2f}s\\n"
	metrics_text += f"Terms: {num_terms}"

	plt.text(
		0.05,
		0.95,
		metrics_text,
		transform=plt.gca().transAxes,
		fontsize=10,
		verticalalignment="top",
		bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
	)

	# Plot 2: Residuals
	plt.subplot(1, 2, 2)
	residuals = y_pred - y_true
	plt.scatter(y_pred, residuals, alpha=0.6, edgecolors="k", linewidth=0.5)
	plt.axhline(y=0, color="r", linestyle="--", lw=2)
	plt.xlabel("Predicted QPU Seconds", fontsize=12)
	plt.ylabel("Residuals (Predicted - Actual)", fontsize=12)
	plt.title(f"Residual Plot", fontsize=13)
	plt.grid(True, alpha=0.3)

	plt.tight_layout()
	plt.savefig(output_path, dpi=150, bbox_inches="tight")
	logger.info(f"Saved plot: {output_path}")
	plt.close()
