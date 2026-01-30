"""CLI for comprehensive model validation."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from resource_estimator.logging_config import setup_logging
from resource_estimator.model import create_prediction_function, prepare_training_data, train_polynomial_model
from resource_estimator.utils import load_data_from_csv

logger = setup_logging()


def validate_model_predictions(data, predict_fn, max_error_pct=50.0, max_negative=0):
	"""Validate model predictions against actual data.

	Args:
		data: List of data points with actual QPU seconds
		predict_fn: Prediction function to test
		max_error_pct: Maximum allowed error percentage (default: 50%)
		max_negative: Maximum allowed negative predictions (default: 0)

	Returns:
		True if validation passes, False otherwise
	"""
	errors = []
	negative_count = 0
	predictions = []
	actuals = []

	logger.info(f"Validating model on {len(data)} data points...")

	for i, row in data.iterrows():
		# Extract parameters
		qubits = row.get("num_qubits") or row.get("qubits")
		depth = row["depth"]
		batches = row.get("batches") or row.get("num_circuits")
		shots = row["shots"]
		actual = row["qpu_seconds"]

		# Make prediction
		predicted = predict_fn(qubits, depth, batches, shots)

		# Track predictions
		predictions.append(predicted)
		actuals.append(actual)

		# Check for negative predictions
		if predicted < 0:
			negative_count += 1
			logger.error(f"Row {i}: NEGATIVE prediction {predicted:.2f}s (actual: {actual:.2f}s)")
			logger.error(f"  Parameters: q={qubits}, d={depth}, b={batches}, s={shots}")

		# Calculate error
		error_pct = abs(predicted - actual) / actual * 100
		errors.append(error_pct)

		# Log problematic predictions
		if error_pct > max_error_pct:
			logger.warning(f"Row {i}: High error {error_pct:.1f}% (predicted: {predicted:.2f}s, actual: {actual:.2f}s)")
			logger.warning(f"  Parameters: q={qubits}, d={depth}, b={batches}, s={shots}")

	# Calculate statistics
	errors_array = np.array(errors)
	predictions_array = np.array(predictions)
	actuals_array = np.array(actuals)

	mean_error = np.mean(errors_array)
	median_error = np.median(errors_array)
	max_error = np.max(errors_array)
	min_error = np.min(errors_array)

	# Print summary
	logger.info("\n" + "=" * 60)
	logger.info("VALIDATION SUMMARY")
	logger.info("=" * 60)
	logger.info(f"Total data points: {len(data)}")
	logger.info(f"Negative predictions: {negative_count}")
	logger.info(f"\nError Statistics:")
	logger.info(f"  Mean error: {mean_error:.1f}%")
	logger.info(f"  Median error: {median_error:.1f}%")
	logger.info(f"  Min error: {min_error:.1f}%")
	logger.info(f"  Max error: {max_error:.1f}%")
	logger.info(f"\nPrediction Range:")
	logger.info(f"  Min predicted: {predictions_array.min():.2f}s")
	logger.info(f"  Max predicted: {predictions_array.max():.2f}s")
	logger.info(f"  Min actual: {actuals_array.min():.2f}s")
	logger.info(f"  Max actual: {actuals_array.max():.2f}s")

	# Check validation criteria
	validation_passed = True
	logger.info("\n" + "=" * 60)
	logger.info("VALIDATION CRITERIA")
	logger.info("=" * 60)

	# Check 1: No negative predictions
	if negative_count > max_negative:
		logger.error(f"❌ FAILED: {negative_count} negative predictions (max allowed: {max_negative})")
		validation_passed = False
	else:
		logger.info("✅ PASSED: No negative predictions")

	# Check 2: Mean error within threshold
	if mean_error > max_error_pct:
		logger.error(f"❌ FAILED: Mean error {mean_error:.1f}% exceeds threshold {max_error_pct}%")
		validation_passed = False
	else:
		logger.info(f"✅ PASSED: Mean error {mean_error:.1f}% within threshold {max_error_pct}%")

	# Check 3: All predictions are finite
	if not np.all(np.isfinite(predictions_array)):
		logger.error("❌ FAILED: Model produces non-finite predictions (inf/nan)")
		validation_passed = False
	else:
		logger.info("✅ PASSED: All predictions are finite")

	# Check 4: All predictions are positive
	if np.any(predictions_array < 0):
		logger.error("❌ FAILED: Model produces negative predictions")
		validation_passed = False
	else:
		logger.info("✅ PASSED: All predictions are positive")

	logger.info("=" * 60)

	if validation_passed:
		logger.info("🎉 VALIDATION PASSED!")
	else:
		logger.error("💥 VALIDATION FAILED!")

	return validation_passed


def generate_validation_plots(data, predict_fn, output_dir: Path):
	"""Generate plots showing how QPU seconds varies with each feature.

	Args:
		data: DataFrame with collected data
		predict_fn: Prediction function
		output_dir: Directory to save plots
	"""
	logger.info(f"\nGenerating validation plots in {output_dir}...")
	output_dir.mkdir(parents=True, exist_ok=True)

	# Prepare data
	df = data.copy()
	if "num_qubits" in df.columns and "qubits" not in df.columns:
		df["qubits"] = df["num_qubits"]
	if "num_circuits" in df.columns and "batches" not in df.columns:
		df["batches"] = df["num_circuits"]
	df["kshots"] = df["shots"] / 1000.0

	# Add predictions
	predictions = []
	for _, row in df.iterrows():
		pred = predict_fn(row["qubits"], row["depth"], row["batches"], row["shots"])
		predictions.append(pred)
	df["predicted_qpu_seconds"] = predictions

	# Create figure with 2x2 subplots
	fig, axes = plt.subplots(2, 2, figsize=(14, 10))
	fig.suptitle("QPU Seconds vs Features (Actual vs Predicted)", fontsize=16, fontweight="bold")

	features = [
		("qubits", "Number of Qubits", axes[0, 0]),
		("depth", "Circuit Depth", axes[0, 1]),
		("batches", "Number of Batches", axes[1, 0]),
		("kshots", "Shots (thousands)", axes[1, 1]),
	]

	for feature, label, ax in features:
		# Plot actual data
		ax.scatter(
			df[feature],
			df["qpu_seconds"],
			alpha=0.6,
			s=50,
			color="blue",
			label="Actual",
			edgecolors="black",
			linewidth=0.5,
		)

		# Plot predictions
		ax.scatter(
			df[feature], df["predicted_qpu_seconds"], alpha=0.6, s=50, color="red", label="Predicted", marker="x"
		)

		ax.set_xlabel(label, fontsize=11, fontweight="bold")
		ax.set_ylabel("QPU Seconds", fontsize=11, fontweight="bold")
		ax.legend(loc="best", fontsize=9)
		ax.grid(True, alpha=0.3, linestyle="--")

		# Add correlation coefficient
		corr = np.corrcoef(df[feature], df["qpu_seconds"])[0, 1]
		ax.text(
			0.05,
			0.95,
			f"Correlation: {corr:.3f}",
			transform=ax.transAxes,
			fontsize=9,
			verticalalignment="top",
			bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
		)

	plt.tight_layout()
	plot_path = output_dir / "feature_analysis.png"
	plt.savefig(plot_path, dpi=150, bbox_inches="tight")
	plt.close()
	logger.info(f"Saved feature analysis plot to {plot_path}")

	# Create prediction vs actual plot
	fig, ax = plt.subplots(figsize=(8, 8))
	ax.scatter(df["qpu_seconds"], df["predicted_qpu_seconds"], alpha=0.6, s=50, edgecolors="black", linewidth=0.5)

	# Add diagonal line (perfect prediction)
	min_val = min(df["qpu_seconds"].min(), df["predicted_qpu_seconds"].min())
	max_val = max(df["qpu_seconds"].max(), df["predicted_qpu_seconds"].max())
	ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect Prediction")

	ax.set_xlabel("Actual QPU Seconds", fontsize=12, fontweight="bold")
	ax.set_ylabel("Predicted QPU Seconds", fontsize=12, fontweight="bold")
	ax.set_title("Predicted vs Actual QPU Seconds", fontsize=14, fontweight="bold")
	ax.legend(loc="best", fontsize=10)
	ax.grid(True, alpha=0.3, linestyle="--")

	# Add R² score
	from sklearn.metrics import r2_score

	r2 = r2_score(df["qpu_seconds"], df["predicted_qpu_seconds"])
	ax.text(
		0.05,
		0.95,
		f"R² = {r2:.4f}",
		transform=ax.transAxes,
		fontsize=12,
		verticalalignment="top",
		bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
	)

	plt.tight_layout()
	plot_path = output_dir / "predicted_vs_actual.png"
	plt.savefig(plot_path, dpi=150, bbox_inches="tight")
	plt.close()
	logger.info(f"Saved predicted vs actual plot to {plot_path}")

	# Create residual plot
	fig, ax = plt.subplots(figsize=(10, 6))
	residuals = df["predicted_qpu_seconds"] - df["qpu_seconds"]
	ax.scatter(df["qpu_seconds"], residuals, alpha=0.6, s=50, edgecolors="black", linewidth=0.5)
	ax.axhline(y=0, color="r", linestyle="--", linewidth=2)
	ax.set_xlabel("Actual QPU Seconds", fontsize=12, fontweight="bold")
	ax.set_ylabel("Residual (Predicted - Actual)", fontsize=12, fontweight="bold")
	ax.set_title("Residual Plot", fontsize=14, fontweight="bold")
	ax.grid(True, alpha=0.3, linestyle="--")

	# Add statistics
	mean_residual = residuals.mean()
	std_residual = residuals.std()
	ax.text(
		0.05,
		0.95,
		f"Mean: {mean_residual:.2f}\nStd: {std_residual:.2f}",
		transform=ax.transAxes,
		fontsize=10,
		verticalalignment="top",
		bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
	)

	plt.tight_layout()
	plot_path = output_dir / "residuals.png"
	plt.savefig(plot_path, dpi=150, bbox_inches="tight")
	plt.close()
	logger.info(f"Saved residual plot to {plot_path}")

	logger.info(f"\n✅ Generated 3 validation plots in {output_dir}")


def main():
	"""Main entry point for validation."""
	parser = argparse.ArgumentParser(description="Validate quantum resource estimation model")
	parser.add_argument("--data", type=str, required=True, help="Input CSV file to validate against")
	parser.add_argument("--degree", type=int, default=3, help="Polynomial degree (default: 3)")
	parser.add_argument("--alpha", type=float, default=0.01, help="Regularization strength (default: 0.01)")
	parser.add_argument(
		"--max-error", type=float, default=50.0, help="Maximum allowed mean error percentage (default: 50%%)"
	)
	parser.add_argument("--max-negative", type=int, default=0, help="Maximum allowed negative predictions (default: 0)")
	parser.add_argument(
		"--plots", type=str, help="Output directory for plots (e.g., plots/). If not specified, no plots generated."
	)
	args = parser.parse_args()

	try:
		# Load data
		data = load_data_from_csv(args.data)

		# Prepare training data and train model
		X, y = prepare_training_data(data)
		model, poly, metrics = train_polynomial_model(X, y, args.degree, args.alpha)

		# Create prediction function
		feature_names = X.columns.tolist()
		predict_fn = create_prediction_function(model, poly, feature_names)

		# Validate
		validation_passed = validate_model_predictions(data, predict_fn, args.max_error, args.max_negative)

		# Generate plots if requested
		if args.plots:
			generate_validation_plots(data, predict_fn, Path(args.plots))

		# Exit with appropriate code
		sys.exit(0 if validation_passed else 1)

	except Exception as e:
		logger.error(f"Error during validation: {e}", exc_info=True)
		sys.exit(1)


if __name__ == "__main__":
	main()
