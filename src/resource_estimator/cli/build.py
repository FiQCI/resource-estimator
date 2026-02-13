"""CLI for model building."""

import argparse
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.optimize import differential_evolution

from resource_estimator.logging_config import setup_logging
from resource_estimator.model import (
	extract_model_coefficients,
	format_javascript_model,
	prepare_training_data,
	train_polynomial_model,
)
from resource_estimator.utils import load_data_from_csv, save_model_as_json, update_javascript_model

logger = setup_logging()


def analytical_model(params, batches, shots):
	"""
	Analytical runtime model using batching efficiency formula.

	T = T_init + efficiency(batches) × batches × shots × throughput
	where efficiency(batches) = base^min(batches, cap)

	Args:
		params: [T_init, efficiency_base, throughput_coef, batch_cap]
		batches: Number of circuit batches
		shots: Number of shots per circuit

	Returns:
		Predicted runtime in seconds
	"""
	T_init, efficiency_base, throughput_coef, batch_cap = params

	# Clamp efficiency_base to (0, 1) range for stability
	efficiency_base = max(0.5, min(0.999, efficiency_base))

	# Batch efficiency: exponential decay capped at batch_cap
	efficiency = np.power(efficiency_base, np.minimum(batches, batch_cap))

	# Runtime calculation
	runtime = T_init + efficiency * batches * shots * throughput_coef

	return runtime


def objective(params, batches, shots, y_true):
	"""Objective function: minimize RMSE."""
	y_pred = analytical_model(params, batches, shots)
	return np.sqrt(mean_squared_error(y_true, y_pred))


def fit_analytical_model(data_path: str, output_dir: str):
	"""Fit analytical model to VTT Q50 data.

	Args:
		data_path: Path to CSV data file
		output_dir: Output directory for plots and configuration
	"""
	output_path = Path(output_dir)
	output_path.mkdir(parents=True, exist_ok=True)

	# Load data
	df = pd.read_csv(data_path)
	logger.info(f"Loaded {len(df)} samples from {data_path}")

	# Prepare features
	batches = df["batches"].values
	shots = df["shots"].values
	y_true = df["qpu_seconds"].values

	# Optimize using differential evolution
	logger.info("Optimizing analytical model parameters...")
	bounds = [
		(0.5, 5.0),  # T_init: 0.5-5 seconds
		(0.85, 0.99),  # efficiency_base: decay rate
		(0.0001, 0.001),  # throughput_coef: shot scaling
		(5.0, 20.0),  # batch_cap: cap for efficiency
	]

	result = differential_evolution(
		objective, bounds, args=(batches, shots, y_true), maxiter=1000, seed=42, polish=True, workers=1
	)

	optimal_params = result.x

	# Evaluate final model
	y_pred = analytical_model(optimal_params, batches, shots)
	r2 = r2_score(y_true, y_pred)
	rmse = np.sqrt(mean_squared_error(y_true, y_pred))
	mae = mean_absolute_error(y_true, y_pred)

	logger.info(f"Model performance: R²={r2:.4f}, RMSE={rmse:.2f}s, MAE={mae:.2f}s")

	# Generate JavaScript configuration
	js_config = {
		"name": "VTT Q50",
		"max_qubits": 54,
		"model_type": "analytical",
		"T_init": float(optimal_params[0]),
		"efficiency_base": float(optimal_params[1]),
		"throughput_coef": float(optimal_params[2]),
		"batch_cap": float(optimal_params[3]),
	}

	# Save JSON configuration
	config_path = output_path / "vtt-q50_analytical_config.json"
	with open(config_path, "w") as f:
		json.dump(js_config, f, indent=2)
	logger.info(f"Saved config: {config_path}")

	# Generate validation plot
	plt.figure(figsize=(14, 5))

	# Plot 1: Actual vs Predicted
	plt.subplot(1, 3, 1)
	plt.scatter(y_true, y_pred, alpha=0.5, s=20)
	min_val = min(y_true.min(), y_pred.min())
	max_val = max(y_true.max(), y_pred.max())
	plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=2)
	plt.xlabel("Actual QPU Seconds")
	plt.ylabel("Predicted QPU Seconds")
	plt.title(f"Analytical Model: R² = {r2:.4f}")
	plt.grid(True, alpha=0.3)

	# Plot 2: Residuals
	plt.subplot(1, 3, 2)
	residuals = y_true - y_pred
	plt.scatter(y_pred, residuals, alpha=0.5, s=20)
	plt.axhline(y=0, color="r", linestyle="--", lw=2)
	plt.xlabel("Predicted QPU Seconds")
	plt.ylabel("Residuals")
	plt.title("Residual Plot")
	plt.grid(True, alpha=0.3)

	# Plot 3: Efficiency curve
	plt.subplot(1, 3, 3)
	batch_range = np.arange(1, 26)
	efficiency = np.power(optimal_params[1], np.minimum(batch_range, optimal_params[3]))
	plt.plot(batch_range, efficiency, "b-", lw=2)
	plt.axvline(x=optimal_params[3], color="r", linestyle="--", alpha=0.5, label=f"Cap at {optimal_params[3]:.0f}")
	plt.xlabel("Number of Batches")
	plt.ylabel("Efficiency Factor")
	plt.title("Batching Efficiency Curve")
	plt.grid(True, alpha=0.3)
	plt.legend()

	plt.tight_layout()
	plot_path = output_path / "vtt-q50_analytical_model.png"
	plt.savefig(plot_path, dpi=150, bbox_inches="tight")
	logger.info(f"Saved plot: {plot_path}")
	plt.close()

	# Generate public-facing plot (simple actual vs predicted)
	plt.figure(figsize=(8, 6))
	plt.scatter(y_true, y_pred, alpha=0.6, edgecolors="k", linewidth=0.5)

	min_val = min(y_true.min(), y_pred.min())
	max_val = max(y_true.max(), y_pred.max())
	plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Prediction")

	plt.xlabel("Actual QPU Seconds", fontsize=12)
	plt.ylabel("Predicted QPU Seconds", fontsize=12)
	plt.title("VTT Q50", fontsize=14, fontweight="bold")

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

	public_plot_path = output_path / "actual_vs_predicted-vtt-q50.png"
	plt.savefig(public_plot_path, dpi=300, bbox_inches="tight")
	logger.info(f"Saved public plot: {public_plot_path}")
	plt.close()

	return js_config, r2, rmse, mae


def build_helmi_model(data_path: str, output_dir: str, max_qubits: int = 5):
	"""Build Helmi polynomial model with depth parameter.

	Args:
		data_path: Path to training data CSV
		output_dir: Output directory for plots and configuration
		max_qubits: Maximum qubits supported (default: 5)
	"""
	output_path = Path(output_dir)
	output_path.mkdir(parents=True, exist_ok=True)

	device = "helmi"
	device_name = "Helmi"

	config = {
		"data_path": data_path,
		"degree": 3,
		"alpha": 0.001,
		"max_qubits": 5,
		"rename_columns": {"num_qubits": "qubits", "num_circuits": "batches", "shots": "kshots"},
	}

	# Load data
	df = pd.read_csv(config["data_path"])

	# Handle different column names
	if config["rename_columns"]:
		df = df.rename(columns=config["rename_columns"])

	# Prepare features (WITH depth for Helmi)
	X = df[["qubits", "depth", "batches", "kshots"]].copy()
	y = df["qpu_seconds"].values

	# Split data
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	# Create polynomial features
	poly = PolynomialFeatures(degree=config["degree"], include_bias=False)
	X_train_poly = poly.fit_transform(X_train)
	X_test_poly = poly.transform(X_test)

	# Train model
	model = Ridge(alpha=config["alpha"])
	model.fit(X_train_poly, y_train)

	# Predictions
	y_test_pred = model.predict(X_test_poly)

	# Metrics
	test_r2 = r2_score(y_test, y_test_pred)
	test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
	test_mae = mean_absolute_error(y_test, y_test_pred)
	num_terms = X_train_poly.shape[1]

	# Print metrics
	logger.info(
		f"Polynomial model (degree {config['degree']}, alpha {config['alpha']}): R²={test_r2:.4f}, RMSE={test_rmse:.2f}s, {num_terms} terms"
	)

	# Generate JavaScript configuration for Helmi
	feature_names = poly.get_feature_names_out(["qubits", "depth", "batches", "kshots"])
	intercept = model.intercept_
	coefficients = model.coef_

	terms = []
	for coef, feature in zip(coefficients, feature_names):
		if abs(coef) > 1e-10:
			term = feature.replace(" ", "*")
			terms.append({"term": term, "coefficient": float(coef)})

	js_config = {
		"name": device_name,
		"max_qubits": config["max_qubits"],
		"model_type": "polynomial",
		"degree": config["degree"],
		"use_log_transform": False,
		"epsilon": 0,
		"intercept": float(intercept),
		"terms": terms,
	}

	# Save JSON configuration
	config_path = output_path / f"{device}_model_config.json"
	with open(config_path, "w") as f:
		json.dump(js_config, f, indent=2)
	logger.info(f"Saved config: {config_path}")

	# Generate validation plot
	plt.figure(figsize=(10, 6))
	plt.scatter(y_test, y_test_pred, alpha=0.6, edgecolors="k", linewidth=0.5)

	min_val = min(y_test.min(), y_test_pred.min())
	max_val = max(y_test.max(), y_test_pred.max())
	plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Prediction")

	plt.xlabel("Actual QPU Seconds", fontsize=12)
	plt.ylabel("Predicted QPU Seconds", fontsize=12)
	plt.title(
		f"{device_name} - Degree {config['degree']} Polynomial Model (with Depth)", fontsize=14, fontweight="bold"
	)

	metrics_text = f"R² = {test_r2:.4f}\\n"
	metrics_text += f"RMSE = {test_rmse:.2f}s\\n"
	metrics_text += f"MAE = {test_mae:.2f}s\\n"
	metrics_text += f"Terms = {num_terms}"

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

	plot_path = output_path / f"{device}_final_model.png"
	plt.savefig(plot_path, dpi=300, bbox_inches="tight")
	logger.info(f"Saved plot: {plot_path}")
	plt.close()


def build_vtt_q50_model(output_dir: str):
	"""Build VTT Q50 analytical model.

	Args:
		output_dir: Output directory for plots and configuration
	"""
	logger.info("=" * 70)
	logger.info("Building VTT Q50 Analytical Model")
	logger.info("=" * 70)

	output_path = Path(output_dir)
	output_path.mkdir(parents=True, exist_ok=True)

	logger.info("\\nVTT Q50")
	logger.info("-" * 70)

	vtt_data_path = "data_analysis/data/vtt-q50.csv"
	js_config, r2, rmse, mae = fit_analytical_model(vtt_data_path, str(output_path))

	logger.info("\\n" + "=" * 70)
	logger.info("VTT Q50 model building complete!")
	logger.info("=" * 70)
	logger.info(f"\\nOutputs saved to: {output_path}")


def build_all_models(helmi_data_path: str, vtt_data_path: str, output_dir: str):
	"""Build models for all devices with validation plots.

	Args:
		helmi_data_path: Path to Helmi training data CSV
		vtt_data_path: Path to VTT Q50 training data CSV
		output_dir: Output directory for plots and configurations
	"""
	logger.info("Building models for all devices...")

	output_path = Path(output_dir)
	output_path.mkdir(parents=True, exist_ok=True)

	# Build Helmi model (polynomial with depth)
	device = "helmi"
	device_name = "Helmi"
	logger.info(f"Building {device_name} polynomial model...")

	config = {
		"data_path": helmi_data_path,
		"degree": 3,
		"alpha": 0.001,
		"max_qubits": 5,
		"rename_columns": {"num_qubits": "qubits", "num_circuits": "batches", "shots": "kshots"},
	}

	# Load data
	df = pd.read_csv(config["data_path"])

	# Handle different column names
	if config["rename_columns"]:
		df = df.rename(columns=config["rename_columns"])

	# Prepare features (WITH depth for Helmi)
	X = df[["qubits", "depth", "batches", "kshots"]].copy()
	y = df["qpu_seconds"].values

	# Split data
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	# Create polynomial features
	poly = PolynomialFeatures(degree=config["degree"], include_bias=False)
	X_train_poly = poly.fit_transform(X_train)
	X_test_poly = poly.transform(X_test)

	# Train model
	model = Ridge(alpha=config["alpha"])
	model.fit(X_train_poly, y_train)

	# Predictions
	y_train_pred = model.predict(X_train_poly)
	y_test_pred = model.predict(X_test_poly)

	# Metrics
	train_r2 = r2_score(y_train, y_train_pred)
	test_r2 = r2_score(y_test, y_test_pred)
	train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
	test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
	test_mae = mean_absolute_error(y_test, y_test_pred)
	num_terms = X_train_poly.shape[1]
	num_negative = np.sum(y_test_pred < 0)

	# Print metrics
	logger.info(f"  Model: Polynomial (with depth)")
	logger.info(f"  Degree: {config['degree']}")
	logger.info(f"  Alpha: {config['alpha']}")
	logger.info(f"  Train R²: {train_r2:.4f}")
	logger.info(f"  Test R²: {test_r2:.4f}")
	logger.info(f"  Train RMSE: {train_rmse:.2f}s")
	logger.info(f"  Test RMSE: {test_rmse:.2f}s")
	logger.info(f"  Test MAE: {test_mae:.2f}s")
	logger.info(f"  Number of terms: {num_terms}")
	logger.info(f"  Negative predictions: {num_negative}")

	# Generate JavaScript configuration for Helmi (keep original format)
	feature_names = poly.get_feature_names_out(["qubits", "depth", "batches", "kshots"])
	intercept = model.intercept_
	coefficients = model.coef_

	terms = []
	for coef, feature in zip(coefficients, feature_names):
		if abs(coef) > 1e-10:
			term = feature.replace(" ", "*")
			terms.append({"term": term, "coefficient": float(coef)})

	js_config = {
		"name": device_name,
		"max_qubits": config["max_qubits"],
		"model_type": "polynomial",
		"degree": config["degree"],
		"use_log_transform": False,
		"epsilon": 0,
		"intercept": float(intercept),
		"terms": terms,
	}

	# Save JSON configuration
	config_path = output_path / f"{device}_model_config.json"
	with open(config_path, "w") as f:
		json.dump(js_config, f, indent=2)
	logger.info(f"  Config saved: {config_path}")

	# Generate validation plot for Helmi
	plt.figure(figsize=(10, 6))
	plt.scatter(y_test, y_test_pred, alpha=0.6, edgecolors="k", linewidth=0.5)

	min_val = min(y_test.min(), y_test_pred.min())
	max_val = max(y_test.max(), y_test_pred.max())
	plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Prediction")

	plt.xlabel("Actual QPU Seconds", fontsize=12)
	plt.ylabel("Predicted QPU Seconds", fontsize=12)
	plt.title(
		f"{device_name} - Degree {config['degree']} Polynomial Model (with Depth)", fontsize=14, fontweight="bold"
	)

	metrics_text = f"R² = {test_r2:.4f}\\n"
	metrics_text += f"RMSE = {test_rmse:.2f}s\\n"
	metrics_text += f"MAE = {test_mae:.2f}s\\n"
	metrics_text += f"Terms = {num_terms}"

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

	plot_path = output_path / f"{device}_final_model.png"
	plt.savefig(plot_path, dpi=300, bbox_inches="tight")
	logger.info(f"Saved plot: {plot_path}")
	plt.close()

	# Build VTT Q50 analytical model
	logger.info("Building VTT Q50 analytical model...")
	js_config_vtt, r2, rmse, mae = fit_analytical_model(vtt_data_path, str(output_path))

	logger.info(f"All models saved to: {output_path}")


def main():
	"""Main entry point for model building."""
	parser = argparse.ArgumentParser(description="Build quantum resource estimation model")
	parser.add_argument("--data", type=str, required=True, help="Input CSV file")
	parser.add_argument("--device", type=str, required=True, help="Device identifier (e.g., 'helmi', 'vtt-q50')")
	parser.add_argument("--device-name", type=str, help="Display name (defaults to device ID)")
	parser.add_argument(
		"--model-type",
		type=str,
		choices=["polynomial", "analytical"],
		default="polynomial",
		help="Model type: 'polynomial' or 'analytical' (default: polynomial)",
	)
	parser.add_argument(
		"--output-dir",
		type=str,
		default="plots/final-models",
		help="Output directory for plots and configs (default: plots/final-models)",
	)
	parser.add_argument("--output-json", type=str, help="Output JSON file")
	parser.add_argument(
		"--degree", type=int, default=3, help="Polynomial degree (default: 3, only for polynomial models)"
	)
	parser.add_argument(
		"--alpha", type=float, default=0.01, help="Regularization strength (default: 0.01, only for polynomial models)"
	)
	parser.add_argument(
		"--no-log-transform",
		action="store_true",
		help="Disable log-transform (use simple polynomial regression instead)",
	)
	parser.add_argument(
		"--update-frontend",
		type=str,
		help="Path to ResourceEstimatorModel.js to automatically update (e.g., src/utils/ResourceEstimatorModel.js)",
	)
	parser.add_argument("--max-qubits", type=int, help="Maximum qubits for device (optional)")
	args = parser.parse_args()

	# Handle analytical model building
	if args.model_type == "analytical":
		try:
			output_path = Path(args.output_dir)
			output_path.mkdir(parents=True, exist_ok=True)

			logger.info(f"Building analytical model for {args.device}...")
			js_config, r2, rmse, mae = fit_analytical_model(args.data, args.output_dir)

			# Update device name and max_qubits if provided
			if args.device_name:
				js_config["name"] = args.device_name
			if args.max_qubits:
				js_config["max_qubits"] = args.max_qubits

			# Save with device-specific name
			config_path = output_path / f"{args.device}_analytical_config.json"
			with open(config_path, "w") as f:
				json.dump(js_config, f, indent=2)
			logger.info(f"\nConfig saved: {config_path}")

			# Update frontend if requested
			if args.update_frontend:
				# Generate JavaScript code from config
				js_code = f"""// {js_config["name"]} - Analytical Model
// Auto-generated configuration
{{
  name: "{js_config["name"]}",
  max_qubits: {js_config["max_qubits"]},
  model_type: "analytical",
  T_init: {js_config["T_init"]},
  efficiency_base: {js_config["efficiency_base"]},
  throughput_coef: {js_config["throughput_coef"]},
  batch_cap: {js_config["batch_cap"]}
}}"""
				update_javascript_model(Path(args.update_frontend), args.device, js_code)
				logger.info(f"Updated frontend model in {args.update_frontend}")

			logger.info("\nAnalytical model building completed successfully!")
			return

		except Exception as e:
			logger.error(f"Error building analytical model: {e}", exc_info=True)
			sys.exit(1)

	# Validate required arguments for polynomial model

	try:
		# Load and prepare data
		data = load_data_from_csv(args.data)
		X, y = prepare_training_data(data)

		# Train model
		use_log_transform = not args.no_log_transform
		model, poly, metrics = train_polynomial_model(
			X, y, args.degree, args.alpha, use_log_transform=use_log_transform
		)

		# Extract coefficients
		feature_names = X.columns.tolist()
		coefficients = extract_model_coefficients(model, poly, feature_names)

		# Format JavaScript (include epsilon only if log-transform is used)
		device_name = args.device_name or args.device.replace("-", " ").title()
		epsilon = getattr(model, "epsilon_", None) if use_log_transform else None
		js_code = format_javascript_model(coefficients, device_name, args.device, epsilon)

		# Print results
		logger.info("\nModel Metrics:")
		logger.info(f"  R² Score: {metrics['r2_score']:.4f}")
		logger.info(f"  RMSE: {metrics['rmse']:.4f}")
		logger.info(f"  MAE: {metrics['mae']:.4f}")
		logger.info("\nJavaScript Model Configuration:")
		print("\n" + js_code + "\n")

		# Save JSON if requested
		if args.output_json:
			save_model_as_json(coefficients, metrics, Path(args.output_json))

		# Update frontend JavaScript file if requested
		if args.update_frontend:
			update_javascript_model(Path(args.update_frontend), args.device, js_code)
			logger.info(f"Updated frontend model in {args.update_frontend}")

		logger.info("Model building completed successfully!")

	except Exception as e:
		logger.error(f"Error: {e}", exc_info=True)
		sys.exit(1)


if __name__ == "__main__":
	main()
