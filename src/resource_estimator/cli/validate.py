"""CLI for model validation."""

import argparse
import sys
from pathlib import Path

from resource_estimator.logging_config import setup_logging
from resource_estimator.model import prepare_training_data, train_polynomial_model
from resource_estimator.utils import load_data_from_csv
from resource_estimator.validation import generate_all_plots

logger = setup_logging()


def main():
	"""Main entry point for model validation."""
	parser = argparse.ArgumentParser(description="Validate quantum resource estimation model")
	parser.add_argument("--data", type=str, required=True, help="Input CSV file")
	parser.add_argument("--device", type=str, required=True, help="Device identifier")
	parser.add_argument("--output", type=str, default="plots", help="Output directory (default: plots)")
	parser.add_argument("--degree", type=int, default=2, help="Polynomial degree (default: 2)")
	parser.add_argument("--alpha", type=float, default=0.01, help="Regularization strength (default: 0.01)")
	args = parser.parse_args()

	try:
		# Load and prepare data
		data = load_data_from_csv(args.data)
		X, y = prepare_training_data(data)

		# Add kshots column to data for plotting
		data["kshots"] = data["shots"] / 1000.0
		if "num_qubits" in data.columns and "qubits" not in data.columns:
			data["qubits"] = data["num_qubits"]
		if "num_circuits" in data.columns and "batches" not in data.columns:
			data["batches"] = data["num_circuits"]

		# Train model
		model, poly, metrics = train_polynomial_model(X, y, args.degree, args.alpha)

		# Generate plots
		device_name = args.device.replace("-", " ").title()
		output_dir = Path(args.output)
		results = generate_all_plots(data, model, poly, device_name, output_dir)

		# Print metrics
		logger.info("\nValidation Metrics:")
		logger.info(f"  R² Score: {results['r2_score']:.4f}")
		logger.info(f"  RMSE: {results['rmse']:.4f}")
		logger.info(f"  MAE: {results['mae']:.4f}")
		logger.info(f"  MAPE: {results['mape']:.2f}%")
		logger.info("\nValidation completed successfully!")

	except Exception as e:
		logger.error(f"Error: {e}", exc_info=True)
		sys.exit(1)


if __name__ == "__main__":
	main()
