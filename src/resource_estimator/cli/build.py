"""CLI for model building."""

import argparse
import sys

from resource_estimator.logging_config import setup_logging
from resource_estimator.model import (
	extract_model_coefficients,
	format_javascript_model,
	prepare_training_data,
	train_polynomial_model,
)
from resource_estimator.utils import load_data_from_csv, save_model_as_json

logger = setup_logging()


def main():
	"""Main entry point for model building."""
	parser = argparse.ArgumentParser(description="Build quantum resource estimation model")
	parser.add_argument("--data", type=str, required=True, help="Input CSV file")
	parser.add_argument("--device", type=str, required=True, help="Device identifier (e.g., 'helmi')")
	parser.add_argument("--device-name", type=str, help="Display name (defaults to device ID)")
	parser.add_argument("--output-json", type=str, help="Output JSON file")
	parser.add_argument("--degree", type=int, default=2, help="Polynomial degree (default: 2)")
	parser.add_argument("--alpha", type=float, default=0.01, help="Regularization strength (default: 0.01)")
	args = parser.parse_args()

	try:
		# Load and prepare data
		data = load_data_from_csv(args.data)
		X, y = prepare_training_data(data)

		# Train model
		model, poly, metrics = train_polynomial_model(X, y, args.degree, args.alpha)

		# Extract coefficients
		feature_names = X.columns.tolist()
		coefficients = extract_model_coefficients(model, poly, feature_names)

		# Format JavaScript
		device_name = args.device_name or args.device.replace("-", " ").title()
		js_code = format_javascript_model(coefficients, device_name, args.device)

		# Print results
		logger.info("\nModel Metrics:")
		logger.info(f"  R² Score: {metrics['r2_score']:.4f}")
		logger.info(f"  RMSE: {metrics['rmse']:.4f}")
		logger.info(f"  MAE: {metrics['mae']:.4f}")
		logger.info("\nJavaScript Model Configuration:")
		print("\n" + js_code + "\n")

		# Save JSON if requested
		if args.output_json:
			from pathlib import Path

			save_model_as_json(coefficients, metrics, Path(args.output_json))

		logger.info("Model building completed successfully!")

	except Exception as e:
		logger.error(f"Error: {e}", exc_info=True)
		sys.exit(1)


if __name__ == "__main__":
	main()
