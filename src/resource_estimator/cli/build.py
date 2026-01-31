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
from resource_estimator.utils import load_data_from_csv, save_model_as_json, update_javascript_model

logger = setup_logging()


def main():
	"""Main entry point for model building."""
	parser = argparse.ArgumentParser(description="Build quantum resource estimation model")
	parser.add_argument("--data", type=str, required=True, help="Input CSV file")
	parser.add_argument("--device", type=str, required=True, help="Device identifier (e.g., 'helmi')")
	parser.add_argument("--device-name", type=str, help="Display name (defaults to device ID)")
	parser.add_argument("--output-json", type=str, help="Output JSON file")
	parser.add_argument("--degree", type=int, default=3, help="Polynomial degree (default: 3)")
	parser.add_argument("--alpha", type=float, default=0.01, help="Regularization strength (default: 0.01)")
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
	args = parser.parse_args()

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
			from pathlib import Path

			save_model_as_json(coefficients, metrics, Path(args.output_json))

		# Update frontend JavaScript file if requested
		if args.update_frontend:
			from pathlib import Path

			update_javascript_model(Path(args.update_frontend), args.device, js_code)
			logger.info(f"Updated frontend model in {args.update_frontend}")

		logger.info("Model building completed successfully!")

	except Exception as e:
		logger.error(f"Error: {e}", exc_info=True)
		sys.exit(1)


if __name__ == "__main__":
	main()
