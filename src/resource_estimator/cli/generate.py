"""CLI for data generation."""

import argparse
import sys

from resource_estimator.data_collection import collect_timing_data, connect_to_backend
from resource_estimator.logging_config import setup_logging
from resource_estimator.utils import export_data_to_csv

logger = setup_logging()


def main():
	"""Main entry point for data generation."""
	parser = argparse.ArgumentParser(description="Generate quantum resource estimation data")
	parser.add_argument("--server-url", type=str, required=True, help="IQM server URL")
	parser.add_argument("--output", type=str, required=True, help="Output CSV file path")
	parser.add_argument("--samples", type=int, default=50, help="Number of samples (default: 50)")
	parser.add_argument("--no-isolated", action="store_true", help="Skip isolated parameter sweeps")
	parser.add_argument(
		"--job-timeout",
		type=float,
		default=900.0,
		help="Timeout for job completion in seconds (default: 900, same as iqm-client DEFAULT_TIMEOUT_SECONDS)",
	)
	parser.add_argument(
		"--checkpoint",
		type=str,
		help="Path to checkpoint file for incremental saves and resume capability (default: <output>.checkpoint.csv)",
	)
	args = parser.parse_args()

	try:
		backend = connect_to_backend(args.server_url)

		# Use provided checkpoint path or create default based on output file
		checkpoint_path = args.checkpoint if args.checkpoint else f"{args.output}.checkpoint.csv"

		data = collect_timing_data(
			backend=backend,
			num_samples=args.samples,
			include_isolated=not args.no_isolated,
			job_timeout=args.job_timeout,
			checkpoint_path=checkpoint_path,
		)

		if not data:
			logger.error("No data was collected!")
			sys.exit(1)

		export_data_to_csv(data, args.output)
		logger.info(f"Successfully generated {len(data)} data points")

	except Exception as e:
		logger.error(f"Error: {e}", exc_info=True)
		sys.exit(1)


if __name__ == "__main__":
	main()
