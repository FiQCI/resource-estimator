"""Utility functions for data I/O."""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_data_from_csv(file_path: str | Path) -> pd.DataFrame:
	"""Load timing data from CSV file.

	Args:
		file_path: Path to CSV file

	Returns:
		DataFrame with timing data
	"""
	logger.info(f"Loading data from {file_path}")
	df = pd.DataFrame(pd.read_csv(file_path))

	# Only check for NaN in critical columns that exist (not error column which is expected to be empty)
	critical_columns = ["num_qubits", "depth", "batches", "shots", "qpu_seconds"]
	existing_critical_columns = [col for col in critical_columns if col in df.columns]

	if existing_critical_columns and df[existing_critical_columns].isna().any().any():
		logger.warning("Data contains missing values in critical columns - dropping them")
		df = df.dropna(subset=existing_critical_columns)

	logger.info(f"Loaded {len(df)} data points")
	return df


def export_data_to_csv(data: list[dict], file_path: str | Path):
	"""Export timing data to CSV file.

	Args:
		data: List of data dictionaries
		file_path: Output file path
	"""
	df = pd.DataFrame(data)

	# Ensure output directory exists
	Path(file_path).parent.mkdir(parents=True, exist_ok=True)

	df.to_csv(file_path, index=False)
	logger.info(f"Exported {len(df)} data points to {file_path}")


def save_model_as_json(coefficients: dict, metrics: dict, output_path: Path):
	"""Save model parameters as JSON.

	Args:
		coefficients: Model coefficients
		metrics: Model metrics
		output_path: Output file path
	"""
	import json

	# Resolve and validate path to prevent traversal
	output_path = Path(output_path).resolve()

	data = {
		"intercept": coefficients.get("intercept", 0.0),
		"coefficients": {k: v for k, v in coefficients.items() if k != "intercept"},
		"metrics": metrics,
	}

	output_path.parent.mkdir(parents=True, exist_ok=True)

	with open(output_path, "w") as f:
		json.dump(data, f, indent=2)

	logger.info(f"Saved model to {output_path}")
