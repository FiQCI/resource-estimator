"""Utility functions for data I/O."""

import logging
from pathlib import Path
import json
import re

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

	# Convert to Path object and resolve to absolute path
	output_path = Path(output_path)

	# Check for path traversal attempts before resolving
	if ".." in str(output_path):
		raise ValueError("Path traversal detected in output path")

	# Resolve to absolute path
	output_path = output_path.resolve()

	data = {
		"intercept": coefficients.get("intercept", 0.0),
		"coefficients": {k: v for k, v in coefficients.items() if k != "intercept"},
		"metrics": metrics,
	}

	output_path.parent.mkdir(parents=True, exist_ok=True)

	with output_path.open("w") as f:
		json.dump(data, f, indent=2)

	logger.info(f"Saved model to {output_path}")


def update_javascript_model(js_file_path: str | Path, device_id: str, js_device_config: str):
	"""Update the JavaScript model file with a new device configuration.

	Args:
		js_file_path: Path to ResourceEstimatorModel.js
		device_id: Device identifier (e.g., 'vtt-q50')
		js_device_config: JavaScript device configuration block

	Raises:
		FileNotFoundError: If the JavaScript file doesn't exist
		ValueError: If the file format is unexpected
	"""
	js_file_path = Path(js_file_path)

	if not js_file_path.exists():
		raise FileNotFoundError(f"JavaScript file not found: {js_file_path}")

	# Read the current file
	content = js_file_path.read_text()

	# Find the DEVICE_PARAMS object
	device_params_pattern = r"const DEVICE_PARAMS = \{(.*?)\};"
	match = re.search(device_params_pattern, content, re.DOTALL)

	if not match:
		raise ValueError("Could not find DEVICE_PARAMS object in JavaScript file")

	# Extract the current DEVICE_PARAMS content
	device_params_content = match.group(1)

	# Check if the device already exists using a more robust pattern
	# This pattern matches the device key and then counts braces to find the complete configuration
	def find_device_config(content, device_id):
		"""Find the complete device configuration including nested structures."""
		# Look for the device key
		pattern = rf"'{device_id}':\s*\{{"
		match = re.search(pattern, content)
		if not match:
			return None

		start_pos = match.start()
		brace_start = match.end() - 1  # Position of opening brace

		# Count braces to find matching closing brace
		brace_count = 1
		pos = brace_start + 1
		while pos < len(content) and brace_count > 0:
			if content[pos] == "{":
				brace_count += 1
			elif content[pos] == "}":
				brace_count -= 1
			pos += 1

		if brace_count == 0:
			end_pos = pos
			return (start_pos, end_pos, content[start_pos:end_pos])
		return None

	existing_device = find_device_config(device_params_content, device_id)

	if existing_device:
		# Replace existing device configuration
		logger.info(f"Replacing existing configuration for device '{device_id}'")
		start_pos, end_pos, old_config = existing_device
		new_device_params_content = (
			device_params_content[:start_pos] + js_device_config.strip() + device_params_content[end_pos:]
		)
	else:
		# Add new device configuration
		logger.info(f"Adding new device '{device_id}' to configuration")
		# Add comma after last device if needed
		new_device_params_content = device_params_content.rstrip()
		if not new_device_params_content.endswith(","):
			new_device_params_content += ","
		new_device_params_content += f"\n\t{js_device_config.strip()}"

	# Reconstruct the full content
	new_content = content.replace(device_params_content, new_device_params_content)

	# Create backup
	backup_path = js_file_path.with_suffix(".js.bak")
	backup_path.write_text(content)
	logger.info(f"Created backup at {backup_path}")

	# Write the updated content
	js_file_path.write_text(new_content)
	logger.info(f"Successfully updated {js_file_path}")
