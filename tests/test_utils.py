"""Tests for utils module."""

import pandas as pd

from resource_estimator.utils import load_data_from_csv, export_data_to_csv, save_model_as_json


def test_export_and_load_data(tmp_path):
	"""Test data export and load round-trip."""
	data = [
		{"num_qubits": 2, "depth": 10, "batches": 1, "shots": 1000, "qpu_seconds": 5.0},
		{"num_qubits": 3, "depth": 20, "batches": 2, "shots": 2000, "qpu_seconds": 12.0},
	]

	file_path = tmp_path / "test_data.csv"

	# Export
	export_data_to_csv(data, file_path)
	assert file_path.exists()

	# Load
	loaded_df = load_data_from_csv(file_path)
	assert len(loaded_df) == 2
	assert "num_qubits" in loaded_df.columns
	assert "qpu_seconds" in loaded_df.columns


def test_load_data_with_missing_values(tmp_path):
	"""Test loading data with missing values."""
	file_path = tmp_path / "data_with_na.csv"

	df = pd.DataFrame({"num_qubits": [2, 3, None, 5], "depth": [10, 20, 15, 25], "qpu_seconds": [5.0, 12.0, 8.5, 18.0]})
	df.to_csv(file_path, index=False)

	loaded_df = load_data_from_csv(file_path)

	# Should drop rows with missing values
	assert len(loaded_df) == 3


def test_save_model_as_json(tmp_path):
	"""Test model JSON export."""
	coefficients = {"intercept": 1.5, "qubits": 0.5, "depth": 0.3}
	metrics = {"r2_score": 0.95, "rmse": 0.5}

	output_path = tmp_path / "model.json"
	save_model_as_json(coefficients, metrics, output_path)

	assert output_path.exists()

	import json

	with open(output_path) as f:
		data = json.load(f)

	assert data["intercept"] == 1.5
	assert "qubits" in data["coefficients"]
	assert data["metrics"]["r2_score"] == 0.95
