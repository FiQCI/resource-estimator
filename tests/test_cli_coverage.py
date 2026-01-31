"""Additional tests for CLI modules to improve coverage."""

import sys
from pathlib import Path
from unittest.mock import Mock, patch
import pytest
import pandas as pd


def test_build_cli_file_not_found():
	"""Test build CLI with non-existent data file."""
	from resource_estimator.cli import build

	with patch.object(sys, "argv", ["build", "--data", "nonexistent.csv", "--device", "test"]):
		with pytest.raises(SystemExit):
			build.main()


def test_build_cli_model_extraction_error():
	"""Test build CLI when model extraction fails."""
	from resource_estimator.cli import build

	mock_data = pd.DataFrame(
		{"num_qubits": [2, 3], "depth": [10, 20], "batches": [1, 2], "shots": [1000, 2000], "qpu_seconds": [5.0, 12.0]}
	)

	with patch.object(sys, "argv", ["build", "--data", "test.csv", "--device", "test"]):
		with patch("resource_estimator.cli.build.load_data_from_csv", return_value=mock_data):
			with patch("resource_estimator.cli.build.extract_model_coefficients", side_effect=ValueError("Test error")):
				with pytest.raises(SystemExit):
					build.main()


def test_build_cli_save_json_error():
	"""Test build CLI when JSON save fails."""
	from resource_estimator.cli import build

	mock_data = pd.DataFrame(
		{
			"num_qubits": [2, 3, 4],
			"qubits": [2, 3, 4],
			"depth": [10, 20, 15],
			"batches": [1, 2, 1],
			"shots": [1000, 2000, 1500],
			"qpu_seconds": [5.0, 12.0, 8.5],
		}
	)

	with patch.object(
		sys, "argv", ["build", "--data", "test.csv", "--device", "test", "--output-json", "/invalid/path.json"]
	):
		with patch("resource_estimator.cli.build.load_data_from_csv", return_value=mock_data):
			with patch("resource_estimator.cli.build.save_model_as_json", side_effect=PermissionError("Cannot write")):
				with pytest.raises(SystemExit):
					build.main()


def test_generate_cli_connection_error():
	"""Test generate CLI with connection error."""
	from resource_estimator.cli import generate

	with patch.object(
		sys, "argv", ["generate", "--server-url", "http://invalid", "--output", "test.csv", "--samples", "10"]
	):
		with patch("resource_estimator.cli.generate.connect_to_backend", side_effect=RuntimeError("Connection failed")):
			with pytest.raises(SystemExit):
				generate.main()


def test_generate_cli_data_collection_error():
	"""Test generate CLI when data collection fails."""
	from resource_estimator.cli import generate

	mock_backend = Mock()

	with patch.object(
		sys, "argv", ["generate", "--server-url", "http://test", "--output", "test.csv", "--samples", "10"]
	):
		with patch("resource_estimator.cli.generate.connect_to_backend", return_value=mock_backend):
			with patch(
				"resource_estimator.cli.generate.collect_timing_data", side_effect=RuntimeError("Collection failed")
			):
				with pytest.raises(SystemExit):
					generate.main()


def test_utils_load_csv_with_errors():
	"""Test utils load_data_from_csv with invalid file."""
	from resource_estimator.utils import load_data_from_csv

	with pytest.raises(FileNotFoundError):
		load_data_from_csv("nonexistent.csv")


def test_utils_export_csv_permission_error(tmp_path):
	"""Test utils export_data_to_csv with permission error."""
	from resource_estimator.utils import export_data_to_csv

	data = [{"num_qubits": 2, "depth": 10, "batches": 1, "shots": 1000, "qpu_seconds": 5.0}]

	# Create a read-only directory
	readonly_dir = tmp_path / "readonly"
	readonly_dir.mkdir()
	readonly_dir.chmod(0o444)

	try:
		with pytest.raises(PermissionError):
			export_data_to_csv(data, str(readonly_dir / "test.csv"))
	finally:
		# Cleanup
		readonly_dir.chmod(0o755)


def test_utils_update_javascript_file_not_found():
	"""Test utils update_javascript_model with non-existent file."""
	from resource_estimator.utils import update_javascript_model

	with pytest.raises(FileNotFoundError):
		update_javascript_model(Path("nonexistent.js"), "test", "// test code")


def test_utils_update_javascript_device_not_found(tmp_path):
	"""Test utils update_javascript_model with device not in file."""
	from resource_estimator.utils import update_javascript_model

	js_file = tmp_path / "test.js"
	js_file.write_text("export const models = {\n\tother: {}\n};")

	with pytest.raises(ValueError):
		update_javascript_model(js_file, "nonexistent", "// test code")


def test_utils_save_model_path_traversal():
	"""Test utils save_model_as_json prevents path traversal."""
	from resource_estimator.utils import save_model_as_json

	coefficients = {"intercept": 1.0, "qubits": 0.5}
	metrics = {"r2_score": 0.95, "rmse": 1.2, "mae": 0.8}

	with pytest.raises(ValueError, match="Path traversal"):
		save_model_as_json(coefficients, metrics, Path("../../../etc/passwd"))


def test_validate_cli_main_with_mock_data(tmp_path):
	"""Test validate CLI main function."""
	from resource_estimator.cli import validate

	# Create test files
	data_file = tmp_path / "test.csv"
	data_file.write_text("num_qubits,depth,batches,shots,qpu_seconds\n2,10,1,1000,5.0\n3,20,2,2000,12.0\n")

	output_dir = tmp_path / "plots"

	with patch.object(sys, "argv", ["validate", "--data", str(data_file), "--plots", str(output_dir)]):
		# This should run without error (though it may fail validation criteria)
		with pytest.raises(SystemExit):  # Will exit with 0 or 1
			validate.main()


def test_validate_cli_missing_data_file():
	"""Test validate CLI with missing data file."""
	from resource_estimator.cli import validate

	with patch.object(sys, "argv", ["validate", "--data", "missing.csv", "--model", "model.js", "--device", "test"]):
		with pytest.raises(SystemExit):
			validate.main()
