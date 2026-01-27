"""Tests for CLI modules."""

import sys
from unittest.mock import Mock, patch

import pandas as pd
import pytest


@patch("resource_estimator.cli.generate.connect_to_backend")
@patch("resource_estimator.cli.generate.collect_timing_data")
@patch("resource_estimator.cli.generate.export_data_to_csv")
def test_generate_cli_success(mock_export, mock_collect, mock_connect, tmp_path):
	"""Test successful data generation CLI."""
	from resource_estimator.cli.generate import main

	mock_backend = Mock()
	mock_connect.return_value = mock_backend
	mock_collect.return_value = [{"qubits": 2, "qpu_seconds": 5.0}]

	output_file = tmp_path / "data.csv"

	with patch.object(
		sys,
		"argv",
		["generate", "--server-url", "https://test.server", "--output", str(output_file), "--samples", "10"],
	):
		main()

	mock_connect.assert_called_once_with("https://test.server")
	mock_collect.assert_called_once()
	mock_export.assert_called_once()


@patch("resource_estimator.cli.generate.connect_to_backend")
def test_generate_cli_connection_error(mock_connect):
	"""Test CLI with connection error."""
	from resource_estimator.cli.generate import main

	mock_connect.side_effect = RuntimeError("Connection failed")

	with patch.object(sys, "argv", ["generate", "--server-url", "https://test.server", "--output", "data.csv"]):
		with pytest.raises(SystemExit):
			main()


@patch("resource_estimator.cli.build.load_data_from_csv")
@patch("resource_estimator.cli.build.prepare_training_data")
@patch("resource_estimator.cli.build.train_polynomial_model")
@patch("resource_estimator.cli.build.extract_model_coefficients")
def test_build_cli_success(mock_extract, mock_train, mock_prepare, mock_load, tmp_path):
	"""Test successful model building CLI."""
	from resource_estimator.cli.build import main

	# Mock data
	df = pd.DataFrame(
		{"qubits": [2, 3], "depth": [10, 20], "batches": [1, 2], "kshots": [1.0, 2.0], "qpu_seconds": [5.0, 12.0]}
	)

	mock_load.return_value = df
	mock_prepare.return_value = (df[["qubits", "depth", "batches", "kshots"]], df["qpu_seconds"].values)

	mock_model = Mock()
	mock_poly = Mock()
	mock_train.return_value = (mock_model, mock_poly, {"r2_score": 0.95, "rmse": 0.5, "mae": 0.3})

	mock_extract.return_value = {"intercept": 1.0, "qubits": 0.5}

	data_file = tmp_path / "data.csv"
	df.to_csv(data_file, index=False)

	with patch.object(sys, "argv", ["build", "--data", str(data_file), "--device", "test"]):
		main()

	mock_load.assert_called_once()
	mock_train.assert_called_once()


@patch("resource_estimator.cli.validate.load_data_from_csv")
@patch("resource_estimator.cli.validate.prepare_training_data")
@patch("resource_estimator.cli.validate.train_polynomial_model")
@patch("resource_estimator.cli.validate.generate_all_plots")
def test_validate_cli_success(mock_plots, mock_train, mock_prepare, mock_load, tmp_path):
	"""Test successful validation CLI."""
	from resource_estimator.cli.validate import main

	# Mock data
	df = pd.DataFrame(
		{
			"qubits": [2, 3],
			"depth": [10, 20],
			"batches": [1, 2],
			"shots": [1000, 2000],
			"kshots": [1.0, 2.0],
			"qpu_seconds": [5.0, 12.0],
		}
	)

	mock_load.return_value = df
	mock_prepare.return_value = (df[["qubits", "depth", "batches", "kshots"]], df["qpu_seconds"].values)

	mock_model = Mock()
	mock_poly = Mock()
	mock_train.return_value = (mock_model, mock_poly, {"r2_score": 0.95, "rmse": 0.5, "mae": 0.3})

	mock_plots.return_value = {"r2_score": 0.95, "rmse": 0.5, "mae": 0.3, "mape": 5.0}

	data_file = tmp_path / "data.csv"
	df.to_csv(data_file, index=False)
	output_dir = tmp_path / "plots"

	with patch.object(
		sys, "argv", ["validate", "--data", str(data_file), "--device", "test", "--output", str(output_dir)]
	):
		main()

	mock_load.assert_called_once()
	mock_train.assert_called_once()
	mock_plots.assert_called_once()
