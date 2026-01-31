"""Extended tests for model module to increase coverage."""

import numpy as np
import pandas as pd

from resource_estimator.model import (
	create_prediction_function,
	extract_model_coefficients,
	format_javascript_model,
	prepare_training_data,
	train_polynomial_model,
)


def test_prepare_training_data_with_num_circuits():
	"""Test prepare_training_data with num_circuits column."""
	data = pd.DataFrame(
		{
			"num_qubits": [2, 3],
			"depth": [10, 20],
			"num_circuits": [1, 2],
			"shots": [1000, 2000],
			"qpu_seconds": [5.0, 12.0],
		}
	)

	X, y = prepare_training_data(data)

	assert "batches" in X.columns
	assert "kshots" in X.columns
	assert len(y) == 2


def test_prepare_training_data_from_list():
	"""Test prepare_training_data with list input."""
	data = [
		{"num_qubits": 2, "depth": 10, "batches": 1, "shots": 1000, "qpu_seconds": 5.0},
		{"num_qubits": 3, "depth": 20, "batches": 2, "shots": 2000, "qpu_seconds": 12.0},
	]

	X, y = prepare_training_data(data)

	assert isinstance(X, pd.DataFrame)
	assert isinstance(y, np.ndarray)
	assert len(X) == 2
	assert len(y) == 2


def test_train_polynomial_model_degree_2():
	"""Test training with degree=2."""
	data = pd.DataFrame(
		{
			"qubits": [2, 3, 4, 5],
			"depth": [10, 20, 15, 25],
			"batches": [1, 2, 1, 3],
			"kshots": [1.0, 2.0, 1.5, 3.0],
			"qpu_seconds": [5.0, 12.0, 8.5, 18.0],
		}
	)

	X = data[["qubits", "depth", "batches", "kshots"]]
	y = data["qpu_seconds"].values

	model, poly, metrics = train_polynomial_model(X, y, degree=2, alpha=0.01)

	assert model is not None
	assert poly is not None
	assert "r2_score" in metrics
	assert "rmse" in metrics
	assert "mae" in metrics
	assert hasattr(model, "epsilon_")


def test_create_prediction_function_basic():
	"""Test prediction function creation and usage."""
	data = pd.DataFrame(
		{
			"qubits": [2, 3, 4],
			"depth": [10, 20, 15],
			"batches": [1, 2, 1],
			"kshots": [1.0, 2.0, 1.5],
			"qpu_seconds": [5.0, 12.0, 8.5],
		}
	)

	X = data[["qubits", "depth", "batches", "kshots"]]
	y = data["qpu_seconds"].values

	model, poly, _ = train_polynomial_model(X, y, degree=2, alpha=0.01)
	predict_fn = create_prediction_function(model, poly, X.columns.tolist())

	# Test prediction
	result = predict_fn(2, 10, 1, 1000)

	assert isinstance(result, float)
	assert result >= 0  # Should be positive due to log-transform


def test_extract_model_coefficients():
	"""Test coefficient extraction."""
	data = pd.DataFrame(
		{
			"qubits": [2, 3, 4],
			"depth": [10, 20, 15],
			"batches": [1, 2, 1],
			"kshots": [1.0, 2.0, 1.5],
			"qpu_seconds": [5.0, 12.0, 8.5],
		}
	)

	X = data[["qubits", "depth", "batches", "kshots"]]
	y = data["qpu_seconds"].values

	model, poly, _ = train_polynomial_model(X, y, degree=2, alpha=0.01)
	coefficients = extract_model_coefficients(model, poly, X.columns.tolist())

	assert "intercept" in coefficients
	assert isinstance(coefficients["intercept"], float)
	assert len(coefficients) > 1  # Should have more than just intercept


def test_format_javascript_model_basic():
	"""Test JavaScript model formatting."""
	coefficients = {"intercept": 1.5, "qubits": 0.3, "depth": 0.2, "qubits^2": -0.01, "qubits depth": 0.05}

	js_code = format_javascript_model(coefficients, "Test Device", "test-device", epsilon=0.001)

	assert "'test-device'" in js_code
	assert "name: 'Test Device'" in js_code
	assert "logTransform: true" in js_code
	assert "epsilon: 0.001000" in js_code
	assert "intercept: 1.500000" in js_code
	assert "type: 'single'" in js_code
	assert "type: 'power'" in js_code
	assert "type: 'interaction'" in js_code


def test_format_javascript_model_with_kshots():
	"""Test JavaScript formatting with kshots terms."""
	coefficients = {
		"intercept": 2.0,
		"k_shots": 0.5,  # Should be converted to kshots
		"k_shots^2": -0.02,
	}

	js_code = format_javascript_model(coefficients, "Device", "device", epsilon=0.001)

	assert "kshots" in js_code
	assert "k_shots" not in js_code  # Should be converted


def test_model_predictions_are_positive():
	"""Test that log-transform ensures positive predictions."""
	data = pd.DataFrame(
		{
			"qubits": list(range(1, 11)),
			"depth": [10, 20, 15, 25, 30, 12, 18, 22, 28, 35],
			"batches": [1, 2, 1, 3, 2, 1, 2, 3, 1, 2],
			"kshots": [1.0, 2.0, 1.5, 3.0, 2.5, 1.0, 2.0, 3.0, 1.5, 2.5],
			"qpu_seconds": [5.0, 12.0, 8.5, 18.0, 22.0, 6.5, 11.0, 17.0, 15.0, 20.0],
		}
	)

	X = data[["qubits", "depth", "batches", "kshots"]]
	y = data["qpu_seconds"].values

	model, poly, _ = train_polynomial_model(X, y, degree=3, alpha=0.01)
	predict_fn = create_prediction_function(model, poly, X.columns.tolist())

	# Test various parameter combinations
	test_cases = [(1, 5, 1, 1000), (5, 10, 2, 2000), (10, 20, 3, 5000), (50, 100, 5, 10000)]

	for qubits, depth, batches, shots in test_cases:
		prediction = predict_fn(qubits, depth, batches, shots)
		assert prediction >= 0, f"Negative prediction for q={qubits}, d={depth}, b={batches}, s={shots}"
		assert np.isfinite(prediction), f"Non-finite prediction for q={qubits}, d={depth}, b={batches}, s={shots}"
