"""Tests for model module."""

import numpy as np
import pandas as pd
import pytest

from resource_estimator.model import (
	extract_model_coefficients,
	format_javascript_model,
	prepare_training_data,
	train_polynomial_model,
	create_prediction_function,
	_parse_coefficient_term,
)


@pytest.fixture
def sample_data():
	"""Create sample training data."""
	return [
		{"num_qubits": 2, "depth": 10, "num_circuits": 1, "shots": 1000, "qpu_seconds": 5.0},
		{"num_qubits": 3, "depth": 20, "num_circuits": 2, "shots": 2000, "qpu_seconds": 12.0},
		{"num_qubits": 4, "depth": 15, "num_circuits": 1, "shots": 1500, "qpu_seconds": 8.5},
		{"num_qubits": 5, "depth": 25, "num_circuits": 3, "shots": 3000, "qpu_seconds": 18.0},
	]


def test_prepare_training_data(sample_data):
	"""Test data preparation."""
	X, y = prepare_training_data(sample_data)

	assert isinstance(X, pd.DataFrame)
	assert len(X) == 4
	assert "kshots" in X.columns
	assert "qubits" in X.columns
	assert "batches" in X.columns
	assert len(y) == 4
	assert np.allclose(y, [5.0, 12.0, 8.5, 18.0])


def test_train_polynomial_model(sample_data):
	"""Test model training."""
	X, y = prepare_training_data(sample_data)
	model, poly, metrics = train_polynomial_model(X, y, degree=3, alpha=0.01)

	assert model is not None
	assert poly is not None
	assert "r2_score" in metrics
	assert "rmse" in metrics
	assert "mae" in metrics
	assert 0 <= metrics["r2_score"] <= 1


def test_extract_model_coefficients(sample_data):
	"""Test coefficient extraction."""
	X, y = prepare_training_data(sample_data)
	model, poly, _ = train_polynomial_model(X, y, degree=3, alpha=0.01)

	feature_names = X.columns.tolist()
	coefficients = extract_model_coefficients(model, poly, feature_names)

	assert "intercept" in coefficients
	assert isinstance(coefficients["intercept"], float)
	assert len(coefficients) > 1  # Should have some polynomial terms


def test_create_prediction_function(sample_data):
	"""Test prediction function creation."""
	X, y = prepare_training_data(sample_data)
	model, poly, _ = train_polynomial_model(X, y, degree=3, alpha=0.01)

	feature_names = X.columns.tolist()
	predict = create_prediction_function(model, poly, feature_names)

	# Test prediction
	result = predict(qubits=3, depth=15, batches=2, shots=2000)

	assert isinstance(result, float)
	assert result > 0


def test_format_javascript_model():
	"""Test JavaScript formatting."""
	coefficients = {"intercept": 1.5, "qubits": 0.5, "batches kshots": 0.3, "qubits^2": 0.1}

	js_code = format_javascript_model(coefficients, "Test Device", "test-device")

	assert "'test-device'" in js_code
	assert "name: 'Test Device'" in js_code
	assert "intercept: 1.500000" in js_code
	assert "type: 'single'" in js_code
	assert "type: 'interaction'" in js_code
	assert "type: 'power'" in js_code


def test_parse_coefficient_term_single():
	"""Test parsing single variable term."""
	result = _parse_coefficient_term("qubits", 0.5)

	assert result["type"] == "single"
	assert result["variable"] == "qubits"
	assert result["coefficient"] == 0.5


def test_parse_coefficient_term_interaction():
	"""Test parsing interaction term."""
	result = _parse_coefficient_term("qubits batches", 0.3)

	assert result["type"] == "interaction"
	assert result["variables"] == ["qubits", "batches"]
	assert result["coefficient"] == 0.3


def test_parse_coefficient_term_power():
	"""Test parsing power term."""
	result = _parse_coefficient_term("qubits^2", 0.1)

	assert result["type"] == "power"
	assert result["variable"] == "qubits"
	assert result["coefficient"] == 0.1
	assert result["exponent"] == 2


def test_parse_coefficient_term_kshots_normalization():
	"""Test k_shots to kshots normalization."""
	result = _parse_coefficient_term("k_shots", 0.2)

	assert result["variable"] == "kshots"
