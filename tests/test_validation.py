"""Tests for validation module."""

import pandas as pd
import pytest
from unittest.mock import patch

from resource_estimator.model import train_polynomial_model
from resource_estimator.validation import validate_model_predictions, calculate_feature_importance, generate_all_plots


@pytest.fixture
def sample_data():
	"""Create sample data."""
	return pd.DataFrame(
		{
			"qubits": [2, 3, 4, 5, 3, 4],
			"depth": [10, 20, 15, 25, 18, 22],
			"batches": [1, 2, 1, 3, 2, 2],
			"kshots": [1.0, 2.0, 1.5, 3.0, 2.5, 2.0],
			"qpu_seconds": [5.0, 12.0, 8.5, 18.0, 13.5, 11.0],
		}
	)


@pytest.fixture
def trained_model(sample_data):
	"""Create a trained model."""
	X = sample_data[["qubits", "depth", "batches", "kshots"]]
	y = sample_data["qpu_seconds"].values
	model, poly, _ = train_polynomial_model(X, y, degree=3, alpha=0.01)
	return model, poly, X, y


def test_validate_model_predictions(trained_model):
	"""Test model validation."""
	model, poly, X, y = trained_model
	results = validate_model_predictions(model, poly, X, y)

	assert "r2_score" in results
	assert "rmse" in results
	assert "mae" in results
	assert "mape" in results
	assert "y_pred" in results
	assert "errors" in results

	assert 0 <= results["r2_score"] <= 1
	assert results["rmse"] >= 0
	assert results["mae"] >= 0
	assert results["mape"] >= 0
	assert len(results["y_pred"]) == len(y)
	assert len(results["errors"]) == len(y)


def test_calculate_feature_importance(trained_model):
	"""Test feature importance calculation."""
	model, poly, X, _ = trained_model
	feature_names = X.columns.tolist()

	importance = calculate_feature_importance(model, poly, feature_names)

	assert len(importance) == len(feature_names)
	assert all(0 <= v <= 1 for v in importance.values())
	assert abs(sum(importance.values()) - 1.0) < 0.01  # Should sum to ~1


@patch("resource_estimator.validation.plt")
def test_generate_all_plots(mock_plt, sample_data, trained_model, tmp_path):
	"""Test plot generation."""
	model, poly, _, _ = trained_model
	output_dir = tmp_path / "plots"

	results = generate_all_plots(sample_data, model, poly, "Test Device", output_dir)

	assert "r2_score" in results
	assert "rmse" in results

	# Verify plot functions were called
	assert mock_plt.figure.call_count > 0
	assert mock_plt.savefig.call_count > 0
