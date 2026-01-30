"""Test JavaScript and Python model consistency."""

import subprocess
from pathlib import Path

import pytest

from resource_estimator.model import create_prediction_function, prepare_training_data, train_polynomial_model
from resource_estimator.utils import load_data_from_csv


# Skip if production data doesn't exist
pytestmark = pytest.mark.skipif(
	not Path("data_analysis/data/vtt-q50.csv").exists(), reason="Production data file not found"
)


@pytest.fixture
def trained_model_deg3():
	"""Create degree-3 trained model with production data."""
	data = load_data_from_csv("data_analysis/data/vtt-q50.csv")
	X, y = prepare_training_data(data)
	model, poly, metrics = train_polynomial_model(X, y, degree=3, alpha=0.01)
	predict_fn = create_prediction_function(model, poly, X.columns.tolist())
	return predict_fn


def test_python_predictions_match_expected(trained_model_deg3):
	"""Test that Python predictions match expected values for degree-3 model."""
	predict_fn = trained_model_deg3

	# Test cases: (qubits, depth, batches, shots, expected_prediction)
	test_cases = [
		(2, 5, 1, 1000, 1.22),
		(2, 5, 3, 1000, 2.03),
		(2, 5, 6, 1000, 3.56),
		(6, 5, 1, 1000, 1.33),
		(12, 5, 1, 1000, 1.44),
		(2, 1, 1, 1000, 1.10),
		(2, 12, 1, 1000, 1.41),
		(2, 23, 1, 1000, 1.57),
		(2, 5, 1, 6444, 3.22),
		(2, 5, 1, 50000, 24.44),
	]

	tolerance = 0.1  # 0.1 second tolerance

	for q, d, b, s, expected in test_cases:
		predicted = predict_fn(q, d, b, s)
		error_msg = f"q={q}, d={d}, b={b}, s={s}: predicted={predicted:.2f}, expected={expected:.2f}"
		assert abs(predicted - expected) < tolerance, error_msg


def test_python_no_negative_predictions(trained_model_deg3):
	"""Test that model never produces negative predictions."""
	predict_fn = trained_model_deg3

	# Test a range of parameters
	test_params = [(1, 1, 1, 1000), (2, 5, 1, 1000), (10, 50, 10, 10000), (50, 100, 20, 50000)]

	for q, d, b, s in test_params:
		predicted = predict_fn(q, d, b, s)
		assert predicted >= 0, f"Negative prediction: q={q}, d={d}, b={b}, s={s}, predicted={predicted}"


def test_js_python_consistency_via_subprocess():
	"""Test that JavaScript and Python produce consistent predictions."""
	# Get repository root (parent of tests directory)
	repo_root = Path(__file__).parent.parent
	js_test_file = repo_root / "test_js_python_consistency.js"

	# Run the Node.js test from repository root
	result = subprocess.run(["node", str(js_test_file)], capture_output=True, text=True, cwd=str(repo_root))

	# Check that the test passed
	assert result.returncode == 0, f"JS-Python consistency test failed:\n{result.stdout}\n{result.stderr}"
	assert "🎉" in result.stdout, "JS-Python consistency test did not pass successfully"
	assert "10 passed, 0 failed" in result.stdout, "Not all tests passed"
