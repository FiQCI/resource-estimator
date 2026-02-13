"""Tests for analytical model fitting and prediction."""

import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import json

from resource_estimator.cli.build import analytical_model, objective, fit_analytical_model


class TestAnalyticalModel:
	"""Test suite for analytical model functions."""

	def test_analytical_model_basic(self):
		"""Test analytical model with basic parameters."""
		params = [1.0, 0.95, 0.0005, 10.0]  # T_init, efficiency_base, throughput, batch_cap
		batches = np.array([1, 5, 10])
		shots = np.array([1000, 5000, 10000])

		result = analytical_model(params, batches, shots)

		# Check result shape and type
		assert result.shape == batches.shape
		assert isinstance(result, np.ndarray)

		# Check all predictions are positive
		assert np.all(result > 0)

		# Check predictions increase with more shots/batches
		assert result[0] < result[1] < result[2]

	def test_analytical_model_edge_cases(self):
		"""Test analytical model with edge case parameters."""
		params = [0.5, 0.99, 0.0001, 20.0]

		# Single batch, single shot
		result = analytical_model(params, np.array([1]), np.array([1]))
		assert result[0] > 0

		# Large batches (beyond cap)
		large_batches = np.array([50, 100, 200])
		shots = np.array([1000, 1000, 1000])
		result = analytical_model(params, large_batches, shots)
		assert np.all(result > 0)

		# Efficiency should saturate beyond batch_cap
		# So runtime per shot should be similar for large batch counts
		runtime_per_shot = result / (large_batches * shots)
		assert np.std(runtime_per_shot) < 0.001  # Low variance when saturated

	def test_analytical_model_efficiency_decay(self):
		"""Test that runtime increases with batches below cap."""
		params = [1.0, 0.9, 0.001, 20.0]  # cap=20
		shots = 1000

		# Test batches well below cap
		batches_low = np.array([1, 5, 10])
		result_low = analytical_model(params, batches_low, np.full(3, shots))

		# Total runtime should increase with more batches (when below cap)
		assert result_low[0] < result_low[1] < result_low[2]

	def test_analytical_model_parameter_bounds(self):
		"""Test analytical model respects parameter constraints."""
		# Efficiency base should be clamped to [0.5, 0.999]
		params_low = [1.0, 0.3, 0.001, 10.0]  # efficiency too low
		params_high = [1.0, 1.1, 0.001, 10.0]  # efficiency too high

		batches = np.array([5])
		shots = np.array([1000])

		result_low = analytical_model(params_low, batches, shots)
		result_high = analytical_model(params_high, batches, shots)

		# Both should produce valid positive results
		assert result_low[0] > 0
		assert result_high[0] > 0

	def test_objective_function(self):
		"""Test objective function for optimization."""
		params = [1.0, 0.95, 0.0005, 10.0]
		batches = np.array([1, 5, 10, 15, 20])
		shots = np.array([1000, 2000, 3000, 4000, 5000])

		# Generate synthetic true values
		y_true = analytical_model(params, batches, shots) + np.random.normal(0, 1, len(batches))

		# Calculate RMSE
		rmse = objective(params, batches, shots, y_true)

		assert isinstance(rmse, (float, np.floating))
		assert rmse >= 0  # RMSE must be non-negative

		# Test that perfect fit gives near-zero RMSE
		y_perfect = analytical_model(params, batches, shots)
		rmse_perfect = objective(params, batches, shots, y_perfect)
		assert rmse_perfect < 1e-10

	def test_fit_analytical_model_integration(self):
		"""Integration test for full analytical model fitting pipeline."""
		# Create synthetic data
		true_params = [0.9, 0.98, 0.0006, 15.0]
		n_samples = 50

		np.random.seed(42)
		batches = np.random.randint(1, 25, n_samples)
		shots = np.random.randint(1000, 50000, n_samples)
		qpu_seconds = analytical_model(true_params, batches, shots) + np.random.normal(0, 2, n_samples)

		# Create temporary CSV
		df = pd.DataFrame({"batches": batches, "shots": shots, "qpu_seconds": qpu_seconds})

		with tempfile.TemporaryDirectory() as tmpdir:
			data_path = Path(tmpdir) / "test_data.csv"
			output_dir = Path(tmpdir) / "output"

			df.to_csv(data_path, index=False)

			# Run fitting
			js_config, r2, rmse, mae = fit_analytical_model(str(data_path), str(output_dir))

			# Validate outputs
			assert isinstance(js_config, dict)
			assert "name" in js_config
			assert "model_type" in js_config
			assert js_config["model_type"] == "analytical"
			assert "T_init" in js_config
			assert "efficiency_base" in js_config
			assert "throughput_coef" in js_config
			assert "batch_cap" in js_config

			# Validate metrics
			assert 0 <= r2 <= 1
			assert rmse >= 0
			assert mae >= 0

			# Check files were created
			config_file = output_dir / "vtt-q50_analytical_config.json"
			plot_file = output_dir / "vtt-q50_analytical_model.png"

			assert config_file.exists()
			assert plot_file.exists()

			# Validate config file content
			with open(config_file) as f:
				saved_config = json.load(f)
			assert saved_config == js_config

	def test_fit_analytical_model_realistic_data(self):
		"""Test fitting with realistic VTT Q50-like data."""
		# Create realistic synthetic data mimicking VTT Q50
		params = [0.88, 0.986, 0.000625, 19.0]
		n_samples = 100

		np.random.seed(123)
		batches = np.concatenate([np.random.randint(1, 10, n_samples // 2), np.random.randint(10, 25, n_samples // 2)])
		shots = np.random.choice([1000, 5000, 10000, 25000, 50000], n_samples)
		qpu_seconds = analytical_model(params, batches, shots) + np.random.normal(0, 5, n_samples)
		qpu_seconds = np.maximum(qpu_seconds, 1.0)  # Ensure positive

		df = pd.DataFrame({"batches": batches, "shots": shots, "qpu_seconds": qpu_seconds})

		with tempfile.TemporaryDirectory() as tmpdir:
			data_path = Path(tmpdir) / "realistic_data.csv"
			output_dir = Path(tmpdir) / "output"

			df.to_csv(data_path, index=False)

			js_config, r2, rmse, mae = fit_analytical_model(str(data_path), str(output_dir))

			# Model should fit reasonably well
			assert r2 > 0.85, f"R² too low: {r2}"
			assert rmse < 50, f"RMSE too high: {rmse}"

			# Parameters should be reasonable
			assert 0.5 < js_config["T_init"] < 5.0
			assert 0.85 < js_config["efficiency_base"] < 0.99
			assert 0.0001 < js_config["throughput_coef"] < 0.001
			assert 5.0 < js_config["batch_cap"] < 20.0

	def test_analytical_model_array_sizes(self):
		"""Test analytical model handles different array sizes."""
		params = [1.0, 0.95, 0.0005, 10.0]

		# Test with different sizes
		for size in [1, 10, 100, 1000]:
			batches = np.random.randint(1, 20, size)
			shots = np.random.randint(1000, 10000, size)

			result = analytical_model(params, batches, shots)

			assert len(result) == size
			assert np.all(result > 0)

	def test_analytical_model_vectorization(self):
		"""Test analytical model is properly vectorized."""
		params = [1.0, 0.95, 0.0005, 10.0]

		# Compare vectorized vs loop
		batches = np.array([1, 5, 10, 15, 20])
		shots = np.array([1000, 2000, 3000, 4000, 5000])

		# Vectorized
		result_vec = analytical_model(params, batches, shots)

		# Loop (for comparison)
		result_loop = np.zeros(len(batches))
		for i in range(len(batches)):
			result_loop[i] = analytical_model(params, np.array([batches[i]]), np.array([shots[i]]))[0]

		# Should be identical
		np.testing.assert_array_almost_equal(result_vec, result_loop)
