"""Tests for server limits validation."""

from unittest.mock import Mock

import pytest

from resource_estimator.data_collection import ServerLimits, run_single_experiment, validate_job_parameters


def test_server_limits_defaults():
	"""Test default server limits."""
	limits = ServerLimits()
	assert limits.max_shots == 10_000_000
	assert limits.max_circuits == 10_000
	assert limits.max_instructions_per_circuit == 100_000
	assert limits.max_queued_jobs == 100


def test_server_limits_custom():
	"""Test custom server limits."""
	limits = ServerLimits(max_shots=5000, max_circuits=100, max_instructions_per_circuit=10000)
	assert limits.max_shots == 5000
	assert limits.max_circuits == 100
	assert limits.max_instructions_per_circuit == 10000


def test_validate_job_parameters_valid():
	"""Test validation with valid parameters."""
	is_valid, error = validate_job_parameters(num_qubits=3, depth=10, batches=5, shots=1000)
	assert is_valid is True
	assert error is None


def test_validate_job_parameters_exceeds_shots():
	"""Test validation when shots exceed limit."""
	limits = ServerLimits(max_shots=5000)
	is_valid, error = validate_job_parameters(num_qubits=3, depth=10, batches=5, shots=10000, limits=limits)
	assert is_valid is False
	assert "shots" in error.lower()
	assert "5000" in error


def test_validate_job_parameters_exceeds_circuits():
	"""Test validation when circuits exceed limit."""
	limits = ServerLimits(max_circuits=100)
	is_valid, error = validate_job_parameters(num_qubits=3, depth=10, batches=200, shots=1000, limits=limits)
	assert is_valid is False
	assert "batches" in error.lower() or "circuits" in error.lower()
	assert "100" in error


def test_validate_job_parameters_exceeds_instructions():
	"""Test validation when estimated instructions exceed limit."""
	limits = ServerLimits(max_instructions_per_circuit=100)
	# With depth=50, num_qubits=10: estimated = 50*10*2 + 10 = 1010 instructions > 100
	is_valid, error = validate_job_parameters(num_qubits=10, depth=50, batches=5, shots=1000, limits=limits)
	assert is_valid is False
	assert "instructions" in error.lower()
	assert "100" in error


def test_validate_job_parameters_edge_case():
	"""Test validation at boundary values."""
	limits = ServerLimits(max_shots=10000, max_circuits=100, max_instructions_per_circuit=1000)

	# Exactly at limit should be valid
	is_valid, error = validate_job_parameters(num_qubits=2, depth=10, batches=100, shots=10000, limits=limits)
	assert is_valid is True
	assert error is None


def test_run_single_experiment_with_invalid_parameters():
	"""Test that run_single_experiment fails with invalid parameters."""
	backend = Mock()
	backend.architecture.qubits = [0, 1, 2, 3, 4]

	# Create restrictive limits
	limits = ServerLimits(max_shots=100, max_circuits=5, max_instructions_per_circuit=50)

	# Try to run with shots exceeding limit
	result = run_single_experiment(backend, num_qubits=2, depth=5, batches=1, shots=200, timeout=600.0, limits=limits)

	# Should fail with validation error
	assert result["error"] is not None
	assert "validation" in result["error"].lower() or "shots" in result["error"].lower()
	assert result["qpu_seconds"] is None


def test_run_single_experiment_with_valid_parameters():
	"""Test that run_single_experiment works with valid parameters."""
	from unittest.mock import patch

	backend = Mock()
	backend.architecture.qubits = [0, 1, 2, 3, 4]
	backend.num_qubits = 5

	# Mock the backend.run to avoid actual execution
	mock_job = Mock()
	mock_result = Mock()

	# Create timeline entries
	from datetime import datetime, timedelta

	class TimelineEntry:
		def __init__(self, status, timestamp):
			self.status = status
			self.timestamp = timestamp

	start_time = datetime.now()
	end_time = start_time + timedelta(seconds=5)

	mock_result.timeline = [TimelineEntry("execution_started", start_time), TimelineEntry("execution_ended", end_time)]

	mock_job.result.return_value = mock_result
	backend.run.return_value = mock_job

	limits = ServerLimits()

	# Mock transpile to avoid backend issues
	with patch("resource_estimator.data_collection.transpile") as mock_transpile:
		mock_transpile.return_value = [Mock(data=[1, 2, 3])]  # Mock transpiled circuit

		result = run_single_experiment(
			backend, num_qubits=2, depth=5, batches=1, shots=1000, timeout=600.0, limits=limits
		)

	# Should succeed
	assert result["error"] is None
	assert result["qpu_seconds"] is not None
	assert result["qpu_seconds"] == pytest.approx(5.0, rel=0.1)


def test_validate_multiple_violations():
	"""Test validation with multiple limit violations."""
	limits = ServerLimits(max_shots=1000, max_circuits=5, max_instructions_per_circuit=50)

	# Violates shots limit
	is_valid, error = validate_job_parameters(num_qubits=2, depth=5, batches=1, shots=2000, limits=limits)
	assert is_valid is False
	assert "shots" in error.lower()

	# Violates circuits limit
	is_valid, error = validate_job_parameters(num_qubits=2, depth=5, batches=10, shots=500, limits=limits)
	assert is_valid is False
	assert "batches" in error.lower() or "circuits" in error.lower()

	# Violates instructions limit
	is_valid, error = validate_job_parameters(num_qubits=10, depth=10, batches=1, shots=500, limits=limits)
	assert is_valid is False
	assert "instructions" in error.lower()
