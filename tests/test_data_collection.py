"""Tests for data_collection module."""

import pytest
from unittest.mock import Mock, patch

from resource_estimator.data_collection import (
	create_random_circuit,
	generate_latin_hypercube_samples,
	run_single_experiment,
)


def test_create_random_circuit():
	"""Test random circuit generation."""
	circuit = create_random_circuit(3, 5)

	assert circuit.num_qubits == 3
	assert len(circuit) > 0  # Has some gates


def test_generate_latin_hypercube_samples():
	"""Test Latin Hypercube sampling."""
	param_ranges = {"batches": (1, 10), "qubits": (2, 5), "depth": (5, 20), "shots": (1000, 5000)}

	samples = generate_latin_hypercube_samples(param_ranges, num_samples=10)

	assert len(samples) == 10

	for sample in samples:
		assert 1 <= sample["batches"] <= 10
		assert 2 <= sample["qubits"] <= 5
		assert 5 <= sample["depth"] <= 20
		assert 1000 <= sample["shots"] <= 5000


@patch("resource_estimator.data_collection.transpile")
def test_run_single_experiment_success(mock_transpile):
	"""Test successful experiment run."""
	from datetime import datetime, timedelta, timezone

	# Mock backend
	backend = Mock()
	backend.run = Mock()

	# Mock job result with timeline (iqm-client>=33.0.2 format)
	mock_result = Mock()

	# Create mock timeline entries
	start_time = datetime(2026, 1, 20, 9, 37, 59, 259703, tzinfo=timezone.utc)
	end_time = start_time + timedelta(seconds=50.5)

	timeline_entry_start = Mock()
	timeline_entry_start.status = "execution_started"
	timeline_entry_start.timestamp = start_time

	timeline_entry_end = Mock()
	timeline_entry_end.status = "execution_ended"
	timeline_entry_end.timestamp = end_time

	mock_result.timeline = [timeline_entry_start, timeline_entry_end]

	mock_job = Mock()
	mock_job.result = Mock(return_value=mock_result)
	backend.run.return_value = mock_job

	# Mock transpile
	mock_transpile.return_value = [Mock()]

	# Run experiment
	result = run_single_experiment(backend, num_qubits=3, depth=5, batches=2, shots=1000, timeout=600.0)

	assert result["error"] is None
	assert result["qpu_seconds"] == 50.5
	assert result["num_qubits"] == 3
	assert result["depth"] == 5
	assert result["batches"] == 2
	assert result["shots"] == 1000


@patch("resource_estimator.data_collection.transpile")
def test_run_single_experiment_failure(mock_transpile):
	"""Test experiment failure handling."""
	backend = Mock()
	backend.run = Mock(side_effect=RuntimeError("Backend error"))

	mock_transpile.return_value = [Mock()]

	result = run_single_experiment(backend, num_qubits=3, depth=5, batches=1, shots=1000, timeout=600.0)

	assert result["error"] is not None
	assert "Backend error" in result["error"]
	assert result["qpu_seconds"] is None


@patch("resource_estimator.data_collection.IQMProvider")
def test_connect_to_backend_success(mock_provider):
	"""Test successful backend connection."""
	from resource_estimator.data_collection import connect_to_backend

	mock_backend = Mock()
	mock_backend.name = "test-backend"
	mock_provider.return_value.get_backend.return_value = mock_backend

	backend = connect_to_backend("https://test.server")

	assert backend.name == "test-backend"


@patch("resource_estimator.data_collection.IQMProvider")
def test_connect_to_backend_failure(mock_provider):
	"""Test backend connection failure."""
	from resource_estimator.data_collection import connect_to_backend

	mock_provider.side_effect = RuntimeError("Connection failed")

	with pytest.raises(RuntimeError, match="Failed to connect"):
		connect_to_backend("https://test.server")
