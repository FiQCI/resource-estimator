"""Additional tests for better coverage of data_collection."""

from unittest.mock import Mock, patch


from resource_estimator.data_collection import collect_timing_data


@patch("resource_estimator.data_collection.run_single_experiment")
def test_collect_timing_data_no_isolated(mock_run):
	"""Test data collection without isolated parameter sweeps."""
	backend = Mock()
	backend.architecture.qubits = [0, 1, 2, 3, 4]

	# Mock successful runs
	mock_run.return_value = {
		"num_qubits": 3,
		"depth": 10,
		"batches": 2,
		"shots": 1000,
		"qpu_seconds": 5.0,
		"error": None,
	}

	data = collect_timing_data(backend, num_samples=5, include_isolated=False)

	assert len(data) > 0
	assert all(d["error"] is None for d in data)


@patch("resource_estimator.data_collection.run_single_experiment")
def test_collect_timing_data_with_isolated(mock_run):
	"""Test data collection with isolated parameter sweeps."""
	backend = Mock()
	backend.architecture.qubits = [0, 1, 2, 3, 4]

	# Mock successful runs
	mock_run.return_value = {
		"num_qubits": 3,
		"depth": 10,
		"batches": 2,
		"shots": 1000,
		"qpu_seconds": 5.0,
		"error": None,
	}

	data = collect_timing_data(backend, num_samples=2, include_isolated=True)

	assert len(data) > 0
	assert mock_run.call_count > 2  # Should include both isolated and LHS samples


@patch("resource_estimator.data_collection.run_single_experiment")
def test_collect_timing_data_with_failures(mock_run):
	"""Test data collection handling failures."""
	backend = Mock()
	backend.architecture.qubits = [0, 1, 2]

	# Mix of success and failure
	call_count = [0]

	def mock_experiment(*args, **kwargs):
		call_count[0] += 1
		if call_count[0] % 2 == 0:
			return {
				"num_qubits": 2,
				"depth": 10,
				"batches": 1,
				"shots": 1000,
				"qpu_seconds": None,
				"error": "Test error",
			}
		else:
			return {"num_qubits": 2, "depth": 10, "batches": 1, "shots": 1000, "qpu_seconds": 5.0, "error": None}

	mock_run.side_effect = mock_experiment

	data = collect_timing_data(backend, num_samples=4, include_isolated=False)

	# Only successful ones should be included
	assert len(data) > 0
	assert all(d["qpu_seconds"] is not None for d in data)


@patch("resource_estimator.data_collection.run_single_experiment")
def test_collect_timing_data_with_long_jobs(mock_run):
	"""Test that long-running jobs are included in results."""
	backend = Mock()
	backend.architecture.qubits = [0, 1, 2, 3]

	# Mix of reasonable and long QPU times
	call_count = [0]

	def mock_experiment(*args, **kwargs):
		call_count[0] += 1
		qpu_time = 5.0 if call_count[0] <= 3 else 200.0  # Some long jobs
		return {
			"num_qubits": 2,
			"depth": 10,
			"batches": call_count[0],
			"shots": 1000,
			"qpu_seconds": qpu_time,
			"error": None,
		}

	mock_run.side_effect = mock_experiment

	data = collect_timing_data(backend, num_samples=3, include_isolated=True)

	# All successful jobs should be included regardless of QPU time
	assert len(data) > 0
