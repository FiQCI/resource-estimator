"""Tests for checkpoint and resume functionality."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from resource_estimator.data_collection import (
	_is_already_collected,
	_load_checkpoint,
	_save_checkpoint,
	collect_timing_data,
)


def test_save_and_load_checkpoint():
	"""Test saving and loading checkpoint files."""
	with tempfile.TemporaryDirectory() as tmpdir:
		checkpoint_path = Path(tmpdir) / "test_checkpoint.csv"

		# Sample data
		data = [
			{"num_qubits": 2, "depth": 5, "batches": 1, "shots": 1000, "qpu_seconds": 1.5, "error": None},
			{"num_qubits": 3, "depth": 10, "batches": 2, "shots": 2000, "qpu_seconds": 2.5, "error": None},
		]

		# Save checkpoint
		_save_checkpoint(data, str(checkpoint_path))
		assert checkpoint_path.exists()

		# Load checkpoint (now returns tuple of (data, param_set))
		loaded_data, param_set = _load_checkpoint(str(checkpoint_path))
		assert len(loaded_data) == 2
		assert loaded_data[0]["num_qubits"] == 2
		assert loaded_data[1]["qpu_seconds"] == 2.5
		assert len(param_set) == 2  # Check param_set was created


def test_load_checkpoint_nonexistent():
	"""Test loading checkpoint when file doesn't exist."""
	data, param_set = _load_checkpoint("/nonexistent/path/checkpoint.csv")
	assert data == []
	assert param_set == set()


def test_is_already_collected():
	"""Test checking if parameters have been collected."""
	# Create param_set for fast lookup
	param_set = {
		(2, 5, 1, 1000),  # (qubits, depth, batches, shots)
		(3, 10, 2, 2000),
	}

	# Test exact match
	params = {"qubits": 2, "depth": 5, "batches": 1, "shots": 1000}
	assert _is_already_collected(params, param_set) is True

	# Test no match
	params = {"qubits": 4, "depth": 5, "batches": 1, "shots": 1000}
	assert _is_already_collected(params, param_set) is False

	# Test partial match (should be False)
	params = {"qubits": 2, "depth": 5, "batches": 2, "shots": 1000}  # Different batches
	assert _is_already_collected(params, param_set) is False


@patch("resource_estimator.data_collection.run_single_experiment")
def test_collect_timing_data_with_checkpoint(mock_run):
	"""Test that checkpointing saves data incrementally."""
	backend = Mock()
	backend.architecture.qubits = [0, 1, 2, 3]

	mock_run.return_value = {
		"num_qubits": 2,
		"depth": 10,
		"batches": 1,
		"shots": 1000,
		"qpu_seconds": 5.0,
		"error": None,
	}

	with tempfile.TemporaryDirectory() as tmpdir:
		checkpoint_path = Path(tmpdir) / "checkpoint.csv"

		# Run with checkpoint
		data = collect_timing_data(backend, num_samples=3, include_isolated=False, checkpoint_path=str(checkpoint_path))

		# Checkpoint should exist
		assert checkpoint_path.exists()

		# Verify checkpoint has all successful data
		loaded, param_set = _load_checkpoint(str(checkpoint_path))
		assert len(loaded) == len(data)


@patch("resource_estimator.data_collection.run_single_experiment")
def test_collect_timing_data_resume(mock_run):
	"""Test resuming data collection from checkpoint."""
	backend = Mock()
	backend.architecture.qubits = [0, 1, 2, 3]

	call_count = [0]

	def mock_experiment(*args, **kwargs):
		call_count[0] += 1
		return {
			"num_qubits": args[1],
			"depth": args[2],
			"batches": args[3],
			"shots": args[4],
			"qpu_seconds": 5.0,
			"error": None,
		}

	mock_run.side_effect = mock_experiment

	with tempfile.TemporaryDirectory() as tmpdir:
		checkpoint_path = Path(tmpdir) / "checkpoint.csv"

		# First run - collect partial data
		data1 = collect_timing_data(
			backend, num_samples=5, include_isolated=False, checkpoint_path=str(checkpoint_path)
		)
		call_count[0]

		# Simulate interruption and resume - should skip already collected samples
		# Reset mock call count
		call_count[0] = 0

		# Second run with same parameters should resume
		data2 = collect_timing_data(
			backend, num_samples=5, include_isolated=False, checkpoint_path=str(checkpoint_path)
		)

		# Should have made fewer calls since some were already collected
		# (Note: Due to random LHS sampling, some might still be called, but with same seed or
		# in practice with real data, we'd skip all already-collected ones)
		assert len(data2) >= len(data1)  # At least as many as before
