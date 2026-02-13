"""Data collection module for quantum resource estimation."""

import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from iqm.qiskit_iqm import IQMProvider
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.random import random_circuit
from qiskit.providers import JobStatus
from scipy.stats import qmc
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class ServerLimits:
	"""IQM server limits."""

	max_shots: int = 10_000_000
	max_circuits: int = 10_000
	max_instructions_per_circuit: int = 100_000
	max_queued_jobs: int = 100


def validate_job_parameters(
	num_qubits: int, depth: int, batches: int, shots: int, limits: ServerLimits | None = None
) -> tuple[bool, str | None]:
	"""Validate job parameters against server limits.

	Args:
		num_qubits: Number of qubits
		depth: Circuit depth
		batches: Number of circuits
		shots: Number of shots
		limits: Server limits (uses defaults if None)

	Returns:
		Tuple of (is_valid, error_message)
	"""
	if limits is None:
		limits = ServerLimits()

	# Check shots limit
	if shots > limits.max_shots:
		return False, f"Shots ({shots}) exceeds server limit ({limits.max_shots})"

	# Check circuits limit
	if batches > limits.max_circuits:
		return False, f"Batches/circuits ({batches}) exceeds server limit ({limits.max_circuits})"

	# Estimate instructions per circuit
	# Random circuit with depth and num_qubits typically has:
	# - depth layers of gates (each layer has ~1-2 gates per qubit)
	# - measurements at the end (1 per qubit)
	# Conservative estimate: depth * num_qubits * 2 + num_qubits
	estimated_instructions = depth * num_qubits * 2 + num_qubits

	if estimated_instructions > limits.max_instructions_per_circuit:
		return (
			False,
			f"Estimated instructions (~{estimated_instructions}) may exceed server limit ({limits.max_instructions_per_circuit}). "
			f"Consider reducing depth ({depth}) or qubits ({num_qubits}).",
		)

	# Note: We don't validate payload size in MB because:
	# 1. Server doesn't specify this limit in its API
	# 2. The limit varies and we don't want to reject valid combinations
	# 3. If payload is too large, server returns error which we catch and log
	# The error handler in run_single_experiment() provides detailed logging for payload errors

	return True, None


def wait_for_job_completion(job: Any, timeout: float = 1800.0, poll_interval: float = 5.0) -> Any:
	"""Wait for a job to complete by polling its status.

	This is more robust than job.result(timeout) because it allows jobs to remain
	queued beyond the timeout without failing. We poll the job status periodically
	and only fail if the job reaches an error/cancelled state.

	Args:
		job: The quantum job to wait for
		timeout: Maximum time to wait in seconds (default: 1800s = 30 min)
		poll_interval: Time between status checks in seconds (default: 5s)

	Returns:
		Job result if successful

	Raises:
		RuntimeError: If job fails, is cancelled, or times out
	"""
	start_time = time.time()
	last_status = None
	status_change_count = 0

	while True:
		current_status = job.status()
		elapsed = time.time() - start_time

		# Log status changes
		if current_status != last_status:
			status_change_count += 1
			logger.info(
				f"Job {job.job_id()} status: {current_status} (elapsed: {elapsed:.1f}s, change #{status_change_count})"
			)
			last_status = current_status

		# Check for terminal statuses
		if current_status == JobStatus.DONE:
			logger.info(f"Job {job.job_id()} completed successfully after {elapsed:.1f}s")
			return job.result()

		if current_status == JobStatus.ERROR:
			raise RuntimeError(f"Job {job.job_id()} failed with error status after {elapsed:.1f}s")

		if current_status == JobStatus.CANCELLED:
			raise RuntimeError(f"Job {job.job_id()} was cancelled after {elapsed:.1f}s")

		# Check timeout
		if elapsed > timeout:
			raise RuntimeError(
				f"Job {job.job_id()} timed out after {timeout:.1f}s. "
				f"Last status: {current_status}. "
				f"Job may still be queued on backend - consider increasing timeout or checking backend queue."
			)

		# Wait before next poll
		time.sleep(poll_interval)


def create_random_circuit(num_qubits: int, depth: int) -> QuantumCircuit:
	"""Create a random quantum circuit for timing measurements.

	Args:
		num_qubits: Number of qubits
		depth: Circuit depth (number of layers)

	Returns:
		Random quantum circuit
	"""
	circuit = random_circuit(num_qubits, depth, measure=True, seed=None)
	return circuit


def run_single_experiment(
	backend: Any,
	num_qubits: int,
	depth: int,
	batches: int,
	shots: int,
	timeout: float = 1800.0,
	limits: ServerLimits | None = None,
) -> dict[str, Any]:
	"""Run a single quantum experiment and measure execution time.

	This function now uses status polling instead of blocking timeout, which allows
	jobs to remain queued on the backend without timing out prematurely.

	Args:
		backend: Quantum backend
		num_qubits: Number of qubits
		depth: Circuit depth
		batches: Number of circuits in batch
		shots: Number of measurement shots
		timeout: Maximum time to wait for job completion in seconds (default: 1800s = 30 min).
		         This includes both queue time and execution time.
		limits: Server limits for validation (uses defaults if None)

	Returns:
		Dictionary with experiment results including qpu_seconds
	"""
	result_template = {
		"num_qubits": num_qubits,
		"depth": depth,
		"batches": batches,
		"shots": shots,
		"qpu_seconds": None,
		"error": None,
	}

	try:
		# Validate parameters against server limits
		is_valid, error_msg = validate_job_parameters(num_qubits, depth, batches, shots, limits)
		if not is_valid:
			raise ValueError(f"Parameter validation failed: {error_msg}")

		# Generate random circuits
		circuits = [create_random_circuit(num_qubits, depth) for _ in range(batches)]

		# Transpile for backend
		transpiled = transpile(circuits, backend=backend)

		# Run job
		job = backend.run(transpiled, shots=shots)
		job_id = job.job_id()
		logger.info(f"Submitted job {job_id} (qubits={num_qubits}, depth={depth}, batches={batches}, shots={shots})")

		# Wait for job completion using status polling (more robust than blocking timeout)
		result = wait_for_job_completion(job, timeout=timeout, poll_interval=5.0)

		# Extract timing from result.timeline (iqm-client>=33.0.2)
		exec_start = next(e for e in result.timeline if e.status == "execution_started")
		exec_end = next(e for e in result.timeline if e.status == "execution_ended")
		qpu_seconds = (exec_end.timestamp - exec_start.timestamp).total_seconds()

		result_template["qpu_seconds"] = qpu_seconds
		return result_template

	except Exception as e:
		# Log detailed parameters when experiment fails
		error_msg = str(e)
		logger.warning(
			f"Experiment failed: {error_msg}\n"
			f"  Parameters: qubits={num_qubits}, depth={depth}, batches={batches}, shots={shots}\n"
			f"  Estimated instructions: {num_qubits * depth * 10}\n"
			f"  Total instructions: {batches * num_qubits * depth * 10}",
			exc_info=True,
		)

		# Special handling for payload size errors
		if "payload" in error_msg.lower() or "too large" in error_msg.lower():
			logger.error(
				f"❌ PAYLOAD TOO LARGE ERROR:\n"
				f"   Circuit: {num_qubits} qubits × {depth} depth = ~{num_qubits * depth * 10} instructions/circuit\n"
				f"   Batch: {batches} circuits\n"
				f"   Total: ~{batches * num_qubits * depth * 10} instructions in batch\n"
				f"   Shots: {shots}\n"
				f"   This combination exceeds server payload limits!"
			)

		result_template["error"] = str(e)
		return result_template


def generate_latin_hypercube_samples(
	param_ranges: dict[str, tuple[int, int]], num_samples: int
) -> list[dict[str, int]]:
	"""Generate parameter samples using Latin Hypercube sampling.

	Args:
		param_ranges: Dictionary of parameter ranges
		num_samples: Number of samples to generate

	Returns:
		List of parameter dictionaries
	"""
	sampler = qmc.LatinHypercube(d=len(param_ranges))
	sample = sampler.random(n=num_samples)

	samples = []
	param_names = list(param_ranges.keys())

	for point in sample:
		params = {}
		for i, name in enumerate(param_names):
			min_val, max_val = param_ranges[name]
			params[name] = int(point[i] * (max_val - min_val) + min_val)
		samples.append(params)

	return samples


def _run_isolated_sweeps(
	backend: Any, param_ranges: dict, base_params: dict, job_timeout: float, limits: ServerLimits | None = None
) -> list[dict[str, Any]]:
	"""Run isolated parameter sweeps.

	Args:
		backend: Quantum backend
		param_ranges: Dictionary of parameter ranges
		base_params: Base parameters for the sweep
		job_timeout: Timeout for individual jobs
		limits: Server limits for validation
	"""
	all_data = []
	for param_name, (min_val, max_val) in param_ranges.items():
		num_points = min(10, max_val - min_val + 1)
		values = np.linspace(min_val, max_val, num_points, dtype=int) if num_points > 1 else [min_val]

		for val in tqdm(values, desc=f"Sweep {param_name}"):
			params = base_params.copy()
			params[param_name] = int(val)

			result = run_single_experiment(
				backend, params["qubits"], params["depth"], params["batches"], params["shots"], job_timeout, limits
			)

			if result["error"] is None and result["qpu_seconds"] is not None:
				all_data.append(result)
	return all_data


def _save_checkpoint(data: list[dict[str, Any]], checkpoint_path: str) -> None:
	"""Save collected data to checkpoint file.

	Args:
		data: List of data points to save
		checkpoint_path: Path to checkpoint file
	"""
	import pandas as pd
	from pathlib import Path

	Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
	pd.DataFrame(data).to_csv(checkpoint_path, index=False)
	logger.debug(f"Checkpoint saved: {len(data)} data points")


def _load_checkpoint(checkpoint_path: str) -> tuple[list[dict[str, Any]], set[tuple]]:
	"""Load data from checkpoint file if it exists.

	Args:
		checkpoint_path: Path to checkpoint file

	Returns:
		Tuple of (list of previously collected data points, set of param tuples for fast lookup)
	"""
	import pandas as pd
	from pathlib import Path

	if not Path(checkpoint_path).exists():
		return [], set()

	try:
		df = pd.read_csv(checkpoint_path)
		data = df.to_dict("records")
		logger.info(f"Resuming from checkpoint: {len(data)} data points already collected")

		# Create set of tuples for O(1) lookup
		param_set = set()
		for point in data:
			qubits = point.get("num_qubits") or point.get("qubits")
			param_tuple = (qubits, point.get("depth"), point.get("batches"), point.get("shots"))
			param_set.add(param_tuple)

		return data, param_set
	except Exception as e:
		logger.warning(f"Failed to load checkpoint: {e}. Starting fresh.")
		return [], set()


def _is_already_collected(params: dict[str, int], param_set: set[tuple]) -> bool:
	"""Check if parameters have already been collected (O(1) lookup).

	Args:
		params: Parameter dictionary to check (keys: qubits, depth, batches, shots)
		param_set: Set of already collected parameter tuples for fast lookup

	Returns:
		True if this parameter combination has been collected
	"""
	param_qubits = params.get("qubits") or params.get("num_qubits")
	param_tuple = (param_qubits, params.get("depth"), params.get("batches"), params.get("shots"))
	return param_tuple in param_set


def collect_timing_data(
	backend: Any,
	num_samples: int = 50,
	include_isolated: bool = True,
	job_timeout: float = 1800.0,
	checkpoint_path: str | None = None,
	limits: ServerLimits | None = None,
) -> list[dict[str, Any]]:
	"""Collect quantum timing data through systematic sampling.

	Args:
		backend: Quantum backend
		num_samples: Number of comprehensive samples
		include_isolated: Include isolated parameter sweeps
		job_timeout: Timeout for individual job completion in seconds (default: 1800s = 30 min).
		             Increased from 900s to better handle backend queue delays.
		checkpoint_path: Path to checkpoint file for saving/resuming data collection
		limits: Server limits for validation (uses defaults if None)

	Returns:
		List of timing data points
	"""
	if limits is None:
		limits = ServerLimits()
		logger.info(
			f"Using default server limits: max_shots={limits.max_shots}, max_circuits={limits.max_circuits}, max_instructions={limits.max_instructions_per_circuit}"
		)

	max_qubits = len(backend.architecture.qubits)
	# Adjust parameter ranges to respect server limits
	max_safe_batches = min(25, limits.max_circuits)
	max_safe_shots = min(50000, limits.max_shots)
	# Ensure depth * qubits * 2 + qubits < max_instructions (with safety margin)
	max_safe_depth = min(100, (limits.max_instructions_per_circuit - max_qubits) // (max_qubits * 2))

	param_ranges = {
		"batches": (1, max_safe_batches),
		"qubits": (1, max_qubits),
		"depth": (1, max_safe_depth),
		"shots": (1000, max_safe_shots),
	}
	base_params = {"batches": 1, "qubits": 2, "depth": 5, "shots": 1000}

	logger.info(
		f"Parameter ranges adjusted for server limits: batches=(1,{max_safe_batches}), depth=(1,{max_safe_depth}), shots=(1000,{max_safe_shots})"
	)

	# Load existing data if resuming (returns data list and param set for fast lookup)
	all_data, param_set = _load_checkpoint(checkpoint_path) if checkpoint_path else ([], set())

	# Isolated parameter sweeps
	if include_isolated:
		logger.info("Running isolated parameter sweeps...")
		isolated_data = _run_isolated_sweeps(backend, param_ranges, base_params, job_timeout, limits)
		all_data.extend(isolated_data)
		if checkpoint_path:
			_save_checkpoint(all_data, checkpoint_path)
			# Update param_set with new data
			for data_point in isolated_data:
				qubits = data_point.get("num_qubits") or data_point.get("qubits")
				param_tuple = (qubits, data_point.get("depth"), data_point.get("batches"), data_point.get("shots"))
				param_set.add(param_tuple)

	# Latin Hypercube sampling
	logger.info(f"Collecting {num_samples} samples via Latin Hypercube...")
	samples = generate_latin_hypercube_samples(param_ranges, num_samples)

	# Skip already collected samples when resuming (O(1) lookup with set)
	skipped = 0

	for params in tqdm(samples, desc="Comprehensive sampling"):
		if checkpoint_path and _is_already_collected(params, param_set):
			skipped += 1
			continue

		result = run_single_experiment(
			backend, params["qubits"], params["depth"], params["batches"], params["shots"], job_timeout, limits
		)

		if result["error"] is None and result["qpu_seconds"] is not None:
			all_data.append(result)
			logger.info(f"✓ Collected sample {len(all_data)}/{num_samples}: qpu_seconds={result['qpu_seconds']:.2f}s")

			# Update param_set with new data
			if checkpoint_path:
				qubits = result.get("num_qubits") or params.get("qubits")
				param_tuple = (qubits, result.get("depth"), result.get("batches"), result.get("shots"))
				param_set.add(param_tuple)
				_save_checkpoint(all_data, checkpoint_path)
				logger.debug(f"Checkpoint saved: {len(all_data)} samples")

	if checkpoint_path and skipped > 0:
		logger.info(f"Skipped {skipped} already-collected samples")

	logger.info(f"Collected {len(all_data)} data points")
	return all_data


def connect_to_backend(server_url: str) -> Any:
	"""Connect to IQM quantum backend.

	Args:
		server_url: IQM server URL

	Returns:
		Backend instance

	Raises:
		RuntimeError: If connection fails
	"""
	try:
		provider = IQMProvider(server_url)
		backend = provider.get_backend()
		logger.info(f"Connected to backend: {backend.name}")
		return backend
	except Exception as e:
		raise RuntimeError(f"Failed to connect to {server_url}: {e}") from e
