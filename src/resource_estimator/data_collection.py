"""Data collection module for quantum resource estimation."""

import logging
from typing import Any

import numpy as np
from iqm.qiskit_iqm import IQMProvider
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.random import random_circuit
from scipy.stats import qmc
from tqdm import tqdm

logger = logging.getLogger(__name__)


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
	backend: Any, num_qubits: int, depth: int, batches: int, shots: int, timeout: float = 600.0
) -> dict[str, Any]:
	"""Run a single quantum experiment and measure execution time.

	Args:
		backend: Quantum backend
		num_qubits: Number of qubits
		depth: Circuit depth
		batches: Number of circuits in batch
		shots: Number of measurement shots
		timeout: Job timeout in seconds

	Returns:
		Dictionary with experiment results including qpu_seconds
	"""
	try:
		# Generate random circuits
		circuits = [create_random_circuit(num_qubits, depth) for _ in range(batches)]

		# Transpile for backend
		transpiled = transpile(circuits, backend=backend)

		# Run job
		job = backend.run(transpiled, shots=shots)
		result = job.result(timeout=timeout)

		# Extract timing from result.timeline (iqm-client>=33.0.2)
		qpu_seconds = 0.0
		metadata_found = False

		try:
			if hasattr(result, "timeline") and result.timeline:
				# Find execution_started and execution_ended entries
				exec_start_time = None
				exec_end_time = None

				for entry in result.timeline:
					if hasattr(entry, "status") and hasattr(entry, "timestamp"):
						if entry.status == "execution_started":
							exec_start_time = entry.timestamp
						elif entry.status == "execution_ended":
							exec_end_time = entry.timestamp

				# Calculate QPU seconds from timeline
				if exec_start_time and exec_end_time:
					qpu_seconds = (exec_end_time - exec_start_time).total_seconds()
					metadata_found = True
					logger.debug(
						f"Extracted QPU time from timeline: {qpu_seconds:.3f}s "
						f"(qubits={num_qubits}, depth={depth}, batches={batches}, shots={shots})"
					)

		except (AttributeError, TypeError, KeyError) as e:
			logger.debug(f"Timeline extraction failed: {e}")

		if not metadata_found:
			logger.warning(
				f"No timing data found in result.timeline for experiment "
				f"(qubits={num_qubits}, depth={depth}, batches={batches}, shots={shots}). "
				f"QPU time set to 0.0"
			)

		return {
			"num_qubits": num_qubits,
			"depth": depth,
			"batches": batches,
			"shots": shots,
			"qpu_seconds": qpu_seconds,
			"error": None,
		}

	except Exception as e:
		logger.warning(f"Experiment failed: {e}")
		return {
			"num_qubits": num_qubits,
			"depth": depth,
			"batches": batches,
			"shots": shots,
			"qpu_seconds": None,
			"error": str(e),
		}


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
	backend: Any, param_ranges: dict, base_params: dict, max_runtime: float
) -> list[dict[str, Any]]:
	"""Run isolated parameter sweeps."""
	all_data = []
	for param_name, (min_val, max_val) in param_ranges.items():
		num_points = min(10, max_val - min_val + 1)
		values = np.linspace(min_val, max_val, num_points, dtype=int) if num_points > 1 else [min_val]

		for val in tqdm(values, desc=f"Sweep {param_name}"):
			params = base_params.copy()
			params[param_name] = int(val)

			result = run_single_experiment(
				backend, params["qubits"], params["depth"], params["batches"], params["shots"], max_runtime * 1.5
			)

			if result["error"] is None and result["qpu_seconds"] is not None:
				all_data.append(result)
				if result["qpu_seconds"] > max_runtime * 1.5:
					break
	return all_data


def collect_timing_data(
	backend: Any, num_samples: int = 50, include_isolated: bool = True, max_runtime: float = 120.0
) -> list[dict[str, Any]]:
	"""Collect quantum timing data through systematic sampling.

	Args:
		backend: Quantum backend
		num_samples: Number of comprehensive samples
		include_isolated: Include isolated parameter sweeps
		max_runtime: Maximum runtime per job

	Returns:
		List of timing data points
	"""
	max_qubits = len(backend.architecture.qubits)
	param_ranges = {"batches": (1, 25), "qubits": (1, max_qubits), "depth": (1, 100), "shots": (1000, 50000)}
	base_params = {"batches": 1, "qubits": 2, "depth": 5, "shots": 1000}
	all_data = []

	# Isolated parameter sweeps
	if include_isolated:
		logger.info("Running isolated parameter sweeps...")
		all_data.extend(_run_isolated_sweeps(backend, param_ranges, base_params, max_runtime))

	# Latin Hypercube sampling
	logger.info(f"Collecting {num_samples} samples via Latin Hypercube...")
	samples = generate_latin_hypercube_samples(param_ranges, num_samples)

	for params in tqdm(samples, desc="Comprehensive sampling"):
		result = run_single_experiment(
			backend, params["qubits"], params["depth"], params["batches"], params["shots"], max_runtime * 1.5
		)

		if result["error"] is None and result["qpu_seconds"] is not None:
			all_data.append(result)

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
