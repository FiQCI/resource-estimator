"""JavaScript model export utilities."""

import json
import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


def format_polynomial_model(
	coefficients: Dict[str, float], intercept: float, device_name: str, max_qubits: int
) -> Dict[str, Any]:
	"""Format polynomial model for JavaScript export.

	Args:
		coefficients: Dictionary mapping term names to coefficients
		intercept: Model intercept
		device_name: Name of the device
		max_qubits: Maximum number of qubits

	Returns:
		JavaScript configuration dictionary
	"""
	# Convert coefficient terms to JavaScript format
	terms = []
	for term, coef in coefficients.items():
		if term == "intercept":
			continue  # Skip intercept, handled separately

		js_term = _format_term_for_javascript(term, coef)
		if js_term:
			terms.append(js_term)

	config = {
		"name": device_name,
		"max_qubits": max_qubits,
		"model_type": "polynomial",
		"intercept": round(intercept, 6),
		"terms": terms,
	}

	logger.info(f"Formatted polynomial model with {len(terms)} terms")
	return config


def format_analytical_model(params: Dict[str, float], device_name: str, max_qubits: int) -> Dict[str, Any]:
	"""Format analytical model for JavaScript export.

	Args:
		params: Model parameters (T_init, efficiency_base, throughput_coef, batch_cap)
		device_name: Name of the device
		max_qubits: Maximum number of qubits

	Returns:
		JavaScript configuration dictionary
	"""
	config = {
		"name": device_name,
		"max_qubits": max_qubits,
		"model_type": "analytical",
		"T_init": params["T_init"],
		"efficiency_base": params["efficiency_base"],
		"throughput_coef": params["throughput_coef"],
		"batch_cap": params["batch_cap"],
	}

	logger.info(f"Formatted analytical model for {device_name}")
	return config


def save_model_config(config: Dict[str, Any], output_path: Path) -> None:
	"""Save model configuration to JSON file.

	Args:
		config: Model configuration dictionary
		output_path: Path to save JSON file
	"""
	with open(output_path, "w") as f:
		json.dump(config, f, indent=2)
	logger.info(f"Saved config: {output_path}")


def update_javascript_model_file(js_file_path: Path, device_key: str, config: Dict[str, Any]) -> None:
	"""Update ResourceEstimatorModel.js with new device configuration.

	Args:
		js_file_path: Path to ResourceEstimatorModel.js
		device_key: Device key (e.g., 'helmi', 'vtt-q50')
		config: Model configuration to insert
	"""
	logger.info(f"Updating {js_file_path} with {device_key} config")

	if not js_file_path.exists():
		raise FileNotFoundError(f"JavaScript file not found: {js_file_path}")

	# Read the file
	content = js_file_path.read_text()

	# Format the new device config as JavaScript
	config_str = _format_config_as_javascript(config, indent=2)

	# Find the DEVICE_PARAMS object and replace the device entry
	# Look for the device key pattern
	import re

	# Pattern to match the device entry including any trailing comma
	pattern = rf'(\s*)("{device_key}":\s*{{[^}}]*}}),?'

	# Check if device exists
	match = re.search(pattern, content, re.DOTALL)

	if match:
		# Replace existing device config
		indent = match.group(1)
		# Add trailing comma if next line isn't closing brace
		next_char_pos = match.end()
		add_comma = next_char_pos < len(content) and content[next_char_pos : next_char_pos + 10].strip()[0] != "}"

		replacement = f'{indent}"{device_key}": {config_str}' + ("," if add_comma else "")
		new_content = content[: match.start()] + replacement + content[match.end() :]
		logger.info(f"Updated existing {device_key} configuration")
	else:
		# Add new device config before closing brace of DEVICE_PARAMS
		pattern = r"(const\s+DEVICE_PARAMS\s*=\s*{[^}]*)(})"
		match = re.search(pattern, content, re.DOTALL)

		if not match:
			raise ValueError("Could not find DEVICE_PARAMS object in JavaScript file")

		# Add with proper indentation and comma
		insertion = f',\n\t"{device_key}": {config_str}\n'
		new_content = content[: match.end(1)] + insertion + content[match.start(2) :]
		logger.info(f"Added new {device_key} configuration")

	# Create backup
	backup_path = js_file_path.with_suffix(".js.bak")
	backup_path.write_text(content)
	logger.info(f"Created backup: {backup_path}")

	# Write updated content
	js_file_path.write_text(new_content)
	logger.info(f"✓ Updated {js_file_path}")


def _format_term_for_javascript(term: str, coef: float) -> Dict[str, Any] | None:
	"""Format a polynomial term for JavaScript.

	Args:
		term: Term name (e.g., 'batches', 'kshots^2', 'batches kshots')
		coef: Coefficient value

	Returns:
		Dictionary with 'coefficient' and 'variables' keys, or None if term should be skipped
	"""
	# Parse the term into variables and their powers
	parts = term.split()

	if len(parts) == 0:
		return None

	variables = []
	for part in parts:
		if "^" in part:
			var, power = part.split("^")
			power = int(power)
		else:
			var = part
			power = 1

		variables.append({"name": var, "power": power})

	return {"coefficient": round(coef, 9), "variables": variables}


def _format_config_as_javascript(config: Dict[str, Any], indent: int = 0) -> str:
	"""Format configuration dictionary as JavaScript object literal.

	Args:
		config: Configuration dictionary
		indent: Number of tabs for indentation

	Returns:
		JavaScript object literal string
	"""
	lines = ["{"]
	indent_str = "\t" * (indent + 1)

	for key, value in config.items():
		if key == "terms" and isinstance(value, list):
			# Format terms array
			lines.append(f"{indent_str}{key}: [")
			for term in value:
				term_str = _format_term_as_javascript(term, indent + 2)
				lines.append(term_str + ",")
			lines.append(f"{indent_str}]" + ("," if key != list(config.keys())[-1] else ""))
		elif isinstance(value, str):
			lines.append(f'{indent_str}{key}: "{value}"' + ("," if key != list(config.keys())[-1] else ""))
		elif isinstance(value, (int, float)):
			lines.append(f"{indent_str}{key}: {value}" + ("," if key != list(config.keys())[-1] else ""))
		else:
			# Fallback to JSON
			json_str = json.dumps(value)
			lines.append(f"{indent_str}{key}: {json_str}" + ("," if key != list(config.keys())[-1] else ""))

	lines.append("\t" * indent + "}")
	return "\n".join(lines)


def _format_term_as_javascript(term: Dict[str, Any], indent: int) -> str:
	"""Format a term dictionary as JavaScript object literal.

	Args:
		term: Term dictionary with 'coefficient' and 'variables'
		indent: Number of tabs for indentation

	Returns:
		JavaScript object literal string
	"""
	indent_str = "\t" * indent
	lines = [f"{indent_str}{{"]

	lines.append(f"{indent_str}\tcoefficient: {term['coefficient']},")
	lines.append(f"{indent_str}\tvariables: [")

	for var in term["variables"]:
		lines.append(f'{indent_str}\t\t{{ name: "{var["name"]}", power: {var["power"]} }},')

	lines.append(f"{indent_str}\t]")
	lines.append(f"{indent_str}}}")

	return "\n".join(lines)
