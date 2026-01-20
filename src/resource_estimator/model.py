"""Model training and building module."""

import logging
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

logger = logging.getLogger(__name__)


def prepare_training_data(data: list[dict] | pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
	"""Prepare data for model training.

	Args:
		data: Raw timing data

	Returns:
		Tuple of (features DataFrame, target array)
	"""
	if isinstance(data, list):
		df = pd.DataFrame(data)
	else:
		df = data.copy()

	# Standardize column names
	if "num_circuits" in df.columns and "batches" not in df.columns:
		df["batches"] = df["num_circuits"]
	if "num_qubits" in df.columns and "qubits" not in df.columns:
		df["qubits"] = df["num_qubits"]

	# Create normalized features
	df["kshots"] = df["shots"] / 1000.0

	# Select features
	feature_cols = ["qubits", "depth", "batches", "kshots"]
	X = df[feature_cols]
	y = df["qpu_seconds"].values

	return X, y


def train_polynomial_model(
	X: pd.DataFrame, y: np.ndarray, degree: int = 2, alpha: float = 0.01
) -> tuple[Ridge, PolynomialFeatures, dict]:
	"""Train polynomial ridge regression model.

	Args:
		X: Feature DataFrame
		y: Target array
		degree: Polynomial degree
		alpha: Regularization strength

	Returns:
		Tuple of (trained model, polynomial transformer, metrics dict)
	"""
	logger.info(f"Training polynomial model (degree={degree}, alpha={alpha})")

	# Create polynomial features
	poly = PolynomialFeatures(degree=degree, include_bias=True)
	X_poly = poly.fit_transform(X)

	# Train model
	model = Ridge(alpha=alpha)
	model.fit(X_poly, y)

	# Calculate metrics
	y_pred = model.predict(X_poly)
	metrics = {
		"r2_score": r2_score(y, y_pred),
		"rmse": np.sqrt(mean_squared_error(y, y_pred)),
		"mae": np.mean(np.abs(y - y_pred)),
	}

	logger.info(f"Model R²: {metrics['r2_score']:.4f}, RMSE: {metrics['rmse']:.4f}")

	return model, poly, metrics


def extract_model_coefficients(model: Ridge, poly: PolynomialFeatures, feature_names: list[str]) -> dict[str, float]:
	"""Extract model coefficients as a dictionary.

	Args:
		model: Trained model
		poly: Polynomial transformer
		feature_names: Original feature names

	Returns:
		Dictionary mapping term names to coefficients
	"""
	coefficients = {"intercept": float(model.intercept_)}

	poly_feature_names = poly.get_feature_names_out(feature_names)

	for name, coef in zip(poly_feature_names, model.coef_):
		if abs(coef) > 1e-6 and name != "1":
			coefficients[name] = float(coef)

	return coefficients


def create_prediction_function(model: Ridge, poly: PolynomialFeatures, feature_names: list[str]) -> Callable:
	"""Create a prediction function from trained model.

	Args:
		model: Trained model
		poly: Polynomial transformer
		feature_names: Feature names in order

	Returns:
		Prediction function
	"""

	def predict(qubits: int, depth: int, batches: int, shots: int) -> float:
		"""Predict QPU seconds.

		Args:
			qubits: Number of qubits
			depth: Circuit depth
			batches: Number of circuits
			shots: Number of shots

		Returns:
			Predicted QPU seconds
		"""
		features = np.array([[qubits, depth, batches, shots / 1000.0]])
		features_poly = poly.transform(features)
		return float(model.predict(features_poly)[0])

	return predict


def format_javascript_model(coefficients: dict[str, float], device_name: str, device_id: str) -> str:
	"""Format model as JavaScript code for frontend.

	Args:
		coefficients: Model coefficients
		device_name: Display name
		device_id: Device identifier

	Returns:
		JavaScript code string
	"""
	intercept = coefficients.get("intercept", 0.0)
	terms = []

	# Parse coefficient terms
	for term_name, coef in coefficients.items():
		if term_name == "intercept":
			continue

		parsed = _parse_coefficient_term(term_name, coef)
		if parsed:
			terms.append(parsed)

	# Sort by absolute coefficient value
	terms.sort(key=lambda x: abs(x["coefficient"]), reverse=True)

	# Generate JavaScript
	js_lines = [
		f"\t'{device_id}': {{",
		f"\t\tname: '{device_name}',",
		f"\t\tintercept: {intercept:.6f},",
		"\t\tterms: [",
	]

	for term in terms:
		if term["type"] == "single":
			js_lines.append(
				f"\t\t\t{{type: 'single', variable: '{term['variable']}', coefficient: {term['coefficient']:.6f}}},"
			)
		elif term["type"] == "interaction":
			vars_str = ", ".join([f"'{v}'" for v in term["variables"]])
			js_lines.append(
				f"\t\t\t{{type: 'interaction', variables: [{vars_str}], coefficient: {term['coefficient']:.6f}}},"
			)
		elif term["type"] == "power":
			js_lines.append(
				f"\t\t\t{{type: 'power', variable: '{term['variable']}', coefficient: {term['coefficient']:.6f}, exponent: {term['exponent']}}},"
			)

	js_lines.extend(["\t\t]", "\t}"])

	return "\n".join(js_lines)


def _parse_coefficient_term(term_name: str, coefficient: float) -> dict | None:
	"""Parse a coefficient term name into structured format.

	Args:
		term_name: Term name from sklearn
		coefficient: Coefficient value

	Returns:
		Parsed term dictionary or None
	"""
	term_name = term_name.replace("k_shots", "kshots")

	# Power terms: var^n or var**n
	if "^" in term_name or "**" in term_name:
		parts = term_name.replace("**", "^").split("^")
		if len(parts) == 2:
			return {
				"type": "power",
				"variable": parts[0].strip(),
				"coefficient": coefficient,
				"exponent": int(parts[1]),
			}

	# Interaction terms: var1 var2
	if " " in term_name:
		variables = [v.strip() for v in term_name.split()]
		if len(variables) == 2:
			return {"type": "interaction", "variables": variables, "coefficient": coefficient}

	# Single variable terms
	if term_name in ["qubits", "depth", "batches", "kshots"]:
		return {"type": "single", "variable": term_name, "coefficient": coefficient}

	return None
