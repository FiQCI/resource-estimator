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
	X: pd.DataFrame, y: np.ndarray, degree: int = 3, alpha: float = 0.01
) -> tuple[Ridge, PolynomialFeatures, dict]:
	"""Train polynomial ridge regression model with log-transform.

	Uses log-transform to ensure positive predictions and handle wide-range data.

	Args:
		X: Feature DataFrame
		y: Target array (must be positive)
		degree: Polynomial degree
		alpha: Regularization strength

	Returns:
		Tuple of (trained model, polynomial transformer, metrics dict)

	Note:
		The model trains on log(y) to guarantee positive predictions.
		Use the returned predict function which handles exp() automatically.
	"""
	logger.info(f"Training polynomial model (degree={degree}, alpha={alpha}, log-transform=True)")

	# Create polynomial features WITHOUT bias (Ridge has its own intercept)
	poly = PolynomialFeatures(degree=degree, include_bias=False)
	X_poly = poly.fit_transform(X)

	# Train on log-transformed target for positive predictions
	# Add small epsilon to avoid log(0) edge case
	epsilon = 0.001
	y_log = np.log(y + epsilon)

	model = Ridge(alpha=alpha, fit_intercept=True)
	model.fit(X_poly, y_log)

	# Make predictions (transform back to original space)
	y_log_pred = model.predict(X_poly)
	y_pred = np.exp(y_log_pred) - epsilon

	# Calculate metrics in original space
	metrics = {
		"r2_score": r2_score(y, y_pred),
		"rmse": np.sqrt(mean_squared_error(y, y_pred)),
		"mae": np.mean(np.abs(y - y_pred)),
	}

	# Validation checks
	neg_count = (y_pred < 0).sum()
	if neg_count > 0:
		logger.warning(f"Model produces {neg_count} negative predictions! This should not happen with log-transform.")

	mean_error_pct = np.mean(np.abs((y_pred - y) / y)) * 100
	logger.info(
		f"Model R²: {metrics['r2_score']:.4f}, RMSE: {metrics['rmse']:.4f}, "
		f"MAE: {metrics['mae']:.4f}, Mean Error: {mean_error_pct:.1f}%"
	)

	# Store epsilon in model for predictions
	model.epsilon_ = epsilon

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
		model: Trained model (trained on log-transformed target)
		poly: Polynomial transformer
		feature_names: Feature names in order

	Returns:
		Prediction function that returns positive QPU seconds
	"""
	# Get epsilon from model if available (from log-transform training)
	epsilon = getattr(model, "epsilon_", 0.001)

	def predict(qubits: int, depth: int, batches: int, shots: int) -> float:
		"""Predict QPU seconds.

		Args:
			qubits: Number of qubits
			depth: Circuit depth
			batches: Number of circuits
			shots: Number of shots

		Returns:
			Predicted QPU seconds (always positive due to log-transform)
		"""
		# Create DataFrame with feature names to avoid sklearn warning
		features = pd.DataFrame([[qubits, depth, batches, shots / 1000.0]], columns=feature_names)
		features_poly = poly.transform(features)
		log_pred = model.predict(features_poly)[0]
		# Transform back from log space
		pred = np.exp(log_pred) - epsilon
		# Ensure positive (should always be true with log-transform)
		return max(0.0, float(pred))

	return predict


def format_javascript_model(
	coefficients: dict[str, float], device_name: str, device_id: str, epsilon: float = 0.001
) -> str:
	"""Format model as JavaScript code for frontend.

	Args:
		coefficients: Model coefficients (in log-space)
		device_name: Display name
		device_id: Device identifier
		epsilon: Epsilon value used in log-transform (default: 0.001)

	Returns:
		JavaScript code string

	Note:
		The model is trained on log-transformed targets, so the frontend
		must apply exp(prediction) - epsilon to get actual QPU seconds.
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
		"\t\tlogTransform: true,  // Model trained on log(y), must apply exp()",
		f"\t\tepsilon: {epsilon:.6f},",
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
				f"\t\t\t{{type: 'power', variable: '{term['variable']}', "
				f"coefficient: {term['coefficient']:.6f}, exponent: {term['exponent']}}},"
			)

	js_lines.extend(["\t\t]", "\t}"])

	return "\n".join(js_lines)


def _parse_coefficient_term(term_name: str, coefficient: float) -> dict | None:
	"""Parse a coefficient term name into structured format.

	Supports only degree-2 terms for JavaScript compatibility:
	- Single: qubits, depth, batches, kshots
	- Power: qubits^2, depth^2, etc.
	- Interaction: qubits depth, qubits batches, etc.

	Higher-order terms (degree-3+) are silently skipped.

	Args:
		term_name: Term name from sklearn
		coefficient: Coefficient value

	Returns:
		Parsed term dictionary or None if term cannot be represented
	"""
	term_name = term_name.replace("k_shots", "kshots")

	# Check for higher-order terms (e.g., "qubits^2 depth", "qubits depth batches")
	# These cannot be represented in JavaScript model - skip them
	if " " in term_name and ("^" in term_name or "**" in term_name):
		# This is a mixed term like "qubits^2 depth" - skip
		return None

	# Power terms: var^n or var**n (only if no spaces, i.e., pure power)
	if ("^" in term_name or "**" in term_name) and " " not in term_name:
		parts = term_name.replace("**", "^").split("^")
		if len(parts) == 2:
			try:
				exponent = int(parts[1].strip())
				return {"type": "power", "variable": parts[0].strip(), "coefficient": coefficient, "exponent": exponent}
			except ValueError:
				# Can't parse exponent - skip this term
				return None

	# Interaction terms: var1 var2 (exactly 2 variables, no powers)
	if " " in term_name and "^" not in term_name and "**" not in term_name:
		variables = [v.strip() for v in term_name.split()]
		if len(variables) == 2:
			return {"type": "interaction", "variables": variables, "coefficient": coefficient}
		# More than 2 variables - skip (e.g., "qubits depth batches")
		return None

	# Single variable terms
	if term_name in ["qubits", "depth", "batches", "kshots"]:
		return {"type": "single", "variable": term_name, "coefficient": coefficient}

	return None
