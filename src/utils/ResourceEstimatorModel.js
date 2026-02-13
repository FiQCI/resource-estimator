// ResourceEstimatorModel.js
// Quantum resource estimation model based on statistical analysis

/**
 * Device model configuration
 * Contains parameters and calculation model for different quantum devices
 *
 * Helmi: Degree-2 polynomial (original model)
 * VTT Q50: Degree-4 polynomial WITHOUT depth parameter, R² = 0.9715
 */
const DEVICE_PARAMS = {
	helmi: {
		name: 'Helmi',
		max_qubits: 5,
		model_type: 'polynomial',
		intercept: 2.361885,
		terms: [
			{type: 'interaction', variables: ['batches', 'kshots'], coefficient: 0.432804},
			{type: 'single', variable: 'qubits', coefficient: 0.178790},
			{type: 'single', variable: 'batches', coefficient: -0.027707},
			{type: 'single', variable: 'kshots', coefficient: -0.025819},
			{type: 'interaction', variables: ['qubits', 'batches'], coefficient: 0.009231},
			{type: 'interaction', variables: ['qubits', 'kshots'], coefficient: 0.007381},
			{type: 'power', variable: 'qubits', coefficient: -0.006612, exponent: 2},
			{type: 'single', variable: 'depth', coefficient: 0.003655},
			{type: 'interaction', variables: ['qubits', 'depth'], coefficient: -0.001835},
			{type: 'power', variable: 'kshots', coefficient: 0.000113, exponent: 2},
			{type: 'interaction', variables: ['depth', 'batches'], coefficient: -0.000107},
			{type: 'power', variable: 'batches', coefficient: 0.000104, exponent: 2},
			{type: 'interaction', variables: ['depth', 'kshots'], coefficient: 0.000014}
		]
	},
	'vtt-q50': {
		name: "VTT Q50",
		max_qubits: 54,
		model_type: "analytical",
		T_init: 0.8837048192083147,
		efficiency_base: 0.9859309839575585,
		throughput_coef: 0.0006252263363799867,
		batch_cap: 19.29350870051621
	}
};

/**
 * Calculate a single polynomial term value.
 *
 * @param {string} termName - Term name (e.g., 'qubits^2', 'batches kshots')
 * @param {Object} values - Dictionary with 'qubits', 'batches', 'kshots'
 * @returns {number} Term value
 */
function calculateTerm(termName, values) {
	let result = 1.0;

	// Parse term (e.g., 'qubits^2 batches', 'kshots^3')
	const parts = termName.split(' ');

	for (const part of parts) {
		if (part.includes('^')) {
			// Power term: variable^exponent
			const [variable, exponent] = part.split('^');
			const power = parseInt(exponent, 10);
			result *= Math.pow(values[variable], power);
		} else {
			// Simple variable
			result *= values[part];
		}
	}

	return result;
}

/**
 * Calculate QPU seconds using polynomial model.
 *
 * @param {string} device - Device identifier ('helmi' or 'vtt-q50')
 * @param {Object} params - Dictionary with keys 'batches', 'shots', and 'qubits'
 * @returns {number} Estimated QPU seconds (always positive)
 */
function calculateQPUSeconds(device, params) {
	if (!DEVICE_PARAMS[device]) {
		console.error(`Unknown device: ${device}`);
		return 0;
	}

	// Get device configuration
	const deviceConfig = DEVICE_PARAMS[device];

	// Parse parameters
	const batches = parseInt(params.batches, 10) || 1;
	const shots = parseInt(params.shots, 10) || 1000;
	const qubits = parseInt(params.qubits, 10) || 2;
	const depth = parseInt(params.depth, 10) || 1;

	// Create feature values
	const kshots = shots / 1000.0;
	const featureValues = {
		qubits: qubits,
		batches: batches,
		kshots: kshots,
		depth: depth
	};

	// Calculate prediction: intercept + sum of (coefficient * term_value)
	let prediction = deviceConfig.intercept;

	// Handle analytical model (VTT Q50)
	if (deviceConfig.model_type === 'analytical') {
		// Analytical formula: T = T_init + efficiency(batches) × batches × shots × throughput
		const efficiency = Math.pow(
			deviceConfig.efficiency_base,
			Math.min(batches, deviceConfig.batch_cap)
		);

		prediction = deviceConfig.T_init +
			efficiency * batches * shots * deviceConfig.throughput_coef;
	} else {
		// Polynomial model (Helmi)
		for (const term of deviceConfig.terms) {
			let termValue;

			// Handle old format (Helmi) with type/variable/variables
			if (term.type) {
				if (term.type === 'single') {
					termValue = featureValues[term.variable];
				} else if (term.type === 'power') {
					termValue = Math.pow(featureValues[term.variable], term.exponent);
				} else if (term.type === 'interaction') {
					termValue = 1.0;
					for (const variable of term.variables) {
						termValue *= featureValues[variable];
					}
				}
			} else {
				// Handle new format with name
				termValue = calculateTerm(term.name, featureValues);
			}

			prediction += term.coefficient * termValue;
		}
	}

	// Apply log-transform if needed
	if (deviceConfig.use_log_transform) {
		// Model was trained on log(y), so we need to transform back
		prediction = Math.exp(prediction) - deviceConfig.epsilon;
	}

	// Ensure positive result
	prediction = Math.max(0.0, prediction);

	// Return rounded to 2 decimal places
	return parseFloat(prediction.toFixed(2));
}

// Export model functions and configurations
export { DEVICE_PARAMS, calculateQPUSeconds };
