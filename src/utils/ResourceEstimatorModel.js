// ResourceEstimatorModel.js
// Quantum resource estimation model based on statistical analysis

/**
 * Device model configuration
 * Contains parameters and calculation model for different quantum devices
 */
const DEVICE_PARAMS = {
	helmi: {
		name: 'Helmi',
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
		name: 'VTT Q50',
		logTransform: true,  // Model trained on log(y), must apply exp()
		epsilon: 0.001000,
		intercept: -0.498844,
		terms: [
			{type: 'single', variable: 'batches', coefficient: 0.311092},
			{type: 'single', variable: 'kshots', coefficient: 0.218129},
			{type: 'single', variable: 'depth', coefficient: 0.031569},
			{type: 'single', variable: 'qubits', coefficient: 0.028589},
			{type: 'power', variable: 'batches', coefficient: -0.015949, exponent: 2},
			{type: 'power', variable: 'kshots', coefficient: -0.006516, exponent: 2},
			{type: 'interaction', variables: ['batches', 'kshots'], coefficient: 0.002829},
			{type: 'interaction', variables: ['qubits', 'batches'], coefficient: -0.001355},
			{type: 'power', variable: 'qubits', coefficient: -0.000950, exponent: 2},
			{type: 'interaction', variables: ['qubits', 'kshots'], coefficient: 0.000842},
			{type: 'power', variable: 'depth', coefficient: -0.000753, exponent: 2},
			{type: 'power', variable: 'batches', coefficient: 0.000318, exponent: 3},
			{type: 'interaction', variables: ['qubits', 'depth'], coefficient: -0.000267},
			{type: 'interaction', variables: ['depth', 'batches'], coefficient: 0.000117},
			{type: 'interaction', variables: ['depth', 'kshots'], coefficient: 0.000084},
			{type: 'power', variable: 'kshots', coefficient: 0.000067, exponent: 3},
			{type: 'power', variable: 'qubits', coefficient: 0.000011, exponent: 3},
			{type: 'power', variable: 'depth', coefficient: 0.000004, exponent: 3},
		]
	}
};

/**
 * Calculate the contribution of a single term in the model.
 *
 * @param {Object} term - Term configuration object
 * @param {Object} params - Normalized input parameters
 * @returns {number} Term contribution to the result
 */
function calculateTerm(term, params) {
	const termType = term.type;

	if (termType === 'single') {
		return term.coefficient * params[term.variable];

	} else if (termType === 'interaction') {
		const [var1, var2] = term.variables;
		return term.coefficient * params[var1] * params[var2];

	} else if (termType === 'power') {
		return term.coefficient * Math.pow(params[term.variable], term.exponent);
	}

	return 0.0;
}

/**
 * Calculate QPU seconds based on the device model and input parameters.
 *
 * @param {string} device - Device identifier ('helmi' or 'vtt-q50')
 * @param {Object} params - Dictionary with keys 'batches', 'depth', 'shots', and 'qubits'
 * @returns {number} Estimated QPU seconds (always positive)
 */
function calculateQPUSeconds(device, params) {
	if (!DEVICE_PARAMS[device]) {
		console.error(`Unknown device: ${device}`);
		return 0;
	}

	// Get device configuration
	const deviceConfig = DEVICE_PARAMS[device];

	// Create a normalized parameters dict with kshots
	const normParams = {
		batches: parseInt(params.batches, 10) || 1,
		depth: parseInt(params.depth, 10) || 10,
		shots: parseInt(params.shots, 10) || 1000,
		qubits: parseInt(params.qubits, 10) || 2,
		kshots: (parseInt(params.shots, 10) || 1000) / 1000 // Convert shots to kshots
	};

	// Start with the intercept
	let result = deviceConfig.intercept;

	// Add contribution from each term
	for (const term of deviceConfig.terms) {
		const termValue = calculateTerm(term, normParams);
		result += termValue;
	}

	// If model uses log-transform, apply exp() to get back to original space
	if (deviceConfig.logTransform) {
		const epsilon = deviceConfig.epsilon || 0.001;
		result = Math.exp(result) - epsilon;
	}

	// Ensure positive result
	result = Math.max(0.0, result);

	// Return rounded to 2 decimal places
	return parseFloat(result.toFixed(2));
}

// Export model functions and configurations
export { DEVICE_PARAMS, calculateQPUSeconds };
