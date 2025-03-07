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
		intercept: 3.687455,
		terms: [
			{type: 'interaction', variables: ['batches', 'kshots'], coefficient: 0.411647},
			{type: 'single', variable: 'batches', coefficient: -0.039480},
			{type: 'single', variable: 'kshots', coefficient: -0.023972},
			{type: 'single', variable: 'depth', coefficient: -0.015355},
			{type: 'single', variable: 'qubits', coefficient: -0.012479},
			{type: 'interaction', variables: ['qubits', 'batches'], coefficient: 0.006657},
			{type: 'interaction', variables: ['depth', 'batches'], coefficient: -0.000807},
			{type: 'power', variable: 'batches', coefficient: 0.000738, exponent: 2},
			{type: 'interaction', variables: ['qubits', 'kshots'], coefficient: 0.000660},
			{type: 'power', variable: 'kshots', coefficient: 0.000327, exponent: 2},
			{type: 'interaction', variables: ['depth', 'kshots'], coefficient: 0.000319},
			{type: 'power', variable: 'qubits', coefficient: 0.000241, exponent: 2},
			{type: 'power', variable: 'depth', coefficient: 0.000157, exponent: 2},
			{type: 'interaction', variables: ['qubits', 'depth'], coefficient: -0.000042}
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
 * Safely calculate the depth term contribution, preventing negative scaling issues.
 * 
 * @param {Object} deviceConfig - Device configuration
 * @param {Object} params - Normalized input parameters
 * @returns {number} Depth term contribution (with safety checks)
 */
function calculateDepthTerm(deviceConfig, params) {
	// Find the depth term
	const depthTerm = deviceConfig.terms.find(
		term => term.type === 'single' && term.variable === 'depth'
	);
	
	if (!depthTerm) {
		return 0.0;
	}
	
	// If depth coefficient is negative, we need to handle it specially
	if (depthTerm.coefficient < 0) {
		// Set a reasonable maximum effect the depth can have
		// This prevents very large depths from unrealistically reducing the runtime
		const maxDepthEffect = Math.abs(depthTerm.coefficient) * 100; // Assume 100 is a reasonable depth cap
		
		// Calculate actual depth effect
		const actualDepthEffect = depthTerm.coefficient * params.depth;
		
		// Limit the negative effect to the maximum
		return Math.max(actualDepthEffect, -maxDepthEffect);
	} else {
		// For positive coefficients, calculate normally
		return depthTerm.coefficient * params.depth;
	}
}

/**
 * Calculate QPU seconds based on the device model and input parameters.
 * 
 * @param {string} device - Device identifier ('helmi' or 'vtt-q50')
 * @param {Object} params - Dictionary with keys 'batches', 'depth', 'shots', and 'qubits'
 * @returns {number} Estimated QPU seconds
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
	
	// Handle depth parameter to avoid negative scaling
	const depthTerm = calculateDepthTerm(deviceConfig, normParams);
	
	// Start with the intercept
	let result = deviceConfig.intercept;
	
	// Add contribution from each term
	for (const term of deviceConfig.terms) {
		// Skip the depth term as we handle it separately
		if (term.type === 'single' && term.variable === 'depth') {
			continue;
		}
		
		const termValue = calculateTerm(term, normParams);
		result += termValue;
	}
	
	// Add the safely calculated depth term
	result += depthTerm;
	
	
	// Return rounded to 2 decimal places
	return parseFloat(result.toFixed(2));
}

// Export model functions and configurations
export { DEVICE_PARAMS, calculateQPUSeconds };