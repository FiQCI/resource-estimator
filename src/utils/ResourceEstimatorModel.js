// ResourceEstimatorModel.js
// Quantum resource estimation model based on statistical analysis

/**
 * Device model configuration
 * Contains parameters and calculation model for different quantum devices
 */
const DEVICE_PARAMS = {
	helmi: {
		name: 'Helmi',
		intercept: 2.423064,
		terms: [
			{type: 'interaction', variables: ['batches', 'kshots'], coefficient: 0.432691},
			{type: 'single', variable: 'qubits', coefficient: 0.111159},
			{type: 'single', variable: 'batches', coefficient: -0.039790},
			{type: 'single', variable: 'kshots', coefficient: -0.018588},
			{type: 'interaction', variables: ['batches', 'qubits'], coefficient: 0.015776},
			{type: 'power', variable: 'qubits', coefficient: 0.005476, exponent: 2}
		]
	},
	'vtt-q50': {
		name: 'VTT Q50',
		intercept: 3.591248,
		terms: [
			{type: 'interaction', variables: ['batches', 'kshots'], coefficient: 0.411585},
			{type: 'single', variable: 'kshots', coefficient: -0.024597},
			{type: 'single', variable: 'batches', coefficient: -0.020134},
			{type: 'single', variable: 'depth', coefficient: 0.013455},
			{type: 'interaction', variables: ['batches', 'qubits'], coefficient: 0.005452}
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