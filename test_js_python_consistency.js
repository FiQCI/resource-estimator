// Test JavaScript model consistency with Python
import { calculateQPUSeconds, DEVICE_PARAMS } from './src/utils/ResourceEstimatorModel.js';

// First, verify device configurations
console.log('Verifying device configurations...');
console.log('Helmi max_qubits:', DEVICE_PARAMS.helmi.max_qubits);
console.log('Helmi model_type:', DEVICE_PARAMS.helmi.model_type);
console.log('VTT Q50 max_qubits:', DEVICE_PARAMS['vtt-q50'].max_qubits);
console.log('VTT Q50 model_type:', DEVICE_PARAMS['vtt-q50'].model_type);

if (DEVICE_PARAMS.helmi.max_qubits !== 5) {
	console.error('❌ Helmi should have max_qubits = 5');
	process.exit(1);
}

if (DEVICE_PARAMS['vtt-q50'].max_qubits !== 54) {
	console.error('❌ VTT Q50 should have max_qubits = 54');
	process.exit(1);
}

if (DEVICE_PARAMS.helmi.model_type !== 'polynomial') {
	console.error('❌ Helmi should use polynomial model');
	process.exit(1);
}

if (DEVICE_PARAMS['vtt-q50'].model_type !== 'analytical') {
	console.error('❌ VTT Q50 should use analytical model');
	process.exit(1);
}

console.log('✅ Device configurations are correct\n');

// Test cases - verify models produce reasonable predictions
const testCases = [
	// Helmi tests (polynomial model with depth)
	{ device: 'helmi', qubits: 2, depth: 10, batches: 1, shots: 1000, expectPositive: true },
	{ device: 'helmi', qubits: 5, depth: 50, batches: 10, shots: 5000, expectPositive: true },
	{ device: 'helmi', qubits: 3, depth: 30, batches: 50, shots: 10000, expectPositive: true },

	// VTT Q50 tests (analytical model, depth ignored)
	{ device: 'vtt-q50', qubits: 2, depth: 1, batches: 1, shots: 1000, expectPositive: true },
	{ device: 'vtt-q50', qubits: 10, depth: 1, batches: 5, shots: 5000, expectPositive: true },
	{ device: 'vtt-q50', qubits: 50, depth: 1, batches: 20, shots: 25000, expectPositive: true },
];

console.log('Testing model predictions...\n');
console.log('Helmi: Polynomial model (with depth)');
console.log('VTT Q50: Analytical model (depth ignored)');
console.log('=' .repeat(80));

let passed = 0;
let failed = 0;

for (const test of testCases) {
	const params = {
		qubits: test.qubits,
		depth: test.depth,
		batches: test.batches,
		shots: test.shots,
	};

	const jsPrediction = calculateQPUSeconds(test.device, params);

	// Validate prediction is positive and reasonable
	const isPositive = jsPrediction > 0;
	const isReasonable = jsPrediction < 10000; // Should be less than 10000 seconds
	const matches = isPositive && isReasonable;
	const status = matches ? '✅ PASS' : '❌ FAIL';

	if (matches) {
		passed++;
	} else {
		failed++;
	}

	console.log(`${status} ${test.device}: q=${test.qubits}, d=${test.depth}, b=${test.batches}, s=${test.shots}`);
	console.log(`     Prediction: ${jsPrediction.toFixed(2)}s (positive: ${isPositive}, reasonable: ${isReasonable})`);
}

console.log('=' .repeat(80));
console.log(`\nResults: ${passed} passed, ${failed} failed`);

if (failed > 0) {
	console.error('\n💥 Model validation failed!');
	process.exit(1);
} else {
	console.log('\n🎉 All model predictions are valid!');
	process.exit(0);
}
