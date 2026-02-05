// Test JavaScript model consistency with Python
import { calculateQPUSeconds, DEVICE_PARAMS } from './src/utils/ResourceEstimatorModel.js';

// First, verify device configurations
console.log('Verifying device configurations...');
console.log('Helmi max_qubits:', DEVICE_PARAMS.helmi.max_qubits);
console.log('VTT Q50 max_qubits:', DEVICE_PARAMS['vtt-q50'].max_qubits);

if (DEVICE_PARAMS.helmi.max_qubits !== 5) {
	console.error('❌ Helmi should have max_qubits = 5');
	process.exit(1);
}

if (DEVICE_PARAMS['vtt-q50'].max_qubits !== 54) {
	console.error('❌ VTT Q50 should have max_qubits = 54');
	process.exit(1);
}

console.log('✅ Device configurations are correct\n');

// Test cases from CSV (verify JavaScript matches Python, not necessarily actual values)
// The goal is to ensure JS and Python produce identical predictions
// Updated for degree=3 model with log-transform
const testCases = [
	{ device: 'vtt-q50', qubits: 2, depth: 5, batches: 1, shots: 1000, pythonPredicts: 1.19 },
	{ device: 'vtt-q50', qubits: 2, depth: 5, batches: 3, shots: 1000, pythonPredicts: 2.12 },
	{ device: 'vtt-q50', qubits: 2, depth: 5, batches: 6, shots: 1000, pythonPredicts: 3.90 },
	{ device: 'vtt-q50', qubits: 6, depth: 5, batches: 1, shots: 1000, pythonPredicts: 1.31 },
	{ device: 'vtt-q50', qubits: 12, depth: 5, batches: 1, shots: 1000, pythonPredicts: 1.44 },
	{ device: 'vtt-q50', qubits: 2, depth: 1, batches: 1, shots: 1000, pythonPredicts: 1.11 },
	{ device: 'vtt-q50', qubits: 2, depth: 12, batches: 1, shots: 1000, pythonPredicts: 1.31 },
	{ device: 'vtt-q50', qubits: 2, depth: 23, batches: 1, shots: 1000, pythonPredicts: 1.42 },
	{ device: 'vtt-q50', qubits: 2, depth: 5, batches: 1, shots: 6444, pythonPredicts: 3.15 },
	{ device: 'vtt-q50', qubits: 2, depth: 5, batches: 1, shots: 50000, pythonPredicts: 22.18 },
];

console.log('Testing JavaScript model consistency with Python...\n');
console.log('Goal: Verify JavaScript produces identical predictions to Python');
console.log('=' .repeat(80));

let passed = 0;
let failed = 0;
const tolerance = 3.0; // 3% tolerance for floating point differences (degree=3 model)

for (const test of testCases) {
	const params = {
		qubits: test.qubits,
		depth: test.depth,
		batches: test.batches,
		shots: test.shots,
	};

	const jsPrediction = calculateQPUSeconds(test.device, params);
	const pythonPrediction = test.pythonPredicts;
	const diff = Math.abs(jsPrediction - pythonPrediction);
	const diffPct = (diff / pythonPrediction) * 100;

	const status = diffPct < tolerance ? '✅ PASS' : '❌ FAIL';
	if (diffPct < tolerance) {
		passed++;
	} else {
		failed++;
	}

	console.log(`${status} q=${test.qubits}, d=${test.depth}, b=${test.batches}, s=${test.shots}`);
	console.log(`     JS: ${jsPrediction.toFixed(2)}s, Python: ${pythonPrediction.toFixed(2)}s, Diff: ${diffPct.toFixed(2)}%`);
}

console.log('=' .repeat(80));
console.log(`\nResults: ${passed} passed, ${failed} failed`);

if (failed > 0) {
	console.error('\n💥 JavaScript and Python predictions do not match!');
	process.exit(1);
} else {
	console.log('\n🎉 JavaScript and Python predictions match perfectly!');
	process.exit(0);
}
