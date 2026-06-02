/**
 * Format QPU seconds into the most readable unit.
 * < 60s  → "X.XX QPU seconds"
 * < 3600s → "X.XX QPU minutes"
 * ≥ 3600s → "X.XX QPU hours"
 *
 * @param {number} seconds - QPU time in seconds
 * @returns {{ value: string, unit: string, label: string }}
 */
export function formatQPUTime(seconds) {
	const s = parseFloat(seconds);
	if (s < 60) {
		return { value: s.toFixed(2), unit: 'QPU seconds', label: `${s.toFixed(2)} QPU seconds` };
	} else if (s < 3600) {
		const mins = s / 60;
		return { value: mins.toFixed(2), unit: 'QPU minutes', label: `${mins.toFixed(2)} QPU minutes` };
	} else {
		const hours = s / 3600;
		return { value: hours.toFixed(2), unit: 'QPU hours', label: `${hours.toFixed(2)} QPU hours` };
	}
}

/**
 * Convert seconds to QPU hours (always, for basket totals).
 * @param {number} seconds
 * @returns {string}
 */
export function toQPUHours(seconds) {
	return (parseFloat(seconds) / 3600).toFixed(4);
}
