import React, { useState, useEffect } from 'react';
import DeviceSelector from './DeviceSelector';
import HistoryPanel from './HistoryPanel';
import BasketPanel from './BasketPanel';
import JobHistoryManager from '../utils/JobHistoryManager';
import { DEVICE_PARAMS, calculateQPUSeconds } from '../utils/ResourceEstimatorModel';

const fontFamily = '-apple-system,BlinkMacSystemFont,"Roboto","Segoe UI","Helvetica Neue","Lucida Grande",Arial,sans-serif';

const ParameterInput = ({ label, value, onChange, hint, error, disabled = false }) => {
	const inputContainerStyle = {
		flex: '1 0 45%',
		marginBottom: '0.75rem',
		textAlign: 'left',
		fontFamily: fontFamily,
		padding: '0 0.4rem'
	};

	const labelStyle = {
		display: 'block',
		fontSize: '1.1rem',
		fontWeight: '500',
		color: disabled ? '#9CA3AF' : '#374151',
		marginBottom: '0.35rem',
		fontFamily: fontFamily
	};

	const hintStyle = {
		display: 'block',
		fontSize: '0.85rem',
		color: '#6B7280',
		marginTop: '0.25rem',
		fontFamily: fontFamily
	};

	const errorStyle = {
		display: 'block',
		fontSize: '0.85rem',
		color: '#DC2626',
		marginTop: '0.25rem',
		fontFamily: fontFamily,
		fontWeight: '500'
	};

	const inputStyle = {
		display: 'block',
		width: '100%',
		padding: '0.5rem 0.7rem',
		backgroundColor: disabled ? '#F3F4F6' : 'white',
		border: error ? '2px solid #DC2626' : '1px solid #D1D5DB',
		borderRadius: '0.35rem',
		boxSizing: 'border-box',
		color: disabled ? '#9CA3AF' : '#000',
		fontFamily: fontFamily,
		fontSize: '1.1rem',
		cursor: disabled ? 'not-allowed' : 'text',
		// Hide up/down arrows for number inputs
		WebkitAppearance: 'none',
		MozAppearance: 'textfield'
	};

	// Handle the input change to allow empty values
	const handleInputChange = (e) => {
		const inputValue = e.target.value;
		if (inputValue === '') {
			// Allow empty strings
			onChange('');
		} else {
			const numValue = parseInt(inputValue, 10);
			if (!isNaN(numValue)) {
				onChange(numValue);
			}
		}
	};

	return (
		<div style={inputContainerStyle}>
			<label style={labelStyle}>{label}</label>
			<input
				type="text"
				value={value}
				onChange={handleInputChange}
				style={inputStyle}
				placeholder="Enter value"
				disabled={disabled}
			/>
			{error ? <span style={errorStyle}>{error}</span> : hint && <span style={hintStyle}>{hint}</span>}
		</div>
	);
};

const ResourceEstimator = () => {
	const [selectedDevice, setSelectedDevice] = useState('helmi');
	const [formData, setFormData] = useState({
		batches: 1,
		shots: 1000,
		qubits: 5,
		depth: 1
	});

	const [estimatedQPU, setEstimatedQPU] = useState(null);
	const [history, setHistory] = useState([]);
	const [basket, setBasket] = useState([]);
	const [validationErrors, setValidationErrors] = useState({});

	// Load history and basket from localStorage on component mount
	useEffect(() => {
		setHistory(JobHistoryManager.loadHistory());
		setBasket(JobHistoryManager.loadBasket());
	}, []);

	// Re-validate qubits when device changes
	useEffect(() => {
		if (formData.qubits) {
			validateField('qubits', formData.qubits);
		}
	}, [selectedDevice]);

	const validateField = (field, value) => {
		const numValue = parseInt(value, 10);
		const newErrors = { ...validationErrors };

		if (field === 'qubits') {
			const deviceConfig = DEVICE_PARAMS[selectedDevice];
			if (value === '' || value === undefined || value === null) {
				newErrors.qubits = 'Qubits is required';
			} else if (numValue < 1) {
				newErrors.qubits = 'Must be at least 1 qubit';
			} else if (deviceConfig.max_qubits && numValue > deviceConfig.max_qubits) {
				newErrors.qubits = `${deviceConfig.name} supports max ${deviceConfig.max_qubits} qubits`;
			} else {
				delete newErrors.qubits;
			}
		} else {
			// Validate other fields for positive values
			if (value === '' || value === undefined || value === null) {
				newErrors[field] = 'This field is required';
			} else if (numValue < 1) {
				newErrors[field] = 'Must be a positive integer';
			} else {
				delete newErrors[field];
			}
		}

		setValidationErrors(newErrors);
	};

	const handleInputChange = (field, value) => {
		console.log(`Updating ${field} to ${value}`);
		setFormData(prevState => ({
			...prevState,
			[field]: value
		}));
		validateField(field, value);
	};

	const calculateQPU = () => {
		// Validate all fields first
		const requiredFields = ['batches', 'shots', 'qubits'];
		if (selectedDevice === 'helmi') {
			requiredFields.push('depth');
		}
		requiredFields.forEach(field => validateField(field, formData[field]));

		// Check if there are any validation errors
		const hasErrors = Object.keys(validationErrors).length > 0 ||
			requiredFields.some(field => formData[field] === '' || formData[field] === undefined || formData[field] === null);

		if (hasErrors) {
			return;
		}

		// Convert any string values to numbers
		const numericFormData = {
			batches: parseInt(formData.batches, 10),
			shots: parseInt(formData.shots, 10),
			qubits: parseInt(formData.qubits, 10)
		}

		// Add depth for Helmi
		if (selectedDevice === 'helmi') {
			numericFormData.depth = parseInt(formData.depth, 10);
		}

		// Calculate QPU seconds using our model
		const qpuSeconds = calculateQPUSeconds(selectedDevice, numericFormData);
		setEstimatedQPU(qpuSeconds);

		// Create estimation object
		const estimation = {
			device: selectedDevice,
			deviceName: DEVICE_PARAMS[selectedDevice].name,
			params: { ...numericFormData },
			qpuSeconds: qpuSeconds,
		};

		// Add to history and update state
		const updatedHistory = JobHistoryManager.addToHistory(estimation);
		setHistory(updatedHistory);
	};

	// Add an estimation to the basket
	const handleAddToBasket = (estimation) => {
		const updatedBasket = JobHistoryManager.addToBasket(estimation);
		setBasket(updatedBasket);
	};

	// Remove an estimation from the basket
	const handleRemoveFromBasket = (index) => {
		const updatedBasket = JobHistoryManager.removeFromBasket(index);
		setBasket(updatedBasket);
	};

	// Clear the basket
	const handleClearBasket = () => {
		const emptyBasket = JobHistoryManager.clearBasket();
		setBasket(emptyBasket);
	};

	// Clear the history
	const handleClearHistory = () => {
		if (confirm('Are you sure you want to clear all estimation history?')) {
			const emptyHistory = JobHistoryManager.clearHistory();
			setHistory(emptyHistory);
		}
	};

	// Styles for the main container with the new layout
	const containerStyle = {
		maxWidth: '100%',
		margin: '0 auto',
		display: 'flex',
		fontFamily: fontFamily
	};

	// Left column styles for the estimator form
	const leftColumnStyle = {
		flex: '2',
		marginRight: '1.05rem',
	};

	// Right column styles for history and basket
	const rightColumnStyle = {
		flex: '1',
		minWidth: '300px',
	};

	const formContainerStyle = {
		backgroundColor: 'white',
		padding: '1.05rem',
		borderRadius: '0.7rem',
		boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)',
		width: '100%',
		boxSizing: 'border-box',
		marginBottom: '1.05rem',
		fontFamily: fontFamily
	};

	const buttonStyle = {
		backgroundColor: '#2e52a5',
		color: 'white',
		fontWeight: '500',
		padding: '0.55rem 1.4rem',
		borderRadius: '0.35rem',
		border: 'none',
		cursor: 'pointer',
		transition: 'background-color 0.2s',
		fontFamily: fontFamily,
		fontSize: '1.1rem'
	};

	const formInputsContainerStyle = {
		display: 'flex',
		flexWrap: 'wrap',
		margin: '0 -0.4rem'
	};

	const resultContainerStyle = {
		backgroundColor: 'white',
		padding: '1.05rem',
		borderRadius: '0.7rem',
		boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)',
		textAlign: 'center',
		width: '100%',
		boxSizing: 'border-box',
		fontFamily: fontFamily,
		marginBottom: '1.05rem'
	};

	const resultValueStyle = {
		backgroundColor: '#f0f7ff',
		padding: '0.7rem 1.4rem',
		borderRadius: '0.7rem',
		border: '1px solid #d0e3ff',
		display: 'inline-block',
		fontFamily: fontFamily
	};

	return (
		<div style={containerStyle}>
			{/* Left column with estimator form */}
			<div style={leftColumnStyle}>
				<DeviceSelector
					selectedDevice={selectedDevice}
					onDeviceSelect={setSelectedDevice}
				/>

				<div style={formContainerStyle}>
					<h2 style={{
						textAlign: 'center',
						fontSize: '1.5rem',
						fontWeight: '500',
						color: '#333333',
						margin: '0 0 1rem 0',
						fontFamily: fontFamily
					}}>
						Input Parameters
					</h2>

					<div style={formInputsContainerStyle}>
						<ParameterInput
							label="Circuits in Batch"
							value={formData.batches}
							onChange={(value) => handleInputChange('batches', value)}
							error={validationErrors.batches}
						/>

						<ParameterInput
							label="Shots per Circuit"
							value={formData.shots}
							onChange={(value) => handleInputChange('shots', value)}
							error={validationErrors.shots}
						/>

						<ParameterInput
							label="Number of Qubits"
							value={formData.qubits}
							onChange={(value) => handleInputChange('qubits', value)}
							hint={`Max: ${DEVICE_PARAMS[selectedDevice].max_qubits} qubits`}
							error={validationErrors.qubits}
						/>

						<ParameterInput
							label="Circuit Depth"
							value={formData.depth}
							onChange={(value) => handleInputChange('depth', value)}
							hint={selectedDevice === 'helmi' ? 'Number of layers in circuit' : 'Not used for VTT Q50'}
							error={validationErrors.depth}
							disabled={selectedDevice === 'vtt-q50'}
						/>
					</div>

					<div style={{marginTop: '1rem', textAlign: 'center'}}>
						<button
							onClick={calculateQPU}
							style={buttonStyle}
						>
							Calculate QPU Estimate
						</button>
					</div>
				</div>

				{estimatedQPU !== null && (
					<>
						<div style={resultContainerStyle}>
							<h2 style={{
								fontSize: '1.5rem',
								fontWeight: '500',
								color: '#333333',
								margin: '0 0 0.7rem 0',
								fontFamily: fontFamily
							}}>
								Estimate
							</h2>
							<div style={resultValueStyle}>
								<div style={{
									fontSize: '1.05rem',
									color: '#333333',
									marginBottom: '0.35rem',
									fontFamily: fontFamily
								}}>
									Estimated QPU Seconds
								</div>
								<div style={{
									fontSize: '2.1rem',
									fontWeight: 'bold',
									color: '#333333',
									fontFamily: fontFamily
								}}>
									{estimatedQPU}
								</div>
							</div>
						</div>

						<div style={{textAlign: 'center', marginBottom: '1.05rem'}}>
							<button
								onClick={() => handleAddToBasket({
									device: selectedDevice,
									deviceName: DEVICE_PARAMS[selectedDevice].name,
									params: { ...formData },
									qpuSeconds: estimatedQPU,
									id: Date.now(),
									timestamp: new Date().toISOString()
								})}
								style={{
									backgroundColor: '#10b981', // Green color
									color: 'white',
									fontWeight: '500',
									padding: '0.55rem 1.4rem',
									borderRadius: '0.35rem',
									border: 'none',
									cursor: 'pointer',
									transition: 'background-color 0.2s',
									fontFamily: fontFamily,
									fontSize: '1.1rem'
								}}
							>
								Add to Basket
							</button>
						</div>
					</>
				)}
			</div>

			{/* */}
			<div style={rightColumnStyle}>
				<BasketPanel
					basket={basket}
					onRemoveFromBasket={handleRemoveFromBasket}
					onClearBasket={handleClearBasket}
				/>

				<HistoryPanel
					history={history}
					onAddToBasket={handleAddToBasket}
					onClearHistory={handleClearHistory}
				/>
			</div>
		</div>
	);
};

export default ResourceEstimator;
