import React from 'react';

const fontFamily = '-apple-system,BlinkMacSystemFont,"Roboto","Segoe UI","Helvetica Neue","Lucida Grande",Arial,sans-serif';

const BasketPanel = ({ basket, onRemoveFromBasket, onClearBasket }) => {
	// Calculate total QPU seconds
	const totalQPUSeconds = basket.reduce((total, item) => total + parseFloat(item.qpuSeconds), 0).toFixed(2);

	// Basket panel container styles
	const containerStyle = {
		backgroundColor: 'white',
		padding: '1.05rem',
		borderRadius: '0.7rem',
		boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)',
		marginBottom: '1.05rem',
		maxHeight: '21rem',
		overflowY: 'auto',
		fontFamily: fontFamily
	};

	// Basket panel title styles
	const titleStyle = {
		textAlign: 'center',
		fontSize: '1.5rem',
		fontWeight: '500',
		color: '#333333',
		margin: '0 0 1.05rem 0',
		fontFamily: fontFamily
	};

	// Basket item styles
	const itemStyle = {
		borderBottom: '1px solid #e5e7eb',
		padding: '0.7rem 0',
		marginBottom: '0.7rem',
	};

	// Remove button styles
	const removeButtonStyle = {
		backgroundColor: '#ef4444',
		color: 'white',
		fontWeight: '500',
		padding: '0.35rem 0.7rem',
		borderRadius: '0.35rem',
		border: 'none',
		cursor: 'pointer',
		transition: 'background-color 0.2s',
		fontFamily: fontFamily,
		fontSize: '0.9rem'
	};

	// Clear basket button styles
	const clearButtonStyle = {
		backgroundColor: '#6B7280',
		color: 'white',
		fontWeight: '500',
		padding: '0.35rem 0.7rem',
		borderRadius: '0.35rem',
		border: 'none',
		cursor: 'pointer',
		transition: 'background-color 0.2s',
		fontFamily: fontFamily,
		fontSize: '0.9rem',
		marginRight: '0.7rem'
	};

	// Total container styles
	const totalContainerStyle = {
		display: 'flex',
		justifyContent: 'space-between',
		alignItems: 'center',
		padding: '0.7rem 0',
		borderTop: '2px solid #e5e7eb',
		marginTop: '0.7rem'
	};

	// Parameter styles
	const paramStyle = {
		padding: '0.2rem 0.4rem',
		backgroundColor: '#f3f4f6',
		borderRadius: '0.35rem',
		fontSize: '0.85rem',
		color: '#333333'
	};

	return (
		<div style={containerStyle}>
			<h2 style={titleStyle}>Job Basket</h2>

			{basket.length === 0 ? (
				<p style={{textAlign: 'center', color: '#333333'}}>No jobs in basket</p>
			) : (
				<>
					{basket.map((item, index) => (
						<div key={item.id || index} style={itemStyle}>
							<div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center'}}>
								<h3 style={{fontSize: '1.1rem', fontWeight: '500', margin: '0'}}>
									Job {index + 1} - {item.deviceName || 'Unknown Device'}
								</h3>
								<button
									style={removeButtonStyle}
									onClick={() => onRemoveFromBasket(index)}
								>
									Remove
								</button>
							</div>

							<div style={{display: 'flex', flexWrap: 'wrap', gap: '0.35rem', margin: '0.35rem 0', fontSize: '0.9rem'}}>
								<span style={{...paramStyle, backgroundColor: '#e6f0ff', border: '1px solid #b8d0ff'}}>
									Device: {item.deviceName}
								</span>
								<span style={paramStyle}>C: {item.params.batches}</span>
								<span style={paramStyle}>D: {item.params.depth}</span>
								<span style={paramStyle}>S: {item.params.shots}</span>
								<span style={paramStyle}>Q: {item.params.qubits}</span>
								<span style={{...paramStyle, backgroundColor: '#f0f7ff', border: '1px solid #d0e3ff', color: '#333333'}}>
									QPU: {item.qpuSeconds}s
								</span>
							</div>
						</div>
					))}

					<div style={totalContainerStyle}>
						<div>
							<div style={{
								backgroundColor: '#f0f7ff',
								padding: '0.7rem 1.05rem',
								borderRadius: '0.35rem',
								border: '1px solid #d0e3ff',
								display: 'inline-block'
							}}>
								<div style={{fontSize: '0.9rem', color: '#333333', marginBottom: '0.35rem'}}>
									Total QPU Seconds
								</div>
								<div style={{fontSize: '1.5rem', fontWeight: 'bold', color: '#333333'}}>
									{totalQPUSeconds}
								</div>
							</div>
						</div>

						<div>
							<button
								style={clearButtonStyle}
								onClick={onClearBasket}
							>
								Clear Basket
							</button>
						</div>
					</div>
				</>
			)}
		</div>
	);
};

export default BasketPanel;
