import React from 'react';

const fontFamily = '-apple-system,BlinkMacSystemFont,"Roboto","Segoe UI","Helvetica Neue","Lucida Grande",Arial,sans-serif';

/**
 * Format timestamp to user-friendly string
 * @param {string} timestamp - ISO string timestamp
 * @returns {string} Formatted date string
 */
const formatTimestamp = (timestamp) => {
	try {
		const date = new Date(timestamp);
		return new Intl.DateTimeFormat('en-GB', {
			day: '2-digit',
			month: 'short',
			hour: '2-digit',
			minute: '2-digit'
		}).format(date);
	} catch {
		return 'Unknown date';
	}
};

const HistoryPanel = ({ history, onAddToBasket, onClearHistory }) => {
	// History panel container styles
	const containerStyle = {
		backgroundColor: 'white',
		padding: '1.05rem',
		borderRadius: '0.7rem',
		boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)',
		marginBottom: '1.05rem',
		maxHeight: '28rem',
		overflowY: 'auto',
		fontFamily: fontFamily
	};

	// History item styles
	const itemStyle = {
		borderBottom: '1px solid #e5e7eb',
		padding: '0.7rem 0',
		marginBottom: '0.7rem',
	};

	// History panel header container style
	const headerContainerStyle = {
		display: 'flex',
		justifyContent: 'space-between',
		alignItems: 'center',
		marginBottom: '1.05rem'
	};

	// History panel title styles
	const titleStyle = {
		fontSize: '1.5rem',
		fontWeight: '500',
		color: '#333333',
		margin: 0,
		fontFamily: fontFamily
	};

	// Clear history button styles
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
		fontSize: '0.9rem'
	};

	// History item parameters container
	const paramsStyle = {
		display: 'flex',
		flexWrap: 'wrap',
		gap: '0.35rem',
		margin: '0.35rem 0',
		fontSize: '0.9rem'
	};

	// Parameter styles
	const paramStyle = {
		padding: '0.2rem 0.4rem',
		backgroundColor: '#f3f4f6',
		borderRadius: '0.35rem',
		fontSize: '0.85rem',
		color: '#333333'
	};

	// Add to basket button styles
	const buttonStyle = {
		backgroundColor: '#2e52a5',
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

	return (
		<div style={containerStyle}>
			<div style={headerContainerStyle}>
				<h2 style={titleStyle}>Estimation History</h2>
				{history.length > 0 && (
					<button
						style={clearButtonStyle}
						onClick={onClearHistory}
					>
						Clear History
					</button>
				)}
			</div>

			{history.length === 0 ? (
				<p style={{textAlign: 'center', color: '#333333'}}>No previous estimations</p>
			) : (
				history.map((item, index) => (
					<div key={item.id || index} style={itemStyle}>
						<div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center'}}>
							<h3 style={{fontSize: '1.1rem', fontWeight: '500', margin: '0'}}>
								{item.deviceName || 'Unknown Device'}
							</h3>
							<span style={{fontSize: '0.85rem', color: '#333333'}}>
								{formatTimestamp(item.timestamp)}
							</span>
						</div>

						<div style={paramsStyle}>
							<span style={{...paramStyle, backgroundColor: '#e6f0ff', border: '1px solid #b8d0ff'}}>
								Device: {item.deviceName}
							</span>
							<span style={paramStyle}>Circuits: {item.params.batches}</span>
							<span style={paramStyle}>Depth: {item.params.depth}</span>
							<span style={paramStyle}>Shots: {item.params.shots}</span>
							<span style={paramStyle}>Qubits: {item.params.qubits}</span>
						</div>

						<div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '0.7rem'}}>
							<div style={{
								backgroundColor: '#f0f7ff',
								padding: '0.35rem 0.7rem',
								borderRadius: '0.35rem',
								border: '1px solid #d0e3ff',
								fontSize: '1.05rem',
								fontWeight: '500',
								color: '#333333'
							}}>
								<span style={{fontSize: '0.85rem', color: '#333333', marginRight: '0.35rem'}}>QPU Seconds:</span>
								{item.qpuSeconds}
							</div>

							<button
								style={buttonStyle}
								onClick={() => onAddToBasket(item)}
							>
								Add to Basket
							</button>
						</div>
					</div>
				))
			)}
		</div>
	);
};

export default HistoryPanel;
