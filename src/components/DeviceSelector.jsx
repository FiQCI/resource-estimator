import React from 'react';

const fontFamily = '-apple-system,BlinkMacSystemFont,"Roboto","Segoe UI","Helvetica Neue","Lucida Grande",Arial,sans-serif';

const DeviceCard = ({ title, image, isSelected, onClick }) => {
	// Force styles with inline styling to override index.css
	const cardStyle = {
		backgroundColor: 'white',
		padding: '0.7rem',
		borderRadius: '0.35rem',
		boxShadow: isSelected ? 'none' : '0 1px 2px rgba(0, 0, 0, 0.05)',
		cursor: 'pointer',
		transition: 'all 0.2s',
		height: '100%',
		opacity: isSelected ? 1 : 0.5,
		border: isSelected ? '1px solid #2e52a5' : '1px solid #e5e7eb',
		boxSizing: 'border-box',
		display: 'flex',
		flexDirection: 'column',
		alignItems: 'center',
		position: 'relative',
		overflow: 'hidden'
	};

	// Adding a highlight effect instead of thick border
	const highlightStyle = isSelected ? {
		position: 'absolute',
		top: 0,
		left: 0,
		bottom: 0,
		width: '5px',
		backgroundColor: '#2e52a5'
	} : null;

	const imgContainerStyle = {
		marginBottom: '0.7rem',
		width: '100%',
		height: '9.8rem', // Increased by 40% from 7rem
		display: 'flex',
		alignItems: 'center',
		justifyContent: 'center',
		padding: '0.35rem'
	};

	// Force event handler to be directly attached
	const handleCardClick = (e) => {
		e.preventDefault();
		e.stopPropagation();
		console.log(`Clicked on ${title}`);
		onClick();
	};

	return (
		<div
			style={cardStyle}
			className="device-card"
			onClick={handleCardClick}
			data-selected={isSelected ? 'true' : 'false'}
			role="button"
			tabIndex={0}
			onKeyDown={(e) => {
				if (e.key === 'Enter' || e.key === ' ') {
					handleCardClick(e);
				}
			}}
		>
			{isSelected && <div style={highlightStyle}></div>}
			<div style={imgContainerStyle}>
				<img
					src={image}
					alt={`${title} device`}
					style={{maxHeight: '100%', maxWidth: '100%', objectFit: 'contain'}}
				/>
			</div>
			<h3 style={{
				textAlign: 'center', 
				fontSize: '1.1rem', // Increased from 0.875rem 
				fontWeight: '500', 
				color: '#333',
				fontFamily: fontFamily,
				margin: '0 0 0.35rem 0'
			}}>
				{title}
			</h3>
		</div>
	);
};

const DeviceSelector = ({ selectedDevice, onDeviceSelect }) => {
	// Container styles
	const containerStyle = {
		marginBottom: '1.05rem',
		width: '100%',
		backgroundColor: 'white',
		padding: '1.05rem',
		borderRadius: '0.7rem',
		boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)',
		boxSizing: 'border-box',
		fontFamily: fontFamily
	};

	const gridStyle = {
		display: 'grid',
		gridTemplateColumns: 'repeat(2, 1fr)',
		gap: '1.4rem',
		width: '100%',
		margin: '0 auto',
		padding: '0.7rem'
	};

	return (
		<div style={containerStyle}>
			<h2 style={{
				textAlign: 'center', 
				fontSize: '1.5rem',
				fontWeight: '500', 
				color: '#333333', 
				margin: '0 0 1.05rem 0',
				fontFamily: fontFamily
			}}>
				Select Quantum Computer
			</h2>
			<div style={gridStyle}>
				<DeviceCard
					title="Helmi"
					image="/helmi.png"
					isSelected={selectedDevice === 'helmi'}
					onClick={() => onDeviceSelect('helmi')}
				/>
				<DeviceCard
					title="VTT Q50"
					image="/vtt-q50.jpg"
					isSelected={selectedDevice === 'vtt-q50'}
					onClick={() => onDeviceSelect('vtt-q50')}
				/>
			</div>
		</div>
	);
};

export default DeviceSelector;