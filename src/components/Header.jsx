import React from 'react';
import fiqciLogo from '/FiQCI-logo.png';
import { DocumentationButton } from './Documentation';

const Header = ({ onOpenDocumentation }) => {
	const headerStyle = {
		backgroundColor: '#eeeeee',
		padding: '0.5rem 1rem',
		textAlign: 'left',
		width: '100%',
		position: 'relative',
		left: 0,
		right: 0,
		margin: 0,
		fontFamily: '-apple-system,BlinkMacSystemFont,"Roboto","Segoe UI","Helvetica Neue","Lucida Grande",Arial,sans-serif'
	};

	const containerStyle = {
		display: 'flex',
		alignItems: 'center',
		justifyContent: 'space-between',
		paddingLeft: 0
	};

	// Add right padding to ensure button isn't cut off
	const buttonContainerStyle = {
		paddingRight: '20px'
	};

	return (
		<header style={headerStyle}>
			<div style={containerStyle}>
				<div style={{ display: 'flex', alignItems: 'center' }}>
					<img 
						src={fiqciLogo} 
						alt="FiQCI Logo" 
						style={{height: '2.5rem'}}
					/>
				</div>
				<h1 style={{
					fontSize: '1.5rem',
					fontWeight: '500',
					color: '#333333',
					margin: 0,
					fontFamily: '-apple-system,BlinkMacSystemFont,"Roboto","Segoe UI","Helvetica Neue","Lucida Grande",Arial,sans-serif'
				}}>
					FiQCI Resource Estimator
				</h1>
				<div style={buttonContainerStyle}>
					<DocumentationButton onClick={onOpenDocumentation} />
				</div>
			</div>
		</header>
	);
};

export default Header;