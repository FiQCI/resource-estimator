import React, { useState } from 'react';
import Header from './components/Header';
import ResourceEstimator from './components/ResourceEstimator';
import Documentation from './components/Documentation';

function App() {
	// State to control documentation visibility
	const [isDocumentationOpen, setIsDocumentationOpen] = useState(false);

	// Function to open documentation
	const openDocumentation = () => {
		setIsDocumentationOpen(true);
	};

	// Function to close documentation
	const closeDocumentation = () => {
		setIsDocumentationOpen(false);
	};

	// Override any default styles from index.css
	const rootStyle = {
		backgroundColor: '#eeeeee',
		maxWidth: '100vw', // Use viewport width
		width: '100vw',
		margin: 0,
		padding: 0,
		textAlign: 'left',
		overflow: 'hidden',
		boxSizing: 'border-box',
		fontFamily: '-apple-system,BlinkMacSystemFont,"Roboto","Segoe UI","Helvetica Neue","Lucida Grande",Arial,sans-serif',
		minHeight: '100vh',
		display: 'flex',
		flexDirection: 'column'
	};

	const mainStyle = {
		padding: '0.5rem 1.5rem', // Increased horizontal padding for the wider layout
		margin: '0 auto',
		width: '100%',
		boxSizing: 'border-box',
		flex: 1,
		maxWidth: '1400px' // Add max-width to control overall width on larger screens
	};

	return (
		<div style={rootStyle}>
			<Header onOpenDocumentation={openDocumentation} />
			
			<main style={mainStyle}>
				<ResourceEstimator />
			</main>
			
			{/* Documentation component */}
			<Documentation 
				isOpen={isDocumentationOpen} 
				onClose={closeDocumentation} 
			/>
			
			<footer style={{
				backgroundColor: '#2e52a5',
				color: 'white',
				padding: '0.5rem',
				textAlign: 'center',
				width: '100%',
				fontFamily: '-apple-system,BlinkMacSystemFont,"Roboto","Segoe UI","Helvetica Neue","Lucida Grande",Arial,sans-serif'
			}}>
				<div style={{margin: '0 auto', padding: '0 1rem'}}>
					<p style={{fontSize: '0.75rem', margin: '0'}}>
						&copy; {new Date().getFullYear()} FiQCI - Finnish Quantum-Computing Infrastructure. All rights reserved.
					</p>
				</div>
			</footer>
		</div>
	);
}

export default App;