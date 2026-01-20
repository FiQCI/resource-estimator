import React, { useState, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';

const fontFamily = '-apple-system,BlinkMacSystemFont,"Roboto","Segoe UI","Helvetica Neue","Lucida Grande",Arial,sans-serif';

const Documentation = ({ isOpen, onClose }) => {
	const [markdown, setMarkdown] = useState('');
	const [loading, setLoading] = useState(true);
	const [error, setError] = useState(null);

	useEffect(() => {
		if (isOpen) {
			setLoading(true);

			// Get the base URL from Vite's environment variable or use an empty string as fallback
			const baseUrl = import.meta.env.BASE_URL || '/';

			// Construct the path to documentation.md, ensuring there are no double slashes
			const docPath = `${baseUrl.endsWith('/') ? baseUrl.slice(0, -1) : baseUrl}/documentation.md`;

			console.log('Fetching documentation from:', docPath); // For debugging

			fetch(docPath)
				.then(response => {
					if (!response.ok) {
						throw new Error(`Failed to load documentation: ${response.status} ${response.statusText}`);
					}
					return response.text();
				})
				.then(text => {
					setMarkdown(text);
					setLoading(false);
				})
				.catch(err => {
					console.error('Error loading documentation:', err);
					setError(err.message);
					setLoading(false);
				});
		}
	}, [isOpen]);

	// Don't render anything if not open
	if (!isOpen) return null;

	// Overlay styles
	const overlayStyle = {
		position: 'fixed',
		top: 0,
		left: 0,
		right: 0,
		bottom: 0,
		backgroundColor: 'rgba(0, 0, 0, 0.5)',
		zIndex: 1000,
		display: 'flex',
		justifyContent: 'center',
		alignItems: 'center'
	};

	// Documentation panel styles
	const containerStyle = {
		backgroundColor: 'white',
		padding: '0', // Remove padding to allow header to reach edge
		borderRadius: '0.7rem',
		boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1), 0 1px 3px rgba(0, 0, 0, 0.08)',
		width: '90%',
		maxWidth: '1000px',
		maxHeight: '85vh',
		overflowY: 'auto',
		fontFamily: fontFamily,
		position: 'relative',
		display: 'flex',
		flexDirection: 'column'
	};

	// Content container styles
	const contentContainerStyle = {
		padding: '0 1.4rem 1.4rem 1.4rem',
		overflowY: 'auto',
		flex: 1
	};

	// Header styles - not sticky anymore
	const headerStyle = {
		display: 'flex',
		justifyContent: 'space-between',
		alignItems: 'center',
		borderBottom: '1px solid #e5e7eb',
		padding: '1rem 1.4rem',
		backgroundColor: 'white',
		borderTopLeftRadius: '0.7rem',
		borderTopRightRadius: '0.7rem'
	};

	// Title styles
	const titleStyle = {
		fontSize: '1.5rem',
		fontWeight: '500',
		color: '#333333',
		margin: 0,
		fontFamily: fontFamily
	};

	// Close button styles
	const closeButtonStyle = {
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

	// Markdown content styles
	const markdownStyle = {
		color: '#333333',
		lineHeight: '1.6',
		fontSize: '1rem'
	};

	// Custom components for ReactMarkdown
	const components = {
		// Custom image component to handle paths and styling
		img: ({ node, ...props }) => {
			let src = props.src;

			// Get the base URL
			const baseUrl = import.meta.env.BASE_URL || '/';

			// If the path is relative and doesn't start with http or the baseUrl,
			// prefix it with the baseUrl
			if (src && !src.startsWith('http') && !src.startsWith(baseUrl)) {
				// Remove leading slash from src if it exists
				src = src.startsWith('/') ? src.slice(1) : src;
				// Add baseUrl, ensuring no double slashes
				src = `${baseUrl.endsWith('/') ? baseUrl : baseUrl + '/'}${src}`;
			}

			return (
				<img
					{...props}
					src={src}
					alt={props.alt || ''}
					style={{
						maxWidth: '100%',
						height: 'auto',
						margin: '1rem 0',
						borderRadius: '0.35rem'
					}}
				/>
			);
		}
	};

	return (
		<div style={overlayStyle} onClick={onClose}>
			<div style={containerStyle} onClick={e => e.stopPropagation()}>
				<div style={headerStyle}>
					<h2 style={titleStyle}>FiQCI Resource Estimator Documentation</h2>
					<button
						style={closeButtonStyle}
						onClick={onClose}
					>
						Close
					</button>
				</div>
				<div style={contentContainerStyle}>
					<div style={markdownStyle}>
						{loading ? (
							<p>Loading documentation...</p>
						) : error ? (
							<div>
								<p>Error loading documentation: {error}</p>
								<p>Please make sure the documentation.md file exists in the public folder.</p>
							</div>
						) : (
							<ReactMarkdown
								remarkPlugins={[remarkMath]}
								rehypePlugins={[rehypeKatex]}
								components={components}
							>
								{markdown}
							</ReactMarkdown>
						)}
					</div>
				</div>
			</div>
		</div>
	);
};

export default Documentation;

// Documentation button component for reuse
export const DocumentationButton = ({ onClick }) => {
	return (
		<button
			onClick={onClick}
			style={{
				backgroundColor: '#2e52a5',
				color: 'white',
				fontWeight: '500',
				padding: '0.35rem 0.7rem',
				borderRadius: '0.35rem',
				border: 'none',
				cursor: 'pointer',
				transition: 'background-color 0.2s',
				fontFamily: '-apple-system,BlinkMacSystemFont,"Roboto","Segoe UI","Helvetica Neue","Lucida Grande",Arial,sans-serif',
				fontSize: '0.9rem'
			}}
		>
			Documentation
		</button>
	);
};
