import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'

// Override any default dark mode preferences
document.documentElement.classList.add('light-mode');
document.documentElement.style.backgroundColor = '#eeeeee';
document.body.style.backgroundColor = '#eeeeee';

// Apply font family
document.body.style.fontFamily = '-apple-system,BlinkMacSystemFont,"Roboto","Segoe UI","Helvetica Neue","Lucida Grande",Arial,sans-serif';

// Fix the root element to take full width
const rootElement = document.getElementById('root');
if (rootElement) {
	// These styles override the ones in index.css
	rootElement.style.maxWidth = '100%';
	rootElement.style.width = '100%';
	rootElement.style.padding = '0';
	rootElement.style.margin = '0';
	rootElement.style.overflow = 'hidden';
}

createRoot(document.getElementById('root')).render(
	<StrictMode>
		<App />
	</StrictMode>
)
