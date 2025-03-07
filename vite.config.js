import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
	plugins: [react()],
	base: '/resource-estimator/', // Repo path name
	
	// Configure the build output
	build: {
		outDir: 'dist',
		emptyOutDir: true,
	}
})