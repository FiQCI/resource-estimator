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
		// Make sure assets get the correct path
		assetsDir: 'assets',
		// Generate clean URLs for assets
		rollupOptions: {
			output: {
				// Ensure proper chunking and file naming
				entryFileNames: 'assets/[name].[hash].js',
				chunkFileNames: 'assets/[name].[hash].js',
				assetFileNames: 'assets/[name].[hash].[ext]'
			}
		}
	},
	// Ensure proper resolution of file paths
	resolve: {
		alias: {
			'@': '/src'
		}
	}
})