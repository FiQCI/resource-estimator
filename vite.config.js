import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  base: '/resource-estimator/', // Repo path name for GitHub Pages
  
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    // Ensure assets use the correct public path
    assetsDir: 'assets',
    // Make sure to explicitly make inline scripts external
    assetsInlineLimit: 0,
    rollupOptions: {
      output: {
        // Format output to ensure correct paths
        entryFileNames: 'assets/[name].[hash].js',
        chunkFileNames: 'assets/[name].[hash].js',
        assetFileNames: 'assets/[name].[hash].[ext]'
      }
    }
  }
})