import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Get the repo name from package.json or GitHub env variable
const getRepoName = () => {
  // For GitHub Pages deployment, use the repository name as the base
  if (process.env.GITHUB_REPOSITORY) {
    const repoName = process.env.GITHUB_REPOSITORY.split('/')[1];
    return `/${repoName}/`;
  }
  // Fallback - you can replace this with your actual repo name
  return '/';
};

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  base: process.env.NODE_ENV === 'production' ? getRepoName() : '/',
  build: {
    chunkSizeWarningLimit: 800, // Increase the warning limit
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom'],
          markdown: ['react-markdown', 'remark-math', 'rehype-katex', 'katex'],
        },
      },
    },
  },
})
