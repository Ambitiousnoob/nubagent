import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./src/__tests__/setup.js'],
    include: ['src/**/*.{test,spec}.{js,jsx}', 'api/**/*.{test,spec}.{js}'],
    exclude: ['node_modules', 'dist', '.git', '.cache'],
    coverage: {
      reporter: ['text', 'json', 'html'],
      exclude: ['node_modules', 'src/__tests__'],
    },
  },
  esbuild: {
    jsx: 'automatic',
    jsxImportSource: 'react',
  },
});
