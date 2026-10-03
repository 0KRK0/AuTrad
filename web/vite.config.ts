import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/api": "http://localhost:8000" } },
  build: {
    target: "es2022",
    minify: "esbuild",          // minified JS
    cssMinify: true,            // minified CSS
    sourcemap: false,
    assetsInlineLimit: 4096,    // tiny assets inlined; larger ones get hashed filenames for CDN caching
    rollupOptions: {
      output: {
        manualChunks: { react: ["react", "react-dom"] }, // stable vendor chunk, cached across deploys
      },
    },
  },
});
