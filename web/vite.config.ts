import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 5173,
    proxy: {
      // Forward API requests to the backend in local dev.
      // In Docker/Render the frontend and API are same-origin (no proxy needed).
      "/admin": { target: "http://localhost:8000", changeOrigin: true },
      "/barcode": { target: "http://localhost:8000", changeOrigin: true },
      "/receiving": { target: "http://localhost:8000", changeOrigin: true },
      "/customers": { target: "http://localhost:8000", changeOrigin: true },
      "/feedback": { target: "http://localhost:8000", changeOrigin: true },
      "/health": { target: "http://localhost:8000", changeOrigin: true },
    },
  },
});
