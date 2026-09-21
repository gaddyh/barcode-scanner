import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 5173,
    // Allow ngrok tunnels for mobile testing. Vite blocks non-localhost
    // Host headers by default; ngrok forwarding requires explicit allow.
    allowedHosts: [".ngrok-free.app", ".ngrok.io"],
    proxy: {
      // Forward API requests to the backend in local dev.
      // In Docker/Render the frontend and API are same-origin (no proxy needed).
      // timeout: 5 min — large camera photos over mobile networks via ngrok
      // can take a while to upload; the default 30s proxy timeout drops them.
      "/admin": { target: "http://localhost:8000", changeOrigin: true, timeout: 300000 },
      "/barcode": { target: "http://localhost:8000", changeOrigin: true, timeout: 300000 },
      "/receiving": { target: "http://localhost:8000", changeOrigin: true, timeout: 300000 },
      "/customers": { target: "http://localhost:8000", changeOrigin: true, timeout: 300000 },
      "/feedback": { target: "http://localhost:8000", changeOrigin: true, timeout: 300000 },
      "/health": { target: "http://localhost:8000", changeOrigin: true, timeout: 300000 },
    },
  },
});
