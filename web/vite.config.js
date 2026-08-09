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
            "/admin": "http://localhost:8000",
            "/barcode": "http://localhost:8000",
            "/feedback": "http://localhost:8000",
            "/health": "http://localhost:8000",
        },
    },
});
