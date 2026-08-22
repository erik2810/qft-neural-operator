import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The static build is served from https://<user>.github.io/qft-neural-operator/, so asset
// URLs need that prefix in production; dev runs at the root behind the backend proxy.
export default defineConfig(({ mode }) => ({
  base: mode === "production" ? "/qft-neural-operator/" : "/",
  plugins: [react(), tailwindcss()],
  server: {
    port: 5175,
    proxy: {
      "/physics": "http://localhost:8000",
      "/health": "http://localhost:8000",
      "/ws": { target: "ws://localhost:8000", ws: true },
    },
  },
}));
