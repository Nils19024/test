import { defineConfig } from "vitest/config";

export default defineConfig({
  root: ".",
  publicDir: "public",
  test: {
    exclude: ["node_modules/**", "dist/**", "dist-types/**"]
  },
  build: {
    outDir: "dist",
    emptyOutDir: true
  },
  server: {
    port: 5173
  }
});
