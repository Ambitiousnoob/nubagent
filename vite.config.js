import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

function isIgnorableLucideDirectiveWarning(warning) {
  return (
    warning?.code === "MODULE_LEVEL_DIRECTIVE" &&
    warning?.message?.includes('"use client"') &&
    typeof warning?.id === "string" &&
    warning.id.includes("/node_modules/lucide-react/")
  );
}

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      onwarn(warning, warn) {
        if (isIgnorableLucideDirectiveWarning(warning)) {
          return;
        }

        warn(warning);
      },
    },
  },
});
