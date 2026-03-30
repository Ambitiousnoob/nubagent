import React from "react";
import { Moon, Sun } from "lucide-react";
import { useSettingsStore } from "../../store/useSettingsStore.js";

/**
 * ThemeToggle Component
 * Dark/light theme switch
 */
export function ThemeToggle() {
  const { theme, toggleTheme, setTheme } = useSettingsStore();

  return (
    <div className="theme-toggle">
      <button
        className={`theme-toggle__option ${theme === "light" ? "theme-toggle__option--active" : ""}`}
        onClick={() => setTheme("light")}
        aria-label="Light theme"
        aria-pressed={theme === "light"}
      >
        <Sun size={20} />
        <span>Light</span>
      </button>
      <button
        className={`theme-toggle__option ${theme === "dark" ? "theme-toggle__option--active" : ""}`}
        onClick={() => setTheme("dark")}
        aria-label="Dark theme"
        aria-pressed={theme === "dark"}
      >
        <Moon size={20} />
        <span>Dark</span>
      </button>
      <button
        className={`theme-toggle__option ${theme === "system" ? "theme-toggle__option--active" : ""}`}
        onClick={() => setTheme("system")}
        aria-label="System theme"
        aria-pressed={theme === "system"}
      >
        <span className="theme-toggle__system-icon">◐</span>
        <span>System</span>
      </button>
    </div>
  );
}

export default ThemeToggle;
