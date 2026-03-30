import React, { createContext, useContext, useEffect, useState } from "react";
import { useSettingsStore } from "../../store/useSettingsStore.js";

/**
 * Theme Context
 */
const ThemeContext = createContext({
  theme: "light",
  resolvedTheme: "light",
  isDark: false,
  isLight: true,
  toggleTheme: () => {},
  setTheme: () => {},
});

const getSystemTheme = () => {
  if (
    typeof window === "undefined" ||
    typeof window.matchMedia !== "function"
  ) {
    return "light";
  }
  return window.matchMedia("(prefers-color-scheme: dark)").matches
    ? "dark"
    : "light";
};

/**
 * Theme Provider Component
 * Manages dark/light theme across the application
 */
export function ThemeProvider({ children }) {
  const { theme, setTheme, toggleTheme } = useSettingsStore();
  const [systemTheme, setSystemTheme] = useState(getSystemTheme);
  const resolvedTheme = theme === "system" ? systemTheme : theme;

  useEffect(() => {
    if (
      typeof window === "undefined" ||
      typeof window.matchMedia !== "function"
    ) {
      return undefined;
    }

    const mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");
    const handleChange = (event) => {
      setSystemTheme(event.matches ? "dark" : "light");
    };

    setSystemTheme(mediaQuery.matches ? "dark" : "light");
    mediaQuery.addEventListener("change", handleChange);
    return () => mediaQuery.removeEventListener("change", handleChange);
  }, []);

  useEffect(() => {
    const root = document.documentElement;
    root.setAttribute("data-theme", resolvedTheme);
    root.classList.remove("theme-dark", "theme-light");
    root.classList.add(`theme-${resolvedTheme}`);
  }, [resolvedTheme]);

  const contextValue = {
    theme,
    resolvedTheme,
    toggleTheme,
    setTheme,
    isDark: resolvedTheme === "dark",
    isLight: resolvedTheme === "light",
  };

  return (
    <ThemeContext.Provider value={contextValue}>
      {children}
    </ThemeContext.Provider>
  );
}

/**
 * useTheme Hook
 * Access theme context
 */
export function useTheme() {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error("useTheme must be used within a ThemeProvider");
  }
  return context;
}

export default ThemeProvider;
