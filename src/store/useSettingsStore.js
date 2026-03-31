import { create } from "zustand";
import { persist } from "zustand/middleware";

export const useSettingsStore = create(
  persist(
    (set, get) => ({
      theme: "light",

      setTheme: (theme) =>
        set({
          theme: theme === "dark" ? "dark" : "light",
        }),

      toggleTheme: () =>
        set({
          theme: get().theme === "dark" ? "light" : "dark",
        }),

      clearAllSettings: () =>
        set({
          theme: "light",
        }),
    }),
    {
      name: "nubagent-settings-storage",
      version: 1,
    },
  ),
);

export default useSettingsStore;
