import { create } from "zustand";
import { getAppViewFromLocation, normalizeAppView } from "../lib/appRoutes.js";

const getInitialRoute = () => {
  if (typeof window === "undefined") return "docs";
  return normalizeAppView(getAppViewFromLocation(window.location));
};

/**
 * UI store for managing application UI state
 * Handles sidebar, modals, toasts, and other UI elements
 */
export const useUIStore = create((set, get) => ({
  // State
  sidebar: {
    isOpen: true,
    isCollapsed: false,
    activeTab: getInitialRoute(),
  },
  modals: {
    settings: false,
    library: false,
    export: false,
    confirm: false,
    help: false,
  },
  toasts: [],
  currentRoute: getInitialRoute(),
  isMobile: false,

  // Actions
  setSidebarOpen: (isOpen) =>
    set((state) => ({
      sidebar: { ...state.sidebar, isOpen },
    })),

  toggleSidebar: () =>
    set((state) => ({
      sidebar: { ...state.sidebar, isOpen: !state.sidebar.isOpen },
    })),

  setSidebarCollapsed: (isCollapsed) =>
    set((state) => ({
      sidebar: { ...state.sidebar, isCollapsed },
    })),

  setActiveTab: (tab) =>
    set((state) => {
      const nextRoute = normalizeAppView(tab);
      return {
        sidebar: { ...state.sidebar, activeTab: nextRoute },
        currentRoute: nextRoute,
      };
    }),

  openModal: (modal) =>
    set((state) => ({
      modals: { ...state.modals, [modal]: true },
    })),

  closeModal: (modal) =>
    set((state) => ({
      modals: { ...state.modals, [modal]: false },
    })),

  toggleModal: (modal) =>
    set((state) => ({
      modals: { ...state.modals, [modal]: !state.modals[modal] },
    })),

  closeAllModals: () =>
    set({
      modals: {
        settings: false,
        library: false,
        export: false,
        confirm: false,
        help: false,
      },
    }),

  // Toast actions
  addToast: (toast) => {
    const id = `${Date.now()}-${Math.random().toString(16).slice(2)}`;
    const newToast = {
      id,
      title: toast.title || "",
      description: toast.description || "",
      type: toast.type || "default", // default, success, error, warning, info
      duration: toast.duration || 5000,
      action: toast.action,
    };

    set((state) => ({
      toasts: [...state.toasts, newToast],
    }));

    // Auto-dismiss
    if (newToast.duration > 0) {
      setTimeout(() => {
        get().removeToast(id);
      }, newToast.duration);
    }

    return id;
  },

  removeToast: (id) =>
    set((state) => ({
      toasts: state.toasts.filter((t) => t.id !== id),
    })),

  clearToasts: () => set({ toasts: [] }),

  // Route actions
  setRoute: (route) =>
    set((state) => {
      const nextRoute = normalizeAppView(route);
      return {
        currentRoute: nextRoute,
        sidebar: { ...state.sidebar, activeTab: nextRoute },
      };
    }),

  setIsMobile: (isMobile) => set({ isMobile }),

  // Confirm modal helper
  showConfirm: (options = {}) =>
    new Promise((resolve) => {
      set({
        modals: { ...get().modals, confirm: true },
        confirmOptions: {
          title: options.title || "Confirm",
          description: options.description || "Are you sure?",
          confirmText: options.confirmText || "Confirm",
          cancelText: options.cancelText || "Cancel",
          variant: options.variant || "default", // default, danger
          onConfirm: () => resolve(true),
          onCancel: () => resolve(false),
        },
      });
    }),
}));

export default useUIStore;
