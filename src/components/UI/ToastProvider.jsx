import React from "react";
import { Toaster, toast as sonnerToast } from "sonner";
import {
  AlertCircle,
  CheckCircle2,
  Info,
  X,
  AlertTriangle,
} from "lucide-react";

/**
 * Toast Provider Component
 * Wraps Sonner toaster with custom styling
 */
export function ToastProvider() {
  return (
    <Toaster
      position="top-right"
      closeButton
      richColors
      toastOptions={{
        duration: 5000,
        className: "toast",
        classNames: {
          success: "toast--success",
          error: "toast--error",
          warning: "toast--warning",
          info: "toast--info",
          default: "toast--default",
        },
      }}
      icons={{
        success: <CheckCircle2 size={20} />,
        error: <AlertCircle size={20} />,
        warning: <AlertTriangle size={20} />,
        info: <Info size={20} />,
        loading: <div className="toast__spinner" />,
        close: <X size={16} />,
      }}
    />
  );
}

/**
 * useToast Hook
 * Programmatic toast notifications
 */
export function useToast() {
  const invokeToast = (variant, title, description, options = {}) => {
    const method =
      typeof sonnerToast[variant] === "function"
        ? sonnerToast[variant]
        : sonnerToast;
    return method(title, {
      description,
      ...options,
    });
  };

  const toast = (options) => {
    const {
      title = "",
      description,
      type = "default",
      duration,
      action,
    } = options || {};

    return invokeToast(
      type,
      title || description || "",
      description && title ? description : undefined,
      {
        duration,
        action,
      },
    );
  };

  const success = (title, description) => {
    return invokeToast("success", title, description);
  };

  const error = (title, description) => {
    return invokeToast("error", title, description);
  };

  const warning = (title, description) => {
    return invokeToast("warning", title, description);
  };

  const info = (title, description) => {
    return invokeToast("info", title, description);
  };

  const loading = (title, description) => {
    return invokeToast("loading", title, description, { duration: Infinity });
  };

  const dismiss = (id) => {
    sonnerToast.dismiss(id);
  };

  return {
    toast,
    success,
    error,
    warning,
    info,
    loading,
    dismiss,
  };
}

export default ToastProvider;
