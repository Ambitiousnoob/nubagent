import React from 'react';
import { Toaster } from 'sonner';
import { useUIStore } from '../../store/useUIStore.js';
import { AlertCircle, CheckCircle2, Info, X, AlertTriangle } from 'lucide-react';

/**
 * Toast Provider Component
 * Wraps Sonner toaster with custom styling
 */
export function ToastProvider() {
  const { toasts, removeToast } = useUIStore();

  return (
    <Toaster
      position="top-right"
      toastOptions={{
        duration: 5000,
        className: 'toast',
        classNames: {
          success: 'toast--success',
          error: 'toast--error',
          warning: 'toast--warning',
          info: 'toast--info',
          default: 'toast--default',
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
  const { addToast, removeToast } = useUIStore();

  const toast = (options) => {
    return addToast(options);
  };

  const success = (title, description) => {
    return addToast({ title, description, type: 'success' });
  };

  const error = (title, description) => {
    return addToast({ title, description, type: 'error' });
  };

  const warning = (title, description) => {
    return addToast({ title, description, type: 'warning' });
  };

  const info = (title, description) => {
    return addToast({ title, description, type: 'info' });
  };

  const loading = (title, description) => {
    return addToast({ title, description, type: 'loading', duration: 0 });
  };

  const dismiss = (id) => {
    removeToast(id);
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
