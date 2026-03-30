import React, { Component } from "react";
import { AlertTriangle, RefreshCw } from "lucide-react";
import { Button } from "./Button.jsx";

const IS_DEV = Boolean(import.meta.env?.DEV);

/**
 * Error Boundary Component
 * Catches JavaScript errors in child components
 */
export class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
    };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    this.setState({ errorInfo });
    console.error("ErrorBoundary caught:", error, errorInfo);

    // Log to error tracking service if configured
    if (this.props.onError) {
      this.props.onError(error, errorInfo);
    }
  }

  handleReset = () => {
    this.setState({ hasError: false, error: null, errorInfo: null });
    if (this.props.onReset) {
      this.props.onReset();
    }
  };

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      return (
        <div className="error-boundary">
          <div className="error-boundary__icon">
            <AlertTriangle size={48} />
          </div>
          <h2 className="error-boundary__title">Something went wrong</h2>
          <p className="error-boundary__message">
            {this.props.errorMessage ||
              "An unexpected error occurred. Please try again."}
          </p>
          {IS_DEV && this.state.error && (
            <details className="error-boundary__details">
              <summary>Error Details</summary>
              <pre className="error-boundary__stack">
                {this.state.error.toString()}
                {this.state.errorInfo?.componentStack}
              </pre>
            </details>
          )}
          <div className="error-boundary__actions">
            <Button onClick={this.handleReset} icon={<RefreshCw size={16} />}>
              Try Again
            </Button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

/**
 * Functional Error Fallback Component
 * For use with React 18 error boundaries
 */
export function ErrorFallback({ error, resetErrorBoundary }) {
  return (
    <div className="error-boundary">
      <div className="error-boundary__icon">
        <AlertTriangle size={48} />
      </div>
      <h2 className="error-boundary__title">Something went wrong</h2>
      <p className="error-boundary__message">
        An unexpected error occurred. Please try again.
      </p>
      {IS_DEV && error && (
        <details className="error-boundary__details">
          <summary>Error Details</summary>
          <pre className="error-boundary__stack">{error.toString()}</pre>
        </details>
      )}
      <div className="error-boundary__actions">
        <Button onClick={resetErrorBoundary} icon={<RefreshCw size={16} />}>
          Try Again
        </Button>
      </div>
    </div>
  );
}

export default ErrorBoundary;
