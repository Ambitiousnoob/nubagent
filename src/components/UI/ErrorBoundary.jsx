import React, { Component } from "react";
import { AlertTriangle } from "lucide-react";

const IS_DEV = Boolean(import.meta.env?.DEV);

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
    if (!this.state.hasError) {
      return this.props.children;
    }

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
          <button
            type="button"
            className="sidebar__footer-item"
            onClick={this.handleReset}
          >
            Try Again
          </button>
        </div>
      </div>
    );
  }
}

export default ErrorBoundary;
