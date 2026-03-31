import React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";

vi.mock("../Docs.jsx", () => ({
  default: () => <div data-testid="docs-view">Docs view</div>,
}));

vi.mock("../components/UI/ThemeProvider.jsx", () => ({
  ThemeProvider: ({ children }) => <>{children}</>,
}));

vi.mock("../components/UI/ToastProvider.jsx", () => ({
  ToastProvider: () => null,
}));

vi.mock("../components/UI/ErrorBoundary.jsx", () => ({
  ErrorBoundary: ({ children }) => <>{children}</>,
}));

import App from "../App.jsx";
import { useSettingsStore } from "../store/useSettingsStore.js";

describe("App shell", () => {
  beforeEach(() => {
    window.localStorage.clear();
    window.history.replaceState({}, "", "/");
    useSettingsStore.setState({
      theme: "light",
    });
  });

  it("renders the documentation site at the root path", async () => {
    render(<App />);

    expect(await screen.findByTestId("docs-view")).toBeInTheDocument();
    expect(screen.getByText("NubAgent Docs")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /api reference/i })).toBeInTheDocument();
  });

  it("canonicalizes legacy app routes back to the docs homepage", async () => {
    window.history.replaceState({}, "", "/library?session=session-42#api-reference");

    render(<App />);

    expect(await screen.findByTestId("docs-view")).toBeInTheDocument();
    expect(window.location.pathname).toBe("/");
    expect(window.location.search).toBe("");
    expect(window.location.hash).toBe("#api-reference");
  });

  it("toggles the persisted theme from the docs shell", async () => {
    render(<App />);

    const [themeToggle] = await screen.findAllByRole("button", {
      name: /switch to dark theme/i,
    });
    fireEvent.click(themeToggle);

    expect(useSettingsStore.getState().theme).toBe("dark");
  });
});
