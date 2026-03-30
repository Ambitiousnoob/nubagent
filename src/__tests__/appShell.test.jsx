import React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";

vi.mock("../SearchEngine.jsx", () => ({
  default: ({ session }) => (
    <div data-testid="chat-view">{session?.id || "no-session"}</div>
  ),
}));

vi.mock("../Library.jsx", () => ({
  default: () => <div data-testid="library-view">Library view</div>,
}));

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

vi.mock("../components/Settings/SettingsModal.jsx", () => ({
  SettingsModal: () => null,
}));

import App from "../App.jsx";
import { useLibraryStore } from "../store/useLibraryStore.js";
import { useSettingsStore } from "../store/useSettingsStore.js";
import { useUIStore } from "../store/useUIStore.js";

describe("App shell routing", () => {
  beforeEach(() => {
    window.localStorage.clear();
    window.history.replaceState({}, "", "/");

    useUIStore.setState({
      sidebar: {
        isOpen: true,
        isCollapsed: false,
        activeTab: "chat",
      },
      currentRoute: "chat",
      isMobile: false,
    });

    useLibraryStore.setState({
      sessions: [],
      isLoading: false,
      error: null,
      searchQuery: "",
      filters: {
        dateRange: "all",
        hasAttachments: false,
        sortBy: "date",
        sortOrder: "desc",
      },
      selectedSession: null,
      selectedSessions: [],
    });

    useSettingsStore.setState({
      theme: "light",
    });
  });

  it("hydrates the current view from the pathname on refresh", async () => {
    window.history.replaceState({}, "", "/docs");

    render(<App />);

    expect(await screen.findByTestId("docs-view")).toBeInTheDocument();
    expect(useUIStore.getState().currentRoute).toBe("docs");
  });

  it("does not push duplicate history entries for the active route", async () => {
    window.history.replaceState({}, "", "/docs");
    const pushStateSpy = vi.spyOn(window.history, "pushState");

    render(<App />);

    expect(await screen.findByTestId("docs-view")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { current: "page" }));

    expect(pushStateSpy).not.toHaveBeenCalled();
    pushStateSpy.mockRestore();
  });

  it("preserves the active chat session across shell route changes", async () => {
    window.localStorage.setItem(
      "nubagent-library",
      JSON.stringify([
        {
          id: "session-42",
          query: "Loaded session",
          heading: "Loaded session",
          body: "Stored answer",
          createdAt: "2026-03-30T10:00:00.000Z",
          updatedAt: "2026-03-30T10:00:00.000Z",
        },
      ]),
    );
    window.history.replaceState({}, "", "/?session=session-42");

    render(<App />);

    await waitFor(() => {
      expect(screen.getByTestId("chat-view")).toHaveTextContent("session-42");
    });

    fireEvent.click(screen.getByText("Docs"));
    await waitFor(() => {
      expect(screen.getByTestId("docs-view")).toBeInTheDocument();
    });
    expect(`${window.location.pathname}${window.location.search}`).toBe(
      "/docs",
    );

    fireEvent.click(screen.getByText("Research"));
    await waitFor(() => {
      expect(screen.getByTestId("chat-view")).toHaveTextContent("session-42");
    });
    expect(`${window.location.pathname}${window.location.search}`).toBe(
      "/?session=session-42",
    );
  });

  it("drops a stale remembered session id instead of pushing it back into the chat route", async () => {
    window.localStorage.setItem(
      "nubagent-library",
      JSON.stringify([
        {
          id: "session-42",
          query: "Loaded session",
          heading: "Loaded session",
          body: "Stored answer",
          createdAt: "2026-03-30T10:00:00.000Z",
          updatedAt: "2026-03-30T10:00:00.000Z",
        },
      ]),
    );
    window.history.replaceState({}, "", "/?session=session-42");

    render(<App />);

    await waitFor(() => {
      expect(screen.getByTestId("chat-view")).toHaveTextContent("session-42");
    });

    window.localStorage.setItem("nubagent-library", JSON.stringify([]));

    fireEvent.click(screen.getByText("Docs"));
    await waitFor(() => {
      expect(screen.getByTestId("docs-view")).toBeInTheDocument();
    });

    fireEvent.click(screen.getByText("Research"));
    await waitFor(() => {
      expect(screen.getByTestId("chat-view")).toHaveTextContent("no-session");
    });
    expect(`${window.location.pathname}${window.location.search}`).toBe("/");
  });
});
