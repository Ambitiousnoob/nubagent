import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";

describe("Library Store", () => {
  let useLibraryStore;

  beforeEach(() => {
    vi.resetModules();
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-03-30T12:00:00.000Z"));
    useLibraryStore = require("../store/useLibraryStore").useLibraryStore;
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("normalizes saved timestamps so library date filters and sort keep working", async () => {
    window.localStorage.setItem(
      "nubagent-library",
      JSON.stringify([
        {
          id: "session-1",
          query: "Climate report",
          heading: "Climate report",
          body: "Summary",
          createdAt: "2026-03-30T10:00:00.000Z",
          updatedAt: "2026-03-30T10:30:00.000Z",
        },
      ]),
    );

    const loadPromise = useLibraryStore.getState().loadSessions();
    await vi.advanceTimersByTimeAsync(200);
    await loadPromise;

    const state = useLibraryStore.getState();
    expect(typeof state.sessions[0].createdAt).toBe("number");

    state.setFilters({ dateRange: "today" });
    const filtered = state.getFilteredSessions();
    expect(filtered).toHaveLength(1);
    expect(filtered[0].query).toBe("Climate report");
  });

  it("persists edited sessions and the typed library query", () => {
    const added = useLibraryStore.getState().addSession({
      id: "session-2",
      query: "Original question",
      heading: "Original heading",
      body: "Original body",
    });

    const updated = useLibraryStore.getState().updateSession(added.id, {
      query: "Edited question",
      heading: "Edited heading",
      body: "Edited body",
    });

    expect(updated.query).toBe("Edited question");
    expect(updated.body).toBe("Edited body");

    const storedSessions = JSON.parse(
      window.localStorage.getItem("nubagent-library"),
    );
    expect(storedSessions[0].query).toBe("Edited question");
    expect(storedSessions[0].heading).toBe("Edited heading");

    useLibraryStore.getState().setSearchQuery("edited");
    const persistedSearchState = JSON.parse(
      window.localStorage.getItem("nubagent-library-storage"),
    );
    expect(persistedSearchState.state.searchQuery).toBe("edited");
  });

  it("builds root-based session share URLs and parses the shared session id back out", () => {
    const {
      buildSessionShareUrl,
      getSharedSessionIdFromLocation,
    } = require("../lib/library");

    const shareUrl = buildSessionShareUrl("session-42", {
      origin: "https://nubagent.vercel.app",
      pathname: "/docs",
    });

    expect(shareUrl).toBe("https://nubagent.vercel.app/?session=session-42");
    expect(
      getSharedSessionIdFromLocation({ search: "?session=session-42" }),
    ).toBe("session-42");
  });
});
