import { describe, expect, it, vi } from "vitest";
import {
  buildAppViewHref,
  buildAppViewUrl,
  getAppViewFromLocation,
  getAppViewFromPathname,
  normalizeAppView,
  writeAppViewToHistory,
} from "../lib/appRoutes.js";

describe("app route helpers", () => {
  it("normalizes supported app views and falls back to chat", () => {
    expect(normalizeAppView("docs")).toBe("docs");
    expect(normalizeAppView("LIBRARY")).toBe("library");
    expect(normalizeAppView("unknown")).toBe("chat");
  });

  it("maps browser pathnames back to stable app views", () => {
    expect(getAppViewFromPathname("/")).toBe("chat");
    expect(getAppViewFromPathname("/research")).toBe("chat");
    expect(getAppViewFromPathname("/docs/")).toBe("docs");
    expect(getAppViewFromPathname("/library")).toBe("library");
    expect(getAppViewFromLocation({ pathname: "/missing", search: "" })).toBe(
      "chat",
    );
  });

  it("builds stable view urls and only carries session ids on the chat route", () => {
    expect(buildAppViewHref("chat", { sessionId: "session-42" })).toBe(
      "/?session=session-42",
    );
    expect(buildAppViewHref("docs", { sessionId: "session-42" })).toBe("/docs");
    expect(
      buildAppViewUrl("chat", {
        locationLike: {
          origin: "https://nubagent.vercel.app",
          pathname: "/docs",
        },
        sessionId: "session-42",
      }),
    ).toBe("https://nubagent.vercel.app/?session=session-42");
    expect(
      buildAppViewUrl("docs", {
        locationLike: { origin: "https://nubagent.vercel.app", pathname: "/" },
        sessionId: "session-42",
      }),
    ).toBe("https://nubagent.vercel.app/docs");
  });

  it("writes canonical urls into browser history", () => {
    const pushState = vi.fn();
    const replaceState = vi.fn();

    writeAppViewToHistory("library", {
      historyLike: { pushState, replaceState },
      locationLike: {
        origin: "https://nubagent.vercel.app",
        href: "https://nubagent.vercel.app/",
      },
    });
    expect(pushState).toHaveBeenCalledWith(
      {},
      "",
      "https://nubagent.vercel.app/library",
    );

    writeAppViewToHistory("chat", {
      replace: true,
      sessionId: "session-42",
      historyLike: { pushState, replaceState },
      locationLike: {
        origin: "https://nubagent.vercel.app",
        href: "https://nubagent.vercel.app/library",
      },
    });
    expect(replaceState).toHaveBeenCalledWith(
      {},
      "",
      "https://nubagent.vercel.app/?session=session-42",
    );
  });
});
