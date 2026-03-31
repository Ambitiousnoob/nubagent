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
  it("collapses all app views to the docs experience", () => {
    expect(normalizeAppView("docs")).toBe("docs");
    expect(normalizeAppView("LIBRARY")).toBe("docs");
    expect(normalizeAppView("unknown")).toBe("docs");
  });

  it("maps legacy browser pathnames back to the single docs route", () => {
    expect(getAppViewFromPathname("/")).toBe("docs");
    expect(getAppViewFromPathname("/research")).toBe("docs");
    expect(getAppViewFromPathname("/docs/")).toBe("docs");
    expect(getAppViewFromPathname("/library")).toBe("docs");
    expect(getAppViewFromLocation({ pathname: "/missing", search: "" })).toBe("docs");
  });

  it("builds stable root urls for the docs site", () => {
    expect(buildAppViewHref("chat")).toBe("/");
    expect(buildAppViewHref("docs")).toBe("/");
    expect(
      buildAppViewUrl("chat", {
        locationLike: {
          origin: "https://nubagent.vercel.app",
          pathname: "/docs",
        },
      }),
    ).toBe("https://nubagent.vercel.app/");
    expect(
      buildAppViewUrl("docs", {
        locationLike: { origin: "https://nubagent.vercel.app", pathname: "/" },
      }),
    ).toBe("https://nubagent.vercel.app/");
  });

  it("writes the canonical docs root into browser history", () => {
    const pushState = vi.fn();
    const replaceState = vi.fn();

    writeAppViewToHistory("library", {
      historyLike: { pushState, replaceState },
      locationLike: {
        origin: "https://nubagent.vercel.app",
        href: "https://nubagent.vercel.app/docs",
      },
    });
    expect(pushState).toHaveBeenCalledWith({}, "", "https://nubagent.vercel.app/");

    writeAppViewToHistory("docs", {
      replace: true,
      historyLike: { pushState, replaceState },
      locationLike: {
        origin: "https://nubagent.vercel.app",
        href: "https://nubagent.vercel.app/chat",
      },
    });
    expect(replaceState).toHaveBeenCalledWith({}, "", "https://nubagent.vercel.app/");
  });
});
