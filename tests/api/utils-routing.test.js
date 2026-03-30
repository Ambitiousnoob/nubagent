import { describe, expect, it } from "vitest";

const {
  parseRequestPathname,
  resolveUtilsAction,
} = require("../../lib/utils-routing.cjs");

describe("utils alias routing", () => {
  it("parses raw request paths without query strings", () => {
    expect(parseRequestPathname("/api/messenger?hub.challenge=123")).toBe(
      "/api/messenger",
    );
  });

  it("routes /api/health to the health action without ?action", () => {
    expect(
      resolveUtilsAction({
        method: "GET",
        url: "/api/health",
        query: {},
        headers: {},
      }),
    ).toBe("health");
  });

  it("routes /api/messenger to the messenger action without ?action", () => {
    expect(
      resolveUtilsAction({
        method: "GET",
        url: "/api/messenger?hub.mode=subscribe&hub.verify_token=secret&hub.challenge=12345",
        query: {
          "hub.mode": "subscribe",
          "hub.verify_token": "secret",
          "hub.challenge": "12345",
        },
        headers: {},
      }),
    ).toBe("messenger");
  });
});
