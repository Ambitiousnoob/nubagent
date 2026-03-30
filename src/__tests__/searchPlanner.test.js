import { describe, expect, it } from "vitest";
import {
  buildSearchQueries,
  detectQuerySignals,
} from "../lib/searchPlanner.js";

describe("searchPlanner", () => {
  it("detects research and freshness intent", () => {
    const signals = detectQuerySignals(
      "latest RRL on transformer evaluation 2026",
    );
    expect(signals.researchIntent).toBe(true);
    expect(signals.currentIntent).toBe(true);
  });

  it("keeps operator-heavy queries focused", () => {
    const variants = buildSearchQueries("site:openai.com function calling", {
      maxQueries: 4,
    });
    expect(variants[0]).toBe("site:openai.com function calling");
    expect(variants).toContain("site:openai.com function calling evidence");
    expect(variants.length).toBeLessThanOrEqual(4);
  });

  it("adds research-oriented variants for academic queries", () => {
    const variants = buildSearchQueries("RRL on vector databases", {
      maxQueries: 4,
    });
    expect(variants).toContain(
      "RRL on vector databases peer reviewed research",
    );
    expect(variants).toContain("RRL on vector databases evidence");
  });

  it("adds freshness and official-source variants for current topics", () => {
    const variants = buildSearchQueries("latest OpenAI API pricing", {
      maxQueries: 4,
    });
    expect(variants).toContain("latest OpenAI API pricing latest");
    expect(variants).toContain("latest OpenAI API pricing official source");
  });

  it("preserves multiple intent lanes for mixed queries", () => {
    const variants = buildSearchQueries(
      "latest API benchmark for vector databases",
      { maxQueries: 6 },
    );
    expect(variants).toContain(
      "latest API benchmark for vector databases official documentation",
    );
    expect(variants).toContain(
      "latest API benchmark for vector databases benchmark analysis",
    );
    expect(variants).toContain(
      "latest API benchmark for vector databases latest",
    );
  });
});
