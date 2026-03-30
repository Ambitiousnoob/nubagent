import { describe, it, expect } from "vitest";
import { mergeSourcesByCanonicalUrl, rerankSourcesForQuery } from "./rag.js";

describe("mergeSourcesByCanonicalUrl", () => {
  it("preserves provider and query-variant agreement across duplicates", () => {
    const merged = mergeSourcesByCanonicalUrl([
      {
        title: "Quarterly Report",
        url: "https://example.com/report?utm_source=newsletter",
        description: "Provider one",
        source: "duckduckgo",
        providerCount: 1,
        queryVariant: "acme quarterly report",
      },
      {
        title: "Quarterly Report",
        url: "https://example.com/report",
        description: "Provider two has the longer description",
        source: "tavily",
        providerCount: 1,
        queryVariant: "acme quarterly report latest",
      },
    ]);

    expect(merged).toHaveLength(1);
    expect(merged[0].url).toBe("https://example.com/report");
    expect(merged[0].providerCount).toBe(2);
    expect(merged[0].providers).toEqual(
      expect.arrayContaining(["duckduckgo", "tavily"]),
    );
    expect(merged[0].queryHitCount).toBe(2);
    expect(merged[0].queryVariants).toEqual(
      expect.arrayContaining([
        "acme quarterly report",
        "acme quarterly report latest",
      ]),
    );
  });
});

describe("rerankSourcesForQuery", () => {
  it("prefers sources backed by multiple providers and query variants", () => {
    const ranked = rerankSourcesForQuery("acme quarterly report latest", [
      {
        title: "Acme results overview",
        url: "https://blog.example.com/acme-results",
        description: "Single-provider summary post",
        providerCount: 1,
        queryHitCount: 1,
      },
      {
        title: "Acme quarterly report",
        url: "https://investor.acme.com/quarterly-report",
        description: "Official filing with earnings details",
        providerCount: 3,
        queryHitCount: 2,
        date: "2026-03-28",
      },
    ]);

    expect(ranked[0].url).toBe("https://investor.acme.com/quarterly-report");
  });
});
