import { describe, expect, it } from "vitest";
import {
  buildAttributedSourcesFromEvidence,
  extractCitationNumbers,
  findSourceForCitation,
  getDisplaySourceNumber,
} from "../lib/citations.js";

describe("citation utilities", () => {
  it("extracts unique citation numbers in encounter order", () => {
    expect(extractCitationNumbers("A [7] and [2] and [7] again.")).toEqual([
      7, 2,
    ]);
  });

  it("resolves citations by citationIndex instead of array position", () => {
    const sources = [
      { title: "Source Two", url: "https://two.example", citationIndex: 2 },
      { title: "Source Seven", url: "https://seven.example", citationIndex: 7 },
    ];

    expect(findSourceForCitation(7, sources)?.url).toBe(
      "https://seven.example",
    );
    expect(findSourceForCitation(2, sources)?.url).toBe("https://two.example");
    expect(findSourceForCitation(5, sources)).toBeNull();
  });

  it("builds attributed sources sorted by citation index for sparse citations", () => {
    const evidenceEntries = [
      {
        source: {
          title: "Source Seven",
          url: "https://seven.example",
          citationIndex: 7,
        },
        content: "<!-- meta -->\nSeven content",
      },
      {
        source: {
          title: "Source Two",
          url: "https://two.example",
          citationIndex: 2,
        },
        content: "<!-- meta -->\nTwo content",
      },
      {
        source: {
          title: "Source Nine",
          url: "https://nine.example",
          citationIndex: 9,
        },
        content: "<!-- meta -->\nNine content",
      },
    ];

    const sources = buildAttributedSourcesFromEvidence(
      "Findings [7] and [2] with a repeat [7].",
      evidenceEntries,
      10,
    );

    expect(sources.map((source) => source.citationIndex)).toEqual([2, 7]);
    expect(getDisplaySourceNumber(sources[0], 0)).toBe(2);
    expect(getDisplaySourceNumber(sources[1], 1)).toBe(7);
  });
});
