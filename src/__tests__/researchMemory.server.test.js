import { describe, expect, it } from "vitest";
import { loadCommonJsModule } from "./loadCommonJsModule.js";

const { buildResearchMemoryEntriesFromRun, normalizeMemoryEntry } = loadCommonJsModule(
  "/root/.bot/.downloads/nubagent/lib/research-memory.js",
  {
    "./db": {
      getPool: () => {
        throw new Error("db access not expected in this test");
      },
    },
  },
);

describe("research memory indexing", () => {
  it("stores graph relations and run artifacts for proactive reuse", () => {
    const entries = buildResearchMemoryEntriesFromRun({
      runId: "research-run-memory-1",
      query: "should we adopt retrieval caching for our agent stack",
      status: "complete",
      plan: {
        domain: { id: "cs_ml", label: "CS + ML" },
        scope: { id: "decision_support", label: "Decision Support" },
        outputMode: { id: "decision_brief", label: "Decision Brief" },
        queryMatrix: {
          counterHypotheses: ["Caching may hurt freshness if invalidation is weak."],
        },
      },
      final: {
        heading: "Retrieval Caching Decision",
        answer: "Adopt caching with explicit invalidation safeguards.",
      },
      result: {
        decision: {
          decision: "Adopt retrieval caching",
          expected_outcome: "Lower repeated lookup latency.",
        },
      },
      outputs: {
        markdown: "# Retrieval Caching Decision",
        dataset_csv: "type,source_title",
      },
      checkpoints: {
        draft: { id: "draft" },
        decision: { id: "decision" },
      },
      tribunal: {
        targeted_dimension: "contradiction_handling",
      },
      convergence: {
        residual_uncertainty: 0.22,
      },
      postmortem: {
        prompt_patches_applied: ["Keep freshness caveats explicit."],
        phases_that_degraded_score: ["Phase 5 - low contradiction handling"],
        final_score_delta: 11,
      },
      structuredData: {
        paperInventory: [
          {
            title: "Caching Benchmark",
            url: "https://example.com/paper",
          },
        ],
        repositories: [
          {
            url: "https://github.com/example/caching",
            notes: "Includes invalidation benchmarks.",
          },
        ],
        supplementary: [
          {
            title: "Appendix A",
            url: "https://example.com/appendix",
          },
        ],
        concepts: [
          {
            name: "retrieval caching",
            ontology: "ACM CCS",
            canonicalId: "acm:retrieval-caching",
            relations: [{ type: "depends_on", target: "cache invalidation", weight: 0.9 }],
          },
          {
            name: "cache invalidation",
            ontology: "ACM CCS",
            canonicalId: "acm:cache-invalidation",
          },
        ],
        epistemicClaims: [
          {
            claim: "Caching reduces repeated lookup latency but requires strong invalidation.",
            confidence: 0.84,
            evidence_weight: 0.8,
            contradiction_score: 0.24,
          },
        ],
        abstractions: [
          {
            summary: "Latency gains persist when invalidation quality remains high.",
          },
        ],
      },
    });

    const episode = entries.find((entry) => entry.layer === "episode");
    const concept = entries.find((entry) => entry.layer === "concept" && entry.title === "retrieval caching");
    const postmortem = entries.find((entry) => entry.layer === "postmortem");

    expect(episode?.metadata?.relatedConcepts).toEqual(expect.arrayContaining([
      "retrieval caching",
      "cache invalidation",
    ]));
    expect(concept?.metadata?.relations).toEqual(expect.arrayContaining([
      expect.objectContaining({ type: "depends_on", target: "cache invalidation" }),
    ]));
    expect(postmortem?.metadata?.promptPatches).toEqual(expect.arrayContaining([
      "Keep freshness caveats explicit.",
    ]));
  });

  it("normalizes relation metadata into searchable memory entries", () => {
    const entry = normalizeMemoryEntry({
      layer: "episode",
      title: "Decision run",
      content: "Adopt retrieval caching.",
      metadata: {
        query: "should we adopt retrieval caching",
        relations: [
          { type: "depends_on", target: "cache invalidation" },
          { type: "supports", target: "latency reduction" },
        ],
      },
    });

    expect(entry?.metadata?.relations).toEqual(expect.arrayContaining([
      expect.objectContaining({ type: "depends_on", target: "cache invalidation" }),
      expect.objectContaining({ type: "supports", target: "latency reduction" }),
    ]));
    expect(entry?.searchText).toContain("cache invalidation");
    expect(entry?.searchText).toContain("latency reduction");
  });
});
