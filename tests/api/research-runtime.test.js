import { beforeEach, describe, expect, it, vi } from "vitest";
import { loadCommonJsModule } from "../../src/__tests__/loadCommonJsModule.js";

const installRuntimeMocks = ({
  pendingControls = [],
  omitFinalCitations = false,
  sourceMeshOverride = null,
  evidenceEntriesOverride = null,
} = {}) => {
  const runStore = new Map();
  const queuedControls = [...pendingControls];

  const runLiteHostChat = vi.fn(async ({ messages = [] }) => {
    const system = String(messages[0]?.content || "");

    if (system.includes("PositionMapper")) {
      return {
        reply: {
          content: JSON.stringify({
            dominant_position: "Caching improves latency.",
            counter_position: "Freshness can degrade.",
            key_claims: [{ claim: "Caching reduced latency in most evaluations.", citations: [1] }],
            quantitative_signals: [{ metric: "latency", direction: "down" }],
            open_gaps: ["Freshness impact is variable."],
          }),
        },
      };
    }

    if (system.includes("ThesisAgent")) {
      return { reply: { content: "Caching improves latency and stability [1]." } };
    }

    if (system.includes("AntithesisAgent")) {
      return { reply: { content: "Caching can weaken freshness if invalidation is poor [1]." } };
    }

    if (system.includes("SynthesisMediator")) {
      return {
        reply: {
          content: omitFinalCitations
            ? "# Decision Draft\nCaching improves latency, but freshness caveats remain.\n\n## Residual uncertainty\nFreshness impact depends on invalidation quality."
            : "# Decision Draft\nCaching improves latency, but freshness caveats remain [1].\n\n## Residual uncertainty\nFreshness impact depends on invalidation quality [1].\n\n## Sources used\n[1]",
        },
      };
    }

    if (system.includes("Recursive Self-Improvement Loop")) {
      return {
        reply: {
          content: JSON.stringify({
            internal_consistency: 0.9,
            coverage: 0.88,
            user_goal_alignment: 0.9,
            targeted_dimension: "coverage",
            rewrite_brief: "Keep caveats explicit.",
          }),
        },
      };
    }

    if (system.includes("ClaimVerifier")) {
      return { reply: { content: JSON.stringify({ score: 0.92, issues: [], rewrite_brief: "" }) } };
    }

    if (system.includes("CitationVerifier")) {
      return { reply: { content: JSON.stringify({ score: 0.91, issues: [], rewrite_brief: "" }) } };
    }

    if (system.includes("ContradictionVerifier")) {
      return { reply: { content: JSON.stringify({ score: 0.86, issues: [], rewrite_brief: "" }) } };
    }

    if (system.includes("UncertaintyVerifier")) {
      return { reply: { content: JSON.stringify({ score: 0.89, issues: [], rewrite_brief: "" }) } };
    }

    if (system.includes("Decision Intelligence Layer")) {
      return {
        reply: {
          content: JSON.stringify({
            decision: "Adopt caching with explicit invalidation safeguards",
            expected_outcome: "Lower latency without hiding freshness risk.",
            risk_profile: { technical: 0.28, epistemic: 0.18 },
            confidence: 0.82,
            reversibility: "medium",
            rationale: "Evidence supports latency gains, but invalidation quality is decisive.",
            recommended_actions: ["Ship a small rollout", "Track stale-hit rate"],
          }),
        },
      };
    }

    return { reply: { content: "{}" } };
  });

  const mocks = {
    "./litehost-chat": { runLiteHostChat },
    "./rag": {
      rankEvidenceEntriesForQuery: vi.fn((query, entries) => entries),
      getSourceDomain: vi.fn((url) => new URL(url).hostname),
      extractQueryTerms: vi.fn((text) => String(text || "").toLowerCase().split(/[^a-z0-9]+/).filter(Boolean).slice(0, 24)),
    },
    "./research-sources": {
      searchResearchSources: vi.fn(async () => ({
      sources: sourceMeshOverride?.sources || [
        {
          title: "Caching Paper",
          url: "https://example.com/paper",
          tier: "core",
          fetchMode: "full",
          citationIndex: 1,
        },
      ],
      providersUsed: sourceMeshOverride?.providersUsed || ["mock"],
      providerErrors: sourceMeshOverride?.providerErrors || [],
    })),
    },
    "./research-extraction": {
      fetchTieredEvidence: vi.fn(async (sources) => (
        evidenceEntriesOverride || [
          {
            source: { ...sources[0], citationIndex: 1 },
            content: "Caching reduced latency in controlled evaluations while freshness depended on invalidation quality.",
            evidenceBlock: "[1] Caching reduced latency in controlled evaluations while freshness depended on invalidation quality.",
          },
        ]
      )),
      extractResearchArtifacts: vi.fn(async () => ({
      claims: [{ type: "effect_size", metric: "latency", value: -0.3, sampleSizeFlag: false }],
      repositories: [],
      supplementary: [],
      concepts: [{ name: "Caching" }],
      evidencePyramid: [],
      metaAnalysis: { combined_effect_size: 0.42, i_squared: 12, model: "random_effects" },
      statisticalVerification: { verified: true },
      safety: { manipulationCount: 0, dualUseCount: 0 },
    })),
    },
    "./research-memory": {
      loadResearchRun: vi.fn(async (runId) => runStore.get(runId) || null),
      saveResearchRun: vi.fn(async (run) => {
      const saved = {
        ...run,
        state: run.state && typeof run.state === "object" ? { ...run.state } : {},
        checkpoints: { ...(run.checkpoints || {}) },
        events: [...(run.events || [])],
      };
      runStore.set(saved.id || saved.runId, saved);
      return saved;
    }),
      indexResearchRun: vi.fn(async () => true),
      updateResearchMemoryFromRun: vi.fn(async () => true),
      findRelevantResearchContext: vi.fn(async () => ({})),
      formatResearchContext: vi.fn(() => ""),
      pullPendingResearchControls: vi.fn(async () => {
      if (!queuedControls.length) return [];
      return [queuedControls.shift()];
    }),
      markResearchControlsApplied: vi.fn(async () => true),
    },
  };

  const runtime = loadCommonJsModule(
    "/root/.bot/.downloads/nubagent/lib/research-runtime.js",
    mocks,
  );
  return { runtime, runLiteHostChat };
};

describe("research runtime orchestration", () => {
  beforeEach(() => {
    vi.resetModules();
    vi.clearAllMocks();
  });

  it("pauses at the decision checkpoint and resumes to a final report with verifier output", async () => {
    const { runtime } = installRuntimeMocks();

    const paused = await runtime.runResearch({
      query: "should we adopt response caching for the research agent",
      forcedOutputMode: "decision_brief",
      stopAfterCheckpoint: "decision",
    });

    expect(paused.status).toBe("awaiting_input");
    expect(paused.awaitingCheckpoint).toBe("decision");
    expect(paused.checkpoints.decision.decision).toMatchObject({
      decision: "Adopt caching with explicit invalidation safeguards",
      confidence: 0.82,
    });

    const resumed = await runtime.resumeResearch({
      runId: paused.id,
    });

    expect(resumed.status).toBe("complete");
    expect(resumed.result.decision).toMatchObject({
      decision: "Adopt caching with explicit invalidation safeguards",
      reversibility: "medium",
    });
    expect(resumed.verifierSummary.dimensions.claim_support).toBeGreaterThan(0.8);
    expect(resumed.verifierSummary.dimensions.citation_integrity).toBeGreaterThan(0.8);
  });

  it("recompiles the DAG when a force_mode control is queued during execution", async () => {
    const { runtime } = installRuntimeMocks({
      pendingControls: [{ type: "force_mode", mode: "decision_brief" }],
    });

    const run = await runtime.runResearch({
      query: "compare retrieval caching strategies in research systems",
    });

    expect(run.status).toBe("complete");
    expect(run.plan.outputMode.id).toBe("decision_brief");
    expect(run.result.decision).toBeTruthy();
    expect(run.events.some((event) => (
      event.type === "control_applied"
      && event.restartNode === "dialecticalSynthesisEngine"
    ))).toBe(true);
  });

  it("falls back to tiered sources when the final draft has no valid citations", async () => {
    const { runtime } = installRuntimeMocks({
      omitFinalCitations: true,
    });

    const run = await runtime.runResearch({
      query: "should we adopt response caching for the research agent",
    });

    expect(run.status).toBe("complete");
    expect(run.result.sources).toHaveLength(1);
    expect(run.result.sources[0]).toMatchObject({
      title: "Caching Paper",
      url: "https://example.com/paper",
      tier: "core",
    });
    expect(run.result.sourceSelection).toMatchObject({
      mode: "fallback",
      citedNumbers: [],
      totalTieredSources: 1,
    });
  });

  it("emits an explicit no-sources report when retrieval produced no grounded evidence", async () => {
    const { runtime } = installRuntimeMocks({
      sourceMeshOverride: {
        sources: [],
        providersUsed: [],
        providerErrors: [
          "web_search DuckDuckGo returned bot challenge",
          "openalex HTTP 429",
        ],
      },
      evidenceEntriesOverride: [],
    });

    const run = await runtime.runResearch({
      query: "how does quantum computing threaten modern encryption",
      searchProviderKeys: {
        tavily: "tvly-preview-key-1234567890",
      },
    });

    expect(run.status).toBe("complete");
    expect(run.result.sources).toEqual([]);
    expect(run.result.sourceSelection).toMatchObject({
      mode: "no_grounded_sources",
      totalTieredSources: 0,
    });
    expect(run.result.finalText).toContain("# No Sources Retrieved");
    expect(run.result.finalText).toContain("Configured search providers for this run: Tavily.");
    expect(run.result.finalText).toContain("DuckDuckGo returned bot challenge");
    expect(run.result.decision).toBeNull();
    expect(run.result.slides).toBe("");
    expect(run.result.datasetCsv).toBe("");
    expect(run.requestOptions.searchProviders).toEqual(["Tavily"]);
  });
});
