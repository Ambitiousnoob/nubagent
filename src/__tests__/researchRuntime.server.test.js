import { describe, expect, it } from "vitest";
import { loadCommonJsModule } from "./loadCommonJsModule.js";

const planModule = loadCommonJsModule(
  "/root/.bot/.downloads/nubagent/lib/research-plan.js",
);
const { compileResearchPlan } = planModule;
const { buildExportPayload, serializeResearchRun } = loadCommonJsModule(
  "/root/.bot/.downloads/nubagent/lib/research-runtime.js",
  {
    "./litehost-chat": {
      runLiteHostChat: async () => ({ reply: { content: "" } }),
    },
    "./rag": {
      rankEvidenceEntriesForQuery: (_query, entries = []) => entries,
      getSourceDomain: () => "example.com",
    },
    "./research-plan": planModule,
    "./research-sources": {
      searchResearchSources: async () => ({ sources: [], providersUsed: [] }),
    },
    "./research-extraction": {
      fetchTieredEvidence: async () => [],
      extractResearchArtifacts: async () => ({
        claims: [],
        repositories: [],
        supplementary: [],
        concepts: [],
        evidencePyramid: [],
      }),
    },
    "./research-memory": {
      loadResearchRun: async () => null,
      saveResearchRun: async (run) => run,
      indexResearchRun: async () => {},
      updateResearchMemoryFromRun: async () => {},
      findRelevantResearchContext: async () => ({}),
      formatResearchContext: () => "",
      pullPendingResearchControls: async () => [],
    },
  },
);

const buildCompletedRun = () => {
  const plan = compileResearchPlan({
    query: "should we adopt retrieval caching for our agent stack",
    depthPreference: "deep",
    forcedOutputMode: "decision_brief",
    refinementBudget: 5,
  });

  return {
    id: "research-run-serialize-1",
    runId: "research-run-serialize-1",
    status: "complete",
    query: plan.query,
    createdAt: "2026-03-30T00:00:00.000Z",
    updatedAt: "2026-03-30T00:10:00.000Z",
    plan,
    result: {
      heading: "Retrieval Caching Decision",
      body: "Adopt retrieval caching for repeated lookups. [1]",
      finalText: "# Retrieval Caching Decision\n\nAdopt retrieval caching for repeated lookups. [1]\n\n## Sources used\n[1]",
      markdown: "# Retrieval Caching Decision\n\nAdopt retrieval caching for repeated lookups. [1]",
      slides: "1. Retrieval Caching Decision\n2. Research question",
      datasetCsv: "\"type\",\"source_title\",\"source_url\",\"value\",\"metric\",\"notes\"\n\"effect_size\",\"Caching Benchmark\",\"https://example.com/paper\",\"0.42\",\"latency_delta\",\"\"",
      sources: [
        {
          citationIndex: 1,
          title: "Caching Benchmark",
          url: "https://example.com/paper",
          tier: "core",
        },
      ],
      decision: {
        decision: "Adopt retrieval caching",
        expected_outcome: "Lower latency for repeated evidence fetches.",
        risk_profile: {
          technical: 0.22,
          epistemic: 0.18,
        },
        confidence: 0.82,
        reversibility: "high",
      },
    },
    tribunal: {
      refinement_budget: 5,
      refinement_cycles: 2,
      targeted_dimension: "coverage",
      critics: {
        internal_consistency: 0.86,
        coverage: 0.81,
        user_goal_alignment: 0.9,
      },
    },
    convergence: {
      iterations: 2,
      stability_score: 0.88,
      evidence_coverage_delta: 0.19,
      residual_uncertainty: 0.18,
      stop_condition: "residual_disagreement",
    },
    verifierSummary: {
      dimensions: {
        claim_support: 0.91,
        citation_integrity: 0.89,
        contradiction_handling: 0.86,
        uncertainty_calibration: 0.9,
      },
      targetedDimension: "contradiction_handling",
      issues: ["contradiction_handling: Freshness caveat must remain explicit."],
      rewriteBrief: "Keep the freshness caveat explicit.",
      aggregateScore: 0.89,
    },
    checkpoints: {
      draft: {
        id: "draft",
        at: "2026-03-30T00:06:00.000Z",
      },
      decision: {
        id: "decision",
        at: "2026-03-30T00:08:00.000Z",
      },
      final: {
        id: "final",
        at: "2026-03-30T00:10:00.000Z",
      },
    },
    events: [
      {
        type: "checkpoint",
        checkpoint: "draft",
        runId: "research-run-serialize-1",
      },
      {
        type: "control_applied",
        runId: "research-run-serialize-1",
        restartNode: "adversarialQueryForge",
        controls: [{ type: "go_deeper" }],
      },
    ],
    state: {
      tieredSources: [
        {
          citationIndex: 1,
          title: "Caching Benchmark",
          url: "https://example.com/paper",
          tier: "core",
          fetchMode: "full",
        },
      ],
      evidenceEntries: [
        {
          source: {
            title: "Caching Benchmark",
            url: "https://example.com/paper",
          },
          content: "Caching reduced latency by 42%.",
        },
      ],
      extraction: {
        claims: [
          {
            type: "effect_size",
            metric: "latency_delta",
            value: 0.42,
            sourceTitle: "Caching Benchmark",
            sourceUrl: "https://example.com/paper",
          },
        ],
        repositories: [],
        supplementary: [],
        concepts: [{ name: "retrieval caching" }],
        evidencePyramid: [],
      },
      claimLedger: [
        {
          claim: "Caching reduced average latency.",
          citations: [1],
          confidence: 0.84,
        },
      ],
      sourceMesh: {
        citationIntelligence: { strongestCitation: 1 },
      },
    },
  };
};

describe("research runtime serialization", () => {
  it("surfaces verifier outcomes, checkpoints, events, and exports in serialized runs", () => {
    const run = buildCompletedRun();

    const serialized = serializeResearchRun(run);

    expect(serialized.researchMeta.outputMode.id).toBe("decision_brief");
    expect(serialized.researchMeta.subagents.map((item) => item.id)).toEqual(expect.arrayContaining([
      "claimVerifier",
      "citationVerifier",
      "contradictionVerifier",
      "uncertaintyVerifier",
    ]));
    expect(serialized.researchMeta.subagents.find((item) => item.id === "claimVerifier")?.equipment).toEqual(expect.arrayContaining([
      "Claim ledger",
      "Evidence excerpts",
    ]));
    expect(serialized.final.tribunal).toEqual(run.tribunal);
    expect(serialized.final.convergence).toEqual(run.convergence);
    expect(serialized.final.verifierSummary).toEqual(run.verifierSummary);
    expect(serialized.final.claimLedger).toEqual(run.state.claimLedger);
    expect(serialized.final.decision).toEqual(run.result.decision);
    expect(serialized.structuredData.verifierSummary).toEqual(run.verifierSummary);
    expect(serialized.structuredData.decisionLayer).toEqual(run.result.decision);
    expect(serialized.final.exports.markdown).toContain("# Retrieval Caching Decision");
    expect(serialized.outputs.dataset_csv).toContain("\"effect_size\"");
    expect(serialized.checkpoints).toMatchObject({
      draft: expect.objectContaining({ id: "draft" }),
      decision: expect.objectContaining({ id: "decision" }),
      final: expect.objectContaining({ id: "final" }),
    });
    expect(serialized.events).toEqual(expect.arrayContaining([
      expect.objectContaining({ type: "checkpoint", checkpoint: "draft" }),
      expect.objectContaining({ type: "control_applied", restartNode: "adversarialQueryForge" }),
    ]));
  });

  it("builds markdown, csv, and json exports from the completed run payload", () => {
    const run = buildCompletedRun();

    expect(buildExportPayload(run, "markdown")).toEqual({
      contentType: "text/markdown; charset=utf-8",
      body: run.result.markdown,
    });
    expect(buildExportPayload(run, "dataset_csv")).toEqual({
      contentType: "text/csv; charset=utf-8",
      body: run.result.datasetCsv,
    });

    const jsonExport = buildExportPayload(run, "json");

    expect(jsonExport.contentType).toBe("application/json; charset=utf-8");
    expect(JSON.parse(jsonExport.body)).toMatchObject({
      runId: "research-run-serialize-1",
      final: {
        decision: {
          reversibility: "high",
        },
      },
      checkpoints: {
        draft: {
          id: "draft",
        },
      },
    });
  });
});
