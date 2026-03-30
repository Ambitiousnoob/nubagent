import { describe, expect, it, vi } from "vitest";
import { loadCommonJsModule } from "../../src/__tests__/loadCommonJsModule.js";

describe("delegate_research tool", () => {
  it("forwards scope, provider keys, and round-robin research settings into the runtime", async () => {
    const runResearch = vi.fn(async () => ({
      id: "research-run-1",
      status: "complete",
      plan: {
        outputMode: {
          id: "state_of_the_field",
          label: "State of the Field",
        },
      },
      result: {
        heading: "Research Answer",
        finalText: "Quantum threats to encryption summary.",
        sourceSelection: {
          mode: "cited_sources",
        },
        sources: [
          {
            citationIndex: 1,
            title: "Quantum Paper",
            url: "https://example.com/quantum",
            tier: "core",
          },
        ],
      },
      tribunal: {
        final_score: 0.92,
      },
      convergence: {
        stability_score: 0.91,
      },
      verifierSummary: {
        claimVerifier: "pass",
      },
    }));

    const { handler } = loadCommonJsModule(
      "/root/.bot/.downloads/nubagent/lib/tools/delegate_research.js",
      {
        "../research-runtime": {
          runResearch,
          resumeResearch: vi.fn(),
        },
      },
    );

    const result = await handler(
      {
        query: "How does quantum computing threaten modern encryption?",
        depth_preference: "deep",
        output_mode: "state_of_the_field",
        refinement_budget: 4,
        max_queries: 8,
      },
      {
        delegationContext: {
          scopeKey: "scope:test-chat",
        },
        requestBody: {
          searchProviderKeys: {
            tavily: "tvly-test-key-1234567890",
          },
          researchProvider: "openrouter",
          researchModel: "nvidia/nemotron-3-super-120b-a12b:free",
          researchModelChain: [
            "nvidia/nemotron-3-super-120b-a12b:free",
            "meta-llama/llama-3.3-70b-instruct:free",
          ],
          researchRoundRobin: true,
          researchProviderKeys: {
            openrouter: "sk-or-v1-1234567890abcdefghijklmnop",
          },
        },
      },
    );

    expect(runResearch).toHaveBeenCalledWith(
      expect.objectContaining({
        query: "How does quantum computing threaten modern encryption?",
        scopeKey: "scope:test-chat",
        depthPreference: "deep",
        forcedOutputMode: "state_of_the_field",
        refinementBudget: 4,
        maxQueries: 8,
        searchProviderKeys: {
          tavily: "tvly-test-key-1234567890",
        },
        researchProvider: "openrouter",
        researchModel: "nvidia/nemotron-3-super-120b-a12b:free",
        researchModelChain: [
          "nvidia/nemotron-3-super-120b-a12b:free",
          "meta-llama/llama-3.3-70b-instruct:free",
        ],
        researchRoundRobin: true,
        researchProviderKeys: {
          openrouter: "sk-or-v1-1234567890abcdefghijklmnop",
        },
      }),
    );

    expect(JSON.parse(result)).toMatchObject({
      delegated: true,
      runId: "research-run-1",
      heading: "Research Answer",
      outputMode: {
        id: "state_of_the_field",
        label: "State of the Field",
      },
      noGroundedSources: false,
      sourceCount: 1,
      sources: [
        {
          citationIndex: 1,
          title: "Quantum Paper",
          url: "https://example.com/quantum",
          tier: "core",
        },
      ],
    });
  });

  it("resumes delegated research until a complete run is available", async () => {
    const runResearch = vi.fn(async () => ({
      id: "research-run-2",
      runId: "research-run-2",
      status: "in_progress",
    }));
    const resumeResearch = vi.fn(async () => ({
      id: "research-run-2",
      runId: "research-run-2",
      status: "complete",
      plan: {
        outputMode: {
          id: "gap_analysis",
          label: "Gap Analysis",
        },
      },
      result: {
        heading: "Research Answer",
        finalText: "Completed after resume.",
        sourceSelection: {
          mode: "cited_sources",
        },
        sources: [
          {
            citationIndex: 1,
            title: "Gap Paper",
            url: "https://example.com/gap",
            tier: "supporting",
          },
        ],
      },
    }));

    const { handler } = loadCommonJsModule(
      "/root/.bot/.downloads/nubagent/lib/tools/delegate_research.js",
      {
        "../research-runtime": {
          runResearch,
          resumeResearch,
        },
      },
    );

    const result = await handler(
      {
        query: "Where are the evidence gaps?",
      },
      {
        delegationContext: {
          scopeKey: "scope:test-chat",
        },
        requestBody: {
          researchProvider: "openrouter",
          researchModel: "nvidia/nemotron-3-super-120b-a12b:free",
        },
      },
    );

    expect(resumeResearch).toHaveBeenCalledWith(
      expect.objectContaining({
        runId: "research-run-2",
        scopeKey: "scope:test-chat",
        researchProvider: "openrouter",
        researchModel: "nvidia/nemotron-3-super-120b-a12b:free",
      }),
    );
    expect(JSON.parse(result)).toMatchObject({
      delegated: true,
      runId: "research-run-2",
      outputMode: {
        id: "gap_analysis",
        label: "Gap Analysis",
      },
      sourceCount: 1,
      sources: [
        {
          citationIndex: 1,
          title: "Gap Paper",
          url: "https://example.com/gap",
          tier: "supporting",
        },
      ],
    });
  });

  it("keeps resuming delegated research until the runtime reaches a complete state", async () => {
    const runResearch = vi.fn(async () => ({
      id: "research-run-2",
      status: "in_progress",
      plan: {
        outputMode: {
          id: "state_of_the_field",
          label: "State of the Field",
        },
      },
      result: {
        heading: "Research Answer",
        finalText: "Draft pass.",
        sourceSelection: {
          mode: "cited_sources",
        },
        sources: [],
      },
    }));
    const resumeResearch = vi.fn(async () => ({
      id: "research-run-2",
      status: "complete",
      plan: {
        outputMode: {
          id: "state_of_the_field",
          label: "State of the Field",
        },
      },
      result: {
        heading: "Research Answer",
        finalText: "Final completed answer.",
        sourceSelection: {
          mode: "cited_sources",
        },
        sources: [
          {
            citationIndex: 1,
            title: "Completed Source",
            url: "https://example.com/completed-source",
            tier: "supporting",
          },
        ],
      },
    }));

    const { handler } = loadCommonJsModule(
      "/root/.bot/.downloads/nubagent/lib/tools/delegate_research.js",
      {
        "../research-runtime": {
          runResearch,
          resumeResearch,
        },
      },
    );

    const result = await handler(
      {
        query: "Finish the delegated run before replying.",
      },
      {
        delegationContext: {
          scopeKey: "scope:resume-chat",
        },
        requestBody: {},
      },
    );

    expect(runResearch).toHaveBeenCalledTimes(1);
    expect(resumeResearch).toHaveBeenCalledWith(
      expect.objectContaining({
        runId: "research-run-2",
        scopeKey: "scope:resume-chat",
      }),
    );
    expect(JSON.parse(result)).toMatchObject({
      delegated: true,
      runId: "research-run-2",
      sourceCount: 1,
      sources: [
        {
          title: "Completed Source",
          url: "https://example.com/completed-source",
          tier: "supporting",
        },
      ],
    });
  });

  it("throws when delegated research never reaches a complete state", async () => {
    const runResearch = vi.fn(async () => ({
      id: "research-run-3",
      runId: "research-run-3",
      status: "in_progress",
    }));
    const resumeResearch = vi.fn(async () => ({
      id: "research-run-3",
      runId: "research-run-3",
      status: "in_progress",
    }));

    const { handler } = loadCommonJsModule(
      "/root/.bot/.downloads/nubagent/lib/tools/delegate_research.js",
      {
        "../research-runtime": {
          runResearch,
          resumeResearch,
        },
      },
    );

    await expect(
      handler(
        {
          query: "Keep researching until complete.",
        },
        {
          delegationContext: {
            scopeKey: "scope:test-chat",
          },
        },
      ),
    ).rejects.toThrow(
      /delegated research failed \((delegated research stopped making progress|delegated research did not reach a complete state)/i,
    );
    expect(resumeResearch).toHaveBeenCalledTimes(2);
  });

  it("waits through longer multi-pass research runs before returning control to the main agent", async () => {
    const runResearch = vi.fn(async () => ({
      id: "research-run-4",
      runId: "research-run-4",
      status: "in_progress",
      completedNodes: ["cognitiveCommandLayer"],
    }));
    const resumeResearch = vi.fn();
    const inProgressPasses = Array.from({ length: 13 }, (_, index) => ({
      id: "research-run-4",
      runId: "research-run-4",
      status: "in_progress",
      completedNodes: Array.from({ length: index + 2 }, (_, nodeIndex) => `node-${nodeIndex + 1}`),
      updatedAt: `2026-03-30T00:00:${String(index).padStart(2, "0")}Z`,
    }));
    for (const pass of inProgressPasses) {
      resumeResearch.mockResolvedValueOnce(pass);
    }
    resumeResearch.mockResolvedValueOnce({
      id: "research-run-4",
      runId: "research-run-4",
      status: "complete",
      completedNodes: [
        "cognitiveCommandLayer",
        "intelligentCrawlerMesh",
        "deepComprehensionEngine",
        "dialecticalSynthesisEngine",
        "adaptiveDeliveryHub",
      ],
      plan: {
        outputMode: {
          id: "state_of_the_field",
          label: "State of the Field",
        },
      },
      result: {
        heading: "Research Answer",
        finalText: "Completed after multiple resume passes.",
        sourceSelection: {
          mode: "cited_sources",
        },
        sources: [
          {
            citationIndex: 1,
            title: "Long Run Source",
            url: "https://example.com/long-run",
            tier: "core",
          },
        ],
      },
    });

    const { handler } = loadCommonJsModule(
      "/root/.bot/.downloads/nubagent/lib/tools/delegate_research.js",
      {
        "../research-runtime": {
          runResearch,
          resumeResearch,
        },
      },
    );

    const result = await handler(
      {
        query: "Finish every research pass before replying.",
      },
      {
        delegationContext: {
          scopeKey: "scope:long-run-chat",
        },
        requestBody: {},
      },
    );

    expect(runResearch).toHaveBeenCalledTimes(1);
    expect(resumeResearch).toHaveBeenCalledTimes(14);
    expect(JSON.parse(result)).toMatchObject({
      delegated: true,
      runId: "research-run-4",
      sourceCount: 1,
      sources: [
        {
          title: "Long Run Source",
          url: "https://example.com/long-run",
          tier: "core",
        },
      ],
    });
  });
});
