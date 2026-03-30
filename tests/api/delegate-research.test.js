import { describe, expect, it, vi } from "vitest";
import { loadCommonJsModule } from "../../src/__tests__/loadCommonJsModule.js";

describe("delegate_research tool", () => {
  it("forwards scope, provider keys, and round-robin research settings into the runtime", async () => {
    const runResearch = vi.fn(async () => ({
      id: "research-run-1",
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

    expect(runResearch).toHaveBeenCalledWith(expect.objectContaining({
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
    }));

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
});
