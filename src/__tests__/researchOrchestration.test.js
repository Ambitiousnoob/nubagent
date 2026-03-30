import { describe, expect, it } from "vitest";
import { compileResearchPlan } from "../lib/researchOrchestration.js";

describe("client research orchestration plan", () => {
  it("respects explicit depth, forced output mode, and refinement budget for decision-oriented runs", () => {
    const plan = compileResearchPlan({
      query: "should we adopt retrieval caching for our agent stack",
      depthPreference: "deep",
      forcedOutputMode: "decision_brief",
      refinementBudget: 5,
    });

    expect(plan.depthPreference).toBe("deep");
    expect(plan.outputMode.id).toBe("decision_brief");
    expect(plan.refinementBudget).toBe(5);
    expect(plan.dag.nodes.map((node) => node.id)).toContain("decisionIntelligenceLayer");

    const nodeById = new Map(plan.dag.nodes.map((node) => [node.id, node]));
    expect(nodeById.get("decisionIntelligenceLayer")?.dependsOn).toEqual(["recursiveSelfImprovementLoop"]);
    expect(nodeById.get("adaptiveDeliveryHub")?.dependsOn).toEqual(["decisionIntelligenceLayer"]);
    expect(plan.subagents.map((item) => item.id)).toEqual(expect.arrayContaining([
      "decisionIntelligenceLayer",
      "adaptiveDeliveryHub",
    ]));
  });

  it("allocates the verifier swarm in the compiled subagent roster", () => {
    const plan = compileResearchPlan({
      query: "compare the evidence for two RAG retrieval strategies",
      depthPreference: "balanced",
    });

    const ids = plan.subagents.map((item) => item.id);
    expect(ids).toContain("claimVerifier");
    expect(ids).toContain("citationVerifier");
    expect(ids).toContain("contradictionVerifier");
    expect(ids).toContain("uncertaintyVerifier");
  });

  it("adds a scholarly harvester and academic lanes for literature-style research runs", () => {
    const plan = compileResearchPlan({
      query: "How does quantum computing threaten modern encryption?",
      depthPreference: "balanced",
    });

    expect(plan.queryMatrix.scholarlyDiscoveryLanes).toEqual(expect.arrayContaining([
      expect.stringContaining("site:scholar.google.com"),
      expect.stringContaining("site:openalex.org"),
    ]));
    expect(plan.searchQueries).toEqual(expect.arrayContaining([
      expect.stringContaining("site:scholar.google.com"),
    ]));
    expect(plan.subagents.find((item) => item.id === "scholarlySourceHarvester")).toMatchObject({
      label: "ScholarlySourceHarvester",
      equipment: expect.arrayContaining([
        "Google Scholar-style lanes",
        "University-domain sweep",
      ]),
    });
  });

  it("keeps the decision layer out of non-decision client plans while preserving verifier details", () => {
    const plan = compileResearchPlan({
      query: "map the current state of evidence for retrieval caching",
      depthPreference: "balanced",
    });

    expect(plan.outputMode.id).toBe("state_of_the_field");
    expect(plan.dag.nodes.map((node) => node.id)).not.toContain("decisionIntelligenceLayer");
    expect(plan.subagents.map((item) => item.id)).not.toContain("decisionIntelligenceLayer");
    expect(plan.subagents.find((item) => item.id === "claimVerifier")).toMatchObject({
      detail: "4 verifier lanes active",
      equipment: expect.arrayContaining([
        "Claim ledger",
        "Evidence excerpts",
      ]),
    });
  });
});
