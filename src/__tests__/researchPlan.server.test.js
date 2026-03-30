import { describe, expect, it } from "vitest";
import { loadCommonJsModule } from "./loadCommonJsModule.js";

const { compileResearchPlan, topologicalBatches } = loadCommonJsModule(
  "/root/.bot/.downloads/nubagent/lib/research-plan.js",
);

describe("server research plan contract", () => {
  it("adds the decision layer between recursive refinement and delivery for decision outputs", () => {
    const plan = compileResearchPlan({
      query: "should we adopt method X",
      depthPreference: "deep",
      forcedOutputMode: "decision_brief",
      refinementBudget: 4,
    });

    expect(plan.outputMode.id).toBe("decision_brief");
    expect(plan.refinementBudget).toBe(4);

    const ordered = topologicalBatches(plan.dag)
      .flat()
      .map((node) => node.id);

    expect(ordered).toContain("decisionIntelligenceLayer");
    expect(ordered.indexOf("decisionIntelligenceLayer")).toBeGreaterThan(ordered.indexOf("recursiveSelfImprovementLoop"));
    expect(ordered.indexOf("adaptiveDeliveryHub")).toBeGreaterThan(ordered.indexOf("decisionIntelligenceLayer"));

    const nodeById = new Map(plan.dag.nodes.map((node) => [node.id, node]));
    expect(nodeById.get("decisionIntelligenceLayer")?.dependsOn).toEqual(["recursiveSelfImprovementLoop"]);
    expect(nodeById.get("adaptiveDeliveryHub")?.dependsOn).toEqual(["decisionIntelligenceLayer"]);
  });

  it("includes the verifier swarm in the active subagent map", () => {
    const plan = compileResearchPlan({
      query: "latest evidence on replication quality in ML benchmarks",
    });

    const ids = plan.subagents.map((item) => item.id);
    expect(ids).toContain("claimVerifier");
    expect(ids).toContain("citationVerifier");
    expect(ids).toContain("contradictionVerifier");
    expect(ids).toContain("uncertaintyVerifier");
  });

  it("keeps decision-only orchestration out of non-decision research plans", () => {
    const plan = compileResearchPlan({
      query: "survey the evidence for retrieval caching in RAG systems",
      depthPreference: "balanced",
    });

    expect(plan.outputMode.id).toBe("state_of_the_field");
    expect(plan.dag.nodes.map((node) => node.id)).not.toContain("decisionIntelligenceLayer");
    expect(plan.subagents.map((item) => item.id)).not.toContain("decisionIntelligenceLayer");
    expect(plan.subagents.find((item) => item.id === "claimVerifier")).toMatchObject({
      label: "ClaimVerifier",
      detail: "claim support gate",
    });
  });
});
