import { describe, expect, it, vi } from "vitest";
import { loadCommonJsModule } from "../../src/__tests__/loadCommonJsModule.js";

const createRes = () => {
  const res = {
    headers: {},
    statusCode: 200,
    body: "",
    setHeader: vi.fn((key, value) => {
      res.headers[key] = value;
    }),
    status: vi.fn((code) => {
      res.statusCode = code;
      return res;
    }),
    write: vi.fn((value) => {
      res.body += value;
      return true;
    }),
    end: vi.fn((value = "") => {
      res.body += value;
      return res;
    }),
  };
  return res;
};

const buildHandler = ({
  body = null,
  runResearch = vi.fn(),
  resumeResearch = vi.fn(),
  serializeResearchRun = vi.fn((run) => run),
  buildExportPayload = vi.fn(),
} = {}) => {
    const mocks = {
    "../lib/web": {
      readBody: vi.fn(async () => body ?? {}),
    },
    "../lib/state-scope": {
      getScopedStateKeyFromRequest: vi.fn(() => ({ stateKey: "scope:test" })),
      normalizeApiKey: vi.fn((value) => String(value || "").trim()),
      normalizeStateKey: vi.fn((value) => value),
    },
    "../lib/research-memory": {
      listResearchRuns: vi.fn(async () => []),
      loadResearchRun: vi.fn(async (id) => (id ? { id, runId: id } : null)),
      queueResearchControl: vi.fn(),
      saveResearchMemoryEntries: vi.fn(),
      searchResearchMemoryDetailed: vi.fn(),
      formatResearchMemoryContext: vi.fn(() => ""),
    },
    "../lib/research-runtime": {
      ResearchRuntimeError: class ResearchRuntimeError extends Error {
        constructor(message, options = {}) {
          super(message);
          this.failureType = options.failureType || "critical";
          this.nodeId = options.nodeId || "";
        }
      },
      runResearch,
      resumeResearch,
      buildExportPayload,
      serializeResearchRun,
    },
  };

  const handler = loadCommonJsModule(
    "/root/.bot/.downloads/nubagent/api/research.js",
    mocks,
  );

  return {
    handler,
    mocks,
    runResearch,
    resumeResearch,
    serializeResearchRun,
    buildExportPayload,
  };
};

describe("/api/research", () => {
  it("returns metadata that exposes the research control and delivery contract", async () => {
    const { handler } = buildHandler();
    const req = {
      method: "GET",
      query: {},
      headers: {},
      url: "/api/research",
    };
    const res = createRes();

    await handler(req, res);

    expect(res.statusCode).toBe(200);
    const payload = JSON.parse(res.body);
    expect(payload.outputs).toContain("decision_brief");
    expect(payload.outputs).toContain("engineering_action_plan");
    expect(payload.controls).toEqual(expect.arrayContaining([
      expect.objectContaining({ type: "prioritize_speed" }),
      expect.objectContaining({ type: "go_deeper" }),
      expect.objectContaining({ type: "force_mode", mode: "gap_analysis" }),
    ]));
    expect(payload.streamingEvents).toEqual(expect.arrayContaining([
      "checkpoint",
      "summary",
      "control_applied",
      "final",
    ]));
  });

  it("forwards orchestration controls to runResearch", async () => {
    const runResearch = vi.fn(async (payload) => ({
      id: "research-run-1",
      runId: "research-run-1",
      query: payload.query,
      status: "complete",
      plan: {},
      result: {},
      state: {},
    }));
    const serializeResearchRun = vi.fn((run) => ({
      ok: true,
      runId: run.runId,
      forwarded: true,
    }));
    const { handler } = buildHandler({
      body: {
        query: "should we adopt retrieval caching",
        attachments: [{ name: "notes.txt", kind: "text", textContent: "cached responses" }],
        depthPreference: "deep",
        refinementBudget: 5,
        forcedOutputMode: "decision_brief",
        searchProviderKeys: {
          tavily: "tvly-dev-key-1234567890",
          serper: "",
        },
        controls: [{ type: "go_deeper" }],
        stopAfterCheckpoint: "draft",
      },
      runResearch,
      serializeResearchRun,
    });
    const req = {
      method: "POST",
      query: {},
      headers: {},
      url: "/api/research",
    };
    const res = createRes();

    await handler(req, res);

    expect(runResearch).toHaveBeenCalledWith(expect.objectContaining({
      query: "should we adopt retrieval caching",
      attachments: [{ id: undefined, name: "notes.txt", kind: "text", size: 0, dataUrl: "", textContent: "cached responses", truncated: false }],
      scopeKey: "scope:test",
      depthPreference: "deep",
      refinementBudget: 5,
      forcedOutputMode: "decision_brief",
      searchProviderKeys: {
        tavily: "tvly-dev-key-1234567890",
      },
      controls: [{ type: "go_deeper" }],
      stopAfterCheckpoint: "draft",
    }));
    expect(res.statusCode).toBe(200);
    expect(JSON.parse(res.body)).toMatchObject({
      ok: true,
      runId: "research-run-1",
      forwarded: true,
    });
  });

  it("forwards orchestration overrides when resuming a stored run", async () => {
    const resumeResearch = vi.fn(async () => ({
      id: "research-run-2",
      runId: "research-run-2",
      status: "in_progress",
      plan: {},
      result: {},
      state: {},
    }));
    const { handler } = buildHandler({
      body: {
        action: "resume",
        runId: "research-run-2",
        depthPreference: "speed",
        refinementBudget: 2,
        forcedOutputMode: "gap_analysis",
        controls: [{ type: "prioritize_speed" }, { type: "force_mode", mode: "gap_analysis" }],
        stopAfterCheckpoint: "summaries",
      },
      resumeResearch,
    });
    const req = {
      method: "POST",
      query: {},
      headers: {},
      url: "/api/research",
    };
    const res = createRes();

    await handler(req, res);

    expect(resumeResearch).toHaveBeenCalledWith(expect.objectContaining({
      runId: "research-run-2",
      depthPreference: "speed",
      refinementBudget: 2,
      forcedOutputMode: "gap_analysis",
      controls: [{ type: "prioritize_speed" }, { type: "force_mode", mode: "gap_analysis" }],
      stopAfterCheckpoint: "summaries",
    }));
    expect(res.statusCode).toBe(200);
  });

  it("queues steering controls through the control action contract", async () => {
    const { handler, mocks } = buildHandler({
      body: {
        action: "control",
        runId: "research-run-4",
        control: {
          type: "increase_depth",
          area: "hypothesis_2",
        },
      },
    });
    mocks["../lib/research-memory"].queueResearchControl.mockResolvedValue({
      id: "control-1",
      type: "increase_depth",
      area: "hypothesis_2",
      status: "pending",
    });
    const req = {
      method: "POST",
      query: {},
      headers: {},
      url: "/api/research",
    };
    const res = createRes();

    await handler(req, res);

    expect(mocks["../lib/research-memory"].queueResearchControl).toHaveBeenCalledWith("research-run-4", {
      type: "increase_depth",
      area: "hypothesis_2",
    });
    expect(JSON.parse(res.body)).toEqual({
      ok: true,
      control: {
        id: "control-1",
        type: "increase_depth",
        area: "hypothesis_2",
        status: "pending",
      },
    });
  });

  it("returns requested export payloads for stored runs", async () => {
    const run = {
      id: "research-run-5",
      runId: "research-run-5",
      result: {
        markdown: "# Retrieval caching\n\nUse it.",
      },
    };
    const buildExportPayload = vi.fn(() => ({
      contentType: "text/markdown; charset=utf-8",
      body: "# Retrieval caching\n\nUse it.",
    }));
    const { handler, mocks } = buildHandler({
      body: {
        action: "export",
        runId: "research-run-5",
        format: "markdown",
      },
      buildExportPayload,
    });
    mocks["../lib/research-memory"].loadResearchRun.mockResolvedValue(run);
    const req = {
      method: "POST",
      query: {},
      headers: {},
      url: "/api/research",
    };
    const res = createRes();

    await handler(req, res);

    expect(mocks["../lib/research-memory"].loadResearchRun).toHaveBeenCalledWith("research-run-5");
    expect(buildExportPayload).toHaveBeenCalledWith(run, "markdown");
    expect(res.headers["Content-Type"]).toBe("text/markdown; charset=utf-8");
    expect(res.body).toBe("# Retrieval caching\n\nUse it.");
  });

  it("returns serializer-produced decision and verifier artifacts in the response body", async () => {
    const runResearch = vi.fn(async () => ({
      id: "research-run-3",
      runId: "research-run-3",
      status: "complete",
      plan: {},
      result: {},
      state: {},
    }));
    const serializeResearchRun = vi.fn(() => ({
      ok: true,
      runId: "research-run-3",
      final: {
        heading: "Adopt retrieval caching",
        decision: {
          confidence: 0.74,
          reversibility: "medium",
        },
        tribunal: {
          verifiers: {
            claim_support: 0.88,
            citation_integrity: 0.91,
          },
        },
      },
      researchMeta: {
        subagents: [
          { id: "claimVerifier" },
          { id: "citationVerifier" },
          { id: "decisionIntelligenceLayer" },
        ],
      },
    }));
    const { handler } = buildHandler({
      body: { query: "Should we use retrieval caching?" },
      runResearch,
      serializeResearchRun,
    });
    const req = {
      method: "POST",
      query: {},
      headers: {},
      url: "/api/research",
    };
    const res = createRes();

    await handler(req, res);

    const payload = JSON.parse(res.body);
    expect(payload.final.decision).toMatchObject({
      confidence: 0.74,
      reversibility: "medium",
    });
    expect(payload.final.tribunal.verifiers).toMatchObject({
      claim_support: 0.88,
      citation_integrity: 0.91,
    });
    expect(payload.researchMeta.subagents.map((item) => item.id)).toEqual(expect.arrayContaining([
      "claimVerifier",
      "citationVerifier",
      "decisionIntelligenceLayer",
    ]));
  });
});
