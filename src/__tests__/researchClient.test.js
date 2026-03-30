import { describe, expect, it, vi } from "vitest";
import {
  fetchResearchExport,
  runResearchRequest,
  sendResearchControl,
} from "../lib/researchClient.js";

const createHeaders = (values = {}) => ({
  get(name) {
    return values[String(name || "").toLowerCase()] ?? null;
  },
});

const createJsonResponse = (payload) => ({
  ok: true,
  headers: createHeaders({ "content-type": "application/json; charset=utf-8" }),
  json: async () => payload,
  text: async () => JSON.stringify(payload),
});

const createTextResponse = (
  body,
  contentType = "text/plain; charset=utf-8",
) => ({
  ok: true,
  headers: createHeaders({ "content-type": contentType }),
  text: async () => body,
});

const createSseResponse = (chunks = []) => {
  const encoder = new TextEncoder();
  let index = 0;

  return {
    ok: true,
    headers: createHeaders({
      "content-type": "text/event-stream; charset=utf-8",
    }),
    body: {
      getReader() {
        return {
          read: async () => {
            if (index >= chunks.length) {
              return { done: true, value: undefined };
            }
            const value = encoder.encode(chunks[index]);
            index += 1;
            return { done: false, value };
          },
        };
      },
    },
  };
};

describe("research client helpers", () => {
  it("preserves checkpoint, control, and verifier-rich final events from SSE runs", async () => {
    const onEvent = vi.fn();
    const payload = {
      action: "run",
      query: "should we adopt retrieval caching",
      depthPreference: "deep",
      refinementBudget: 5,
      forcedOutputMode: "decision_brief",
    };

    global.fetch.mockResolvedValue(
      createSseResponse([
        'event: checkpoint\ndata: {"type":"checkpoint","runId":"research-run-1","checkpoint":"draft","payload":{"heading":"Draft report"}}\n\n',
        'event: control_applied\ndata: {"type":"control_applied","runId":"research-run-1","restartNode":"adversarialQueryForge","statusText":"1 steering command applied."}\n\n',
        'event: final\ndata: {"type":"final","runId":"research-run-1","final":{"tribunal":{"refinement_cycles":2,"targeted_dimension":"coverage"},"convergence":{"residual_uncertainty":0.24},"claimLedger":[{"claim":"Caching reduced average latency.","citations":[1]}],"exports":{"markdown":"# Retrieval caching\\n\\nUse it."}},"outputs":{"markdown":"# Retrieval caching\\n\\nUse it."}}\n\n',
        "data: [DONE]\n\n",
      ]),
    );

    const result = await runResearchRequest(payload, { onEvent });

    expect(global.fetch).toHaveBeenCalledWith(
      "/api/research",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify(payload),
      }),
    );
    expect(result.events.map((event) => event.type)).toEqual([
      "checkpoint",
      "control_applied",
      "final",
    ]);
    expect(result.final.tribunal.refinement_cycles).toBe(2);
    expect(result.final.claimLedger).toEqual([
      { claim: "Caching reduced average latency.", citations: [1] },
    ]);
    expect(result.outputs.markdown).toContain("# Retrieval caching");
    expect(onEvent).toHaveBeenNthCalledWith(
      1,
      expect.objectContaining({
        type: "checkpoint",
        checkpoint: "draft",
      }),
    );
    expect(onEvent).toHaveBeenNthCalledWith(
      2,
      expect.objectContaining({
        type: "control_applied",
        restartNode: "adversarialQueryForge",
      }),
    );
  });

  it("sends steering controls with the control action envelope", async () => {
    global.fetch.mockResolvedValue(
      createJsonResponse({
        ok: true,
        control: {
          type: "exclude_source",
          sourceType: "preprint",
        },
      }),
    );

    const result = await sendResearchControl("research-run-2", {
      type: "exclude_source",
      sourceType: "preprint",
    });

    expect(global.fetch).toHaveBeenCalledWith(
      "/api/research",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          action: "control",
          runId: "research-run-2",
          control: {
            type: "exclude_source",
            sourceType: "preprint",
          },
        }),
      }),
    );
    expect(result).toEqual({
      ok: true,
      control: {
        type: "exclude_source",
        sourceType: "preprint",
      },
    });
  });

  it("requests research exports by format and returns non-JSON bodies verbatim", async () => {
    global.fetch.mockResolvedValue(
      createTextResponse(
        "# Retrieval caching\n\nUse it.",
        "text/markdown; charset=utf-8",
      ),
    );

    const result = await fetchResearchExport("research-run-3", "markdown");

    expect(global.fetch).toHaveBeenCalledWith(
      "/api/research",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          action: "export",
          runId: "research-run-3",
          format: "markdown",
        }),
      }),
    );
    expect(result).toEqual({
      contentType: "text/markdown; charset=utf-8",
      body: "# Retrieval caching\n\nUse it.",
    });
  });
});
