const crypto = require("node:crypto");
const { readBody } = require("../lib/web");
const { getScopedStateKeyFromRequest, normalizeApiKey, normalizeStateKey } = require("../lib/state-scope");
const {
    listResearchRuns,
    loadResearchRun,
    queueResearchControl,
    saveResearchMemoryEntries,
    searchResearchMemoryDetailed,
    formatResearchMemoryContext,
} = require("../lib/research-memory");
const {
    ResearchRuntimeError,
    runResearch,
    resumeResearch,
    buildExportPayload,
    serializeResearchRun,
} = require("../lib/research-runtime");

const normalizeText = (value) => String(value ?? "").trim();

const writeCorsHeaders = (res) => {
    res.setHeader("Access-Control-Allow-Origin", "*");
    res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization, X-State-Key, X-API-Key");
};

const sendJson = (res, status, payload) => {
    res.status(status);
    res.setHeader("Content-Type", "application/json; charset=utf-8");
    res.end(JSON.stringify(payload));
};

const normalizeScopeKey = (req, body = {}) => {
    const scoped = getScopedStateKeyFromRequest(req);
    if (scoped?.stateKey) return scoped.stateKey;
    const explicit = normalizeStateKey(body?.stateKey || body?.scopeKey || req.query?.stateKey);
    if (explicit) return explicit;
    return `ephemeral:${crypto.createHash("sha256").update(String(req.headers["user-agent"] || "nubagent")).digest("hex").slice(0, 24)}`;
};

const normalizeAction = (req, body = {}) => normalizeText(
    body?.action
    || req.query?.action
    || req.query?.mode
    || (req.method === "GET" || req.method === "HEAD" ? "metadata" : "run"),
).toLowerCase() || "run";

const wantsStream = (req, body = {}) => {
    const responseType = normalizeText(
        body?.responseType
        || body?.format
        || req.query?.responseType,
    ).toLowerCase();
    if (responseType === "stream" || responseType === "sse") return true;
    if (responseType === "json") return false;
    return body?.stream === true;
};

const normalizeAttachments = (attachments = []) => {
    return (Array.isArray(attachments) ? attachments : []).map((attachment) => ({
        id: attachment?.id,
        name: String(attachment?.name || "attachment"),
        kind: attachment?.kind === "image" ? "image" : "text",
        size: Number(attachment?.size || 0) || 0,
        dataUrl: attachment?.kind === "image" ? String(attachment?.dataUrl || "") : "",
        textContent: attachment?.kind === "text" ? String(attachment?.textContent || "") : "",
        truncated: Boolean(attachment?.truncated),
    }));
};

const normalizeDepthPreference = (value) => {
    const normalized = normalizeText(value).toLowerCase();
    return ["speed", "balanced", "deep"].includes(normalized) ? normalized : "";
};

const normalizeForcedOutputMode = (value) => normalizeText(value).toLowerCase().replace(/\s+/g, "_");

const normalizeRefinementBudget = (value) => {
    if (value == null || value === "") return undefined;
    const resolved = Number(value);
    if (!Number.isFinite(resolved)) return undefined;
    return Math.max(1, Math.min(6, Math.floor(resolved)));
};

const normalizeMaxQueries = (value) => {
    if (value == null || value === "") return undefined;
    const resolved = Number(value);
    if (!Number.isFinite(resolved)) return undefined;
    return Math.max(1, Math.min(12, Math.floor(resolved)));
};
const normalizeResearchProvider = (value) => {
    const normalized = normalizeText(value).toLowerCase();
    if (normalized === "openrouter" || normalized === "open-router") return "openrouter";
    if (normalized === "google" || normalized === "gemini") return "google";
    return undefined;
};
const normalizeResearchModel = (value) => normalizeText(value) || undefined;
const normalizeResearchModelChain = (value) => {
    const entries = Array.isArray(value)
        ? value
        : typeof value === "string"
            ? value.split(",")
            : [];
    const normalized = entries
        .map((item) => normalizeText(item))
        .filter(Boolean);
    return normalized.length ? [...new Set(normalized)] : undefined;
};
const normalizeResearchRoundRobin = (value) => {
    if (value == null || value === "") return undefined;
    if (typeof value === "boolean") return value;
    const normalized = normalizeText(value).toLowerCase();
    if (["true", "1", "yes", "round_robin", "round-robin"].includes(normalized)) return true;
    if (["false", "0", "no", "single", "off"].includes(normalized)) return false;
    return undefined;
};

const normalizeSearchProviderKeys = (value = {}) => {
    const candidate = value && typeof value === "object" ? value : {};
    const normalized = {
        tavily: normalizeApiKey(candidate?.tavily),
        serper: normalizeApiKey(candidate?.serper),
        brave: normalizeApiKey(candidate?.brave),
        jina: normalizeApiKey(candidate?.jina),
    };

    return Object.fromEntries(
        Object.entries(normalized).filter(([, apiKey]) => apiKey),
    );
};
const normalizeResearchProviderKeys = (value = {}) => {
    const candidate = value && typeof value === "object" ? value : {};
    const normalized = {
        openrouter: normalizeApiKey(candidate?.openrouter || candidate?.openRouter || candidate?.["open-router"]),
        google: normalizeApiKey(candidate?.google || candidate?.gemini),
    };
    return Object.fromEntries(
        Object.entries(normalized).filter(([, apiKey]) => apiKey),
    );
};

const buildRuntimeRequestOptions = (req, body = {}) => ({
    depthPreference: normalizeDepthPreference(body?.depthPreference || body?.depth || req.query?.depthPreference || req.query?.depth),
    forcedOutputMode: normalizeForcedOutputMode(body?.forcedOutputMode || body?.outputMode || req.query?.forcedOutputMode || req.query?.outputMode),
    refinementBudget: normalizeRefinementBudget(body?.refinementBudget || req.query?.refinementBudget),
    maxQueries: normalizeMaxQueries(body?.maxQueries || req.query?.maxQueries),
    searchProviderKeys: normalizeSearchProviderKeys(body?.searchProviderKeys),
    researchProvider: normalizeResearchProvider(body?.researchProvider || body?.provider || req.query?.researchProvider || req.query?.provider),
    researchModel: normalizeResearchModel(body?.researchModel || body?.model || req.query?.researchModel || req.query?.model),
    researchModelChain: normalizeResearchModelChain(body?.researchModelChain || body?.modelChain || req.query?.researchModelChain || req.query?.modelChain),
    researchRoundRobin: normalizeResearchRoundRobin(body?.researchRoundRobin || body?.roundRobin || req.query?.researchRoundRobin || req.query?.roundRobin),
    researchProviderKeys: normalizeResearchProviderKeys(body?.researchProviderKeys || body?.providerApiKeys || body?.provider_api_keys),
});

const buildRunPreview = (run = {}) => {
    const serialized = serializeResearchRun(run);
    return {
        runId: serialized.runId,
        status: serialized.status,
        query: serialized.query,
        title: run.title || serialized.final?.heading || serialized.query,
        updatedAt: run.updatedAt || serialized.researchMeta?.generatedAt || null,
        researchMeta: serialized.researchMeta,
    };
};

const metadataPayload = (scopeKey = "") => ({
    ok: true,
    endpoint: "/api/research",
    scopeKey,
    actions: {
        run: { method: "POST", stream: true, description: "Run the compiled research DAG." },
        resume: { method: "POST", stream: true, description: "Resume a stored research run from the last checkpoint." },
        control: { method: "POST", stream: false, description: "Queue a steering control for an active run." },
        export: { method: "POST", stream: false, description: "Export a stored research run." },
        get: { method: "GET|POST", stream: false, description: "Fetch a stored research run by id." },
        list: { method: "GET", stream: false, description: "List research runs for the active scope." },
        memory_search: { method: "GET|POST", stream: false, description: "Search the hierarchical research memory." },
        memory_insert: { method: "POST", stream: false, description: "Insert custom research memory entries." },
    },
    controls: [
        { type: "prioritize_speed" },
        { type: "go_deeper" },
        { type: "increase_depth", area: "hypothesis_2" },
        { type: "exclude_source", sourceType: "preprint" },
        { type: "force_mode", mode: "gap_analysis" },
    ],
    outputs: [
        "tutorial",
        "state_of_the_field",
        "controversy_map",
        "gap_analysis",
        "replication_crisis_report",
        "foundational_review",
        "decision_brief",
        "policy_recommendation",
        "engineering_action_plan",
    ],
    exportFormats: [
        "json",
        "markdown",
        "slide_deck_outline",
        "dataset_csv",
        "dataset_json",
    ],
    streamingEvents: [
        "status",
        "checkpoint",
        "inventory",
        "summary",
        "warning",
        "control_applied",
        "final",
        "error",
    ],
});

const writeSseEvent = (res, event = {}) => {
    const type = normalizeText(event?.type || "message") || "message";
    res.write(`event: ${type}\n`);
    res.write(`data: ${JSON.stringify({ ...event, type })}\n\n`);
};

module.exports = async (req, res) => {
    writeCorsHeaders(res);

    if (req.method === "OPTIONS") {
        res.status(204).end();
        return;
    }

    let body = {};
    if (req.method === "POST") {
        try {
            body = await readBody(req);
        } catch {
            sendJson(res, 400, { error: "Invalid JSON body" });
            return;
        }
    }

    const scopeKey = normalizeScopeKey(req, body);
    const action = normalizeAction(req, body);

    if (req.method === "GET" || req.method === "HEAD") {
        try {
            if (action === "get") {
                const runId = normalizeText(req.query?.id || req.query?.runId);
                if (!runId) {
                    sendJson(res, 400, { error: "Provide id or runId." });
                    return;
                }
                const run = await loadResearchRun(runId);
                if (!run) {
                    sendJson(res, 404, { error: "Research run not found." });
                    return;
                }
                sendJson(res, 200, serializeResearchRun(run));
                return;
            }

            if (action === "list") {
                const runs = await listResearchRuns(scopeKey, { limit: req.query?.limit });
                sendJson(res, 200, {
                    ok: true,
                    scopeKey,
                    runs: runs.map((run) => buildRunPreview(run)),
                });
                return;
            }

            if (action === "memory_search") {
                const query = normalizeText(req.query?.query);
                if (!query) {
                    sendJson(res, 400, { error: "Provide a query string." });
                    return;
                }
                const search = await searchResearchMemoryDetailed(scopeKey, query, {
                    limit: req.query?.limit,
                    layers: req.query?.layers ? String(req.query.layers).split(",") : [],
                });
                sendJson(res, 200, {
                    ok: true,
                    scopeKey,
                    query,
                    search: search.meta,
                    results: search.results,
                    context: formatResearchMemoryContext(search.results),
                });
                return;
            }

            sendJson(res, 200, metadataPayload(scopeKey));
        } catch (error) {
            sendJson(res, 500, { error: error?.message || "Research metadata request failed." });
        }
        return;
    }

    if (req.method !== "POST") {
        sendJson(res, 405, { error: "Method not allowed" });
        return;
    }

    try {
        if (action === "control") {
            const runId = normalizeText(body?.runId || body?.id);
            if (!runId) {
                sendJson(res, 400, { error: "Provide runId or id." });
                return;
            }

            const queued = await queueResearchControl(runId, body?.control || body);
            if (!queued) {
                sendJson(res, 400, { error: "Provide a control payload with a type." });
                return;
            }
            sendJson(res, 200, { ok: true, control: queued });
            return;
        }

        if (action === "memory_search") {
            const query = normalizeText(body?.query);
            if (!query) {
                sendJson(res, 400, { error: "Provide a query string." });
                return;
            }
            const search = await searchResearchMemoryDetailed(scopeKey, query, {
                limit: body?.limit,
                layers: Array.isArray(body?.layers) ? body.layers : [],
            });
            sendJson(res, 200, {
                ok: true,
                scopeKey,
                query,
                search: search.meta,
                results: search.results,
                context: formatResearchMemoryContext(search.results),
            });
            return;
        }

        if (action === "memory_insert") {
            const entries = Array.isArray(body?.entries)
                ? body.entries
                : body?.entry
                    ? [body.entry]
                    : [];
            if (!entries.length) {
                sendJson(res, 400, { error: "Provide entries or entry." });
                return;
            }
            const inserted = await saveResearchMemoryEntries(scopeKey, entries);
            sendJson(res, 200, { ok: true, scopeKey, inserted });
            return;
        }

        if (action === "export") {
            const runId = normalizeText(body?.runId || body?.id);
            if (!runId) {
                sendJson(res, 400, { error: "Provide runId or id." });
                return;
            }
            const run = await loadResearchRun(runId);
            if (!run) {
                sendJson(res, 404, { error: "Research run not found." });
                return;
            }
            const exportPayload = buildExportPayload(run, body?.format || body?.mode || "json");
            res.status(200);
            res.setHeader("Content-Type", exportPayload.contentType);
            res.end(exportPayload.body);
            return;
        }

        if (action === "get") {
            const runId = normalizeText(body?.runId || body?.id);
            if (!runId) {
                sendJson(res, 400, { error: "Provide runId or id." });
                return;
            }
            const run = await loadResearchRun(runId);
            if (!run) {
                sendJson(res, 404, { error: "Research run not found." });
                return;
            }
            sendJson(res, 200, serializeResearchRun(run));
            return;
        }

        const stream = wantsStream(req, body);
        const runRequest = async (onEvent = null) => {
            const runtimeOptions = buildRuntimeRequestOptions(req, body);
            if (action === "resume") {
                const runId = normalizeText(body?.runId || body?.id);
                if (!runId) {
                    throw new ResearchRuntimeError("Provide runId or id.", { failureType: "critical" });
                }
                return resumeResearch({
                    runId,
                    scopeKey,
                    ...runtimeOptions,
                    controls: body?.controls,
                    stopAfterCheckpoint: body?.stopAfterCheckpoint,
                    onEvent,
                });
            }

            const query = normalizeText(body?.query || body?.prompt);
            const attachments = normalizeAttachments(body?.attachments);
            if (!query && !attachments.length) {
                throw new ResearchRuntimeError("Provide a query or attachments.", { failureType: "critical" });
            }

            return runResearch({
                query,
                attachments,
                scopeKey,
                ...runtimeOptions,
                controls: body?.controls,
                stopAfterCheckpoint: body?.stopAfterCheckpoint,
                onEvent,
            });
        };

        if (stream) {
            res.status(200);
            res.setHeader("Content-Type", "text/event-stream; charset=utf-8");
            res.setHeader("Cache-Control", "no-cache, no-transform");
            res.setHeader("Connection", "keep-alive");

            try {
                const run = await runRequest(async (event) => {
                    writeSseEvent(res, event);
                });
                writeSseEvent(res, {
                    type: "final",
                    ...serializeResearchRun(run),
                });
                res.write("data: [DONE]\n\n");
                res.end();
            } catch (error) {
                const normalized = error instanceof ResearchRuntimeError
                    ? error
                    : new ResearchRuntimeError(error?.message || "Research request failed.");
                writeSseEvent(res, {
                    type: "error",
                    error: normalized.message,
                    failureType: normalized.failureType,
                    nodeId: normalized.nodeId,
                });
                res.write("data: [DONE]\n\n");
                res.end();
            }
            return;
        }

        const run = await runRequest();
        sendJson(res, 200, serializeResearchRun(run));
    } catch (error) {
        const normalized = error instanceof ResearchRuntimeError
            ? error
            : new ResearchRuntimeError(error?.message || "Research request failed.");
        sendJson(res, normalized.failureType === "critical" ? 500 : 502, {
            error: normalized.message,
            failureType: normalized.failureType,
            nodeId: normalized.nodeId,
        });
    }
};
