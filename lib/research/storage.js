const crypto = require("node:crypto");
const { loadAppState, saveAppState } = require("../db");
const { getScopedStateKeyFromRequest } = require("../state-scope");

const RUN_INDEX_LIMIT = 40;
const MEMORY_EPISODE_LIMIT = 40;
const MEMORY_ABSTRACTION_LIMIT = 60;
const CONTROL_QUEUE_LIMIT = 20;

const normalizeText = (value) => String(value || "").replace(/\s+/g, " ").trim();

const clampInteger = (value, min, max, fallback) => {
    const number = Number(value);
    if (!Number.isFinite(number)) return fallback;
    return Math.min(max, Math.max(min, Math.floor(number)));
};

const createResearchRunId = () => `rr_${Date.now()}_${crypto.randomBytes(4).toString("hex")}`;

const resolveResearchScope = (req) => {
    const scoped = getScopedStateKeyFromRequest(req);
    return {
        stateKey: scoped?.stateKey || "public",
        scope: scoped?.scope || "public",
    };
};

const buildResearchRunKey = (runId) => `research:run:${String(runId || "").trim()}`;
const buildResearchIndexKey = (scopeKey) => `research:index:${String(scopeKey || "public").trim()}`;
const buildResearchMemoryKey = (scopeKey) => `research:memory:${String(scopeKey || "public").trim()}`;
const buildResearchControlsKey = (runId) => `research:controls:${String(runId || "").trim()}`;

const normalizeRunRecord = (run = {}) => ({
    runId: String(run.runId || "").trim(),
    status: normalizeText(run.status || "pending") || "pending",
    scopeKey: normalizeText(run.scopeKey || "public") || "public",
    query: normalizeText(run.query),
    createdAt: run.createdAt || new Date().toISOString(),
    updatedAt: run.updatedAt || new Date().toISOString(),
    attachments: Array.isArray(run.attachments) ? run.attachments : [],
    plan: run.plan && typeof run.plan === "object" ? run.plan : null,
    checkpoints: Array.isArray(run.checkpoints) ? run.checkpoints : [],
    outputs: run.outputs && typeof run.outputs === "object" ? run.outputs : {},
    final: run.final && typeof run.final === "object" ? run.final : null,
    error: normalizeText(run.error),
});

const normalizeResearchMemory = (memory = {}) => ({
    version: 1,
    updatedAt: memory.updatedAt || null,
    episodes: Array.isArray(memory.episodes) ? memory.episodes.slice(-MEMORY_EPISODE_LIMIT) : [],
    concepts: memory.concepts && typeof memory.concepts === "object" ? memory.concepts : {},
    abstractions: Array.isArray(memory.abstractions) ? memory.abstractions.slice(-MEMORY_ABSTRACTION_LIMIT) : [],
    postmortems: Array.isArray(memory.postmortems) ? memory.postmortems.slice(-MEMORY_EPISODE_LIMIT) : [],
});

const normalizeCheckpoint = (checkpoint = {}) => ({
    id: String(checkpoint.id || `${Date.now()}`).trim(),
    phase: normalizeText(checkpoint.phase),
    type: normalizeText(checkpoint.type || "checkpoint") || "checkpoint",
    title: normalizeText(checkpoint.title),
    statusText: normalizeText(checkpoint.statusText),
    summary: normalizeText(checkpoint.summary),
    payload: checkpoint.payload && typeof checkpoint.payload === "object" ? checkpoint.payload : {},
    createdAt: checkpoint.createdAt || new Date().toISOString(),
});

const loadResearchRun = async (runId) => {
    const record = await loadAppState(buildResearchRunKey(runId));
    return record?.state ? normalizeRunRecord(record.state) : null;
};

const saveResearchRun = async (run) => {
    const normalized = normalizeRunRecord(run);
    if (!normalized.runId) throw new Error("Research runId is required.");
    await saveAppState(buildResearchRunKey(normalized.runId), normalized);
    return normalized;
};

const createResearchRun = async (options = {}) => {
    const record = normalizeRunRecord({
        runId: options.runId || createResearchRunId(),
        status: options.status || "running",
        scopeKey: options.scopeKey || "public",
        query: options.query,
        attachments: options.attachments,
        plan: options.plan,
        checkpoints: [],
        outputs: {},
        final: null,
        error: "",
    });
    await saveResearchRun(record);
    return record;
};

const updateResearchRun = async (runId, patch = {}) => {
    const current = await loadResearchRun(runId);
    const next = normalizeRunRecord({
        ...(current || { runId }),
        ...patch,
        updatedAt: new Date().toISOString(),
    });
    await saveResearchRun(next);
    return next;
};

const appendResearchCheckpoint = async (runId, checkpoint) => {
    const current = await loadResearchRun(runId);
    const next = normalizeRunRecord({
        ...(current || { runId }),
        checkpoints: [...(current?.checkpoints || []), normalizeCheckpoint(checkpoint)],
        updatedAt: new Date().toISOString(),
    });
    await saveResearchRun(next);
    return next;
};

const loadResearchIndex = async (scopeKey) => {
    const record = await loadAppState(buildResearchIndexKey(scopeKey));
    return Array.isArray(record?.state?.runs) ? record.state.runs : [];
};

const appendResearchIndex = async (scopeKey, runSummary = {}) => {
    const current = await loadResearchIndex(scopeKey);
    const next = [
        {
            runId: String(runSummary.runId || "").trim(),
            query: normalizeText(runSummary.query),
            status: normalizeText(runSummary.status || "complete") || "complete",
            domain: normalizeText(runSummary.domain),
            outputMode: normalizeText(runSummary.outputMode),
            updatedAt: runSummary.updatedAt || new Date().toISOString(),
        },
        ...current.filter((item) => item?.runId !== runSummary.runId),
    ].slice(0, RUN_INDEX_LIMIT);

    await saveAppState(buildResearchIndexKey(scopeKey), { runs: next });
    return next;
};

const extractConceptTerms = (text = "") => (
    Array.from(new Set(
        String(text || "")
            .toLowerCase()
            .match(/[a-z0-9][a-z0-9+/_-]{2,}/g) || [],
    )).slice(0, 32)
);

const updateResearchMemoryFromRun = async (scopeKey, run = {}) => {
    const existing = normalizeResearchMemory((await loadAppState(buildResearchMemoryKey(scopeKey)))?.state);
    const final = run.final || {};
    const plan = run.plan || {};
    const episode = {
        runId: run.runId,
        query: normalizeText(run.query),
        summary: normalizeText(final.summary || final.heading || final.body || "").slice(0, 1200),
        domain: normalizeText(plan.domain?.label),
        scope: normalizeText(plan.scope?.label),
        outputMode: normalizeText(plan.outputMode?.label),
        updatedAt: run.updatedAt || new Date().toISOString(),
    };

    const conceptText = [
        run.query,
        final.heading,
        final.body,
        ...(Array.isArray(final.claims) ? final.claims.map((claim) => claim?.text || "") : []),
        ...(Array.isArray(final.concepts) ? final.concepts : []),
    ].join(" ");
    const conceptTerms = extractConceptTerms(conceptText);
    const concepts = { ...existing.concepts };
    for (const term of conceptTerms) {
        const current = concepts[term] || {
            label: term,
            mentions: 0,
            lastRunId: "",
            lastSeenAt: "",
            support: [],
        };
        concepts[term] = {
            ...current,
            mentions: Number(current.mentions || 0) + 1,
            lastRunId: run.runId,
            lastSeenAt: run.updatedAt || new Date().toISOString(),
            support: Array.from(new Set([...(Array.isArray(current.support) ? current.support : []), run.runId])).slice(-10),
        };
    }

    const abstractions = [
        ...existing.abstractions.filter((item) => item?.runId !== run.runId),
        ...((Array.isArray(final.abstractions) ? final.abstractions : []).map((item, index) => ({
            id: `${run.runId}:a${index + 1}`,
            runId: run.runId,
            pattern: normalizeText(item?.pattern || item),
            evidence: normalizeText(item?.evidence),
            updatedAt: run.updatedAt || new Date().toISOString(),
        })).filter((item) => item.pattern)),
    ].slice(-MEMORY_ABSTRACTION_LIMIT);

    const postmortems = [
        ...existing.postmortems.filter((item) => item?.runId !== run.runId),
        ...((final.postmortem ? [{
            runId: run.runId,
            queryType: normalizeText(plan.scope?.id || plan.scope?.label),
            domain: normalizeText(plan.domain?.label),
            phasesThatDegradedScore: Array.isArray(final.postmortem.phases_that_degraded_score)
                ? final.postmortem.phases_that_degraded_score
                : [],
            promptPatchesApplied: Array.isArray(final.postmortem.prompt_patches_applied)
                ? final.postmortem.prompt_patches_applied
                : [],
            finalScoreDelta: Number(final.postmortem.final_score_delta || 0),
            updatedAt: run.updatedAt || new Date().toISOString(),
        }] : [])),
    ].slice(-MEMORY_EPISODE_LIMIT);

    const next = normalizeResearchMemory({
        ...existing,
        updatedAt: new Date().toISOString(),
        episodes: [
            episode,
            ...existing.episodes.filter((item) => item?.runId !== run.runId),
        ].slice(0, MEMORY_EPISODE_LIMIT),
        concepts,
        abstractions,
        postmortems,
    });

    await saveAppState(buildResearchMemoryKey(scopeKey), next);
    return next;
};

const listResearchRuns = async (scopeKey, limit = 20) => {
    const runs = await loadResearchIndex(scopeKey);
    return runs.slice(0, clampInteger(limit, 1, RUN_INDEX_LIMIT, 20));
};

const enqueueSteeringCommands = async (runId, commands = []) => {
    const key = buildResearchControlsKey(runId);
    const record = await loadAppState(key);
    const queue = Array.isArray(record?.state?.queue) ? record.state.queue : [];
    const next = [...queue, ...(Array.isArray(commands) ? commands : [commands])].slice(-CONTROL_QUEUE_LIMIT);
    await saveAppState(key, { queue: next });
    return next;
};

const consumeSteeringCommands = async (runId) => {
    const key = buildResearchControlsKey(runId);
    const record = await loadAppState(key);
    const queue = Array.isArray(record?.state?.queue) ? record.state.queue : [];
    if (queue.length) {
        await saveAppState(key, { queue: [] });
    }
    return queue;
};

module.exports = {
    createResearchRunId,
    resolveResearchScope,
    buildResearchRunKey,
    buildResearchMemoryKey,
    createResearchRun,
    loadResearchRun,
    saveResearchRun,
    updateResearchRun,
    appendResearchCheckpoint,
    appendResearchIndex,
    loadResearchIndex,
    loadResearchRunList: listResearchRuns,
    updateResearchMemoryFromRun,
    enqueueSteeringCommands,
    consumeSteeringCommands,
};
