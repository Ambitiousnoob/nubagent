const DEFAULT_OUTPUT_MODE = "state_of_the_field";
const DEFAULT_DEPTH_PREFERENCE = "balanced";
const DEFAULT_REFINEMENT_BUDGET = 3;
const DEFAULT_MAX_QUERIES = 6;
const MAX_ANSWER_CHARS = 14000;
const MAX_SOURCE_COUNT = 8;

const normalizeText = (value) => String(value ?? "").replace(/\u0000/g, "").trim();

const truncateText = (value, max = MAX_ANSWER_CHARS) => {
    const text = String(value ?? "").trim();
    if (!text) return "";
    return text.length > max ? `${text.slice(0, max)}...` : text;
};

const normalizeDepthPreference = (value) => {
    const normalized = normalizeText(value).toLowerCase();
    return ["speed", "balanced", "deep"].includes(normalized) ? normalized : DEFAULT_DEPTH_PREFERENCE;
};

const normalizeOutputMode = (value) => {
    const normalized = normalizeText(value).toLowerCase().replace(/\s+/g, "_");
    return normalized || DEFAULT_OUTPUT_MODE;
};

const normalizeBoundedInteger = (value, fallback, minimum, maximum) => {
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) return fallback;
    return Math.max(minimum, Math.min(maximum, Math.floor(numeric)));
};

const normalizeKeyMap = (value = {}) => {
    const candidate = value && typeof value === "object" ? value : {};
    return Object.fromEntries(
        Object.entries(candidate)
            .map(([key, item]) => [String(key), normalizeText(item)])
            .filter(([, item]) => item),
    );
};

const summarizeSources = (sources = []) => (
    (Array.isArray(sources) ? sources : [])
        .slice(0, MAX_SOURCE_COUNT)
        .map((source) => ({
            citationIndex: Number(source?.citationIndex || 0) || undefined,
            title: normalizeText(source?.title),
            url: normalizeText(source?.url),
            tier: normalizeText(source?.tier || "supporting") || "supporting",
        }))
        .filter((source) => source.title && source.url)
);

const buildDelegatedResearchPayload = (run = {}) => {
    const outputMode = run.plan?.outputMode || null;
    const sources = summarizeSources(run.result?.sources);
    const finalText = truncateText(run.result?.finalText || run.result?.markdown || run.result?.body || "");
    const noGroundedSources = normalizeText(run.result?.sourceSelection?.mode) === "no_grounded_sources";

    return {
        delegated: true,
        runId: run.id || run.runId || "",
        heading: normalizeText(run.result?.heading || "Research Answer") || "Research Answer",
        answer: finalText,
        outputMode: outputMode
            ? {
                id: normalizeText(outputMode.id),
                label: normalizeText(outputMode.label),
            }
            : null,
        sourceSelection: run.result?.sourceSelection || null,
        noGroundedSources,
        sources,
        sourceCount: sources.length,
        tribunal: run.tribunal || null,
        convergence: run.convergence || null,
        verifierSummary: run.verifierSummary || null,
    };
};

const definition = {
    type: "function",
    function: {
        name: "delegate_research",
        strict: true,
        description: "Delegate a complex or source-grounded question to the research orchestration subagents. Use this when the answer needs multi-step web research, verifier passes, conflicting-evidence handling, or stronger source grounding than a direct answer from memory.",
        parameters: {
            type: "object",
            properties: {
                query: {
                    type: "string",
                    description: "The research question or task to delegate to the research subagent swarm.",
                },
                depth_preference: {
                    type: "string",
                    enum: ["speed", "balanced", "deep"],
                    description: "How deep the research run should go.",
                },
                output_mode: {
                    type: "string",
                    description: "Optional output mode such as state_of_the_field, controversy_map, gap_analysis, decision_brief, policy_recommendation, or engineering_action_plan.",
                },
                refinement_budget: {
                    type: "integer",
                    minimum: 1,
                    maximum: 6,
                    description: "Maximum targeted refinement cycles for the verifier tribunal.",
                },
                max_queries: {
                    type: "integer",
                    minimum: 1,
                    maximum: 12,
                    description: "Maximum search lanes to execute in the research runtime.",
                },
            },
            required: ["query"],
            additionalProperties: false,
        },
    },
};

const handler = async (args = {}, context = {}) => {
    const query = normalizeText(args?.query);
    if (!query) return "Error: query is required";

    const requestBody = context?.requestBody && typeof context.requestBody === "object"
        ? context.requestBody
        : {};
    const delegationContext = context?.delegationContext && typeof context.delegationContext === "object"
        ? context.delegationContext
        : {};
    const scopeKey = normalizeText(
        delegationContext.scopeKey
        || requestBody.stateKey
        || requestBody.scopeKey
        || "chat:delegate_research",
    ) || "chat:delegate_research";

    const { runResearch } = require("../research-runtime");

    try {
        const run = await runResearch({
            query,
            scopeKey,
            depthPreference: normalizeDepthPreference(args?.depth_preference || args?.depthPreference || requestBody.depthPreference),
            forcedOutputMode: normalizeOutputMode(args?.output_mode || args?.outputMode || requestBody.forcedOutputMode || requestBody.outputMode),
            refinementBudget: normalizeBoundedInteger(
                args?.refinement_budget ?? args?.refinementBudget ?? requestBody.refinementBudget,
                DEFAULT_REFINEMENT_BUDGET,
                1,
                6,
            ),
            maxQueries: normalizeBoundedInteger(
                args?.max_queries ?? args?.maxQueries ?? requestBody.maxQueries,
                DEFAULT_MAX_QUERIES,
                1,
                12,
            ),
            searchProviderKeys: normalizeKeyMap(requestBody.searchProviderKeys),
            researchProvider: normalizeText(requestBody.researchProvider),
            researchModel: normalizeText(requestBody.researchModel),
            researchModelChain: Array.isArray(requestBody.researchModelChain) ? requestBody.researchModelChain : [],
            researchRoundRobin: requestBody.researchRoundRobin === true,
            researchProviderKeys: normalizeKeyMap(requestBody.researchProviderKeys),
        });

        return JSON.stringify(buildDelegatedResearchPayload(run));
    } catch (error) {
        return `Error: delegated research failed (${normalizeText(error?.message || error || "unknown error") || "unknown error"})`;
    }
};

module.exports = {
    definition,
    handler,
};
