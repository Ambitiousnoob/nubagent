const crypto = require("node:crypto");
const { runLiteHostChat } = require("./litehost-chat");
const { rankEvidenceEntriesForQuery, getSourceDomain } = require("./rag");
const {
    compileResearchPlan,
    buildConvergenceMetrics,
    buildTribunalSummary,
    topologicalBatches,
} = require("./research-plan");
const { searchResearchSources } = require("./research-sources");
const { fetchTieredEvidence, extractResearchArtifacts } = require("./research-extraction");
const {
    loadResearchRun,
    saveResearchRun,
    indexResearchRun,
    updateResearchMemoryFromRun,
    findRelevantResearchContext,
    formatResearchContext,
    pullPendingResearchControls,
    markResearchControlsApplied,
} = require("./research-memory");

const CHECKPOINT_BY_NODE = {
    cognitiveCommandLayer: "plan",
    tieredEpistemicFilter: "inventory",
    deepComprehensionEngine: "summaries",
    dialecticalSynthesisEngine: "draft",
    recursiveSelfImprovementLoop: "tribunal",
    decisionIntelligenceLayer: "decision",
    adaptiveDeliveryHub: "final",
};

const RETRYABLE_NODE_FAILURES = new Set(["soft"]);
const DECISION_OUTPUT_MODES = new Set(["decision_brief", "policy_recommendation", "engineering_action_plan"]);
const VERIFIER_SPECS = Object.freeze({
    claim_support: {
        label: "ClaimVerifier",
        threshold: 0.8,
        system: "You are ClaimVerifier inside Research Framework v3.1. Return strict JSON with keys score, issues, and rewrite_brief. Check every substantive claim in the draft against the supplied evidence and identify unsupported, overreaching, or hallucinated claims.",
    },
    citation_integrity: {
        label: "CitationVerifier",
        threshold: 0.8,
        system: "You are CitationVerifier inside Research Framework v3.1. Return strict JSON with keys score, issues, and rewrite_brief. Check whether each bracketed citation is used faithfully, whether cited sources are Core or Supporting, and whether claims drift beyond the cited evidence.",
    },
    contradiction_handling: {
        label: "ContradictionVerifier",
        threshold: 0.76,
        system: "You are ContradictionVerifier inside Research Framework v3.1. Return strict JSON with keys score, issues, and rewrite_brief. Check whether the draft addresses counter-hypotheses, disconfirming evidence, and disagreement instead of implying false consensus.",
    },
    uncertainty_calibration: {
        label: "UncertaintyVerifier",
        threshold: 0.78,
        system: "You are UncertaintyVerifier inside Research Framework v3.1. Return strict JSON with keys score, issues, and rewrite_brief. Check whether confidence, residual uncertainty, caveats, and limits are calibrated to the available evidence.",
    },
});
const CONTROL_RESTART_ORDER = [
    "cognitiveCommandLayer",
    "adversarialQueryForge",
    "intelligentCrawlerMesh",
    "tieredEpistemicFilter",
    "deepComprehensionEngine",
    "dialecticalSynthesisEngine",
    "recursiveSelfImprovementLoop",
    "decisionIntelligenceLayer",
    "adaptiveDeliveryHub",
];
const STATE_KEYS_BY_NODE = Object.freeze({
    cognitiveCommandLayer: ["plan"],
    activeSafetyAndEthics: ["safety"],
    adversarialQueryForge: ["queryForge"],
    intelligentCrawlerMesh: ["sourceMesh"],
    tieredEpistemicFilter: ["tieredSources", "inventory"],
    deepComprehensionEngine: ["evidenceEntries", "extraction"],
    dialecticalSynthesisEngine: ["positionMaps", "thesis", "antithesis", "draft"],
    recursiveSelfImprovementLoop: ["finalText", "tribunal", "convergence", "claimLedger", "postmortem", "verifierSummary"],
    decisionIntelligenceLayer: ["decisionLayer"],
});

class ResearchRuntimeError extends Error {
    constructor(message, options = {}) {
        super(message);
        this.name = "ResearchRuntimeError";
        this.failureType = options.failureType || "critical";
        this.nodeId = options.nodeId || "";
        this.detail = options.detail || "";
    }
}

const normalizeText = (value) => String(value ?? "").replace(/\u0000/g, "").trim();
const unique = (values = []) => [...new Set((Array.isArray(values) ? values : []).filter(Boolean))];
const clampScore = (value, fallback = 0.5) => {
    const resolved = Number(value);
    if (!Number.isFinite(resolved)) return fallback;
    return Math.max(0, Math.min(1, resolved));
};

const normalizeRefinementBudget = (value, fallback = 3) => {
    const resolved = Number(value);
    if (!Number.isFinite(resolved)) return fallback;
    return Math.max(1, Math.min(6, Math.floor(resolved)));
};
const normalizeDepthPreference = (value, fallback = "balanced") => {
    const resolved = normalizeText(value).toLowerCase();
    return ["speed", "balanced", "deep"].includes(resolved) ? resolved : fallback;
};

const isDecisionOutputMode = (outputModeId = "") => DECISION_OUTPUT_MODES.has(normalizeText(outputModeId).toLowerCase());

const buildRunId = () => `research-${Date.now()}-${crypto.randomUUID().slice(0, 8)}`;

const parseJsonObject = (value) => {
    if (value && typeof value === "object" && !Array.isArray(value)) return value;
    const text = normalizeText(value);
    if (!text) return null;

    const fenced = text.match(/```(?:json)?\s*([\s\S]*?)```/i);
    const candidate = fenced?.[1] || text;
    const match = candidate.match(/\{[\s\S]*\}/);
    if (!match) return null;

    try {
        return JSON.parse(match[0]);
    } catch {
        return null;
    }
};

const runResearchModel = async ({
    messages,
    maxTokens = 2200,
    temperature = 0.1,
}) => {
    const result = await runLiteHostChat({
        model: "nub-agent",
        stream: false,
        use_tools: false,
        research_mode: true,
        max_tokens: maxTokens,
        temperature,
        messages,
    });
    return normalizeText(result?.reply?.content || "");
};

const extractAnswerParts = (text = "") => {
    const clean = String(text || "").replace(/##\s*Sources?[\s\S]*$/i, "").trim();
    const lines = clean.split("\n");
    const heading = (lines[0] || "").replace(/^#+\s*/, "").trim();
    const body = lines.length > 1 ? lines.slice(1).join("\n").trim() : clean;
    return {
        heading: heading || "Answer",
        body,
    };
};

const chunkArray = (items = [], size = 1) => {
    const result = [];
    const chunkSize = Math.max(1, Number(size) || 1);
    for (let index = 0; index < items.length; index += chunkSize) {
        result.push(items.slice(index, index + chunkSize));
    }
    return result;
};

const buildSourceIndex = (sources = []) => (
    (Array.isArray(sources) ? sources : [])
        .map((source) => `[${source.citationIndex}] ${source.title} — ${source.url}`)
        .join("\n")
);

const buildPlanBrief = (plan = {}, memoryContextString = "") => {
    const lines = [
        `Framework: Research Framework v${plan.frameworkVersion || "3.1"}`,
        `Domain: ${plan.domain?.label || "General Research"}${plan.domain?.taxonomy ? ` (${plan.domain.taxonomy})` : ""}`,
        `Scope: ${plan.scope?.label || "Broad Research"}`,
        `Output mode: ${plan.outputMode?.label || "State-of-the-Field"}`,
        `Pareto mode: ${plan.pareto?.mode || "balanced"}`,
        `Search lanes: ${Math.max(1, Number(plan.searchQueries?.length) || 0)}`,
        `Counter-hypotheses: ${(plan.queryMatrix?.counterHypotheses || []).length}`,
        `Safety checks: ${plan.safety?.activeCount || 0}`,
    ];

    if (Array.isArray(plan.intentConfidence?.ambiguousAxes) && plan.intentConfidence.ambiguousAxes.length) {
        lines.push(`Ambiguous axes: ${plan.intentConfidence.ambiguousAxes.join(", ")}`);
    }
    if (plan.continuity?.active && plan.continuity?.overlap) {
        lines.push(`Session continuity overlap: ${Math.round(plan.continuity.overlap * 100)}%`);
    }
    if (memoryContextString) {
        lines.push(`Research memory linked: yes`);
    }

    return lines.join("\n");
};

const buildOutputModeInstruction = (outputMode = {}) => {
    switch (outputMode?.id) {
    case "tutorial":
        return "Organize the response as a tutorial with setup, core concepts, and practical application guidance.";
    case "controversy_map":
        return "Organize the response as a controversy map with the dominant view, strongest counter-view, and unresolved tensions.";
    case "gap_analysis":
        return "Organize the response as a gap analysis with current consensus, missing evidence, and the highest-value next questions.";
    case "replication_crisis_report":
        return "Organize the response as a replication crisis report with reproducibility risks, failed replications, and robustness signals.";
    case "foundational_review":
        return "Organize the response as a foundational review with historical context, seminal ideas, and the current field position.";
    case "decision_brief":
        return "Organize the response as a decision brief with recommendation, tradeoffs, risk profile, confidence, and reversibility.";
    case "policy_recommendation":
        return "Organize the response as a policy recommendation with recommendation, stakeholder impact, risk profile, and implementation caveats.";
    case "engineering_action_plan":
        return "Organize the response as an engineering action plan with recommendation, rollout steps, risk profile, and reversibility.";
    default:
        return "Organize the response as a state-of-the-field briefing with consensus, disagreement, evidence quality, and practical takeaways.";
    }
};

const buildContinuityBlock = (continuity = {}, memoryContextString = "") => {
    const parts = [];
    if (continuity?.active && continuity?.summary) {
        parts.push(`Prior session continuity (${Math.round((continuity.overlap || 0) * 100)}% overlap):\n${continuity.summary}`);
    }
    if (memoryContextString) {
        parts.push(memoryContextString);
    }
    return parts.join("\n\n").trim();
};

const buildCounterHypothesisBlock = (plan = {}) => {
    const counterHypotheses = Array.isArray(plan.queryMatrix?.counterHypotheses)
        ? plan.queryMatrix.counterHypotheses
        : [];
    return counterHypotheses.length
        ? counterHypotheses.map((item) => `- ${item}`).join("\n")
        : "- No explicit counter-hypotheses were generated.";
};

const buildConstraintBlock = (plan = {}) => (
    (Array.isArray(plan.inlineConstraints) ? plan.inlineConstraints : [])
        .map((constraint) => `- ${constraint.constraint}`)
        .join("\n")
);

const buildPromptPatchBlock = (plan = {}) => (
    unique(
        (Array.isArray(plan.promptPatches) ? plan.promptPatches : [])
            .map((patch) => normalizeText(typeof patch === "string" ? patch : patch?.text || patch?.prompt || ""))
            .filter(Boolean),
    )
        .map((patch) => `- ${patch}`)
        .join("\n")
);

const assignCitationIndices = (sources = []) => (
    (Array.isArray(sources) ? sources : []).map((source, index) => ({
        ...source,
        citationIndex: index + 1,
    }))
);

const classifyFailure = (error, nodeId = "") => {
    if (error instanceof ResearchRuntimeError) return error;
    const message = normalizeText(error?.message || error || "Research runtime failure");
    const failureType = /timed out|timeout|429|503|overloaded|retry/i.test(message)
        ? "soft"
        : /fetch|provider|search|read|content|supplementary/i.test(message)
            ? "partial"
            : "critical";
    return new ResearchRuntimeError(message, {
        failureType,
        nodeId,
        detail: message,
    });
};

const createCheckpointPayload = (id, payload = {}) => ({
    id,
    at: new Date().toISOString(),
    ...payload,
});

const emitRuntimeEvent = async (context, type, payload = {}) => {
    const event = {
        type,
        runId: context.run.id || context.run.runId || "",
        at: new Date().toISOString(),
        researchMeta: buildRuntimeMeta({
            run: context.run,
            plan: context.plan,
            state: context.state,
        }),
        ...payload,
    };
    context.run.events = [...(Array.isArray(context.run.events) ? context.run.events : []), event].slice(-80);
    if (typeof context.onEvent === "function") {
        await context.onEvent(event);
    }
};

const persistRuntimeState = async (context) => {
    context.run.updatedAt = new Date().toISOString();
    context.run = await saveResearchRun(context.run);
    return context.run;
};

const writeCheckpoint = async (context, id, payload = {}) => {
    context.run.checkpoints = {
        ...(context.run.checkpoints || {}),
        [id]: createCheckpointPayload(id, payload),
    };
    const phase = (Array.isArray(context.plan?.dag?.nodes) ? context.plan.dag.nodes : [])
        .find((node) => CHECKPOINT_BY_NODE[node.id] === id)?.label || id;
    await emitRuntimeEvent(context, "checkpoint", {
        checkpoint: id,
        phase,
        payload,
    });
    await persistRuntimeState(context);
};

const shouldPauseAfterCheckpoint = (context, checkpointId) => {
    if (!context.stopAfterCheckpoint) return false;
    if (context.stopAfterCheckpoint === true) {
        return checkpointId !== "final";
    }
    return String(context.stopAfterCheckpoint) === String(checkpointId);
};

const buildCitationValidation = (text = "", sources = []) => {
    const citedNumbers = unique(
        [...String(text || "").matchAll(/\[(\d+)]/g)]
            .map((match) => Number(match[1]))
            .filter((value) => Number.isFinite(value)),
    );
    const byCitation = new Map((Array.isArray(sources) ? sources : []).map((source) => [Number(source.citationIndex), source]));
    const unsupported = citedNumbers.filter((citation) => {
        const source = byCitation.get(citation);
        return !source || !["core", "supporting"].includes(source.tier);
    });

    return {
        citedNumbers,
        unsupported,
    };
};

const parseTribunalScores = (text = "") => {
    const parsed = parseJsonObject(text);
    if (!parsed) return {
        internal_consistency: 0.78,
        coverage: 0.78,
        user_goal_alignment: 0.82,
        targeted_dimension: "coverage",
        rewrite_brief: "",
    };

    const critics = parsed.critics && typeof parsed.critics === "object" ? parsed.critics : {};
    return {
        internal_consistency: Math.max(0, Math.min(1, Number(
            critics.internal_consistency ?? critics.internalConsistency ?? parsed.internal_consistency ?? parsed.internalConsistency ?? 0.78,
        ) || 0.78)),
        coverage: Math.max(0, Math.min(1, Number(critics.coverage ?? parsed.coverage ?? 0.78) || 0.78)),
        user_goal_alignment: Math.max(0, Math.min(1, Number(
            critics.user_goal_alignment ?? critics.userGoalAlignment ?? parsed.user_goal_alignment ?? parsed.userGoalAlignment ?? 0.82,
        ) || 0.82)),
        targeted_dimension: normalizeText(parsed.targeted_dimension || parsed.targetedDimension || parsed.focus || "coverage")
            .replace(/\s+/g, "_")
            .toLowerCase(),
        rewrite_brief: normalizeText(parsed.rewrite_brief || parsed.rewriteBrief || parsed.guidance || ""),
    };
};

const parseVerifierAssessment = (text = "", defaults = {}) => {
    const parsed = parseJsonObject(text);
    if (!parsed) {
        return {
            score: clampScore(defaults.score, 0.78),
            issues: [],
            rewrite_brief: "",
        };
    }

    const issues = unique(
        [
            ...(Array.isArray(parsed.issues) ? parsed.issues : []),
            ...(Array.isArray(parsed.unsupported_claims) ? parsed.unsupported_claims : []),
            ...(Array.isArray(parsed.overreach_claims) ? parsed.overreach_claims : []),
            ...(Array.isArray(parsed.broken_citations) ? parsed.broken_citations : []),
            ...(Array.isArray(parsed.weak_citations) ? parsed.weak_citations : []),
            ...(Array.isArray(parsed.omitted_counterevidence) ? parsed.omitted_counterevidence : []),
            ...(Array.isArray(parsed.false_consensus_flags) ? parsed.false_consensus_flags : []),
            ...(Array.isArray(parsed.overconfident_statements) ? parsed.overconfident_statements : []),
            ...(Array.isArray(parsed.missing_caveats) ? parsed.missing_caveats : []),
        ]
            .map((item) => normalizeText(typeof item === "string" ? item : item?.claim || item?.issue || item?.text || ""))
            .filter(Boolean),
    );

    return {
        score: clampScore(
            parsed.score
            ?? parsed.support_score
            ?? parsed.citation_score
            ?? parsed.coverage_score
            ?? parsed.calibration_score
            ?? defaults.score,
            defaults.score ?? 0.78,
        ),
        issues,
        rewrite_brief: normalizeText(parsed.rewrite_brief || parsed.rewriteBrief || parsed.guidance || ""),
    };
};

const pickLowestDimension = (scores = {}) => {
    const entries = Object.entries(scores)
        .filter(([, value]) => Number.isFinite(value))
        .sort((left, right) => left[1] - right[1]);
    return entries[0]?.[0] || "coverage";
};

const averageScore = (values = [], fallback = 0.78) => {
    const numericValues = (Array.isArray(values) ? values : [])
        .map((value) => Number(value))
        .filter((value) => Number.isFinite(value));
    if (!numericValues.length) return fallback;
    return numericValues.reduce((sum, value) => sum + value, 0) / numericValues.length;
};

const buildEvidenceDigest = (evidenceEntries = []) => (
    (Array.isArray(evidenceEntries) ? evidenceEntries : [])
        .slice(0, 6)
        .map((entry) => {
            const citation = entry?.source?.citationIndex ? `[${entry.source.citationIndex}] ` : "";
            const title = normalizeText(entry?.source?.title || "Untitled source");
            const excerpt = normalizeText(entry?.content || entry?.excerpt || entry?.evidenceBlock || "").slice(0, 280);
            return `${citation}${title}\n${excerpt}`;
        })
        .join("\n\n")
);

const computeTextSimilarity = (left = "", right = "") => {
    const leftTokens = new Set(normalizeText(left).toLowerCase().split(/[^a-z0-9]+/i).filter(Boolean));
    const rightTokens = new Set(normalizeText(right).toLowerCase().split(/[^a-z0-9]+/i).filter(Boolean));
    if (!leftTokens.size && !rightTokens.size) return 1;
    if (!leftTokens.size || !rightTokens.size) return 0;

    let intersection = 0;
    for (const token of leftTokens) {
        if (rightTokens.has(token)) intersection += 1;
    }

    const union = new Set([...leftTokens, ...rightTokens]).size || 1;
    return intersection / union;
};

const buildVerifierSummary = ({
    claimAssessment = {},
    citationAssessment = {},
    contradictionAssessment = {},
    uncertaintyAssessment = {},
    citationValidation = {},
} = {}) => {
    const unsupportedCitationPenalty = Math.min(0.35, (citationValidation.unsupported || []).length * 0.08);
    const dimensions = {
        claim_support: clampScore(claimAssessment.score, 0.78),
        citation_integrity: clampScore((citationAssessment.score ?? 0.78) - unsupportedCitationPenalty, 0.72),
        contradiction_handling: clampScore(contradictionAssessment.score, 0.76),
        uncertainty_calibration: clampScore(uncertaintyAssessment.score, 0.78),
    };

    return {
        dimensions,
        unsupportedCitationCount: (citationValidation.unsupported || []).length,
        unsupportedCitationNumbers: citationValidation.unsupported || [],
        issues: unique([
            ...(claimAssessment.issues || []),
            ...(citationAssessment.issues || []),
            ...(contradictionAssessment.issues || []),
            ...(uncertaintyAssessment.issues || []),
        ]),
        assessments: {
            claimVerifier: claimAssessment,
            citationVerifier: citationAssessment,
            contradictionVerifier: contradictionAssessment,
            uncertaintyVerifier: uncertaintyAssessment,
        },
    };
};

const buildRewriteGuidance = ({
    critic = {},
    verifierSummary = {},
    targetedDimension = "",
} = {}) => {
    const dimensionToAssessment = {
        claim_support: verifierSummary.assessments?.claim_support || verifierSummary.assessments?.claimVerifier,
        citation_integrity: verifierSummary.assessments?.citation_integrity || verifierSummary.assessments?.citationVerifier,
        contradiction_handling: verifierSummary.assessments?.contradiction_handling || verifierSummary.assessments?.contradictionVerifier,
        uncertainty_calibration: verifierSummary.assessments?.uncertainty_calibration || verifierSummary.assessments?.uncertaintyVerifier,
    };

    return unique([
        normalizeText(critic.rewrite_brief),
        normalizeText(dimensionToAssessment[targetedDimension]?.rewrite_brief),
        ...((verifierSummary.issues || []).slice(0, 4)),
    ]).filter(Boolean).join(" ").trim();
};

const buildExtractionSnapshot = (extraction = {}) => {
    const claims = Array.isArray(extraction.claims) ? extraction.claims : [];
    const repositories = Array.isArray(extraction.repositories) ? extraction.repositories : [];
    const supplementary = Array.isArray(extraction.supplementary) ? extraction.supplementary : [];
    const concepts = Array.isArray(extraction.concepts) ? extraction.concepts : [];
    const contradictions = Array.isArray(extraction.contradictions) ? extraction.contradictions : [];
    const metaAnalysis = extraction.metaAnalysis || null;
    const safety = extraction.safety || {};
    const sampleSizes = claims
        .filter((claim) => claim.type === "sample_size")
        .map((claim) => Number(claim.value))
        .filter(Number.isFinite);
    const reproducibilityAverage = Number(extraction.codeArtifacts?.reproducibilityAverage || 0);
    const claimPreview = claims
        .slice(0, 4)
        .map((claim) => `${claim.metric || claim.type || "claim"} ${claim.operator || ""} ${claim.value ?? ""}`.trim())
        .filter(Boolean);
    const contradictionPreview = contradictions
        .slice(0, 3)
        .map((item) => normalizeText(item?.claim || item?.summary || item?.issue || item))
        .filter(Boolean);

    return [
        `Extracted claims: ${claims.length}`,
        `Repositories: ${repositories.length}`,
        `Supplementary files: ${supplementary.length}`,
        `Linked concepts: ${concepts.length}`,
        sampleSizes.length
            ? `Sample sizes: ${sampleSizes.length} extracted, min ${Math.min(...sampleSizes)}, max ${Math.max(...sampleSizes)}`
            : "Sample sizes: not available",
        repositories.length
            ? `Repository reproducibility average: ${reproducibilityAverage}`
            : "Repository reproducibility: not available",
        contradictionPreview.length ? `Contradiction signals:\n- ${contradictionPreview.join("\n- ")}` : "Contradiction signals: none captured",
        claimPreview.length ? `Claim preview:\n- ${claimPreview.join("\n- ")}` : "Claim preview: none captured",
        metaAnalysis?.combined_effect_size != null
            ? `Meta-analysis: effect ${metaAnalysis.combined_effect_size}, I2 ${metaAnalysis.i_squared ?? "n/a"}, model ${metaAnalysis.model || "n/a"}`
            : "Meta-analysis: not available",
        `Safety signals: funding ${Number(safety.fundingConflictCount || 0)}, predatory ${Number(safety.predatoryCount || 0)}, dual-use ${Number(safety.dualUseCount || 0)}, manipulation ${Number(safety.manipulationCount || 0)}`,
    ].join("\n");
};

const buildVerifierIssueList = (dimension = "", issues = []) => (
    unique((Array.isArray(issues) ? issues : []).map((issue) => normalizeText(issue)).filter(Boolean))
        .slice(0, 6)
        .map((issue) => `${dimension}: ${issue}`)
);

const runVerifierSwarm = async ({
    plan,
    planBrief,
    constraintBlock,
    counterHypothesisBlock,
    sourceIndex,
    currentDraft,
    evidenceEntries,
    extraction,
    citationValidation,
    thesis = "",
    antithesis = "",
    continuityBlock = "",
}) => {
    const promptPatchBlock = buildPromptPatchBlock(plan);
    const verifierInput = [
        `Research plan:\n${planBrief}`,
        `Constraints:\n${constraintBlock || "- No explicit inline constraints."}`,
        `Counter-hypotheses:\n${counterHypothesisBlock}`,
        ...(promptPatchBlock ? [`Prompt patches:\n${promptPatchBlock}`] : []),
        `Source index:\n${sourceIndex}`,
        `Evidence digest:\n${buildEvidenceDigest(evidenceEntries) || "No evidence digest available."}`,
        `Extraction snapshot:\n${buildExtractionSnapshot(extraction)}`,
        `Unsupported citations count: ${citationValidation.unsupported.length}`,
        thesis ? `Thesis:\n${thesis}` : "",
        antithesis ? `Antithesis:\n${antithesis}` : "",
        continuityBlock ? `Continuity:\n${continuityBlock}` : "",
        `Draft:\n${currentDraft}`,
    ].join("\n\n");

    const verifierEntries = await Promise.all(
        Object.entries(VERIFIER_SPECS).map(async ([dimension, spec]) => {
            const response = await runResearchModel({
                maxTokens: 900,
                messages: [
                    {
                        role: "system",
                        content: spec.system,
                    },
                    {
                        role: "user",
                        content: verifierInput,
                    },
                ],
            });
            const assessment = parseVerifierAssessment(response, { score: spec.threshold });
            if (dimension === "citation_integrity" && citationValidation.unsupported.length) {
                assessment.score = Math.min(assessment.score, 0.56);
                assessment.issues = unique([
                    ...assessment.issues,
                    `Unsupported or non-Core citations: ${citationValidation.unsupported.map((citation) => `[${citation}]`).join(", ")}`,
                ]);
                assessment.rewrite_brief = `${assessment.rewrite_brief} Remove or replace claims that cite peripheral, discarded, or missing sources.`.trim();
            }
            return [dimension, assessment];
        }),
    );

    const assessments = Object.fromEntries(verifierEntries);
    const scores = Object.fromEntries(verifierEntries.map(([dimension, assessment]) => [dimension, assessment.score]));
    const rewriteHints = verifierEntries
        .map(([dimension, assessment]) => {
            const brief = normalizeText(assessment.rewrite_brief);
            return brief ? `${dimension}: ${brief}` : "";
        })
        .filter(Boolean);

    return {
        assessments,
        scores,
        dimensions: scores,
        targetedDimension: pickLowestDimension(scores),
        unsupportedCitationCount: citationValidation.unsupported.length,
        issues: verifierEntries.flatMap(([dimension, assessment]) => buildVerifierIssueList(dimension, assessment.issues)).slice(0, 16),
        rewriteBrief: rewriteHints.join(" ").trim(),
        aggregateScore: Number(averageScore(Object.values(scores), 0.78).toFixed(2)),
    };
};

const buildClaimLedger = ({
    positionMaps = [],
    extraction = {},
    tieredSources = [],
    tribunal = null,
    verifierSummary = null,
}) => {
    const sourceByCitation = new Map((Array.isArray(tieredSources) ? tieredSources : []).map((source) => [Number(source.citationIndex), source]));
    const keyClaims = positionMaps.flatMap((map) => Array.isArray(map?.key_claims) ? map.key_claims : []);
    const extractedClaims = Array.isArray(extraction.claims) ? extraction.claims : [];
    const effectClaims = extractedClaims.filter((claim) => claim.type === "effect_size" || claim.type === "p_value").slice(0, 12);
    const contradictionPenalty = verifierSummary?.scores?.contradiction_handling != null
        ? 1 - clampScore(verifierSummary.scores.contradiction_handling, 0.76)
        : null;
    const uncertaintyPenalty = verifierSummary?.scores?.uncertainty_calibration != null
        ? 1 - clampScore(verifierSummary.scores.uncertainty_calibration, 0.78)
        : null;

    const structuredClaims = keyClaims.map((claim) => {
        const citations = unique((Array.isArray(claim?.citations) ? claim.citations : []).map((value) => Number(value)).filter(Number.isFinite));
        const tierWeights = citations.map((citation) => {
            const source = sourceByCitation.get(citation);
            if (!source) return 0.35;
            if (source.tier === "core") return 0.92;
            if (source.tier === "supporting") return 0.74;
            if (source.tier === "peripheral") return 0.5;
            return 0.25;
        });
        const evidenceWeight = tierWeights.length
            ? Math.max(...tierWeights)
            : 0.42;
        const contradictionScore = contradictionPenalty == null
            ? (tribunal?.targeted_dimension === "internal_consistency" ? 0.24 : 0.16)
            : Math.max(0.1, contradictionPenalty);
        const sensitivity = uncertaintyPenalty == null
            ? (/uncertain|limited|small sample|pilot/i.test(claim?.claim || "") ? 0.44 : 0.24)
            : Math.max(/uncertain|limited|small sample|pilot/i.test(claim?.claim || "") ? 0.44 : 0.24, uncertaintyPenalty);
        const confidence = Math.max(0, Math.min(1, Number((evidenceWeight * (1 - contradictionScore / 2) * (1 - sensitivity / 2)).toFixed(2))));
        return {
            claim: normalizeText(claim?.claim),
            citations,
            confidence,
            evidence_weight: Number(evidenceWeight.toFixed(2)),
            contradiction_score: Number(contradictionScore.toFixed(2)),
            sensitivity: Number(sensitivity.toFixed(2)),
        };
    }).filter((claim) => claim.claim);

    const quantitativeClaims = effectClaims.map((claim) => ({
        claim: `${claim.metric || claim.type} ${claim.operator || ""} ${claim.value}`.trim(),
        citations: [],
        confidence: claim.sampleSizeFlag ? 0.52 : Number((0.46 + ((1 - Math.min(0.45, uncertaintyPenalty ?? 0.22)) * 0.26)).toFixed(2)),
        evidence_weight: claim.sampleSizeFlag ? 0.48 : 0.64,
        contradiction_score: Number((Math.max(0.12, contradictionPenalty ?? 0.18)).toFixed(2)),
        sensitivity: claim.sampleSizeFlag ? 0.5 : Number((Math.max(0.18, uncertaintyPenalty ?? 0.26)).toFixed(2)),
    }));

    return [...structuredClaims, ...quantitativeClaims].slice(0, 20);
};

const buildDecisionLayer = ({ plan, convergence, safety, finalText, tribunal = null, verifierSummary = null }) => {
    if (!isDecisionOutputMode(plan?.outputMode?.id)) {
        return null;
    }

    const { heading } = extractAnswerParts(finalText);
    const epistemicRisk = Number((convergence?.residual_uncertainty ?? 0.28).toFixed(2));
    const technicalRisk = Number((Math.min(0.9, 0.18 + ((safety?.manipulationCount || 0) * 0.08) + ((safety?.dualUseCount || 0) * 0.12))).toFixed(2));
    const verifierScores = Object.values(verifierSummary?.scores || verifierSummary?.dimensions || {});
    const verifierRisk = Number((Math.max(0.08, 1 - clampScore(averageScore(verifierScores, 0.78), 0.78))).toFixed(2));
    const epistemicComposite = Number((Math.max(epistemicRisk, verifierRisk)).toFixed(2));
    const tribunalStrength = averageScore([
        tribunal?.critics?.internal_consistency,
        tribunal?.critics?.coverage,
        tribunal?.critics?.user_goal_alignment,
    ], 0.78);
    const confidence = Number((Math.max(
        0.12,
        Math.min(0.97, ((1 - epistemicComposite) * 0.55) + (tribunalStrength * 0.45)),
    )).toFixed(2));

    return {
        decision: heading || `Act on ${plan.query}`,
        expected_outcome: `Decision frame aligned to ${plan.outputMode.label}.`,
        risk_profile: {
            technical: technicalRisk,
            epistemic: epistemicComposite,
        },
        confidence,
        reversibility: epistemicRisk > 0.45 ? "low" : epistemicRisk > 0.28 ? "medium" : "high",
        support: {
            targeted_dimension: tribunal?.targeted_dimension || "",
            verifier_average: Number(averageScore(verifierScores, 0.78).toFixed(2)),
            verifier_issues: Array.isArray(verifierSummary?.issues) ? verifierSummary.issues.length : 0,
        },
    };
};

const csvEscape = (value) => `"${String(value ?? "").replace(/"/g, "\"\"")}"`;

const buildDatasetCsv = (run = {}) => {
    const rows = [["type", "source_title", "source_url", "value", "metric", "notes"]];
    for (const claim of (run.extraction?.claims || [])) {
        rows.push([
            claim.type,
            claim.sourceTitle || "",
            claim.sourceUrl || "",
            claim.value ?? "",
            claim.metric || claim.operator || "",
            claim.sampleSizeFlag ? "sample_size_flag" : "",
        ]);
    }
    for (const repo of (run.extraction?.repositories || [])) {
        rows.push([
            "repository",
            "",
            repo.url || "",
            repo.reproducibilityScore ?? "",
            "reproducibility_score",
            repo.notes || "",
        ]);
    }
    return rows.map((row) => row.map(csvEscape).join(",")).join("\n");
};

const buildMarkdownExport = (run = {}) => {
    const tags = unique([
        "research",
        run.plan?.domain?.id,
        run.plan?.scope?.id,
        run.plan?.outputMode?.id,
    ]).filter(Boolean);
    const frontmatter = [
        "---",
        `title: ${run.result?.heading || run.query}`,
        `run_id: ${run.id}`,
        `domain: ${run.plan?.domain?.label || ""}`,
        `scope: ${run.plan?.scope?.label || ""}`,
        `output_mode: ${run.plan?.outputMode?.label || ""}`,
        `created_at: ${run.createdAt}`,
        `tags: [${tags.join(", ")}]`,
        "---",
    ].join("\n");

    const concepts = (run.extraction?.concepts || []).slice(0, 12).map((concept) => `- [[${concept.name}]]`).join("\n");
    return [
        frontmatter,
        run.result?.finalText || "",
        concepts ? `## Backlinks\n${concepts}` : "",
    ].filter(Boolean).join("\n\n");
};

const buildSlideOutline = (run = {}) => {
    const { heading, body } = extractAnswerParts(run.result?.finalText || "");
    const bodyLines = body
        .split("\n")
        .map((line) => normalizeText(line))
        .filter((line) => line && !line.startsWith("#"))
        .slice(0, 12);

    return [
        `1. ${heading || run.query}`,
        `2. Research question\n   - ${run.query}`,
        `3. Field position\n   - ${(bodyLines[0] || "See report.")}`,
        `4. Core evidence\n   - ${(bodyLines[1] || "See report.")}`,
        `5. Disagreement and risk\n   - ${(bodyLines[2] || "See report.")}`,
        `6. Decision / next steps\n   - ${(bodyLines[3] || "See report.")}`,
    ].join("\n");
};

const buildRuntimeMeta = ({ run = {}, plan = null, state = null } = {}) => {
    const resolvedPlan = plan || run.plan || {};
    const resolvedState = state || run.state || {};
    const sourceMesh = resolvedState.sourceMesh || {};
    const tieredSources = Array.isArray(resolvedState.tieredSources) ? resolvedState.tieredSources : [];
    const evidenceEntries = Array.isArray(resolvedState.evidenceEntries) ? resolvedState.evidenceEntries : [];
    const extraction = run.extraction || resolvedState.extraction || {};
    const partialFailures = Array.isArray(run.partialFailures) ? run.partialFailures : [];

    return {
        frameworkVersion: resolvedPlan.frameworkVersion || "3.1",
        domain: resolvedPlan.domain || null,
        scope: resolvedPlan.scope || null,
        outputMode: resolvedPlan.outputMode || null,
        depthPreference: resolvedPlan.depthPreference || run.requestedDepthPreference || run.requestOptions?.depthPreference || null,
        intentConfidence: resolvedPlan.intentConfidence || null,
        pareto: resolvedPlan.pareto || null,
        continuity: resolvedPlan.continuity || null,
        promptPatches: resolvedPlan.promptPatches || [],
        safety: run.safety || resolvedState.safety || extraction.safety || resolvedPlan.safety || null,
        dag: resolvedPlan.dag || null,
        dagSummary: resolvedPlan.dagSummary || "",
        queryMatrix: resolvedPlan.queryMatrix || null,
        refinementBudget: Number(resolvedPlan.refinementBudget || 3) || 3,
        searchCount: Array.isArray(resolvedPlan.searchQueries) ? resolvedPlan.searchQueries.length : 0,
        rankedSites: Array.isArray(sourceMesh.sources) ? sourceMesh.sources.length : tieredSources.length,
        fetchPlanned: tieredSources.filter((source) => source?.fetchMode !== "skip").length,
        fetchedSites: evidenceEntries.length,
        fetchAttempts: evidenceEntries.length + partialFailures.length,
        synthesisWorkers: Math.max(1, Array.isArray(resolvedState.positionMaps) ? resolvedState.positionMaps.length : 0),
        tribunal: run.tribunal || resolvedState.tribunal || null,
        convergence: run.convergence || resolvedState.convergence || null,
        verifierSummary: run.verifierSummary || resolvedState.verifierSummary || null,
        decisionLayer: run.result?.decision || resolvedState.decisionLayer || null,
        controls: Array.isArray(run.controls) ? run.controls : [],
        partialFailures,
        ragLexical: true,
        subagents: resolvedPlan.subagents || [],
        generatedAt: run.updatedAt || run.createdAt || null,
        runId: run.id || run.runId || "",
        status: run.status || "in_progress",
        awaitingCheckpoint: run.awaitingCheckpoint || "",
    };
};

const buildOutputsPayload = (run = {}) => {
    const extraction = run.extraction || run.state?.extraction || {};
    return {
        markdown: run.result?.markdown || "",
        slide_deck_outline: run.result?.slides || "",
        dataset_csv: run.result?.datasetCsv || "",
        dataset_json: JSON.stringify({
            claims: extraction.claims || [],
            repositories: extraction.repositories || [],
            supplementary: extraction.supplementary || [],
            epistemicClaims: run.state?.claimLedger || [],
            verifierSummary: run.verifierSummary || run.state?.verifierSummary || null,
            decision: run.result?.decision || run.state?.decisionLayer || null,
            safety: run.safety || extraction.safety || null,
            postmortem: run.postmortem || null,
        }, null, 2),
    };
};

const serializeResearchRun = (run = {}) => {
    const state = run.state || {};
    const extraction = run.extraction || state.extraction || {};
    const outputs = buildOutputsPayload(run);

    return {
        ok: true,
        runId: run.id || run.runId || "",
        status: run.status || "in_progress",
        query: run.query || "",
        plan: run.plan || {},
        requestOptions: run.requestOptions || null,
        final: {
            heading: run.result?.heading || "",
            body: run.result?.body || "",
            answer: run.result?.finalText || "",
            markdown: run.result?.finalText || "",
            sources: Array.isArray(run.result?.sources) ? run.result.sources : [],
            tribunal: run.tribunal || state.tribunal || null,
            convergence: run.convergence || state.convergence || null,
            verifierSummary: run.verifierSummary || state.verifierSummary || null,
            decision: run.result?.decision || state.decisionLayer || null,
            claimLedger: run.result?.claimLedger || state.claimLedger || [],
            exports: outputs,
        },
        researchMeta: buildRuntimeMeta({ run }),
        structuredData: {
            paperInventory: state.tieredSources || [],
            deepComprehension: state.evidenceEntries || [],
            claims: extraction.claims || [],
            repositories: extraction.repositories || [],
            supplementary: extraction.supplementary || [],
            concepts: extraction.concepts || [],
            evidencePyramid: extraction.evidencePyramid || [],
            metaAnalysis: extraction.metaAnalysis || null,
            statisticalVerification: extraction.statisticalVerification || null,
            citationIntelligence: state.sourceMesh?.citationIntelligence || null,
            authorNetwork: state.sourceMesh?.authorNetwork || null,
            temporalTrends: state.sourceMesh?.temporalTrends || null,
            epistemicClaims: state.claimLedger || [],
            verifierSummary: run.verifierSummary || state.verifierSummary || null,
            decisionLayer: run.result?.decision || state.decisionLayer || null,
            safety: run.safety || extraction.safety || null,
            postmortem: run.postmortem || null,
        },
        checkpoints: run.checkpoints || {},
        events: Array.isArray(run.events) ? run.events : [],
        outputs,
    };
};

const resolveControls = (controls = []) => (
    Array.isArray(controls) ? controls.filter((control) => control && typeof control === "object") : []
);

const getForcedOutputMode = (controls = [], fallback = "") => {
    const forced = [...resolveControls(controls)]
        .reverse()
        .find((control) => control.type === "force_mode" && normalizeText(control.mode || control.value));
    return forced ? normalizeText(forced.mode || forced.value) : fallback;
};

const getDepthOverride = (controls = [], fallback = "") => {
    const latestControl = [...resolveControls(controls)]
        .reverse()
        .find((control) => ["increase_depth", "go_deeper", "prioritize_speed"].includes(normalizeText(control.type)));
    if (latestControl?.type === "increase_depth" || latestControl?.type === "go_deeper") return "deep";
    if (latestControl?.type === "prioritize_speed") return "speed";
    return fallback;
};

const applyControlFilters = (sources = [], controls = []) => {
    const excludedSourceTypes = new Set(
        resolveControls(controls)
            .filter((control) => control.type === "exclude_source" && normalizeText(control.sourceType || control.typeId))
            .map((control) => normalizeText(control.sourceType || control.typeId).toLowerCase()),
    );
    if (!excludedSourceTypes.size) return sources;

    return sources.filter((source) => {
        const haystack = [
            source.kind,
            source.provider,
            source.providerLabel,
            source.metadata?.type,
            getSourceDomain(source.url),
        ].map((value) => normalizeText(value).toLowerCase());
        return !haystack.some((value) => excludedSourceTypes.has(value));
    });
};

const normalizeAppliedControls = (controls = []) => resolveControls(controls).map((control) => {
    const payload = control?.payload && typeof control.payload === "object"
        ? control.payload
        : control;
    return {
        id: Number(control?.id || payload?.id || 0) || undefined,
        runId: normalizeText(control?.runId || payload?.runId || ""),
        status: normalizeText(control?.status || payload?.status || ""),
        createdAt: control?.createdAt || payload?.createdAt || "",
        updatedAt: control?.updatedAt || payload?.updatedAt || "",
        ...(payload && typeof payload === "object" ? payload : {}),
        type: normalizeText(control?.type || payload?.type || control?.command || payload?.command || ""),
    };
});

const getRestartNodeForControl = (control = {}) => {
    const type = normalizeText(control?.type || control?.command || "");
    if (type === "prioritize_speed" || type === "go_deeper" || type === "increase_depth") {
        return "adversarialQueryForge";
    }
    if (type === "exclude_source") return "tieredEpistemicFilter";
    if (type === "force_mode") return "dialecticalSynthesisEngine";
    return "recursiveSelfImprovementLoop";
};

const compareRestartNodes = (left = "", right = "") => {
    if (!left) return right;
    if (!right) return left;
    const leftIndex = CONTROL_RESTART_ORDER.indexOf(left);
    const rightIndex = CONTROL_RESTART_ORDER.indexOf(right);
    if (leftIndex < 0) return right;
    if (rightIndex < 0) return left;
    return leftIndex <= rightIndex ? left : right;
};

const getRestartNodeForPlanChange = ({
    previousPlan = {},
    nextPlan = {},
    controls = [],
} = {}) => {
    let restartNode = "";

    if (normalizeText(previousPlan?.depthPreference) !== normalizeText(nextPlan?.depthPreference)) {
        restartNode = compareRestartNodes(restartNode, "adversarialQueryForge");
    }
    if (normalizeText(previousPlan?.outputMode?.id) !== normalizeText(nextPlan?.outputMode?.id)) {
        restartNode = compareRestartNodes(restartNode, "dialecticalSynthesisEngine");
    }
    if (normalizeRefinementBudget(previousPlan?.refinementBudget, 3) !== normalizeRefinementBudget(nextPlan?.refinementBudget, 3)) {
        restartNode = compareRestartNodes(restartNode, "recursiveSelfImprovementLoop");
    }
    for (const control of normalizeAppliedControls(controls)) {
        restartNode = compareRestartNodes(restartNode, getRestartNodeForControl(control));
    }

    return restartNode;
};

const getAffectedNodeIds = (dagOrDags = {}, startNode = "") => {
    const visited = new Set();
    const queue = [startNode].filter(Boolean);
    const adjacency = new Map();
    const dags = Array.isArray(dagOrDags) ? dagOrDags : [dagOrDags];
    for (const dag of dags) {
        for (const edge of Array.isArray(dag?.edges) ? dag.edges : []) {
            const next = adjacency.get(edge.from) || [];
            next.push(edge.to);
            adjacency.set(edge.from, next);
        }
    }

    while (queue.length) {
        const current = queue.shift();
        if (!current || visited.has(current)) continue;
        visited.add(current);
        for (const next of adjacency.get(current) || []) {
            queue.push(next);
        }
    }

    return visited;
};

const rollbackStateFromNode = (context, startNode = "", options = {}) => {
    if (!startNode) return;
    const affectedNodeIds = getAffectedNodeIds(
        [context.plan?.dag || {}, options.previousDag || {}],
        startNode,
    );
    if (!affectedNodeIds.size) return;

    for (const nodeId of affectedNodeIds) {
        for (const key of STATE_KEYS_BY_NODE[nodeId] || []) {
            delete context.state[key];
        }
    }

    context.run.completedNodes = (Array.isArray(context.run.completedNodes) ? context.run.completedNodes : [])
        .filter((nodeId) => !affectedNodeIds.has(nodeId));

    const checkpoints = { ...(context.run.checkpoints || {}) };
    for (const [nodeId, checkpointId] of Object.entries(CHECKPOINT_BY_NODE)) {
        if (affectedNodeIds.has(nodeId)) {
            delete checkpoints[checkpointId];
        }
    }
    context.run.checkpoints = checkpoints;

    if (affectedNodeIds.has("deepComprehensionEngine")) {
        delete context.run.extraction;
        delete context.run.safety;
    }
    if (affectedNodeIds.has("recursiveSelfImprovementLoop")) {
        delete context.run.tribunal;
        delete context.run.convergence;
        delete context.run.postmortem;
        delete context.run.verifierSummary;
    }
    if (affectedNodeIds.has("decisionIntelligenceLayer")) {
        delete context.state.decisionLayer;
    }
    if (affectedNodeIds.has("adaptiveDeliveryHub")) {
        delete context.run.result;
    }

    delete context.run.awaitingCheckpoint;
    context.run.status = "in_progress";
    context.run.state = context.state;
};

const recompilePlanWithControls = (context, appliedControls = []) => {
    const previousDag = context.plan?.dag || context.run.plan?.dag || {};
    const mergedControls = normalizeAppliedControls([
        ...(Array.isArray(context.controls) ? context.controls : []),
        ...appliedControls,
    ]);
    context.controls = mergedControls;
    context.plan = compileResearchPlan({
        query: context.query,
        attachments: context.attachments.length,
        depthPreference: getDepthOverride(mergedControls, context.requestedDepthPreference || context.plan?.depthPreference || ""),
        forcedOutputMode: getForcedOutputMode(mergedControls, context.requestedForcedOutputMode || context.plan?.outputMode?.id || ""),
        refinementBudget: context.requestedRefinementBudget || context.plan?.refinementBudget || 3,
        maxQueries: context.requestedMaxQueries || context.run.requestOptions?.maxQueries || context.plan?.searchQueries?.length || 6,
        memoryContext: context.memoryContext,
        steering: mergedControls,
    });
    const activeNodeIds = new Set((Array.isArray(context.plan?.dag?.nodes) ? context.plan.dag.nodes : []).map((node) => node.id));
    for (const [nodeId, stateKeys] of Object.entries(STATE_KEYS_BY_NODE)) {
        if (activeNodeIds.has(nodeId)) continue;
        for (const key of stateKeys) {
            delete context.state[key];
        }
    }
    context.run.controls = mergedControls;
    context.run.requestOptions = {
        depthPreference: context.requestedDepthPreference || context.plan?.depthPreference || "",
        forcedOutputMode: context.requestedForcedOutputMode || context.plan?.outputMode?.id || "",
        refinementBudget: context.requestedRefinementBudget || context.plan?.refinementBudget || 3,
        maxQueries: context.requestedMaxQueries || context.run.requestOptions?.maxQueries || context.plan?.searchQueries?.length || 6,
    };
    context.run.plan = context.plan;
    context.state.plan = context.plan;
    context.run.completedNodes = (Array.isArray(context.run.completedNodes) ? context.run.completedNodes : [])
        .filter((nodeId) => activeNodeIds.has(nodeId));
    return previousDag;
};

const processPendingControls = async (context) => {
    const runId = context.run.id || context.run.runId || "";
    if (!runId) return "";

    const queued = await pullPendingResearchControls(runId, { limit: 8 });
    if (!queued.length) return "";

    const appliedControls = normalizeAppliedControls(queued);
    let restartNode = "";
    for (const control of appliedControls) {
        restartNode = compareRestartNodes(restartNode, getRestartNodeForControl(control));
    }

    const previousDag = recompilePlanWithControls(context, appliedControls);
    rollbackStateFromNode(context, restartNode, { previousDag });
    await emitRuntimeEvent(context, "control_applied", {
        controls: appliedControls,
        restartNode,
        statusText: `${appliedControls.length} steering command${appliedControls.length === 1 ? "" : "s"} applied.`,
    });
    await markResearchControlsApplied(appliedControls.map((control) => control.id).filter(Boolean));
    await persistRuntimeState(context);
    return restartNode;
};

const buildRuntimeContext = async ({
    query,
    attachments = [],
    scopeKey = "default",
    controls = [],
    depthPreference = "",
    forcedOutputMode = "",
    refinementBudget = 3,
    maxQueries = 6,
    runId = "",
    stopAfterCheckpoint = "",
    onEvent = null,
    resumeRun = null,
}) => {
    const resolvedRunId = normalizeText(runId) || buildRunId();
    const memoryContext = await findRelevantResearchContext(scopeKey, query);
    const filteredControls = resolveControls(controls);
    const requestedDepthPreference = normalizeText(
        depthPreference || resumeRun?.requestOptions?.depthPreference || resumeRun?.plan?.depthPreference || "",
    ).toLowerCase();
    const requestedForcedOutputMode = normalizeText(
        forcedOutputMode || resumeRun?.requestOptions?.forcedOutputMode || resumeRun?.plan?.outputMode?.id || "",
    ).toLowerCase();
    const requestedRefinementBudget = normalizeRefinementBudget(
        refinementBudget || resumeRun?.requestOptions?.refinementBudget || resumeRun?.plan?.refinementBudget || 3,
        3,
    );
    const requestedMaxQueries = Math.max(
        1,
        Number(maxQueries || resumeRun?.requestOptions?.maxQueries || resumeRun?.plan?.searchQueries?.length || 6) || 6,
    );
    const plan = compileResearchPlan({
        query,
        attachments: attachments.length,
        depthPreference: getDepthOverride(filteredControls, requestedDepthPreference),
        forcedOutputMode: getForcedOutputMode(filteredControls, requestedForcedOutputMode),
        refinementBudget: requestedRefinementBudget,
        maxQueries: requestedMaxQueries,
        memoryContext,
        steering: filteredControls,
    });

    const run = resumeRun || {
        id: resolvedRunId,
        runId: resolvedRunId,
        scopeKey,
        query,
        attachments,
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
        status: "in_progress",
        controls: filteredControls,
        requestOptions: {
            depthPreference: requestedDepthPreference,
            forcedOutputMode: requestedForcedOutputMode,
            refinementBudget: requestedRefinementBudget,
            maxQueries: requestedMaxQueries,
        },
        events: [],
        checkpoints: {},
        completedNodes: [],
        state: {},
    };

    return {
        plan,
        query,
        attachments,
        scopeKey,
        controls: filteredControls,
        requestedDepthPreference,
        requestedForcedOutputMode,
        requestedRefinementBudget,
        requestedMaxQueries,
        stopAfterCheckpoint,
        onEvent,
        memoryContext,
        memoryContextString: formatResearchContext(memoryContext),
        run: {
            ...run,
            id: normalizeText(run.id || run.runId || resolvedRunId) || resolvedRunId,
            runId: normalizeText(run.runId || run.id || resolvedRunId) || resolvedRunId,
            query,
            scopeKey,
            attachments,
            controls: filteredControls,
            requestOptions: {
                depthPreference: requestedDepthPreference,
                forcedOutputMode: requestedForcedOutputMode,
                refinementBudget: requestedRefinementBudget,
                maxQueries: requestedMaxQueries,
            },
            plan,
            status: "in_progress",
        },
        state: run.state && typeof run.state === "object" ? { ...run.state } : {},
    };
};

const resolveResumeStartNode = (run = {}) => {
    const checkpoint = run.awaitingCheckpoint || "";
    if (checkpoint === "plan") return "adversarialQueryForge";
    if (checkpoint === "inventory") return "deepComprehensionEngine";
    if (checkpoint === "summaries") return "dialecticalSynthesisEngine";
    if (checkpoint === "draft") return "recursiveSelfImprovementLoop";
    if (checkpoint === "tribunal") {
        return isDecisionOutputMode(run?.plan?.outputMode?.id) ? "decisionIntelligenceLayer" : "adaptiveDeliveryHub";
    }
    if (checkpoint === "decision") return "adaptiveDeliveryHub";
    return "";
};

const executeNode = async (context, node) => {
    const plan = context.plan;
    const planBrief = buildPlanBrief(plan, context.memoryContextString);
    const continuityBlock = buildContinuityBlock(plan.continuity, context.memoryContextString);
    const constraintBlock = buildConstraintBlock(plan);
    const counterHypothesisBlock = buildCounterHypothesisBlock(plan);
    const promptPatchBlock = buildPromptPatchBlock(plan);
    const promptPatchSection = promptPatchBlock ? `\n\nPrompt patches:\n${promptPatchBlock}` : "";

    switch (node.id) {
    case "cognitiveCommandLayer": {
        context.state.plan = plan;
        await emitRuntimeEvent(context, "status", {
            nodeId: node.id,
            label: node.label,
            detail: `Compiled ${plan.scope.label} DAG with ${plan.searchQueries.length} search lanes.`,
        });
        await writeCheckpoint(context, "plan", {
            plan,
            subagents: plan.subagents,
            searchQueries: plan.searchQueries,
            outputMode: plan.outputMode,
            refinementBudget: plan.refinementBudget,
        });
        return;
    }

    case "activeSafetyAndEthics": {
        context.state.safety = {
            ...plan.safety,
            controls: context.controls,
        };
        await emitRuntimeEvent(context, "status", {
            nodeId: node.id,
            label: node.label,
            detail: `${plan.safety.activeCount} safety check${plan.safety.activeCount === 1 ? "" : "s"} active.`,
        });
        return;
    }

    case "adversarialQueryForge": {
        context.state.queryForge = {
            versions: plan.queryMatrix.versions,
            counterHypotheses: plan.queryMatrix.counterHypotheses,
            ontologyMappedVocabulary: plan.queryMatrix.ontologyMappedVocabulary,
        };
        await emitRuntimeEvent(context, "status", {
            nodeId: node.id,
            label: node.label,
            detail: `${plan.queryMatrix.counterHypotheses.length || 1} counter-hypothesis lane${plan.queryMatrix.counterHypotheses.length === 1 ? "" : "s"} prepared.`,
        });
        return;
    }

    case "intelligentCrawlerMesh": {
        const sourceMesh = await searchResearchSources({
            query: context.query,
            plan,
            maxResults: 28,
        });
        context.state.sourceMesh = sourceMesh;
        await emitRuntimeEvent(context, "status", {
            nodeId: node.id,
            label: node.label,
            detail: `${sourceMesh.sources.length} source${sourceMesh.sources.length === 1 ? "" : "s"} collected across ${sourceMesh.providersUsed.length} provider${sourceMesh.providersUsed.length === 1 ? "" : "s"}.`,
        });
        return;
    }

    case "tieredEpistemicFilter": {
        const filteredSources = applyControlFilters(context.state.sourceMesh?.sources || [], context.controls)
            .filter((source) => source.tier !== "discard")
            .slice(0, 30);
        const tieredSources = assignCitationIndices(filteredSources);
        context.state.tieredSources = tieredSources;
        context.state.inventory = {
            core: tieredSources.filter((source) => source.tier === "core").length,
            supporting: tieredSources.filter((source) => source.tier === "supporting").length,
            peripheral: tieredSources.filter((source) => source.tier === "peripheral").length,
        };
        await emitRuntimeEvent(context, "inventory", {
            sources: tieredSources.slice(0, 12),
            counts: context.state.inventory,
        });
        await writeCheckpoint(context, "inventory", {
            counts: context.state.inventory,
            sources: tieredSources.slice(0, 18),
        });
        return;
    }

    case "deepComprehensionEngine": {
        const evidenceEntries = await fetchTieredEvidence(context.state.tieredSources || [], context.query);
        const rankedEvidenceEntries = rankEvidenceEntriesForQuery(context.query, evidenceEntries).slice(0, 24);
        if (!rankedEvidenceEntries.length) {
            throw new ResearchRuntimeError("No readable research evidence was fetched.", {
                nodeId: node.id,
                failureType: "partial",
            });
        }
        context.state.evidenceEntries = rankedEvidenceEntries;
        const extraction = await extractResearchArtifacts({
            query: context.query,
            plan,
            evidenceEntries: rankedEvidenceEntries,
        });
        context.state.extraction = extraction;
        for (const entry of rankedEvidenceEntries.slice(0, 10)) {
            await emitRuntimeEvent(context, "summary", {
                source: entry.source,
                excerpt: entry.content.slice(0, 420),
            });
        }
        await writeCheckpoint(context, "summaries", {
            extractedClaims: extraction.claims.length,
            repositoryCount: extraction.repositories.length,
            supplementaryCount: extraction.supplementary.length,
            evidenceCount: rankedEvidenceEntries.length,
        });
        return;
    }

    case "dialecticalSynthesisEngine": {
        const evidenceEntries = context.state.evidenceEntries || [];
        const chunkSize = Math.max(1, Math.ceil(evidenceEntries.length / 3));
        const evidenceChunks = chunkArray(evidenceEntries, chunkSize).slice(0, 3);
        const sourceIndex = buildSourceIndex(context.state.tieredSources || []);

        const positionMaps = [];
        for (let index = 0; index < evidenceChunks.length; index += 1) {
            const evidenceBlock = evidenceChunks[index].map((entry) => entry.evidenceBlock).join("\n\n---\n\n");
            const positionText = await runResearchModel({
                maxTokens: 1600,
                messages: [
                    {
                        role: "system",
                        content: "You are PositionMapper inside Research Framework v3.1. Return strict JSON with keys dominant_position, counter_position, key_claims, quantitative_signals, and open_gaps. key_claims must be an array of objects with claim and citations. Use only bracketed citation numbers from the evidence.",
                    },
                    {
                        role: "user",
                        content: `Research plan:\n${planBrief}\n\nConstraints:\n${constraintBlock}\n\nCounter-hypotheses:\n${counterHypothesisBlock}${continuityBlock ? `\n\n${continuityBlock}` : ""}${promptPatchSection}\n\nSource index:\n${sourceIndex}\n\nEvidence set:\n${evidenceBlock}`,
                    },
                ],
            });
            const parsed = parseJsonObject(positionText) || {
                dominant_position: positionText,
                counter_position: "",
                key_claims: [],
                quantitative_signals: [],
                open_gaps: [],
            };
            positionMaps.push(parsed);
        }

        const positionMapBlock = positionMaps
            .map((map, index) => `### Position Map ${index + 1}\n${JSON.stringify(map, null, 2)}`)
            .join("\n\n");

        const thesis = await runResearchModel({
            maxTokens: 1400,
            messages: [
                {
                    role: "system",
                    content: "You are ThesisAgent in Research Framework v3.1. Build the strongest case for the dominant position using only the position maps and cited sources. Use bracketed citations [n] only.",
                },
                {
                    role: "user",
                    content: `Research plan:\n${planBrief}${promptPatchSection}\n\nSource index:\n${sourceIndex}\n\nPosition maps:\n${positionMapBlock}`,
                },
            ],
        });

        const antithesis = await runResearchModel({
            maxTokens: 1400,
            messages: [
                {
                    role: "system",
                    content: "You are AntithesisAgent in Research Framework v3.1. Build the strongest counter-case using disconfirming evidence, counter-hypotheses, and the supplied position maps. Use bracketed citations [n] only.",
                },
                {
                    role: "user",
                    content: `Research plan:\n${planBrief}\n\nCounter-hypotheses:\n${counterHypothesisBlock}${promptPatchSection}\n\nSource index:\n${sourceIndex}\n\nPosition maps:\n${positionMapBlock}`,
                },
            ],
        });

        const draft = await runResearchModel({
            maxTokens: 2400,
            messages: [
                {
                    role: "system",
                    content: `You are SynthesisMediator working with NarrativeArchitect in Research Framework v3.1. ${buildOutputModeInstruction(plan.outputMode)} Use only bracketed citations [n]. Start with a single H1 title. Include a short residual uncertainty section and end with ## Sources used listing cited bracket numbers only.`,
                },
                {
                    role: "user",
                    content: `Research plan:\n${planBrief}\n\nConstraints:\n${constraintBlock}\n\nCounter-hypotheses:\n${counterHypothesisBlock}${continuityBlock ? `\n\n${continuityBlock}` : ""}${promptPatchSection}\n\nSource index:\n${sourceIndex}\n\nPosition maps:\n${positionMapBlock}\n\nThesis:\n${thesis}\n\nAntithesis:\n${antithesis}`,
                },
            ],
        });

        context.state.positionMaps = positionMaps;
        context.state.thesis = thesis;
        context.state.antithesis = antithesis;
        context.state.draft = draft;
        await writeCheckpoint(context, "draft", {
            heading: extractAnswerParts(draft).heading,
            draft,
        });
        return;
    }

    case "recursiveSelfImprovementLoop": {
        const sourceIndex = buildSourceIndex(context.state.tieredSources || []);
        let currentDraft = context.state.draft || "";
        let lastSimilarity = 1;
        let tribunal = buildTribunalSummary({
            coverageScore: 0.74,
            contradictionScore: 0.22,
            alignmentScore: 0.8,
            verifiers: {
                claim_support: 0.78,
                citation_integrity: 0.78,
                contradiction_handling: 0.78,
                uncertainty_calibration: 0.78,
            },
            iterations: 1,
            refinementBudget: plan.refinementBudget,
        });
        let verifierSummary = {
            assessments: {},
            scores: {},
            targetedDimension: "claim_support",
            issues: [],
            rewriteBrief: "",
        };

        for (let cycle = 1; cycle <= plan.refinementBudget; cycle += 1) {
            const citationValidation = buildCitationValidation(currentDraft, context.state.tieredSources || []);
            const [criticText, cycleVerifierSummary] = await Promise.all([
                runResearchModel({
                    maxTokens: 900,
                    messages: [
                        {
                            role: "system",
                            content: "You are the Recursive Self-Improvement Loop. Return strict JSON with keys internal_consistency, coverage, user_goal_alignment, targeted_dimension, and rewrite_brief. Score whether major claims are citation-supported, whether hypotheses and counter-hypotheses are covered, and whether the output matches the requested format.",
                        },
                        {
                            role: "user",
                            content: `Research plan:\n${planBrief}\n\nConstraints:\n${constraintBlock}\n\nCounter-hypotheses:\n${counterHypothesisBlock}${promptPatchSection}\n\nSource index:\n${sourceIndex}\n\nUnsupported citations count: ${citationValidation.unsupported.length}\n\nDraft:\n${currentDraft}`,
                        },
                    ],
                }),
                runVerifierSwarm({
                    plan,
                    planBrief,
                    constraintBlock,
                    counterHypothesisBlock,
                    sourceIndex,
                    currentDraft,
                    evidenceEntries: context.state.evidenceEntries,
                    extraction: context.state.extraction,
                    citationValidation,
                    thesis: context.state.thesis,
                    antithesis: context.state.antithesis,
                    continuityBlock,
                }),
            ]);

            const critic = parseTribunalScores(criticText);
            verifierSummary = cycleVerifierSummary;

            const verifierConsistency = averageScore([
                verifierSummary.dimensions?.claim_support,
                verifierSummary.dimensions?.citation_integrity,
            ], 0.78);
            const verifierCoverage = clampScore(verifierSummary.dimensions?.contradiction_handling, 0.76);
            const verifierAlignment = clampScore(verifierSummary.dimensions?.uncertainty_calibration, 0.78);

            critic.internal_consistency = Math.min(critic.internal_consistency, verifierConsistency);
            critic.coverage = Math.min(critic.coverage, verifierCoverage);
            critic.user_goal_alignment = Math.min(critic.user_goal_alignment, verifierAlignment);

            if (citationValidation.unsupported.length || verifierSummary.unsupportedCitationCount) {
                critic.internal_consistency = Math.min(critic.internal_consistency, 0.56);
            }

            const verifierTarget = pickLowestDimension({
                internal_consistency: critic.internal_consistency,
                coverage: critic.coverage,
                user_goal_alignment: critic.user_goal_alignment,
                ...(verifierSummary.dimensions || {}),
            });
            critic.targeted_dimension = verifierTarget;
            critic.rewrite_brief = buildRewriteGuidance({
                critic,
                verifierSummary,
                targetedDimension: verifierTarget,
            });

            tribunal = buildTribunalSummary({
                coverageScore: critic.coverage,
                contradictionScore: 1 - critic.internal_consistency,
                alignmentScore: critic.user_goal_alignment,
                verifiers: verifierSummary.dimensions,
                iterations: cycle,
                refinementBudget: plan.refinementBudget,
            });
            tribunal.targeted_dimension = verifierTarget;
            tribunal.verifier_summary = verifierSummary;

            const verifierThresholdBreach = Object.entries(verifierSummary.dimensions || {})
                .some(([dimension, score]) => score < (VERIFIER_SPECS[dimension]?.threshold || 0.78));
            const needsRewrite = (
                critic.internal_consistency < 0.78
                || critic.coverage < 0.76
                || critic.user_goal_alignment < 0.8
                || verifierThresholdBreach
            );

            if (!needsRewrite || cycle >= plan.refinementBudget) {
                break;
            }

            const targetedDimension = tribunal.targeted_dimension || critic.targeted_dimension || verifierSummary.targetedDimension || "coverage";
            const rewriteBrief = buildRewriteGuidance({
                critic,
                verifierSummary,
                targetedDimension,
            }) || verifierSummary.rewriteBrief;

            const previousDraft = currentDraft;
            currentDraft = await runResearchModel({
                maxTokens: 2400,
                messages: [
                    {
                        role: "system",
                        content: `You are NarrativeArchitect revising a draft after tribunal feedback in Research Framework v3.1. ${buildOutputModeInstruction(plan.outputMode)} Use only bracketed citations [n]. Improve the targeted dimension without weakening the others. Remove unsupported claims. Start with a single H1 title and end with ## Sources used listing cited bracket numbers only.`,
                    },
                    {
                        role: "user",
                        content: `Targeted dimension: ${targetedDimension}\nRewrite brief: ${rewriteBrief || "Strengthen coverage, consistency, user-goal alignment, and verifier-supported grounding."}\n\nVerifier issues:\n${(verifierSummary.issues || []).map((issue) => `- ${issue}`).join("\n") || "- None recorded."}\n\nConstraints:\n${constraintBlock}\n\nCounter-hypotheses:\n${counterHypothesisBlock}${promptPatchSection}\n\nSource index:\n${sourceIndex}\n\nCurrent draft:\n${currentDraft}`,
                    },
                ],
            });
            lastSimilarity = computeTextSimilarity(previousDraft, currentDraft);
        }

        const convergence = buildConvergenceMetrics({
            iterations: tribunal.refinement_cycles,
            refinementBudget: tribunal.refinement_budget,
            coverageScore: tribunal.critics.coverage,
            contradictionScore: 1 - tribunal.critics.internal_consistency,
            verificationScore: verifierSummary.aggregateScore,
            stabilityScore: averageScore([
                lastSimilarity,
                tribunal.critics.internal_consistency,
                tribunal.critics.coverage,
                tribunal.critics.user_goal_alignment,
                verifierSummary.aggregateScore,
            ], 0.8),
        });

        context.state.finalText = currentDraft;
        context.state.tribunal = tribunal;
        context.state.convergence = convergence;
        context.state.verifierSummary = verifierSummary;
        context.state.claimLedger = buildClaimLedger({
            positionMaps: context.state.positionMaps,
            extraction: context.state.extraction,
            tieredSources: context.state.tieredSources,
            tribunal,
            verifierSummary,
        });
        context.state.postmortem = {
            query_type: plan.scope.id,
            domain: plan.domain.label,
            phases_that_degraded_score: unique([
                `Phase 5 - low ${tribunal.targeted_dimension}`,
                ...((verifierSummary.issues || []).slice(0, 3).map((issue) => `Phase 6 - ${issue}`)),
            ]),
            prompt_patches_applied: tribunal.refinement_cycles > 1 ? unique([
                `Strengthened ${tribunal.targeted_dimension} enforcement.`,
                buildRewriteGuidance({
                    critic: { rewrite_brief: "" },
                    verifierSummary,
                    targetedDimension: tribunal.targeted_dimension,
                }),
            ]).filter(Boolean) : [],
            final_score_delta: Number(((
                tribunal.critics.internal_consistency
                + tribunal.critics.coverage
                + tribunal.critics.user_goal_alignment
            ) * 10).toFixed(1)),
            targetedDimension: tribunal.targeted_dimension,
        };
        await writeCheckpoint(context, "tribunal", {
            tribunal,
            verifierSummary,
            convergence,
        });
        return;
    }

    case "decisionIntelligenceLayer": {
        const finalText = context.state.finalText || context.state.draft || "";
        const fallbackDecision = buildDecisionLayer({
            plan,
            convergence: context.state.convergence,
            safety: {
                ...context.state.safety,
                ...(context.state.extraction?.safety || {}),
            },
            finalText,
            tribunal: context.state.tribunal,
            verifierSummary: context.state.verifierSummary,
        });

        if (!fallbackDecision) {
            context.state.decisionLayer = null;
            return;
        }

        let decisionLayer = fallbackDecision;
        try {
            const decisionResponse = await runResearchModel({
                maxTokens: 1200,
                messages: [
                    {
                        role: "system",
                        content: "You are the Decision Intelligence Layer in Research Framework v3.1. Return strict JSON with keys decision, expected_outcome, risk_profile, confidence, reversibility, rationale, and recommended_actions. Keep all numeric risk and confidence values between 0 and 1.",
                    },
                    {
                        role: "user",
                        content: `Research plan:\n${planBrief}${promptPatchSection}\n\nConvergence:\n${JSON.stringify(context.state.convergence || {}, null, 2)}\n\nVerifier summary:\n${JSON.stringify(context.state.verifierSummary || {}, null, 2)}\n\nSafety:\n${JSON.stringify({
                            ...(context.state.safety || {}),
                            ...(context.state.extraction?.safety || {}),
                        }, null, 2)}\n\nClaim ledger:\n${JSON.stringify((context.state.claimLedger || []).slice(0, 8), null, 2)}\n\nFinal report:\n${finalText}`,
                    },
                ],
            });
            const parsedDecision = parseJsonObject(decisionResponse);
            if (parsedDecision) {
                decisionLayer = {
                    ...fallbackDecision,
                    decision: normalizeText(parsedDecision.decision) || fallbackDecision.decision,
                    expected_outcome: normalizeText(parsedDecision.expected_outcome || parsedDecision.expectedOutcome) || fallbackDecision.expected_outcome,
                    risk_profile: {
                        technical: clampScore(
                            parsedDecision.risk_profile?.technical ?? parsedDecision.riskProfile?.technical,
                            fallbackDecision.risk_profile.technical,
                        ),
                        epistemic: clampScore(
                            parsedDecision.risk_profile?.epistemic ?? parsedDecision.riskProfile?.epistemic,
                            fallbackDecision.risk_profile.epistemic,
                        ),
                    },
                    confidence: clampScore(parsedDecision.confidence, fallbackDecision.confidence),
                    reversibility: normalizeText(parsedDecision.reversibility) || fallbackDecision.reversibility,
                    rationale: normalizeText(parsedDecision.rationale || parsedDecision.summary || ""),
                    recommended_actions: unique([
                        ...(Array.isArray(parsedDecision.recommended_actions) ? parsedDecision.recommended_actions : []),
                        ...(Array.isArray(parsedDecision.actions) ? parsedDecision.actions : []),
                    ].map((item) => normalizeText(item)).filter(Boolean)).slice(0, 6),
                };
            }
        } catch {
            decisionLayer = fallbackDecision;
        }

        context.state.decisionLayer = decisionLayer;
        await writeCheckpoint(context, "decision", {
            decision: decisionLayer,
        });
        await emitRuntimeEvent(context, "status", {
            nodeId: node.id,
            label: node.label,
            detail: `Decision packaged with ${decisionLayer.reversibility} reversibility and ${Math.round((decisionLayer.confidence || 0) * 100)}% confidence.`,
        });
        return;
    }

    case "adaptiveDeliveryHub": {
        const finalText = context.state.finalText || context.state.draft || "";
        const { heading, body } = extractAnswerParts(finalText);
        const citedNumbers = unique(
            [...String(finalText || "").matchAll(/\[(\d+)]/g)]
                .map((match) => Number(match[1]))
                .filter((value) => Number.isFinite(value)),
        );
        const citedSources = (context.state.tieredSources || []).filter((source) => citedNumbers.includes(Number(source.citationIndex)));
        const decision = context.state.decisionLayer || buildDecisionLayer({
            plan,
            convergence: context.state.convergence,
            safety: {
                ...context.state.safety,
                ...(context.state.extraction?.safety || {}),
            },
            finalText,
            tribunal: context.state.tribunal,
            verifierSummary: context.state.verifierSummary,
        });

        context.run.result = {
            heading,
            body,
            finalText,
            sources: citedSources,
            markdown: buildMarkdownExport({
                ...context.run,
                plan,
                result: { heading, body, finalText },
                extraction: context.state.extraction,
            }),
            slides: buildSlideOutline({
                ...context.run,
                query: context.query,
                result: { heading, body, finalText },
            }),
            datasetCsv: buildDatasetCsv({
                extraction: context.state.extraction,
            }),
            claimLedger: context.state.claimLedger,
            verifierSummary: context.state.verifierSummary,
            decision,
        };
        context.run.tribunal = context.state.tribunal;
        context.run.convergence = context.state.convergence;
        context.run.verifierSummary = context.state.verifierSummary;
        context.run.extraction = context.state.extraction;
        context.run.safety = {
            ...context.state.safety,
            ...(context.state.extraction?.safety || {}),
        };
        context.run.postmortem = context.state.postmortem;
        context.run.status = "complete";
        await writeCheckpoint(context, "final", {
            heading,
            finalText,
        });
        return;
    }

    default:
        return;
    }
};

const runNodeWithRecovery = async (context, node) => {
    const maxAttempts = 2;
    for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
        try {
            await executeNode(context, node);
            context.run.completedNodes = unique([...(context.run.completedNodes || []), node.id]);
            context.run.state = context.state;
            await persistRuntimeState(context);
            return;
        } catch (error) {
            const failure = classifyFailure(error, node.id);
            if (RETRYABLE_NODE_FAILURES.has(failure.failureType) && attempt < maxAttempts) {
                await emitRuntimeEvent(context, "warning", {
                    nodeId: node.id,
                    detail: `Retrying after soft failure: ${failure.message}`,
                });
                continue;
            }

            if (failure.failureType === "partial") {
                context.run.partialFailures = [
                    ...(Array.isArray(context.run.partialFailures) ? context.run.partialFailures : []),
                    {
                        nodeId: node.id,
                        message: failure.message,
                    },
                ].slice(-20);
                await emitRuntimeEvent(context, "warning", {
                    nodeId: node.id,
                    detail: failure.message,
                });
                context.run.completedNodes = unique([...(context.run.completedNodes || []), node.id]);
                await persistRuntimeState(context);
                return;
            }

            throw failure;
        }
    }
};

const executeResearchRuntimePass = async (context, resumeStartNode = "") => {
    const batches = topologicalBatches(context.plan.dag);
    let shouldExecute = !resumeStartNode;

    for (const batch of batches) {
        const executableBatch = [];
        for (const node of batch) {
            if (!shouldExecute) {
                if (node.id === resumeStartNode) shouldExecute = true;
                else continue;
            }
            if ((context.run.completedNodes || []).includes(node.id) && !resumeStartNode) continue;
            executableBatch.push(node);
        }

        for (const node of executableBatch) {
            await emitRuntimeEvent(context, "status", {
                nodeId: node.id,
                label: node.label,
                detail: "running",
            });
            await runNodeWithRecovery(context, node);
            const restartNode = await processPendingControls(context);
            if (restartNode) {
                return {
                    restartNode,
                    paused: false,
                };
            }
            const checkpointId = CHECKPOINT_BY_NODE[node.id];
            if (checkpointId && shouldPauseAfterCheckpoint(context, checkpointId)) {
                context.run.status = "awaiting_input";
                context.run.awaitingCheckpoint = checkpointId;
                context.run.state = context.state;
                await persistRuntimeState(context);
                return {
                    restartNode: "",
                    paused: true,
                };
            }
        }
    }

    return {
        restartNode: "",
        paused: false,
    };
};

const executeResearchRuntime = async (context, options = {}) => {
    let resumeStartNode = options.resumeStartNode || "";

    while (true) {
        const outcome = await executeResearchRuntimePass(context, resumeStartNode);
        if (outcome.paused) {
            return context.run;
        }
        if (!outcome.restartNode) break;
        resumeStartNode = outcome.restartNode;
    }

    context.run.status = "complete";
    context.run.state = context.state;
    await persistRuntimeState(context);
    return context.run;
};

const finalizeCompletedRun = async (context) => {
    if (context.run.status !== "complete") return context.run;
    await indexResearchRun(context.scopeKey, {
        ...context.run,
        title: context.run.result?.heading,
        domain: context.plan.domain,
        scope: context.plan.scope,
        outputMode: context.plan.outputMode,
        tribunal: context.run.tribunal,
        convergence: context.run.convergence,
    });
    await updateResearchMemoryFromRun(context.scopeKey, {
        ...context.run,
        plan: context.plan,
        domain: context.plan.domain,
        scope: context.plan.scope,
        outputMode: context.plan.outputMode,
    });
    return context.run;
};

const runResearch = async ({
    query,
    attachments = [],
    scopeKey = "default",
    controls = [],
    depthPreference = "",
    forcedOutputMode = "",
    refinementBudget = 3,
    maxQueries = 6,
    stopAfterCheckpoint = "",
    onEvent = null,
}) => {
    const context = await buildRuntimeContext({
        query,
        attachments,
        scopeKey,
        controls,
        depthPreference,
        forcedOutputMode,
        refinementBudget,
        maxQueries,
        stopAfterCheckpoint,
        onEvent,
    });
    await persistRuntimeState(context);
    const run = await executeResearchRuntime(context);
    await finalizeCompletedRun(context);
    return run;
};

const resumeResearch = async ({
    runId,
    scopeKey = "default",
    controls = [],
    depthPreference = "",
    forcedOutputMode = "",
    refinementBudget = 3,
    maxQueries = 6,
    stopAfterCheckpoint = "",
    onEvent = null,
}) => {
    const existingRun = await loadResearchRun(runId);
    if (!existingRun) {
        throw new ResearchRuntimeError("Research run not found.", { failureType: "critical" });
    }
    if (existingRun.status === "complete") return existingRun;

    const context = await buildRuntimeContext({
        query: existingRun.query,
        attachments: existingRun.attachments || [],
        scopeKey: existingRun.scopeKey || scopeKey,
        controls: [...(Array.isArray(existingRun.controls) ? existingRun.controls : []), ...resolveControls(controls)],
        depthPreference: depthPreference || existingRun.plan?.depthPreference || "",
        forcedOutputMode: forcedOutputMode || existingRun.plan?.outputMode?.id || "",
        refinementBudget: refinementBudget || existingRun.plan?.refinementBudget || 3,
        maxQueries: maxQueries || existingRun.requestOptions?.maxQueries || existingRun.plan?.searchQueries?.length || 6,
        runId: existingRun.id,
        stopAfterCheckpoint,
        onEvent,
        resumeRun: existingRun,
    });
    context.state = existingRun.state && typeof existingRun.state === "object" ? { ...existingRun.state } : {};
    context.run.completedNodes = Array.isArray(existingRun.completedNodes) ? existingRun.completedNodes : [];
    const checkpointResumeNode = resolveResumeStartNode(existingRun);
    const overrideRestartNode = getRestartNodeForPlanChange({
        previousPlan: existingRun.plan || {},
        nextPlan: context.plan || {},
        controls,
    });
    const resumeStartNode = compareRestartNodes(checkpointResumeNode, overrideRestartNode);
    if (overrideRestartNode) {
        rollbackStateFromNode(context, resumeStartNode, {
            previousDag: existingRun.plan?.dag || {},
        });
    }
    const run = await executeResearchRuntime(context, { resumeStartNode });
    await finalizeCompletedRun(context);
    return run;
};

const buildExportPayload = (run = {}, format = "json") => {
    switch (String(format || "").toLowerCase()) {
    case "markdown":
    case "obsidian":
    case "notion":
        return {
            contentType: "text/markdown; charset=utf-8",
            body: run.result?.markdown || "",
        };
    case "slide_deck_outline":
    case "slides":
        return {
            contentType: "text/plain; charset=utf-8",
            body: run.result?.slides || "",
        };
    case "dataset_csv":
    case "csv":
    case "dataset":
        return {
            contentType: "text/csv; charset=utf-8",
            body: run.result?.datasetCsv || "",
        };
    case "dataset_json":
        return {
            contentType: "application/json; charset=utf-8",
            body: buildOutputsPayload(run).dataset_json,
        };
    case "json":
    default:
        return {
            contentType: "application/json; charset=utf-8",
            body: JSON.stringify(serializeResearchRun(run), null, 2),
        };
    }
};

module.exports = {
    ResearchRuntimeError,
    buildRunId,
    buildRuntimeMeta,
    serializeResearchRun,
    runResearch,
    resumeResearch,
    loadResearchRun,
    buildExportPayload,
};
