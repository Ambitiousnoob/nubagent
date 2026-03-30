const { extractQueryTerms } = require("./rag");

const RESEARCH_FRAMEWORK_VERSION = "3.1";
const DEFAULT_DEPTH = "balanced";
const DEFAULT_REFINEMENT_BUDGET = 3;

const OPERATOR_HEAVY_RE = /\b(site:|filetype:|intitle:|inurl:|after:|before:)\b/i;
const DOCS_RE = /\b(api|sdk|docs?|documentation|guide|install|setup|reference|spec|error|troubleshoot(?:ing)?)\b/i;
const RESEARCH_RE = /\b(rrl|related literature|literature|study|studies|research|paper|papers|journal|benchmark|evaluation|meta-analysis|peer reviewed)\b/i;
const CURRENT_RE = /\b(latest|recent|today|current|new|newest|breaking|updated?|this week|this month|this year|202\d|release|released|price|stock)\b/i;
const COMPARISON_RE = /\b(vs|versus|compare|comparison|best|top|alternative|alternatives)\b/i;
const QUESTION_RE = /^(who|what|when|where|why|how)\b/i;

const DOMAIN_RULES = [
    {
        id: "biomedical",
        label: "Biomedical",
        taxonomy: "MeSH",
        patterns: /\b(biomedical|biosecurity|clinical|cohort|trial|trials|patient|patients|oncology|genomics|protein|dna|rna|biopsy|medical|medicine|public health|epidemiology|pharmacology)\b/i,
        recencyHalfLifeYears: 2,
        ontologyTerms: ["MeSH", "clinical trial", "systematic review", "cohort study"],
    },
    {
        id: "cs_ml",
        label: "CS + ML",
        taxonomy: "ACM CCS",
        patterns: /\b(llm|language model|transformer|retrieval augmented generation|rag|benchmark|ablation|machine learning|deep learning|computer vision|reinforcement learning|ml safety|alignment|scaling law|inference)\b/i,
        recencyHalfLifeYears: 2,
        ontologyTerms: ["ACM CCS", "benchmark", "ablation study", "reproducibility"],
    },
    {
        id: "social_science",
        label: "Social Science",
        taxonomy: "JEL",
        patterns: /\b(social science|sociology|economics|policy|public policy|behavioral|survey|qualitative|labor market|governance|education|political science)\b/i,
        recencyHalfLifeYears: 6,
        ontologyTerms: ["JEL", "survey instrument", "observational study", "policy analysis"],
    },
    {
        id: "interdisciplinary",
        label: "Interdisciplinary",
        taxonomy: "Mixed ontology",
        patterns: /\b(interdisciplinary|cross-domain|human-ai|socio-technical|sociotechnical|computational social science|digital health|bioinformatics)\b/i,
        recencyHalfLifeYears: 4,
        ontologyTerms: ["cross-disciplinary", "mixed methods", "evidence synthesis", "knowledge graph"],
    },
];

const OUTPUT_MODE_RULES = [
    { id: "tutorial", label: "Tutorial", patterns: /\b(tutorial|how to|how do|walkthrough|guide|step by step)\b/i },
    { id: "state_of_the_field", label: "State-of-the-Field", patterns: /\b(state of the field|overview|landscape|what do we know)\b/i },
    { id: "controversy_map", label: "Controversy Map", patterns: /\b(controversy|debate|argue|disagreement|pros and cons|counter[- ]?argument)\b/i },
    { id: "gap_analysis", label: "Gap Analysis", patterns: /\b(gap analysis|research gap|open question|what is missing|where is the gap)\b/i },
    { id: "replication_crisis_report", label: "Replication Crisis Report", patterns: /\b(replication|reproduce|reproducibility|reproducible|replication crisis)\b/i },
    { id: "foundational_review", label: "Foundational Review", patterns: /\b(foundational|history of|foundations|background|origin|foundational review)\b/i },
    { id: "decision_brief", label: "Decision Brief", patterns: /\b(decision|recommend|should we|should i|choose|tradeoff|go no-go)\b/i },
    { id: "policy_recommendation", label: "Policy Recommendation", patterns: /\b(policy recommendation|regulation|governance|policy|compliance|regulatory)\b/i },
    { id: "engineering_action_plan", label: "Engineering Action Plan", patterns: /\b(engineering action plan|implementation plan|action plan|roadmap|migration plan)\b/i },
];

const SCOPE_RULES = [
    { id: "gap_finding", label: "Gap-finding", patterns: /\b(gap|missing|open question|unexplored|future work)\b/i },
    { id: "replication_study", label: "Replication Study", patterns: /\b(replication|reproducibility|reproduce|rerun|replicate)\b/i },
    { id: "contested_topic", label: "Contested Topic", patterns: /\b(controversy|debate|versus|vs\.?|conflicting|disputed|contested)\b/i },
    { id: "decision_support", label: "Decision Support", patterns: /\b(should|recommend|choose|decision|buy|adopt|prioritize)\b/i },
];

const DOMAIN_FALLBACK = {
    id: "general_research",
    label: "General Research",
    taxonomy: "General ontology",
    recencyHalfLifeYears: 4,
    ontologyTerms: ["systematic review", "benchmark", "evidence synthesis"],
};

const OUTPUT_MODE_FALLBACK = { id: "state_of_the_field", label: "State-of-the-Field" };
const SCOPE_FALLBACK = { id: "broad_research", label: "Broad Research" };

const PIPELINE_PHASES = Object.freeze({
    cognitiveCommandLayer: {
        id: "cognitiveCommandLayer",
        label: "Cognitive Command Layer",
        dependsOn: [],
        parallelizable: false,
        priority: 1,
    },
    adversarialQueryForge: {
        id: "adversarialQueryForge",
        label: "Adversarial Query Forge",
        dependsOn: ["cognitiveCommandLayer"],
        parallelizable: false,
        priority: 0.96,
    },
    intelligentCrawlerMesh: {
        id: "intelligentCrawlerMesh",
        label: "Intelligent Crawler Mesh",
        dependsOn: ["adversarialQueryForge"],
        parallelizable: true,
        priority: 0.94,
    },
    tieredEpistemicFilter: {
        id: "tieredEpistemicFilter",
        label: "Tiered Epistemic Filter",
        dependsOn: ["intelligentCrawlerMesh"],
        parallelizable: true,
        priority: 0.9,
    },
    activeSafetyAndEthics: {
        id: "activeSafetyAndEthics",
        label: "Active Safety & Ethics",
        dependsOn: ["cognitiveCommandLayer"],
        parallelizable: true,
        priority: 0.92,
    },
    deepComprehensionEngine: {
        id: "deepComprehensionEngine",
        label: "Deep Comprehension Engine",
        dependsOn: ["tieredEpistemicFilter", "activeSafetyAndEthics"],
        parallelizable: true,
        priority: 0.88,
    },
    dialecticalSynthesisEngine: {
        id: "dialecticalSynthesisEngine",
        label: "Dialectical Synthesis Engine",
        dependsOn: ["deepComprehensionEngine"],
        parallelizable: false,
        priority: 0.86,
    },
    recursiveSelfImprovementLoop: {
        id: "recursiveSelfImprovementLoop",
        label: "Recursive Self-Improvement Loop",
        dependsOn: ["dialecticalSynthesisEngine"],
        parallelizable: false,
        priority: 0.84,
    },
    decisionIntelligenceLayer: {
        id: "decisionIntelligenceLayer",
        label: "Decision Intelligence Layer",
        dependsOn: ["recursiveSelfImprovementLoop"],
        parallelizable: false,
        priority: 0.83,
    },
    adaptiveDeliveryHub: {
        id: "adaptiveDeliveryHub",
        label: "Adaptive Delivery Hub",
        dependsOn: ["decisionIntelligenceLayer"],
        parallelizable: false,
        priority: 0.82,
    },
});

const SUBAGENT_SPECS = Object.freeze({
    cognitiveCommandLayer: { label: "Cognitive Command Layer", scope: "intent decomposition, graph compilation, Pareto steering, continuity", phaseId: "cognitiveCommandLayer" },
    devilsAdvocateDecomposer: { label: "DevilsAdvocateDecomposer", scope: "counter-hypothesis generation", phaseId: "adversarialQueryForge" },
    domainDetector: { label: "DomainDetector", scope: "domain taxonomy and ontology selection", phaseId: "adversarialQueryForge" },
    queryVersionController: { label: "QueryVersionController", scope: "auditable query transformations", phaseId: "adversarialQueryForge" },
    forwardCitationTracer: { label: "ForwardCitationTracer", scope: "forward citation expansion", phaseId: "intelligentCrawlerMesh" },
    authorNetworkMapper: { label: "AuthorNetworkMapper", scope: "author graph and echo-chamber analysis", phaseId: "intelligentCrawlerMesh" },
    temporalTrendAnalyzer: { label: "TemporalTrendAnalyzer", scope: "topic acceleration and saturation analysis", phaseId: "intelligentCrawlerMesh" },
    temporalRelevanceDecay: { label: "TemporalRelevanceDecay", scope: "field-specific recency weighting", phaseId: "tieredEpistemicFilter" },
    retractedPaperGuard: { label: "RetractedPaperGuard", scope: "retraction and quarantine checks", phaseId: "tieredEpistemicFilter" },
    sampleSizeFilter: { label: "SampleSizeFilter", scope: "sample-size confidence weighting", phaseId: "tieredEpistemicFilter" },
    statisticalClaimExtractor: { label: "StatisticalClaimExtractor", scope: "structured statistical claim extraction", phaseId: "deepComprehensionEngine" },
    codeRepoAnalyzer: { label: "CodeRepoAnalyzer", scope: "linked repository reproducibility analysis", phaseId: "deepComprehensionEngine" },
    supplementaryMaterialParser: { label: "SupplementaryMaterialParser", scope: "supplementary appendix extraction", phaseId: "deepComprehensionEngine" },
    conceptEntityLinker: { label: "ConceptEntityLinker", scope: "concept graph linking", phaseId: "deepComprehensionEngine" },
    statisticalVerifier: { label: "StatisticalVerifier", scope: "cross-paper quantitative checks", phaseId: "deepComprehensionEngine" },
    thesisAgent: { label: "ThesisAgent", scope: "dominant-view argument", phaseId: "dialecticalSynthesisEngine" },
    antithesisAgent: { label: "AntithesisAgent", scope: "counter-view argument", phaseId: "dialecticalSynthesisEngine" },
    synthesisMediator: { label: "SynthesisMediator", scope: "uncertainty-aware reconciliation", phaseId: "dialecticalSynthesisEngine" },
    narrativeArchitect: { label: "NarrativeArchitect", scope: "mode-specific report compilation", phaseId: "dialecticalSynthesisEngine" },
    quantitativeSynthesizer: { label: "QuantitativeSynthesizer", scope: "meta-analysis and heterogeneity synthesis", phaseId: "dialecticalSynthesisEngine" },
    evidencePyramidBuilder: { label: "EvidencePyramidBuilder", scope: "evidence weighting by type", phaseId: "dialecticalSynthesisEngine" },
    evolvingNarrativeTracker: { label: "EvolvingNarrativeTracker", scope: "consensus shift tracking", phaseId: "dialecticalSynthesisEngine" },
    internalConsistencyCritic: { label: "InternalConsistencyCritic", scope: "claim-to-evidence enforcement", phaseId: "recursiveSelfImprovementLoop" },
    claimVerifier: { label: "ClaimVerifier", scope: "claim-by-claim evidence support and hallucination filtering", phaseId: "recursiveSelfImprovementLoop" },
    citationVerifier: { label: "CitationVerifier", scope: "citation integrity and excerpt alignment", phaseId: "recursiveSelfImprovementLoop" },
    contradictionVerifier: { label: "ContradictionVerifier", scope: "counterevidence surfacing and false-consensus prevention", phaseId: "recursiveSelfImprovementLoop" },
    uncertaintyVerifier: { label: "UncertaintyVerifier", scope: "confidence calibration and residual uncertainty", phaseId: "recursiveSelfImprovementLoop" },
    coverageAuditor: { label: "CoverageAuditor", scope: "hypothesis coverage scoring", phaseId: "recursiveSelfImprovementLoop" },
    userGoalAlignmentCritic: { label: "UserGoalAlignmentCritic", scope: "goal and output-mode alignment", phaseId: "recursiveSelfImprovementLoop" },
    decisionIntelligenceLayer: { label: "Decision Intelligence Layer", scope: "decision payload generation and risk shaping", phaseId: "decisionIntelligenceLayer" },
    adaptiveDeliveryHub: { label: "Adaptive Delivery Hub", scope: "streaming, export packaging, delivery", phaseId: "adaptiveDeliveryHub" },
    activeSafetyAndEthics: { label: "Active Safety & Ethics", scope: "cross-cutting safety and ethics", phaseId: "activeSafetyAndEthics" },
    causalRiskAnalyzer: { label: "CausalRiskAnalyzer", scope: "causal misuse chain analysis", phaseId: "activeSafetyAndEthics" },
});

const normalizeText = (value) => String(value || "").replace(/\s+/g, " ").trim();
const clamp01 = (value) => Math.max(0, Math.min(1, Number(value) || 0));
const uniqueList = (values = []) => [...new Set((Array.isArray(values) ? values : []).map(normalizeText).filter(Boolean))];

const detectQuerySignals = (query) => {
    const base = normalizeText(query);
    return {
        operatorHeavy: OPERATOR_HEAVY_RE.test(base),
        docsIntent: DOCS_RE.test(base),
        researchIntent: RESEARCH_RE.test(base),
        currentIntent: CURRENT_RE.test(base),
        comparisonIntent: COMPARISON_RE.test(base),
        questionIntent: QUESTION_RE.test(base),
    };
};

const buildSearchQueries = (query, options = {}) => {
    const base = normalizeText(query);
    const maxQueries = Math.max(1, Number(options.maxQueries) || 4);
    if (!base) return [];

    const variants = [base];
    const signals = detectQuerySignals(base);
    const push = (...values) => variants.push(...values.map(normalizeText).filter(Boolean));

    if (signals.currentIntent) push(`${base} latest`, `${base} official source`);
    if (signals.docsIntent) push(`${base} official documentation`, `${base} reference`);
    if (signals.researchIntent) push(`${base} peer reviewed research`, `${base} evidence`);
    if (signals.comparisonIntent) push(`${base} benchmark analysis`);
    if (signals.questionIntent) push(`${base} expert analysis`);
    if (signals.operatorHeavy) push(`${base} evidence`);
    if (!Object.values(signals).some(Boolean)) push(`${base} overview`, `${base} official source`);

    return uniqueList(variants).slice(0, maxQueries);
};

const detectDepthPreference = (query, explicit = "") => {
    const manual = normalizeText(explicit).toLowerCase();
    if (["speed", "balanced", "deep"].includes(manual)) return manual;

    const normalized = normalizeText(query).toLowerCase();
    if (!normalized) return DEFAULT_DEPTH;
    if (/\b(quick|brief|fast|speed|high level|high-level|tl;dr)\b/.test(normalized)) return "speed";
    if (/\b(deep|deeper|thorough|exhaustive|comprehensive|detailed|deep dive)\b/.test(normalized)) return "deep";
    return DEFAULT_DEPTH;
};

const detectResearchDomain = (query) => {
    const normalized = normalizeText(query);
    if (!normalized) return { ...DOMAIN_FALLBACK, confidence: 0.24 };
    const match = DOMAIN_RULES.find((rule) => rule.patterns.test(normalized));
    return match ? { ...match, confidence: 0.82 } : { ...DOMAIN_FALLBACK, confidence: 0.48 };
};

const detectResearchScope = (query) => {
    const normalized = normalizeText(query);
    if (!normalized) return { ...SCOPE_FALLBACK, confidence: 0.26 };
    const match = SCOPE_RULES.find((rule) => rule.patterns.test(normalized));
    if (match) return { ...match, confidence: 0.78 };

    const signals = detectQuerySignals(normalized);
    if (signals.comparisonIntent || signals.currentIntent) {
        return { id: "field_scan", label: "Field Scan", confidence: 0.62 };
    }

    return { ...SCOPE_FALLBACK, confidence: 0.5 };
};

const detectResearchOutputMode = (query, forcedOutputMode = "") => {
    const forced = normalizeText(forcedOutputMode).toLowerCase();
    if (forced) {
        const match = OUTPUT_MODE_RULES.find((rule) => rule.id === forced || rule.label.toLowerCase() === forced);
        if (match) return { ...match, confidence: 1 };
    }

    const normalized = normalizeText(query);
    if (!normalized) return { ...OUTPUT_MODE_FALLBACK, confidence: 0.32 };
    const match = OUTPUT_MODE_RULES.find((rule) => rule.patterns.test(normalized));
    return match ? { ...match, confidence: 0.8 } : { ...OUTPUT_MODE_FALLBACK, confidence: 0.52 };
};

const detectIntentConfidence = (query, forcedOutputMode = "") => {
    const domain = detectResearchDomain(query);
    const scope = detectResearchScope(query);
    const outputMode = detectResearchOutputMode(query, forcedOutputMode);
    const axes = {
        domain: { value: domain.label, confidence: clamp01(domain.confidence) },
        scope: { value: scope.label, confidence: clamp01(scope.confidence) },
        outputFormat: { value: outputMode.label, confidence: clamp01(outputMode.confidence) },
    };

    return {
        axes,
        ambiguousAxes: Object.entries(axes).filter(([, axis]) => axis.confidence < 0.65).map(([axis]) => axis),
    };
};

const buildCounterHypotheses = (query, scope) => {
    const normalized = normalizeText(query);
    if (!normalized) return [];

    const hypotheses = [
        `Evidence against ${normalized}`,
        `Conflicting findings for ${normalized}`,
    ];
    if (scope.id === "replication_study") hypotheses.push(`Failed replications for ${normalized}`);
    if (scope.id === "gap_finding") hypotheses.push(`Unanswered questions in ${normalized}`);
    if (scope.id === "decision_support") hypotheses.push(`Risks and tradeoffs of ${normalized}`);
    return uniqueList(hypotheses).slice(0, 4);
};

const buildSemanticSeeds = (query, domain) => {
    const normalized = normalizeText(query);
    if (!normalized) return [];
    const seeds = [
        `${normalized} related work`,
        `${normalized} evidence synthesis`,
        `${normalized} benchmark analysis`,
    ];
    if (domain.id === "biomedical") seeds.push(`${normalized} systematic review`);
    if (domain.id === "cs_ml") seeds.push(`${normalized} ablation study`);
    return uniqueList(seeds).slice(0, 4);
};

const buildCitationSnowballSeeds = (query, scope) => {
    const normalized = normalizeText(query);
    if (!normalized) return [];
    const seeds = [`${normalized} seminal paper`, `${normalized} highly cited`];
    if (scope.id === "gap_finding") seeds.push(`${normalized} future work`);
    if (scope.id === "replication_study") seeds.push(`${normalized} replication study`);
    return uniqueList(seeds).slice(0, 4);
};

const buildQueryVersionLog = (query, domain, scope, outputMode, depthPreference = DEFAULT_DEPTH) => {
    const normalized = normalizeText(query);
    if (!normalized) return [];
    const depthLabel = depthPreference === "speed" ? "high signal" : depthPreference === "deep" ? "deep evidence" : "balanced evidence";

    return [
        { version: "v0", query: normalized, rationale: "Original user intent." },
        { version: "v1", query: `${normalized} ${domain.taxonomy}`.trim(), rationale: `DomainDetector applied ${domain.taxonomy} cues.` },
        { version: "v2", query: `${normalized} ${outputMode.label}`.trim(), rationale: `Narrative target aligned to ${outputMode.label}.` },
        { version: "v3", query: `${normalized} ${scope.label}`.trim(), rationale: `Scope-specific expansion for ${scope.label}.` },
        { version: "v4", query: `${normalized} ${depthLabel}`.trim(), rationale: `Cost-quality profile tuned for ${depthPreference}.` },
    ];
};

const buildResearchQueryMatrix = (query, options = {}) => {
    const normalized = normalizeText(query);
    const domain = options.domain || detectResearchDomain(normalized);
    const scope = options.scope || detectResearchScope(normalized);
    const outputMode = options.outputMode || detectResearchOutputMode(normalized, options.forcedOutputMode);
    const depthPreference = options.depthPreference || DEFAULT_DEPTH;

    return {
        keywords: uniqueList(buildSearchQueries(normalized, { maxQueries: 6 })),
        semanticEmbeddings: buildSemanticSeeds(normalized, domain),
        citationSnowballSeeds: buildCitationSnowballSeeds(normalized, scope),
        ontologyMappedVocabulary: uniqueList(domain.ontologyTerms).slice(0, 4),
        counterHypotheses: buildCounterHypotheses(normalized, scope),
        versions: buildQueryVersionLog(normalized, domain, scope, outputMode, depthPreference),
    };
};

const computeOverlapScore = (queryTerms, text) => {
    if (!queryTerms.length) return 0;
    const otherTerms = new Set(extractQueryTerms(text, 32));
    if (!otherTerms.size) return 0;
    let overlap = 0;
    for (const term of queryTerms) {
        if (otherTerms.has(term)) overlap += 1;
    }
    return overlap / Math.max(queryTerms.length, 1);
};

const inferSessionContinuity = (query, memoryContext = {}) => {
    const queryTerms = extractQueryTerms(normalizeText(query), 24);
    const ranked = (Array.isArray(memoryContext.episodes) ? memoryContext.episodes : [])
        .map((episode) => ({
            episode,
            overlap: computeOverlapScore(queryTerms, [episode.query, episode.title, episode.summary].join(" ")),
        }))
        .filter((row) => row.overlap > 0)
        .sort((left, right) => right.overlap - left.overlap)
        .slice(0, 3);

    const best = ranked[0];
    if (!best || best.overlap < 0.25) {
        return {
            active: false,
            overlap: 0,
            summary: "",
            reason: "No strong prior research overlap detected.",
        };
    }

    return {
        active: true,
        overlap: Number(best.overlap.toFixed(2)),
        summary: `${best.episode.title || best.episode.query}\n${best.episode.summary}`,
        reason: "Prior research context was linked into the Cognitive Command Layer.",
        sourceRunId: best.episode.runId,
    };
};

const buildParetoProfile = (depthPreference = DEFAULT_DEPTH) => {
    if (depthPreference === "speed") {
        return {
            mode: "speed",
            speed: 0.86,
            depth: 0.44,
            explanation: "Pareto front biased toward latency and rapid checkpoint delivery.",
        };
    }
    if (depthPreference === "deep") {
        return {
            mode: "deep",
            speed: 0.36,
            depth: 0.9,
            explanation: "Pareto front biased toward coverage, critique cycles, and extraction depth.",
        };
    }
    return {
        mode: "balanced",
        speed: 0.64,
        depth: 0.68,
        explanation: "Pareto front balanced for depth without overextending the graph.",
    };
};

const buildSafetyChecks = (query, domain) => {
    const normalized = normalizeText(query);
    const checks = [
        { id: "dual_use_flagging", label: "Dual-use flagging", active: /\b(biosecurity|surveillance|weapon|exploit|malware|pathogen)\b/i.test(normalized) },
        { id: "funding_conflict_detector", label: "Funding conflict detector", active: /\b(drug|biomedical|policy|regulation|safety|surveillance)\b/i.test(normalized) },
        { id: "predatory_journal_filter", label: "Predatory journal filter", active: true },
        { id: "statistical_manipulation_detector", label: "Statistical manipulation detector", active: ["biomedical", "social_science", "cs_ml"].includes(domain.id) },
    ];

    return {
        checks,
        activeCount: checks.filter((check) => check.active).length,
    };
};

const buildDynamicSearchQueries = (plan, options = {}) => {
    const maxQueries = Math.max(1, Number(options.maxQueries) || 6);
    const matrix = plan.queryMatrix || {};
    return uniqueList([
        ...(matrix.keywords || []),
        ...(matrix.semanticEmbeddings || []),
        ...(matrix.counterHypotheses || []),
        ...(matrix.ontologyMappedVocabulary || []).map((term) => `${plan.query} ${term}`),
        ...(matrix.citationSnowballSeeds || []),
    ]).slice(0, maxQueries);
};

const buildDag = (plan) => {
    const hasQuery = Boolean(plan.query);
    const hasAttachments = Number(plan.attachments || 0) > 0;
    const needsDecisionLayer = ["decision_brief", "policy_recommendation", "engineering_action_plan"].includes(plan?.outputMode?.id);
    const nodes = Object.values(PIPELINE_PHASES)
        .filter((phase) => {
            if ((phase.id === "intelligentCrawlerMesh" || phase.id === "tieredEpistemicFilter") && !hasQuery) return false;
            if (phase.id === "deepComprehensionEngine") return hasQuery || hasAttachments;
            if (phase.id === "decisionIntelligenceLayer") return needsDecisionLayer;
            return true;
        })
        .map((phase) => ({
            ...phase,
        }));

    const activeIds = new Set(nodes.map((node) => node.id));
    const filteredNodes = nodes.map((node) => ({
        ...node,
        dependsOn: node.dependsOn.filter((dependency) => activeIds.has(dependency)),
    }));

    return {
        nodes: filteredNodes,
        edges: filteredNodes.flatMap((node) => node.dependsOn.map((from) => ({ from, to: node.id }))),
    };
};

const buildInlineConstraints = (plan) => ([
    {
        id: "core_or_supporting_citations_only",
        constraint: "All narrative claims must map to at least one Core or Supporting source.",
        enforcedAt: "synthesisMediator",
    },
    {
        id: "user_goal_alignment",
        constraint: `Final output must follow ${plan.outputMode.label} mode and expose residual uncertainty.`,
        enforcedAt: "narrativeArchitect",
    },
    {
        id: "counter_hypothesis_coverage",
        constraint: "Counter-hypotheses must be addressed explicitly when they exist.",
        enforcedAt: "coverageAuditor",
    },
]).map((item) => ({
    ...item,
    severity: item.id === "core_or_supporting_citations_only" ? "critical" : "strong",
}));

const buildSubagents = (plan) => {
    const hasQuery = Boolean(plan.query);
    const attachmentCount = Number(plan.attachments || 0);
    const counterHypothesisCount = (plan.queryMatrix.counterHypotheses || []).length;
    const searchLaneCount = Math.max(1, plan.searchQueries.length || 1);
    const descriptors = [
        {
            id: "cognitiveCommandLayer",
            label: SUBAGENT_SPECS.cognitiveCommandLayer.label,
            scope: SUBAGENT_SPECS.cognitiveCommandLayer.scope,
            count: 1,
            detail: `${plan.intentConfidence.ambiguousAxes.length} low-confidence axis${plan.intentConfidence.ambiguousAxes.length === 1 ? "" : "es"}`,
        },
        {
            id: "domainDetector",
            label: SUBAGENT_SPECS.domainDetector.label,
            scope: SUBAGENT_SPECS.domainDetector.scope,
            count: 1,
            detail: `${plan.domain.label} / ${plan.domain.taxonomy}`,
        },
        {
            id: "queryVersionController",
            label: SUBAGENT_SPECS.queryVersionController.label,
            scope: SUBAGENT_SPECS.queryVersionController.scope,
            count: 1,
            detail: `${plan.queryMatrix.versions.length} logged revisions`,
        },
        {
            id: "activeSafetyAndEthics",
            label: SUBAGENT_SPECS.activeSafetyAndEthics.label,
            scope: SUBAGENT_SPECS.activeSafetyAndEthics.scope,
            count: 1,
            detail: `${plan.safety.activeCount} active safety check${plan.safety.activeCount === 1 ? "" : "s"}`,
        },
        {
            id: "causalRiskAnalyzer",
            label: SUBAGENT_SPECS.causalRiskAnalyzer.label,
            scope: SUBAGENT_SPECS.causalRiskAnalyzer.scope,
            count: 1,
            detail: plan.safety.activeCount ? "risk propagation watch" : "latent risk watch",
        },
    ];

    if (hasQuery) {
        descriptors.push(
            { id: "devilsAdvocateDecomposer", label: SUBAGENT_SPECS.devilsAdvocateDecomposer.label, scope: SUBAGENT_SPECS.devilsAdvocateDecomposer.scope, count: 1, detail: `${counterHypothesisCount || 1} counter-hypothesis lane${counterHypothesisCount === 1 ? "" : "s"}` },
            { id: "forwardCitationTracer", label: SUBAGENT_SPECS.forwardCitationTracer.label, scope: SUBAGENT_SPECS.forwardCitationTracer.scope, count: plan.depthPreference === "deep" ? 2 : 1, detail: `depth ${plan.depthPreference === "deep" ? 2 : 1} citation tracing` },
            { id: "authorNetworkMapper", label: SUBAGENT_SPECS.authorNetworkMapper.label, scope: SUBAGENT_SPECS.authorNetworkMapper.scope, count: 1, detail: "co-authorship watch" },
            { id: "temporalTrendAnalyzer", label: SUBAGENT_SPECS.temporalTrendAnalyzer.label, scope: SUBAGENT_SPECS.temporalTrendAnalyzer.scope, count: 1, detail: `${plan.domain.recencyHalfLifeYears}-year recency horizon` },
            { id: "temporalRelevanceDecay", label: SUBAGENT_SPECS.temporalRelevanceDecay.label, scope: SUBAGENT_SPECS.temporalRelevanceDecay.scope, count: 1, detail: `${plan.domain.recencyHalfLifeYears}-year half-life` },
            { id: "retractedPaperGuard", label: SUBAGENT_SPECS.retractedPaperGuard.label, scope: SUBAGENT_SPECS.retractedPaperGuard.scope, count: 1, detail: "quarantine and audit log" },
            { id: "sampleSizeFilter", label: SUBAGENT_SPECS.sampleSizeFilter.label, scope: SUBAGENT_SPECS.sampleSizeFilter.scope, count: 1, detail: "confidence weighting" },
            { id: "thesisAgent", label: SUBAGENT_SPECS.thesisAgent.label, scope: SUBAGENT_SPECS.thesisAgent.scope, count: 1, detail: "dominant-view argument" },
            { id: "antithesisAgent", label: SUBAGENT_SPECS.antithesisAgent.label, scope: SUBAGENT_SPECS.antithesisAgent.scope, count: 1, detail: "counter-evidence argument" },
            { id: "synthesisMediator", label: SUBAGENT_SPECS.synthesisMediator.label, scope: SUBAGENT_SPECS.synthesisMediator.scope, count: 1, detail: "uncertainty-aware reconciliation" },
            { id: "narrativeArchitect", label: SUBAGENT_SPECS.narrativeArchitect.label, scope: SUBAGENT_SPECS.narrativeArchitect.scope, count: 1, detail: plan.outputMode.label },
            { id: "quantitativeSynthesizer", label: SUBAGENT_SPECS.quantitativeSynthesizer.label, scope: SUBAGENT_SPECS.quantitativeSynthesizer.scope, count: 1, detail: "lightweight meta-analysis" },
            { id: "evidencePyramidBuilder", label: SUBAGENT_SPECS.evidencePyramidBuilder.label, scope: SUBAGENT_SPECS.evidencePyramidBuilder.scope, count: 1, detail: "evidence-type weighting" },
            { id: "evolvingNarrativeTracker", label: SUBAGENT_SPECS.evolvingNarrativeTracker.label, scope: SUBAGENT_SPECS.evolvingNarrativeTracker.scope, count: 1, detail: "consensus shift tracking" },
        );
    }

    if (hasQuery || attachmentCount) {
        descriptors.push(
            { id: "statisticalClaimExtractor", label: SUBAGENT_SPECS.statisticalClaimExtractor.label, scope: SUBAGENT_SPECS.statisticalClaimExtractor.scope, count: 1, detail: attachmentCount ? `${attachmentCount} attachment${attachmentCount === 1 ? "" : "s"} + fetched evidence` : "structured claim schema" },
            { id: "codeRepoAnalyzer", label: SUBAGENT_SPECS.codeRepoAnalyzer.label, scope: SUBAGENT_SPECS.codeRepoAnalyzer.scope, count: 1, detail: hasQuery ? "repo-linked paper checks" : "attachment code review" },
            { id: "supplementaryMaterialParser", label: SUBAGENT_SPECS.supplementaryMaterialParser.label, scope: SUBAGENT_SPECS.supplementaryMaterialParser.scope, count: 1, detail: "supplementary appendix sweep" },
            { id: "conceptEntityLinker", label: SUBAGENT_SPECS.conceptEntityLinker.label, scope: SUBAGENT_SPECS.conceptEntityLinker.scope, count: 1, detail: `${plan.domain.taxonomy} + knowledge graph` },
            { id: "statisticalVerifier", label: SUBAGENT_SPECS.statisticalVerifier.label, scope: SUBAGENT_SPECS.statisticalVerifier.scope, count: 1, detail: "sandboxed recomputation hooks" },
        );
    }

    descriptors.push(
        { id: "internalConsistencyCritic", label: SUBAGENT_SPECS.internalConsistencyCritic.label, scope: SUBAGENT_SPECS.internalConsistencyCritic.scope, count: 1, detail: `${plan.refinementBudget} cycle budget` },
        { id: "claimVerifier", label: SUBAGENT_SPECS.claimVerifier.label, scope: SUBAGENT_SPECS.claimVerifier.scope, count: 1, detail: "claim support gate" },
        { id: "citationVerifier", label: SUBAGENT_SPECS.citationVerifier.label, scope: SUBAGENT_SPECS.citationVerifier.scope, count: 1, detail: "citation integrity gate" },
        { id: "contradictionVerifier", label: SUBAGENT_SPECS.contradictionVerifier.label, scope: SUBAGENT_SPECS.contradictionVerifier.scope, count: 1, detail: "counterevidence gate" },
        { id: "uncertaintyVerifier", label: SUBAGENT_SPECS.uncertaintyVerifier.label, scope: SUBAGENT_SPECS.uncertaintyVerifier.scope, count: 1, detail: "confidence calibration gate" },
        { id: "coverageAuditor", label: SUBAGENT_SPECS.coverageAuditor.label, scope: SUBAGENT_SPECS.coverageAuditor.scope, count: 1, detail: plan.scope.label },
        { id: "userGoalAlignmentCritic", label: SUBAGENT_SPECS.userGoalAlignmentCritic.label, scope: SUBAGENT_SPECS.userGoalAlignmentCritic.scope, count: 1, detail: plan.outputMode.label },
        ...(["decision_brief", "policy_recommendation", "engineering_action_plan"].includes(plan.outputMode.id)
            ? [{ id: "decisionIntelligenceLayer", label: SUBAGENT_SPECS.decisionIntelligenceLayer.label, scope: SUBAGENT_SPECS.decisionIntelligenceLayer.scope, count: 1, detail: plan.outputMode.label }]
            : []),
        { id: "adaptiveDeliveryHub", label: SUBAGENT_SPECS.adaptiveDeliveryHub.label, scope: SUBAGENT_SPECS.adaptiveDeliveryHub.scope, count: 1, detail: `${searchLaneCount} streaming lane${searchLaneCount === 1 ? "" : "s"}` },
    );

    return descriptors;
};

const buildConvergenceMetrics = ({
    iterations = 1,
    refinementBudget = DEFAULT_REFINEMENT_BUDGET,
    coverageScore = 0.7,
    contradictionScore = 0.2,
    verificationScore = null,
    stabilityScore = null,
} = {}) => {
    const resolvedIterations = Math.max(1, Number(iterations) || 1);
    const resolvedCoverage = clamp01(coverageScore);
    const resolvedContradiction = clamp01(contradictionScore);
    const resolvedVerification = verificationScore == null ? null : clamp01(verificationScore);
    const resolvedStability = stabilityScore == null
        ? clamp01(
            0.58
            + (resolvedIterations * 0.1)
            + (resolvedCoverage * 0.16)
            - (resolvedContradiction * 0.2)
            + (resolvedVerification == null ? 0 : (resolvedVerification * 0.16)),
        )
        : clamp01(stabilityScore);
    const evidenceCoverageDelta = Number((1 - resolvedCoverage).toFixed(2));
    const residualUncertainty = Number((Math.max(0.08, (1 - resolvedStability) + (resolvedContradiction * 0.32))).toFixed(2));
    const exhaustedBudget = resolvedIterations >= Math.max(1, Number(refinementBudget) || DEFAULT_REFINEMENT_BUDGET);

    return {
        iterations: resolvedIterations,
        stability_score: Number(resolvedStability.toFixed(2)),
        evidence_coverage_delta: evidenceCoverageDelta,
        residual_uncertainty: residualUncertainty,
        stop_condition: exhaustedBudget
            ? "refinement_budget_exhausted"
            : (resolvedStability < 0.95 && evidenceCoverageDelta > 0.03 ? "residual_disagreement" : "stability_reached"),
    };
};

const buildTribunalSummary = ({
    coverageScore = 0.7,
    contradictionScore = 0.2,
    alignmentScore = 0.75,
    verifiers = {},
    iterations = 1,
    refinementBudget = DEFAULT_REFINEMENT_BUDGET,
} = {}) => {
    const resolvedCoverage = clamp01(coverageScore);
    const resolvedContradiction = clamp01(contradictionScore);
    const resolvedAlignment = clamp01(alignmentScore);
    const normalizedVerifiers = {
        claim_support: clamp01(verifiers.claim_support),
        citation_integrity: clamp01(verifiers.citation_integrity),
        contradiction_handling: clamp01(verifiers.contradiction_handling),
        uncertainty_calibration: clamp01(verifiers.uncertainty_calibration),
    };
    const dimensionScores = [
        ["coverage", resolvedCoverage],
        ["internal_consistency", 1 - resolvedContradiction],
        ["user_goal_alignment", resolvedAlignment],
        ["claim_support", normalizedVerifiers.claim_support ?? 1],
        ["citation_integrity", normalizedVerifiers.citation_integrity ?? 1],
        ["contradiction_handling", normalizedVerifiers.contradiction_handling ?? 1],
        ["uncertainty_calibration", normalizedVerifiers.uncertainty_calibration ?? 1],
    ];
    const [lowestDimension] = dimensionScores.reduce((lowest, current) => (
        current[1] < lowest[1] ? current : lowest
    ));

    return {
        refinement_budget: Math.max(1, Number(refinementBudget) || DEFAULT_REFINEMENT_BUDGET),
        refinement_cycles: Math.max(1, Number(iterations) || 1),
        targeted_dimension: lowestDimension,
        critics: {
            internal_consistency: Number((1 - resolvedContradiction).toFixed(2)),
            coverage: Number(resolvedCoverage.toFixed(2)),
            user_goal_alignment: Number(resolvedAlignment.toFixed(2)),
        },
        verifiers: Object.fromEntries(
            Object.entries(normalizedVerifiers)
                .filter(([, score]) => score > 0)
                .map(([dimension, score]) => [dimension, Number(score.toFixed(2))]),
        ),
    };
};

const summarizeDag = (dag = {}) => (
    (Array.isArray(dag.nodes) ? dag.nodes : []).map((node) => node.label).join(" -> ")
);

const topologicalBatches = (dag = {}) => {
    const nodes = new Map((Array.isArray(dag.nodes) ? dag.nodes : []).map((node) => [node.id, node]));
    const pending = new Set(nodes.keys());
    const completed = new Set();
    const batches = [];

    while (pending.size) {
        const ready = [...pending]
            .map((id) => nodes.get(id))
            .filter((node) => node.dependsOn.every((dependency) => completed.has(dependency)))
            .sort((left, right) => right.priority - left.priority);

        if (!ready.length) {
            throw new Error("Research DAG contains a dependency cycle.");
        }

        const parallelBatch = ready.filter((node) => node.parallelizable);
        if (parallelBatch.length) {
            batches.push(parallelBatch);
            for (const node of parallelBatch) {
                pending.delete(node.id);
                completed.add(node.id);
            }
        } else {
            const [node] = ready;
            batches.push([node]);
            pending.delete(node.id);
            completed.add(node.id);
        }
    }

    return batches;
};

const compileResearchPlan = (options = {}) => {
    const query = normalizeText(options.query);
    const attachments = Math.max(0, Number(options.attachments) || 0);
    const depthPreference = detectDepthPreference(query, options.depthPreference);
    const domain = detectResearchDomain(query);
    const scope = detectResearchScope(query);
    const outputMode = detectResearchOutputMode(query, options.forcedOutputMode);
    const intentConfidence = detectIntentConfidence(query, options.forcedOutputMode);
    const memoryContext = options.memoryContext || {};
    const continuity = inferSessionContinuity(query, memoryContext);
    const queryMatrix = buildResearchQueryMatrix(query, {
        domain,
        scope,
        outputMode,
        depthPreference,
    });
    const pareto = buildParetoProfile(depthPreference);
    const safety = buildSafetyChecks(query, domain);
    const refinementBudget = Math.max(1, Number(options.refinementBudget) || DEFAULT_REFINEMENT_BUDGET);

    const basePlan = {
        frameworkVersion: RESEARCH_FRAMEWORK_VERSION,
        query,
        attachments,
        depthPreference,
        domain,
        scope,
        outputMode,
        intentConfidence,
        continuity,
        queryMatrix,
        pareto,
        safety,
        refinementBudget,
        steering: Array.isArray(options.steering) ? options.steering : [],
        promptPatches: Array.isArray(memoryContext.promptPatches) ? memoryContext.promptPatches : [],
    };

    const searchQueries = buildDynamicSearchQueries(basePlan, {
        maxQueries: options.maxQueries || 6,
    });
    const dag = buildDag({ ...basePlan, searchQueries });
    const inlineConstraints = buildInlineConstraints(basePlan);
    const plan = {
        ...basePlan,
        searchQueries,
        dag,
        dagSummary: summarizeDag(dag),
        inlineConstraints,
    };

    return {
        ...plan,
        subagents: buildSubagents(plan),
    };
};

module.exports = {
    RESEARCH_FRAMEWORK_VERSION,
    DEFAULT_REFINEMENT_BUDGET,
    PIPELINE_PHASES,
    SUBAGENT_SPECS,
    detectDepthPreference,
    detectResearchDomain,
    detectResearchScope,
    detectResearchOutputMode,
    detectIntentConfidence,
    buildResearchQueryMatrix,
    buildDynamicSearchQueries,
    buildConvergenceMetrics,
    buildTribunalSummary,
    summarizeDag,
    topologicalBatches,
    compileResearchPlan,
};
