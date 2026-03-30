const { extractQueryTerms } = require("../rag");

const RESEARCH_FRAMEWORK_VERSION = "3.1";
const DEFAULT_DEPTH = "balanced";
const REFINEMENT_BUDGET = 3;

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
    { id: "tutorial", label: "Tutorial", patterns: /\b(tutorial|how to|walkthrough|guide|step by step)\b/i },
    { id: "state_of_the_field", label: "State-of-the-Field", patterns: /\b(state of the field|state-of-the-field|overview|landscape|survey)\b/i },
    { id: "controversy_map", label: "Controversy Map", patterns: /\b(controversy|debate|disagreement|pros and cons|counter[- ]?argument)\b/i },
    { id: "gap_analysis", label: "Gap Analysis", patterns: /\b(gap analysis|research gap|open question|what is missing|where is the gap)\b/i },
    { id: "replication_crisis_report", label: "Replication Crisis Report", patterns: /\b(replication|reproduce|reproducibility|replication crisis)\b/i },
    { id: "foundational_review", label: "Foundational Review", patterns: /\b(foundational|history of|foundations|background|origin)\b/i },
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

const PIPELINE_PHASES = Object.freeze([
    {
        id: "cognitiveCommandLayer",
        label: "Cognitive Command Layer",
        dependsOn: [],
        priority: 1,
        parallelizable: false,
    },
    {
        id: "activeSafetyAndEthics",
        label: "Active Safety & Ethics",
        dependsOn: ["cognitiveCommandLayer"],
        priority: 0.96,
        parallelizable: true,
    },
    {
        id: "adversarialQueryForge",
        label: "Adversarial Query Forge",
        dependsOn: ["cognitiveCommandLayer"],
        priority: 0.94,
        parallelizable: false,
    },
    {
        id: "intelligentCrawlerMesh",
        label: "Intelligent Crawler Mesh",
        dependsOn: ["adversarialQueryForge"],
        priority: 0.9,
        parallelizable: true,
    },
    {
        id: "tieredEpistemicFilter",
        label: "Tiered Epistemic Filter",
        dependsOn: ["intelligentCrawlerMesh", "activeSafetyAndEthics"],
        priority: 0.88,
        parallelizable: true,
    },
    {
        id: "deepComprehensionEngine",
        label: "Deep Comprehension Engine",
        dependsOn: ["tieredEpistemicFilter", "activeSafetyAndEthics"],
        priority: 0.86,
        parallelizable: true,
    },
    {
        id: "dialecticalSynthesisEngine",
        label: "Dialectical Synthesis Engine",
        dependsOn: ["deepComprehensionEngine"],
        priority: 0.84,
        parallelizable: false,
    },
    {
        id: "recursiveSelfImprovementLoop",
        label: "Recursive Self-Improvement Loop",
        dependsOn: ["dialecticalSynthesisEngine"],
        priority: 0.82,
        parallelizable: false,
    },
    {
        id: "decisionIntelligenceLayer",
        label: "Decision Intelligence Layer",
        dependsOn: ["recursiveSelfImprovementLoop"],
        priority: 0.8,
        parallelizable: false,
    },
    {
        id: "adaptiveDeliveryHub",
        label: "Adaptive Delivery Hub",
        dependsOn: ["decisionIntelligenceLayer"],
        priority: 0.78,
        parallelizable: false,
    },
]);

const SUBAGENT_SPECS = Object.freeze({
    cognitiveCommandLayer: { label: "Cognitive Command Layer", phaseId: "cognitiveCommandLayer", scope: "Intent decomposition, DAG compilation, Pareto steering, and session continuity." },
    devilsAdvocateDecomposer: { label: "DevilsAdvocateDecomposer", phaseId: "adversarialQueryForge", scope: "Counter-hypothesis generation and adversarial decomposition." },
    domainDetector: { label: "DomainDetector", phaseId: "adversarialQueryForge", scope: "Domain taxonomy routing and ontology-aware heuristics." },
    queryVersionController: { label: "QueryVersionController", phaseId: "adversarialQueryForge", scope: "Auditable query rewrites and rollback-ready reasoning." },
    forwardCitationTracer: { label: "ForwardCitationTracer", phaseId: "intelligentCrawlerMesh", scope: "Citation snowball expansion and seed reinforcement." },
    authorNetworkMapper: { label: "AuthorNetworkMapper", phaseId: "intelligentCrawlerMesh", scope: "Co-authorship clustering and echo-chamber detection." },
    temporalTrendAnalyzer: { label: "TemporalTrendAnalyzer", phaseId: "intelligentCrawlerMesh", scope: "Emerging vs saturated topic velocity analysis." },
    temporalRelevanceDecay: { label: "TemporalRelevanceDecay", phaseId: "tieredEpistemicFilter", scope: "Field-specific recency weighting." },
    retractedPaperGuard: { label: "RetractedPaperGuard", phaseId: "tieredEpistemicFilter", scope: "Retraction quarantine and audit logging." },
    sampleSizeFilter: { label: "SampleSizeFilter", phaseId: "tieredEpistemicFilter", scope: "Sample-size confidence adjustment." },
    statisticalClaimExtractor: { label: "StatisticalClaimExtractor", phaseId: "deepComprehensionEngine", scope: "Effect sizes, p-values, sample sizes, and intervals." },
    codeRepoAnalyzer: { label: "CodeRepoAnalyzer", phaseId: "deepComprehensionEngine", scope: "Repo completeness and reproducibility scoring." },
    supplementaryMaterialParser: { label: "SupplementaryMaterialParser", phaseId: "deepComprehensionEngine", scope: "Supplementary file ingestion and appendix recovery." },
    conceptEntityLinker: { label: "ConceptEntityLinker", phaseId: "deepComprehensionEngine", scope: "Concept linking into the hierarchical memory graph." },
    statisticalVerifier: { label: "StatisticalVerifier", phaseId: "deepComprehensionEngine", scope: "Cross-paper quantitative checks and recomputation hooks." },
    thesisAgent: { label: "ThesisAgent", phaseId: "dialecticalSynthesisEngine", scope: "Dominant-view argument construction." },
    antithesisAgent: { label: "AntithesisAgent", phaseId: "dialecticalSynthesisEngine", scope: "Counter-position argument construction." },
    synthesisMediator: { label: "SynthesisMediator", phaseId: "dialecticalSynthesisEngine", scope: "Conflict reconciliation with explicit uncertainty." },
    narrativeArchitect: { label: "NarrativeArchitect", phaseId: "dialecticalSynthesisEngine", scope: "Narrative output compilation." },
    quantitativeSynthesizer: { label: "QuantitativeSynthesizer", phaseId: "dialecticalSynthesisEngine", scope: "Lightweight meta-analysis and heterogeneity." },
    evidencePyramidBuilder: { label: "EvidencePyramidBuilder", phaseId: "dialecticalSynthesisEngine", scope: "Evidence hierarchy weighting." },
    evolvingNarrativeTracker: { label: "EvolvingNarrativeTracker", phaseId: "dialecticalSynthesisEngine", scope: "Consensus shift tracking over time." },
    internalConsistencyCritic: { label: "InternalConsistencyCritic", phaseId: "recursiveSelfImprovementLoop", scope: "Claim-to-evidence consistency enforcement." },
    coverageAuditor: { label: "CoverageAuditor", phaseId: "recursiveSelfImprovementLoop", scope: "Hypothesis coverage and gap detection." },
    userGoalAlignmentCritic: { label: "UserGoalAlignmentCritic", phaseId: "recursiveSelfImprovementLoop", scope: "Output depth and format alignment." },
    deterministicDagRuntime: { label: "Deterministic DAG Runtime", phaseId: "cognitiveCommandLayer", scope: "Scheduling, checkpoints, and failure recovery." },
    inlineConstraintSystem: { label: "Inline Constraint System", phaseId: "recursiveSelfImprovementLoop", scope: "Constraint emission and immediate rewrite triggers." },
    probabilisticEpistemicLayer: { label: "Probabilistic Epistemic Layer", phaseId: "recursiveSelfImprovementLoop", scope: "Computable uncertainty propagation." },
    decisionIntelligenceLayer: { label: "Decision Intelligence Layer", phaseId: "decisionIntelligenceLayer", scope: "Decision payload generation and risk shaping." },
    adaptiveDeliveryHub: { label: "Adaptive Delivery Hub", phaseId: "adaptiveDeliveryHub", scope: "Streaming checkpoints, exports, and API packaging." },
    activeSafetyAndEthics: { label: "Active Safety & Ethics", phaseId: "activeSafetyAndEthics", scope: "Dual-use, conflict, predatory-journal, and stats-risk checks." },
    causalRiskAnalyzer: { label: "CausalRiskAnalyzer", phaseId: "activeSafetyAndEthics", scope: "Technique-to-risk chain modeling." },
});

const normalizeText = (value) => String(value || "").replace(/\s+/g, " ").trim();

const clamp01 = (value) => Math.max(0, Math.min(1, Number(value) || 0));

const uniqueList = (values = []) => [...new Set((Array.isArray(values) ? values : [values]).map(normalizeText).filter(Boolean))];

const buildSearchQueries = (query, maxQueries = 6) => {
    const normalized = normalizeText(query);
    if (!normalized) return [];

    const variants = [
        normalized,
        `${normalized} evidence`,
        `${normalized} latest evidence`,
        `${normalized} systematic review`,
        `${normalized} benchmark`,
        `${normalized} cited by`,
        `${normalized} future work`,
        `${normalized} contradiction`,
    ];

    return uniqueList(variants).slice(0, maxQueries);
};

const resolveRule = (normalizedQuery, rules, fallback, confidenceIfMatched, confidenceFallback) => {
    if (!normalizedQuery) {
        return {
            ...fallback,
            confidence: confidenceFallback * 0.55,
        };
    }

    const matched = rules.find((rule) => rule.patterns.test(normalizedQuery));
    if (matched) {
        return {
            ...matched,
            confidence: confidenceIfMatched,
        };
    }

    return {
        ...fallback,
        confidence: confidenceFallback,
    };
};

const DOMAIN_FALLBACK = {
    id: "general_research",
    label: "General Research",
    taxonomy: "General ontology",
    recencyHalfLifeYears: 4,
    ontologyTerms: ["systematic review", "benchmark", "evidence synthesis"],
};

const OUTPUT_MODE_FALLBACK = {
    id: "state_of_the_field",
    label: "State-of-the-Field",
};

const SCOPE_FALLBACK = {
    id: "broad_research",
    label: "Broad Research",
};

const detectResearchDomain = (query) => resolveRule(normalizeText(query), DOMAIN_RULES, DOMAIN_FALLBACK, 0.84, 0.48);
const detectResearchOutputMode = (query) => resolveRule(normalizeText(query), OUTPUT_MODE_RULES, OUTPUT_MODE_FALLBACK, 0.82, 0.56);
const detectResearchScope = (query) => resolveRule(normalizeText(query), SCOPE_RULES, SCOPE_FALLBACK, 0.8, 0.5);

const detectIntentConfidence = (query) => {
    const domain = detectResearchDomain(query);
    const scope = detectResearchScope(query);
    const outputMode = detectResearchOutputMode(query);

    const axes = {
        domain: { value: domain.label, confidence: clamp01(domain.confidence) },
        scope: { value: scope.label, confidence: clamp01(scope.confidence) },
        outputFormat: { value: outputMode.label, confidence: clamp01(outputMode.confidence) },
    };

    return {
        axes,
        ambiguousAxes: Object.entries(axes)
            .filter(([, axis]) => axis.confidence < 0.65)
            .map(([key]) => key),
    };
};

const buildCounterHypotheses = (query, scope) => {
    const normalized = normalizeText(query);
    if (!normalized) return [];

    const hypotheses = [
        `Evidence against: ${normalized}`,
        `Conflicting findings for: ${normalized}`,
    ];

    if (scope.id === "replication_study") hypotheses.push(`Failed replications for: ${normalized}`);
    if (scope.id === "gap_finding") hypotheses.push(`Unanswered questions in: ${normalized}`);
    if (scope.id === "decision_support") hypotheses.push(`Risks and tradeoffs of: ${normalized}`);

    return uniqueList(hypotheses).slice(0, 4);
};

const buildQueryVersionLog = (query, domain, scope, outputMode, depthPreference) => {
    const normalized = normalizeText(query);
    if (!normalized) return [];

    return [
        { version: "v0", query: normalized, rationale: "Original user intent." },
        { version: "v1", query: `${normalized} ${domain.taxonomy}`.trim(), rationale: `DomainDetector applied ${domain.taxonomy} cues.` },
        { version: "v2", query: `${normalized} ${outputMode.label}`.trim(), rationale: `Output mode aligned to ${outputMode.label}.` },
        { version: "v3", query: `${normalized} ${scope.label}`.trim(), rationale: `Scope-specific expansion for ${scope.label}.` },
        { version: "v4", query: `${normalized} ${depthPreference === "deep" ? "deep evidence" : depthPreference === "speed" ? "high signal" : "balanced evidence"}`.trim(), rationale: `Cost-quality profile tuned for ${depthPreference}.` },
    ];
};

const computeOverlapScore = (queryTerms, text) => {
    if (!queryTerms.length) return 0;
    const otherTerms = new Set(extractQueryTerms(text, 24));
    if (!otherTerms.size) return 0;
    let overlap = 0;
    for (const term of queryTerms) {
        if (otherTerms.has(term)) overlap += 1;
    }
    return overlap / Math.max(queryTerms.length, 1);
};

const inferSessionContinuity = (query, episodes = []) => {
    const normalized = normalizeText(query);
    const queryTerms = extractQueryTerms(normalized, 24);
    const ranked = (Array.isArray(episodes) ? episodes : [])
        .map((episode) => ({
            episode,
            overlap: computeOverlapScore(queryTerms, [episode.query, episode.summary, episode.outputMode].filter(Boolean).join(" ")),
        }))
        .filter((row) => row.overlap > 0)
        .sort((left, right) => right.overlap - left.overlap)
        .slice(0, 3);

    if (!ranked.length || ranked[0].overlap < 0.25) {
        return {
            active: false,
            overlap: 0,
            sourceRunId: "",
            summary: "",
            reason: "No strong prior-run overlap detected.",
        };
    }

    const best = ranked[0];
    return {
        active: true,
        overlap: Number(best.overlap.toFixed(2)),
        sourceRunId: best.episode.runId || "",
        summary: normalizeText(best.episode.summary || best.episode.query || "").slice(0, 720),
        reason: "Prior research memory was injected into Phase 1.",
    };
};

const buildParetoProfile = (depthPreference = DEFAULT_DEPTH) => {
    if (depthPreference === "speed") {
        return {
            mode: "speed",
            speed: 0.88,
            depth: 0.42,
            explanation: "Pareto front biased toward faster checkpoint delivery.",
        };
    }
    if (depthPreference === "deep") {
        return {
            mode: "deep",
            speed: 0.38,
            depth: 0.92,
            explanation: "Pareto front biased toward coverage, extraction depth, and refinement cycles.",
        };
    }
    return {
        mode: "balanced",
        speed: 0.64,
        depth: 0.7,
        explanation: "Pareto front balanced for throughput and quality.",
    };
};

const buildSafetyChecks = (query, domain) => {
    const normalized = normalizeText(query);
    const checks = [
        { id: "dual_use_flagging", label: "Dual-use flagging", active: /\b(biosecurity|surveillance|weapon|exploit|malware|pathogen)\b/i.test(normalized) },
        { id: "funding_conflict_detector", label: "Funding conflict detector", active: /(\b(drug|biomedical|policy|regulation|safety|surveillance)\b)/i.test(normalized) },
        { id: "predatory_journal_filter", label: "Predatory journal filter", active: true },
        { id: "statistical_manipulation_detector", label: "Statistical manipulation detector", active: ["biomedical", "social_science", "cs_ml"].includes(domain.id) },
    ];

    return {
        checks,
        activeCount: checks.filter((item) => item.active).length,
    };
};

const buildResearchQueryMatrix = (query, options = {}) => {
    const normalized = normalizeText(query);
    const domain = options.domain || detectResearchDomain(normalized);
    const scope = options.scope || detectResearchScope(normalized);
    const outputMode = options.outputMode || detectResearchOutputMode(normalized);
    const depthPreference = options.depthPreference || DEFAULT_DEPTH;

    const keywords = buildSearchQueries(normalized, 6);
    const semanticEmbeddings = uniqueList([
        `${normalized} related work`,
        `${normalized} evidence synthesis`,
        `${normalized} benchmark analysis`,
        domain.id === "biomedical" ? `${normalized} systematic review` : "",
        domain.id === "cs_ml" ? `${normalized} ablation study` : "",
    ]).slice(0, 4);
    const citationSnowballSeeds = uniqueList([
        `${normalized} seminal paper`,
        `${normalized} highly cited`,
        scope.id === "gap_finding" ? `${normalized} future work` : "",
        scope.id === "replication_study" ? `${normalized} replication study` : "",
    ]).slice(0, 4);
    const ontologyMappedVocabulary = uniqueList(domain.ontologyTerms).slice(0, 4);
    const counterHypotheses = buildCounterHypotheses(normalized, scope);
    const versions = buildQueryVersionLog(normalized, domain, scope, outputMode, depthPreference);

    return {
        keywords,
        semanticEmbeddings,
        citationSnowballSeeds,
        ontologyMappedVocabulary,
        counterHypotheses,
        versions,
    };
};

const buildDynamicSearchQueries = (plan, maxQueries = 6) => uniqueList([
    ...(plan.queryMatrix?.keywords || []),
    ...(plan.queryMatrix?.semanticEmbeddings || []),
    ...(plan.queryMatrix?.counterHypotheses || []),
    ...((plan.queryMatrix?.ontologyMappedVocabulary || []).map((term) => `${plan.query} ${term}`)),
    ...(plan.queryMatrix?.citationSnowballSeeds || []),
]).slice(0, maxQueries);

const buildInitialHypotheses = (plan) => {
    const base = uniqueList([
        plan.query,
        `${plan.query} evidence summary`,
        ...((plan.queryMatrix?.counterHypotheses || []).slice(0, 2)),
    ]).filter(Boolean);

    return base.slice(0, 5).map((statement, index) => ({
        id: `h${index + 1}`,
        statement,
        polarity: /^evidence against:|^conflicting findings/i.test(statement) ? "counter" : "primary",
    }));
};

const buildDag = (plan) => {
    const hasQuery = Boolean(plan.query);
    const hasAttachments = Number(plan.attachments || 0) > 0;
    const nodes = PIPELINE_PHASES
        .filter((phase) => {
            if (!hasQuery && ["adversarialQueryForge", "intelligentCrawlerMesh", "tieredEpistemicFilter"].includes(phase.id)) return false;
            if (!hasQuery && !hasAttachments && ["deepComprehensionEngine", "dialecticalSynthesisEngine"].includes(phase.id)) return false;
            return true;
        })
        .map((phase) => ({
            ...phase,
            dependsOn: phase.dependsOn.filter((dependency) => PIPELINE_PHASES.some((item) => item.id === dependency)),
        }));

    const edges = nodes.flatMap((node) => node.dependsOn.map((dependency) => ({ from: dependency, to: node.id })));
    return { nodes, edges };
};

const createSubagentDescriptor = (id, detail, count = 1) => {
    const spec = SUBAGENT_SPECS[id];
    if (!spec) return null;
    return {
        id,
        label: spec.label,
        scope: spec.scope,
        phaseId: spec.phaseId,
        count: Math.max(1, Number(count) || 1),
        detail: normalizeText(detail),
    };
};

const buildSubagents = (plan) => {
    const hasQuery = Boolean(plan.query);
    const descriptors = [
        createSubagentDescriptor("cognitiveCommandLayer", `${plan.intentConfidence.ambiguousAxes.length} low-confidence axis(es)`),
        createSubagentDescriptor("deterministicDagRuntime", `${plan.dag.nodes.length} node(s)`),
        createSubagentDescriptor("domainDetector", `${plan.domain.label} / ${plan.domain.taxonomy}`),
        createSubagentDescriptor("queryVersionController", `${plan.queryMatrix.versions.length} revision(s)`),
        createSubagentDescriptor("activeSafetyAndEthics", `${plan.safety.activeCount} active check(s)`),
        createSubagentDescriptor("causalRiskAnalyzer", plan.safety.activeCount ? "risk propagation watch" : "latent risk watch"),
        createSubagentDescriptor("internalConsistencyCritic", `${plan.refinementBudget} cycle budget`),
        createSubagentDescriptor("coverageAuditor", plan.scope.label),
        createSubagentDescriptor("userGoalAlignmentCritic", plan.outputMode.label),
        createSubagentDescriptor("inlineConstraintSystem", "claim-to-source constraints"),
        createSubagentDescriptor("probabilisticEpistemicLayer", "computable uncertainty"),
        createSubagentDescriptor("decisionIntelligenceLayer", plan.outputMode.label),
        createSubagentDescriptor("adaptiveDeliveryHub", plan.outputMode.label),
    ];

    if (hasQuery) {
        descriptors.push(
            createSubagentDescriptor("devilsAdvocateDecomposer", `${plan.queryMatrix.counterHypotheses.length || 1} counter-lane(s)`),
            createSubagentDescriptor("forwardCitationTracer", plan.depthPreference === "deep" ? "depth 2 citation tracing" : "depth 1 citation tracing", plan.depthPreference === "deep" ? 2 : 1),
            createSubagentDescriptor("authorNetworkMapper", "co-authorship watch"),
            createSubagentDescriptor("temporalTrendAnalyzer", `${plan.domain.recencyHalfLifeYears}-year horizon`),
            createSubagentDescriptor("temporalRelevanceDecay", `${plan.domain.recencyHalfLifeYears}-year half-life`),
            createSubagentDescriptor("retractedPaperGuard", "quarantine and audit log"),
            createSubagentDescriptor("sampleSizeFilter", "confidence weighting"),
            createSubagentDescriptor("thesisAgent", "dominant-view case"),
            createSubagentDescriptor("antithesisAgent", "counter-view case"),
            createSubagentDescriptor("synthesisMediator", "uncertainty-aware reconciliation"),
            createSubagentDescriptor("narrativeArchitect", plan.outputMode.label),
            createSubagentDescriptor("quantitativeSynthesizer", "lightweight meta-analysis"),
            createSubagentDescriptor("evidencePyramidBuilder", "evidence weighting"),
            createSubagentDescriptor("evolvingNarrativeTracker", "consensus shift tracking"),
        );
    }

    descriptors.push(
        createSubagentDescriptor("statisticalClaimExtractor", plan.attachments ? `${plan.attachments} attachment(s) + fetched evidence` : "structured claim schema"),
        createSubagentDescriptor("codeRepoAnalyzer", "repo-linked reproducibility"),
        createSubagentDescriptor("supplementaryMaterialParser", "supplementary appendix sweep"),
        createSubagentDescriptor("conceptEntityLinker", `${plan.domain.taxonomy} + memory graph`),
        createSubagentDescriptor("statisticalVerifier", "cross-paper recomputation"),
    );

    return descriptors.filter(Boolean);
};

const applySteeringCommands = (plan, commands = []) => {
    const next = JSON.parse(JSON.stringify(plan || {}));
    const steering = Array.isArray(commands) ? commands : [];

    next.steering = [
        ...(Array.isArray(next.steering) ? next.steering : []),
        ...steering,
    ];

    for (const command of steering) {
        const type = String(command?.type || command?.command || "").trim();
        if (type === "increase_depth") {
            next.depthPreference = "deep";
            next.pareto = buildParetoProfile("deep");
        }
        if (type === "prioritize_speed") {
            next.depthPreference = "speed";
            next.pareto = buildParetoProfile("speed");
        }
        if (type === "force_mode") {
            const requested = normalizeText(command?.mode || command?.value);
            const matched = OUTPUT_MODE_RULES.find((item) => item.id === requested || item.label.toLowerCase() === requested.toLowerCase());
            next.outputMode = matched ? { ...matched, confidence: 1 } : next.outputMode;
        }
        if (type === "exclude_source") {
            const excluded = normalizeText(command?.sourceType || command?.value);
            next.excludedSourceTypes = uniqueList([...(next.excludedSourceTypes || []), excluded]);
        }
    }

    next.queryMatrix = buildResearchQueryMatrix(next.query, {
        domain: next.domain,
        scope: next.scope,
        outputMode: next.outputMode,
        depthPreference: next.depthPreference,
    });
    next.searchQueries = buildDynamicSearchQueries(next, 6);
    next.dag = buildDag(next);
    next.subagents = buildSubagents(next);
    return next;
};

const compileResearchPlan = (options = {}) => {
    const query = normalizeText(options.query);
    const attachments = Math.max(0, Number(options.attachments) || 0);
    const depthPreference = ["speed", "balanced", "deep"].includes(options.depthPreference)
        ? options.depthPreference
        : DEFAULT_DEPTH;
    const domain = detectResearchDomain(query);
    const scope = detectResearchScope(query);
    const outputMode = detectResearchOutputMode(query);
    const intentConfidence = detectIntentConfidence(query);
    const continuity = inferSessionContinuity(query, options.episodes || []);
    const queryMatrix = buildResearchQueryMatrix(query, {
        domain,
        scope,
        outputMode,
        depthPreference,
    });
    const pareto = buildParetoProfile(depthPreference);
    const safety = buildSafetyChecks(query, domain);
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
        refinementBudget: Math.max(1, Number(options.refinementBudget) || REFINEMENT_BUDGET),
    };
    const searchQueries = buildDynamicSearchQueries(basePlan, options.maxQueries || 6);
    const dag = buildDag({
        ...basePlan,
        searchQueries,
    });
    const plan = {
        ...basePlan,
        searchQueries,
        dag,
    };
    return {
        ...plan,
        hypotheses: buildInitialHypotheses(plan),
        subagents: buildSubagents(plan),
    };
};

const summarizeDag = (dag = {}) => (Array.isArray(dag.nodes) ? dag.nodes : []).map((node) => node.label).join(" -> ");

module.exports = {
    RESEARCH_FRAMEWORK_VERSION,
    REFINEMENT_BUDGET,
    SUBAGENT_SPECS,
    compileResearchPlan,
    applySteeringCommands,
    summarizeDag,
    buildInitialHypotheses,
    inferSessionContinuity,
};
