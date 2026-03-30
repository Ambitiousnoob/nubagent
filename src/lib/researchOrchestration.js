import { buildSearchQueries, detectQuerySignals } from "./searchPlanner.js";
import { extractQueryTerms } from "./rag.js";

export const RESEARCH_FRAMEWORK_VERSION = "3.1";
export const REFINEMENT_BUDGET = 3;

const DEFAULT_DEPTH = "balanced";
const SCHOLARLY_SIGNAL_RE =
  /\b(scholar|scholarly|academic|peer[- ]?reviewed|journal|journals|paper|papers|study|studies|literature|meta-analysis|systematic review|doi|citation|citations|university|thesis|dissertation)\b/i;

const DOMAIN_RULES = [
  {
    id: "biomedical",
    label: "Biomedical",
    taxonomy: "MeSH",
    patterns:
      /\b(biomedical|biosecurity|clinical|cohort|trial|trials|patient|patients|oncology|genomics|protein|dna|rna|biopsy|medical|medicine|public health|epidemiology|pharmacology)\b/i,
    recencyHalfLifeYears: 2,
    ontologyTerms: [
      "MeSH",
      "clinical trial",
      "systematic review",
      "cohort study",
    ],
  },
  {
    id: "cs_ml",
    label: "CS + ML",
    taxonomy: "ACM CCS",
    patterns:
      /\b(llm|language model|transformer|retrieval augmented generation|rag|benchmark|ablation|machine learning|deep learning|computer vision|reinforcement learning|ml safety|alignment|scaling law|inference)\b/i,
    recencyHalfLifeYears: 2,
    ontologyTerms: [
      "ACM CCS",
      "benchmark",
      "ablation study",
      "reproducibility",
    ],
  },
  {
    id: "social_science",
    label: "Social Science",
    taxonomy: "JEL",
    patterns:
      /\b(social science|sociology|economics|policy|public policy|behavioral|survey|qualitative|labor market|governance|education|political science)\b/i,
    recencyHalfLifeYears: 6,
    ontologyTerms: [
      "JEL",
      "survey instrument",
      "observational study",
      "policy analysis",
    ],
  },
  {
    id: "interdisciplinary",
    label: "Interdisciplinary",
    taxonomy: "Mixed ontology",
    patterns:
      /\b(interdisciplinary|cross-domain|human-ai|socio-technical|sociotechnical|computational social science|digital health|bioinformatics)\b/i,
    recencyHalfLifeYears: 4,
    ontologyTerms: [
      "cross-disciplinary",
      "mixed methods",
      "evidence synthesis",
      "knowledge graph",
    ],
  },
];

const OUTPUT_MODE_RULES = [
  {
    id: "tutorial",
    label: "Tutorial",
    patterns: /\b(tutorial|how to|how do|walkthrough|guide|step by step)\b/i,
  },
  {
    id: "controversy_map",
    label: "Controversy Map",
    patterns:
      /\b(controversy|debate|argue|disagreement|pros and cons|counter[- ]?argument)\b/i,
  },
  {
    id: "gap_analysis",
    label: "Gap Analysis",
    patterns:
      /\b(gap analysis|research gap|open question|what is missing|where is the gap)\b/i,
  },
  {
    id: "replication_crisis_report",
    label: "Replication Crisis Report",
    patterns:
      /\b(replication|reproduce|reproducibility|reproducible|replication crisis)\b/i,
  },
  {
    id: "foundational_review",
    label: "Foundational Review",
    patterns:
      /\b(foundational|history of|foundations|background|origin|foundational review)\b/i,
  },
  {
    id: "decision_brief",
    label: "Decision Brief",
    patterns:
      /\b(decision|recommend|should we|should i|choose|tradeoff|go no-go)\b/i,
  },
  {
    id: "policy_recommendation",
    label: "Policy Recommendation",
    patterns:
      /\b(policy recommendation|regulation|governance|policy|compliance|regulatory)\b/i,
  },
  {
    id: "engineering_action_plan",
    label: "Engineering Action Plan",
    patterns:
      /\b(engineering action plan|implementation plan|action plan|roadmap|migration plan)\b/i,
  },
];

const SCOPE_RULES = [
  {
    id: "gap_finding",
    label: "Gap-finding",
    patterns: /\b(gap|missing|open question|unexplored|future work)\b/i,
  },
  {
    id: "replication_study",
    label: "Replication Study",
    patterns: /\b(replication|reproducibility|reproduce|rerun|replicate)\b/i,
  },
  {
    id: "contested_topic",
    label: "Contested Topic",
    patterns:
      /\b(controversy|debate|versus|vs\.?|conflicting|disputed|contested)\b/i,
  },
  {
    id: "decision_support",
    label: "Decision Support",
    patterns: /\b(should|recommend|choose|decision|buy|adopt|prioritize)\b/i,
  },
];

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

const PIPELINE_PHASES = Object.freeze({
  cognitiveCommandLayer: {
    id: "cognitiveCommandLayer",
    label: "Cognitive Command Layer",
    summary:
      "Intent confidence decomposition, DAG compilation, Pareto steering, and session continuity.",
    dependsOn: [],
    parallelizable: false,
    critical: true,
  },
  adversarialQueryForge: {
    id: "adversarialQueryForge",
    label: "Adversarial Query Forge",
    summary:
      "Counter-hypothesis generation, domain detection, and auditable query version control.",
    dependsOn: ["cognitiveCommandLayer"],
    parallelizable: false,
    critical: true,
  },
  intelligentCrawlerMesh: {
    id: "intelligentCrawlerMesh",
    label: "Intelligent Crawler Mesh",
    summary:
      "Adaptive source mesh, scholarly harvesting, citation tracing, author networks, and temporal trend analysis.",
    dependsOn: ["adversarialQueryForge"],
    parallelizable: true,
    critical: true,
  },
  tieredEpistemicFilter: {
    id: "tieredEpistemicFilter",
    label: "Tiered Epistemic Filter",
    summary:
      "Core/supporting/peripheral/discard gating plus temporal, retraction, and sample-size guardrails.",
    dependsOn: ["intelligentCrawlerMesh"],
    parallelizable: true,
    critical: true,
  },
  activeSafetyAndEthics: {
    id: "activeSafetyAndEthics",
    label: "Active Safety & Ethics",
    summary:
      "Dual-use, conflicts, predatory journal, and statistical manipulation checks.",
    dependsOn: ["cognitiveCommandLayer"],
    parallelizable: true,
    critical: true,
  },
  deepComprehensionEngine: {
    id: "deepComprehensionEngine",
    label: "Deep Comprehension Engine",
    summary:
      "Structured claim extraction, code/repo review, supplementary parsing, entity linking, and statistical verification.",
    dependsOn: ["tieredEpistemicFilter", "activeSafetyAndEthics"],
    parallelizable: true,
    critical: true,
  },
  dialecticalSynthesisEngine: {
    id: "dialecticalSynthesisEngine",
    label: "Dialectical Synthesis Engine",
    summary:
      "Position mapping, thesis/antithesis debate, and narrative compilation.",
    dependsOn: ["deepComprehensionEngine"],
    parallelizable: false,
    critical: true,
  },
  recursiveSelfImprovementLoop: {
    id: "recursiveSelfImprovementLoop",
    label: "Recursive Self-Improvement Loop",
    summary:
      "Verifier swarm and critic-enforced refinement budget with hallucination checks, coverage, and alignment control.",
    dependsOn: ["dialecticalSynthesisEngine"],
    parallelizable: false,
    critical: true,
  },
  decisionIntelligenceLayer: {
    id: "decisionIntelligenceLayer",
    label: "Decision Intelligence Layer",
    summary:
      "Decision framing, risk shaping, reversibility, and recommendation packaging.",
    dependsOn: ["recursiveSelfImprovementLoop"],
    parallelizable: false,
    critical: true,
  },
  adaptiveDeliveryHub: {
    id: "adaptiveDeliveryHub",
    label: "Adaptive Delivery Hub",
    summary:
      "Streaming checkpoints, output-mode adaptation, and delivery packaging.",
    dependsOn: ["decisionIntelligenceLayer"],
    parallelizable: false,
    critical: true,
  },
});

const SUBAGENT_EQUIPMENT = Object.freeze({
  cognitiveCommandLayer: [
    "Intent axes",
    "DAG compiler",
    "Pareto controls",
    "Session continuity memory",
  ],
  devilsAdvocateDecomposer: [
    "Counter-hypothesis lanes",
    "Disconfirming prompts",
    "Scope challenges",
  ],
  domainDetector: ["Domain taxonomy", "Ontology vocabulary", "Recency norms"],
  queryVersionController: [
    "Revision log",
    "Rationale ledger",
    "Rollback pointer",
  ],
  forwardCitationTracer: [
    "OpenAlex cited-by API",
    "Citation snowball seeds",
    "Seed paper set",
  ],
  scholarlySourceHarvester: [
    "Google Scholar-style lanes",
    "Semantic Scholar lanes",
    "OpenAlex lanes",
    "University-domain sweep",
    "Institutional repositories",
  ],
  authorNetworkMapper: [
    "Author graph",
    "Co-authorship edges",
    "Echo-chamber heuristics",
  ],
  temporalTrendAnalyzer: [
    "Publication year buckets",
    "Velocity chart",
    "Recency horizon",
  ],
  temporalRelevanceDecay: [
    "Field half-life",
    "Recency weighting",
    "Freshness score",
  ],
  retractedPaperGuard: [
    "Retraction signals",
    "Quarantine log",
    "DOI/title cross-check",
  ],
  sampleSizeFilter: [
    "Sample-size parser",
    "Domain threshold",
    "Confidence weighting",
  ],
  statisticalClaimExtractor: [
    "Effect-size parser",
    "P-value parser",
    "Sample-size schema",
  ],
  codeRepoAnalyzer: [
    "Repo URL extractor",
    "README audit",
    "Dependency manifest scan",
  ],
  supplementaryMaterialParser: [
    "Appendix scan",
    "Method recovery",
    "Supplementary evidence cache",
  ],
  conceptEntityLinker: [
    "Ontology vocabulary",
    "Concept graph",
    "Entity normalizer",
  ],
  statisticalVerifier: [
    "Sandboxed recomputation",
    "Claim ledger",
    "Meta-analysis hooks",
  ],
  thesisAgent: [
    "Core/supporting source set",
    "Position map",
    "Dominant-view brief",
  ],
  antithesisAgent: [
    "Counterevidence clusters",
    "Contradiction map",
    "Disconfirming source set",
  ],
  synthesisMediator: [
    "Debate transcripts",
    "Uncertainty schema",
    "Inline constraints",
  ],
  narrativeArchitect: [
    "Output-mode template",
    "Citation slots",
    "Delivery formatter",
  ],
  quantitativeSynthesizer: [
    "Effect-size aggregator",
    "Heterogeneity scan",
    "Evidence weights",
  ],
  evidencePyramidBuilder: [
    "Evidence-type classifier",
    "Tier weights",
    "Study design labels",
  ],
  evolvingNarrativeTracker: [
    "Temporal trend points",
    "Consensus shift map",
    "Year buckets",
  ],
  internalConsistencyCritic: [
    "Claim ledger",
    "Source-to-claim map",
    "Rewrite gate",
  ],
  claimVerifier: ["Claim ledger", "Evidence excerpts", "Support threshold"],
  citationVerifier: [
    "Citation map",
    "Excerpt alignment",
    "Source-strength check",
  ],
  contradictionVerifier: [
    "Counterevidence map",
    "Stance clusters",
    "False-consensus check",
  ],
  uncertaintyVerifier: [
    "Confidence scores",
    "Residual uncertainty",
    "Calibration rules",
  ],
  taskFocusVerifier: [
    "Primary question",
    "Anchor terms",
    "Scope contract",
    "Off-topic drift check",
  ],
  coverageAuditor: ["Hypothesis list", "Coverage matrix", "Gap report"],
  userGoalAlignmentCritic: [
    "Output-mode target",
    "Depth preference",
    "Steering controls",
  ],
  decisionIntelligenceLayer: [
    "Decision payload",
    "Risk profile",
    "Reversibility frame",
  ],
  adaptiveDeliveryHub: [
    "Checkpoint stream",
    "Markdown export",
    "Slide outline",
    "Dataset export",
  ],
  activeSafetyAndEthics: [
    "Dual-use flags",
    "Funding conflict scan",
    "Predatory journal scan",
    "Manipulation scan",
  ],
  causalRiskAnalyzer: [
    "Technique-to-risk chain",
    "Misuse vector map",
    "Severity scores",
  ],
});

const attachSubagentEquipment = (specs) =>
  Object.freeze(
    Object.fromEntries(
      Object.entries(specs).map(([id, spec]) => [
        id,
        {
          ...spec,
          equipment: SUBAGENT_EQUIPMENT[id] || [],
        },
      ]),
    ),
  );

export const RESEARCH_SUBAGENT_SPECS = attachSubagentEquipment({
  cognitiveCommandLayer: {
    label: "Cognitive Command Layer",
    scope:
      "Intent decomposition, DAG compilation, Pareto steering, and session continuity.",
    phaseId: "cognitiveCommandLayer",
  },
  devilsAdvocateDecomposer: {
    label: "DevilsAdvocateDecomposer",
    scope: "Counter-hypothesis generation and disconfirming evidence lanes.",
    phaseId: "adversarialQueryForge",
  },
  domainDetector: {
    label: "DomainDetector",
    scope: "Domain taxonomy detection and ontology-aware heuristics.",
    phaseId: "adversarialQueryForge",
  },
  queryVersionController: {
    label: "QueryVersionController",
    scope: "Auditable query transformations with rollback-ready rationale.",
    phaseId: "adversarialQueryForge",
  },
  forwardCitationTracer: {
    label: "ForwardCitationTracer",
    scope: "Forward citation expansion and seed reinforcement.",
    phaseId: "intelligentCrawlerMesh",
  },
  scholarlySourceHarvester: {
    label: "ScholarlySourceHarvester",
    scope:
      "Google Scholar-style discovery, institutional repository, and university-domain harvesting via permitted search paths.",
    phaseId: "intelligentCrawlerMesh",
  },
  authorNetworkMapper: {
    label: "AuthorNetworkMapper",
    scope: "Co-authorship clustering and echo-chamber detection.",
    phaseId: "intelligentCrawlerMesh",
  },
  temporalTrendAnalyzer: {
    label: "TemporalTrendAnalyzer",
    scope: "Emerging-vs-saturated thread analysis and publication velocity.",
    phaseId: "intelligentCrawlerMesh",
  },
  temporalRelevanceDecay: {
    label: "TemporalRelevanceDecay",
    scope: "Field-specific recency weighting and relevance decay.",
    phaseId: "tieredEpistemicFilter",
  },
  retractedPaperGuard: {
    label: "RetractedPaperGuard",
    scope: "Retraction quarantine and audit logging.",
    phaseId: "tieredEpistemicFilter",
  },
  sampleSizeFilter: {
    label: "SampleSizeFilter",
    scope: "Field-sensitive sample-size confidence adjustment.",
    phaseId: "tieredEpistemicFilter",
  },
  statisticalClaimExtractor: {
    label: "StatisticalClaimExtractor",
    scope: "Effect sizes, intervals, p-values, and sample-size extraction.",
    phaseId: "deepComprehensionEngine",
  },
  codeRepoAnalyzer: {
    label: "CodeRepoAnalyzer",
    scope: "Repo completeness and reproducibility scoring.",
    phaseId: "deepComprehensionEngine",
  },
  supplementaryMaterialParser: {
    label: "SupplementaryMaterialParser",
    scope: "Supplementary file ingestion and methodology recovery.",
    phaseId: "deepComprehensionEngine",
  },
  conceptEntityLinker: {
    label: "ConceptEntityLinker",
    scope: "Knowledge-graph entity linking and concept alignment.",
    phaseId: "deepComprehensionEngine",
  },
  statisticalVerifier: {
    label: "StatisticalVerifier",
    scope: "Sandboxed statistical cross-checks and recomputation hooks.",
    phaseId: "deepComprehensionEngine",
  },
  thesisAgent: {
    label: "ThesisAgent",
    scope: "Constructs the strongest case for the dominant position.",
    phaseId: "dialecticalSynthesisEngine",
  },
  antithesisAgent: {
    label: "AntithesisAgent",
    scope: "Builds the strongest counter-case from disconfirming evidence.",
    phaseId: "dialecticalSynthesisEngine",
  },
  synthesisMediator: {
    label: "SynthesisMediator",
    scope: "Reconciles conflict with explicit uncertainty.",
    phaseId: "dialecticalSynthesisEngine",
  },
  narrativeArchitect: {
    label: "NarrativeArchitect",
    scope: "Compiles the final narrative in the detected output mode.",
    phaseId: "dialecticalSynthesisEngine",
  },
  quantitativeSynthesizer: {
    label: "QuantitativeSynthesizer",
    scope: "Lightweight meta-analysis, heterogeneity, and aggregate signals.",
    phaseId: "dialecticalSynthesisEngine",
  },
  evidencePyramidBuilder: {
    label: "EvidencePyramidBuilder",
    scope: "Evidence-type weighting and pyramid construction.",
    phaseId: "dialecticalSynthesisEngine",
  },
  evolvingNarrativeTracker: {
    label: "EvolvingNarrativeTracker",
    scope: "Consensus shift detection over time.",
    phaseId: "dialecticalSynthesisEngine",
  },
  internalConsistencyCritic: {
    label: "InternalConsistencyCritic",
    scope: "Claim-to-evidence consistency enforcement.",
    phaseId: "recursiveSelfImprovementLoop",
  },
  claimVerifier: {
    label: "ClaimVerifier",
    scope:
      "Claim-by-claim evidence support checks and hallucination filtering.",
    phaseId: "recursiveSelfImprovementLoop",
  },
  citationVerifier: {
    label: "CitationVerifier",
    scope: "Citation integrity, excerpt alignment, and source-strength checks.",
    phaseId: "recursiveSelfImprovementLoop",
  },
  contradictionVerifier: {
    label: "ContradictionVerifier",
    scope: "Counterevidence surfacing and false-consensus prevention.",
    phaseId: "recursiveSelfImprovementLoop",
  },
  uncertaintyVerifier: {
    label: "UncertaintyVerifier",
    scope: "Confidence calibration and residual-uncertainty enforcement.",
    phaseId: "recursiveSelfImprovementLoop",
  },
  taskFocusVerifier: {
    label: "TaskFocusVerifier",
    scope: "Primary-question alignment and drift prevention.",
    phaseId: "recursiveSelfImprovementLoop",
  },
  coverageAuditor: {
    label: "CoverageAuditor",
    scope: "Hypothesis coverage scoring and gap detection.",
    phaseId: "recursiveSelfImprovementLoop",
  },
  userGoalAlignmentCritic: {
    label: "UserGoalAlignmentCritic",
    scope: "Output depth and format alignment checks.",
    phaseId: "recursiveSelfImprovementLoop",
  },
  decisionIntelligenceLayer: {
    label: "Decision Intelligence Layer",
    scope:
      "Decision payload generation, risk shaping, and reversibility framing.",
    phaseId: "decisionIntelligenceLayer",
  },
  adaptiveDeliveryHub: {
    label: "Adaptive Delivery Hub",
    scope: "Streaming checkpoints, format packaging, and export lanes.",
    phaseId: "adaptiveDeliveryHub",
  },
  activeSafetyAndEthics: {
    label: "Active Safety & Ethics",
    scope:
      "Cross-cutting safety, funding, predatory journal, and dual-use checks.",
    phaseId: "activeSafetyAndEthics",
  },
  causalRiskAnalyzer: {
    label: "CausalRiskAnalyzer",
    scope: "Technique-to-risk chain modeling and propagation scoring.",
    phaseId: "activeSafetyAndEthics",
  },
});

export const SUBAGENT_STAGE_LABELS = Object.freeze(
  Object.entries(RESEARCH_SUBAGENT_SPECS).reduce((acc, [id, spec]) => {
    const phase = PIPELINE_PHASES[spec.phaseId];
    acc[id] = phase?.label || spec.label;
    return acc;
  }, {}),
);

export const PLANNING_STEPS = Object.freeze([
  [
    "Decomposing",
    "domain, scope, and output ambiguity into independent confidence axes",
  ],
  [
    "Compiling",
    "a query-specific research DAG with safety and refinement hooks",
  ],
  [
    "Forging",
    "counter-hypotheses, ontology cues, and auditable query revisions",
  ],
  [
    "Allocating",
    "dialectical synthesis, verifier swarm, tribunal critics, and adaptive delivery owners",
  ],
]);

const clamp01 = (value) => Math.max(0, Math.min(1, Number(value) || 0));

const normalizeText = (value) =>
  String(value || "")
    .replace(/\s+/g, " ")
    .trim();

const normalizeVariant = (value) => normalizeText(value);

const uniqueList = (values = []) => [
  ...new Set(
    (Array.isArray(values) ? values : [values])
      .map(normalizeVariant)
      .filter(Boolean),
  ),
];

const pluralize = (count, singular, plural = `${singular}s`) =>
  `${count} ${count === 1 ? singular : plural}`;

export const createSubagentDescriptor = (id, options = {}) => {
  const spec = RESEARCH_SUBAGENT_SPECS[id];
  if (!spec) return null;

  return {
    id,
    label: spec.label,
    scope: spec.scope,
    phaseId: spec.phaseId,
    count: Math.max(1, Number(options.count) || 1),
    detail: options.detail ? String(options.detail) : "",
    equipment: uniqueList([
      ...(Array.isArray(spec.equipment) ? spec.equipment : []),
      ...(Array.isArray(options.equipment) ? options.equipment : []),
    ]).slice(0, 6),
  };
};

export const countSubagentAssignments = (items = []) =>
  (Array.isArray(items) ? items : []).reduce(
    (sum, item) => sum + Math.max(1, Number(item?.count) || 1),
    0,
  );

export const formatSubagentDescriptor = (descriptor) => {
  if (!descriptor?.label) return "";
  const countPrefix = descriptor.count > 1 ? `${descriptor.count}x ` : "";
  return `${countPrefix}${descriptor.label}${descriptor.detail ? ` (${descriptor.detail})` : ""}`;
};

export const summarizeSubagents = (items = []) =>
  (Array.isArray(items) ? items : [])
    .map((item) => formatSubagentDescriptor(item))
    .filter(Boolean)
    .join("; ");

export const formatSubagentStatus = (id, detail) => {
  const label = RESEARCH_SUBAGENT_SPECS[id]?.label;
  return label ? `${label}: ${detail}` : detail;
};

export const detectResearchDomain = (query) => {
  const normalized = normalizeText(query);
  if (!normalized) {
    return {
      ...DOMAIN_FALLBACK,
      confidence: 0.24,
    };
  }

  const match = DOMAIN_RULES.find((rule) => rule.patterns.test(normalized));
  if (match) {
    return {
      ...match,
      confidence: 0.82,
    };
  }

  return {
    ...DOMAIN_FALLBACK,
    confidence: 0.44,
  };
};

export const detectResearchOutputMode = (query, forcedOutputMode = "") => {
  const forced = normalizeText(forcedOutputMode).toLowerCase();
  if (forced) {
    const forcedMatch = OUTPUT_MODE_RULES.find(
      (rule) => rule.id === forced || rule.label.toLowerCase() === forced,
    );
    if (forcedMatch) {
      return {
        ...forcedMatch,
        confidence: 1,
      };
    }
  }

  const normalized = normalizeText(query);
  if (!normalized) {
    return {
      ...OUTPUT_MODE_FALLBACK,
      confidence: 0.32,
    };
  }

  const match = OUTPUT_MODE_RULES.find((rule) =>
    rule.patterns.test(normalized),
  );
  if (match) {
    return {
      ...match,
      confidence: 0.8,
    };
  }

  return {
    ...OUTPUT_MODE_FALLBACK,
    confidence: 0.52,
  };
};

export const detectResearchScope = (query) => {
  const normalized = normalizeText(query);
  if (!normalized) {
    return {
      ...SCOPE_FALLBACK,
      confidence: 0.26,
    };
  }

  const match = SCOPE_RULES.find((rule) => rule.patterns.test(normalized));
  if (match) {
    return {
      ...match,
      confidence: 0.78,
    };
  }

  const signals = detectQuerySignals(normalized);
  if (signals.comparisonIntent || signals.currentIntent) {
    return {
      id: "field_scan",
      label: "Field Scan",
      confidence: 0.62,
    };
  }

  return {
    ...SCOPE_FALLBACK,
    confidence: 0.48,
  };
};

export const detectIntentConfidence = (query, forcedOutputMode = "") => {
  const domain = detectResearchDomain(query);
  const scope = detectResearchScope(query);
  const outputMode = detectResearchOutputMode(query, forcedOutputMode);

  const axes = {
    domain: {
      value: domain.label,
      confidence: clamp01(domain.confidence),
    },
    scope: {
      value: scope.label,
      confidence: clamp01(scope.confidence),
    },
    outputFormat: {
      value: outputMode.label,
      confidence: clamp01(outputMode.confidence),
    },
  };

  const ambiguousAxes = Object.entries(axes)
    .filter(([, axis]) => axis.confidence < 0.65)
    .map(([axis]) => axis);

  return {
    axes,
    ambiguousAxes,
  };
};

const buildCounterHypotheses = (query, scope) => {
  const normalized = normalizeText(query);
  if (!normalized) return [];

  const hypotheses = [
    `Evidence against: ${normalized}`,
    `Conflicting findings for: ${normalized}`,
  ];

  if (scope.id === "replication_study") {
    hypotheses.push(`Failed replications for: ${normalized}`);
  }
  if (scope.id === "gap_finding") {
    hypotheses.push(`Unanswered questions in: ${normalized}`);
  }
  if (scope.id === "decision_support") {
    hypotheses.push(`Risks and tradeoffs of: ${normalized}`);
  }

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

  if (domain.id === "biomedical") {
    seeds.push(`${normalized} systematic review`);
  }
  if (domain.id === "cs_ml") {
    seeds.push(`${normalized} ablation study`);
  }

  return uniqueList(seeds).slice(0, 4);
};

const buildCitationSnowballSeeds = (query, scope) => {
  const normalized = normalizeText(query);
  if (!normalized) return [];

  const seeds = [`${normalized} seminal paper`, `${normalized} highly cited`];

  if (scope.id === "gap_finding") {
    seeds.push(`${normalized} future work`);
  }
  if (scope.id === "replication_study") {
    seeds.push(`${normalized} replication study`);
  }

  return uniqueList(seeds).slice(0, 4);
};

const shouldActivateScholarlyHarvest = (query, domain, scope, outputMode) => {
  const normalized = normalizeText(query);
  if (!normalized) return false;
  if (SCHOLARLY_SIGNAL_RE.test(normalized)) return true;
  if (domain?.id && domain.id !== DOMAIN_FALLBACK.id) return true;
  if (
    ["gap_finding", "replication_study", "contested_topic"].includes(scope?.id)
  )
    return true;
  if (
    [
      "state_of_the_field",
      "controversy_map",
      "gap_analysis",
      "replication_crisis_report",
      "foundational_review",
    ].includes(outputMode?.id)
  )
    return true;
  return false;
};

const buildScholarlyDiscoveryLanes = (query, domain, scope, outputMode) => {
  const normalized = normalizeText(query);
  if (
    !normalized ||
    !shouldActivateScholarlyHarvest(normalized, domain, scope, outputMode)
  ) {
    return [];
  }

  const lanes = [
    `${normalized} site:scholar.google.com`,
    `${normalized} site:semanticscholar.org`,
    `${normalized} site:openalex.org`,
    `${normalized} site:.edu`,
  ];

  if (domain?.id === "biomedical") {
    lanes.push(
      `${normalized} site:pubmed.ncbi.nlm.nih.gov`,
      `${normalized} site:nih.gov`,
    );
  }
  if (domain?.id === "cs_ml") {
    lanes.push(
      `${normalized} site:arxiv.org`,
      `${normalized} site:paperswithcode.com`,
    );
  }
  if (domain?.id === "social_science") {
    lanes.push(`${normalized} site:jstor.org`, `${normalized} site:ssrn.com`);
  }
  if (scope?.id === "replication_study") {
    lanes.push(`${normalized} site:osf.io`);
  }

  return uniqueList(lanes).slice(0, 5);
};

const buildScholarlyProviderBias = (domain, scope) => {
  const providers = ["openalex", "crossref"];
  if (domain?.id === "cs_ml") {
    providers.push("papers_with_code");
  }
  if (domain?.id === "social_science") {
    providers.push("jstor");
  }
  if (domain?.id === "biomedical" || scope?.id === "replication_study") {
    providers.push("ieee_xplore");
  }
  return uniqueList(providers).slice(0, 5);
};

export const buildQueryVersionLog = (
  query,
  domain,
  scope,
  outputMode,
  depthPreference = DEFAULT_DEPTH,
) => {
  const normalized = normalizeText(query);
  if (!normalized) return [];

  return [
    {
      version: "v0",
      query: normalized,
      rationale: "Original user intent.",
    },
    {
      version: "v1",
      query: `${normalized} ${domain.taxonomy}`.trim(),
      rationale: `DomainDetector applied ${domain.taxonomy} cues.`,
    },
    {
      version: "v2",
      query: `${normalized} ${outputMode.label}`.trim(),
      rationale: `Narrative target aligned to ${outputMode.label}.`,
    },
    {
      version: "v3",
      query: `${normalized} ${scope.label}`.trim(),
      rationale: `Scope-specific expansion for ${scope.label}.`,
    },
    {
      version: "v4",
      query:
        `${normalized} ${depthPreference === "speed" ? "high signal" : depthPreference === "deep" ? "deep evidence" : "balanced evidence"}`.trim(),
      rationale: `Cost-quality profile tuned for ${depthPreference}.`,
    },
  ].filter((entry) => entry.query);
};

export const buildResearchQueryMatrix = (query, options = {}) => {
  const normalized = normalizeText(query);
  const domain = options.domain || detectResearchDomain(normalized);
  const scope = options.scope || detectResearchScope(normalized);
  const outputMode = options.outputMode || detectResearchOutputMode(normalized);
  const depthPreference = options.depthPreference || DEFAULT_DEPTH;

  const keywords = uniqueList(
    buildSearchQueries(normalized, { maxQueries: 6 }),
  );
  const semanticEmbeddings = buildSemanticSeeds(normalized, domain);
  const citationSnowballSeeds = buildCitationSnowballSeeds(normalized, scope);
  const ontologyMappedVocabulary = uniqueList(domain.ontologyTerms).slice(0, 4);
  const counterHypotheses = buildCounterHypotheses(normalized, scope);
  const scholarlyDiscoveryLanes = buildScholarlyDiscoveryLanes(
    normalized,
    domain,
    scope,
    outputMode,
  );
  const scholarlyProviderBias = buildScholarlyProviderBias(domain, scope);
  const versions = buildQueryVersionLog(
    normalized,
    domain,
    scope,
    outputMode,
    depthPreference,
  );

  return {
    keywords,
    semanticEmbeddings,
    citationSnowballSeeds,
    ontologyMappedVocabulary,
    counterHypotheses,
    scholarlyDiscoveryLanes,
    scholarlyProviderBias,
    scholarlyHarvestActive: scholarlyDiscoveryLanes.length > 0,
    versions,
  };
};

export const buildDynamicSearchQueries = (plan, options = {}) => {
  const maxQueries = Math.max(1, Number(options.maxQueries) || 4);
  const matrix = plan?.queryMatrix || {};
  const scholarlyLanes = matrix.scholarlyDiscoveryLanes || [];
  const primary = uniqueList([
    ...(matrix.keywords || []).slice(0, scholarlyLanes.length ? 2 : 3),
    ...scholarlyLanes.slice(0, 2),
    ...(matrix.semanticEmbeddings || []).slice(0, 1),
    ...(matrix.citationSnowballSeeds || []).slice(0, 1),
    ...(matrix.counterHypotheses || []).slice(0, 1),
    ...(matrix.ontologyMappedVocabulary || [])
      .slice(0, 1)
      .map((term) => `${plan.query} ${term}`),
  ]);
  const fallback = uniqueList([
    ...(matrix.keywords || []),
    ...scholarlyLanes,
    ...(matrix.semanticEmbeddings || []),
    ...(matrix.counterHypotheses || []),
    ...(matrix.ontologyMappedVocabulary || []).map(
      (term) => `${plan.query} ${term}`,
    ),
    ...(matrix.citationSnowballSeeds || []),
  ]);
  return uniqueList([...primary, ...fallback]).slice(0, maxQueries);
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

export const inferSessionContinuity = (
  query,
  savedSessions = [],
  options = {},
) => {
  const normalized = normalizeText(query);
  const queryTerms = extractQueryTerms(normalized, 24);
  const excludedId = options.excludeId ? String(options.excludeId) : "";

  const ranked = (Array.isArray(savedSessions) ? savedSessions : [])
    .filter((session) => session?.id && session.id !== excludedId)
    .map((session) => {
      const body = normalizeText(
        [session.query, session.heading, session.body]
          .filter(Boolean)
          .join(" "),
      );
      return {
        session,
        overlap: computeOverlapScore(queryTerms, body),
      };
    })
    .filter((row) => row.overlap > 0)
    .sort((left, right) => right.overlap - left.overlap)
    .slice(0, 3);

  const best = ranked[0];
  if (!best || best.overlap < 0.25) {
    return {
      active: false,
      reason: "No strong prior-session overlap detected.",
      sourceSessionId: "",
      overlap: 0,
      summary: "",
    };
  }

  const session = best.session;
  const summary = normalizeText(
    `${session.heading || session.query || "Prior session"}\n${String(session.body || "").slice(0, 720)}`,
  );

  return {
    active: true,
    reason: "Prior research context was automatically linked into Phase 1.",
    sourceSessionId: session.id,
    overlap: Number(best.overlap.toFixed(2)),
    summary,
  };
};

const buildTaskFocusContract = (query, plan = {}) => {
  const normalizedQuery = normalizeText(query);
  const anchorTerms = extractQueryTerms(normalizedQuery, 10).slice(0, 8);
  const scopeLabel = plan.scope?.label || "Broad Research";
  const outputLabel = plan.outputMode?.label || "State-of-the-Field";
  const primaryQuestion =
    normalizedQuery || "Answer the current user request directly.";
  const summary = normalizedQuery
    ? `${scopeLabel} / ${outputLabel} run anchored to: ${primaryQuestion}`
    : `${scopeLabel} / ${outputLabel} run anchored to the current user request.`;

  return {
    summary,
    primaryQuestion,
    anchorTerms,
    overlapThreshold: anchorTerms.length >= 4 ? 0.32 : 0.22,
    scopeContract: `Stay inside ${scopeLabel} and ${outputLabel} mode while answering: ${primaryQuestion}`,
    requiredOutcomes: uniqueList([
      `Directly answer: ${primaryQuestion}`,
      `Keep every major section materially tied to ${primaryQuestion}`,
      scopeLabel === "Gap-finding"
        ? "Prioritize missing evidence and open questions over generic background."
        : "",
      scopeLabel === "Decision Support"
        ? "Keep the narrative centered on the actual decision being asked."
        : "",
      outputLabel === "Tutorial"
        ? "Only include tutorial steps that directly support the original question."
        : "",
    ]),
    forbiddenDriftBehaviors: uniqueList([
      "Do not switch to adjacent topics, technologies, or policy debates unless they directly answer the primary question.",
      "Do not pad the answer with generic background that does not change the conclusion.",
      "Do not introduce recommendations, implementation advice, or tutorials unless the selected output mode requires them.",
      "Do not treat unsupported assumptions, analogies, or side examples as part of the answer.",
    ]),
  };
};

const createSubagentTaskContract = (plan = {}, descriptor = {}) =>
  normalizeText(
    [
      plan?.taskFocus?.scopeContract || "",
      descriptor?.scope ? `Your assigned scope: ${descriptor.scope}.` : "",
      `Do not drift beyond the original question: ${normalizeText(plan?.taskFocus?.primaryQuestion || plan?.query || "current task")}.`,
    ].join(" "),
  );

const buildDag = (plan) => {
  const hasQuery = Boolean(plan.query);
  const hasAttachments = Number(plan.attachments || 0) > 0;
  const needsDecisionLayer = [
    "decision_brief",
    "policy_recommendation",
    "engineering_action_plan",
  ].includes(plan?.outputMode?.id);
  const nodes = Object.values(PIPELINE_PHASES)
    .filter((phase) => {
      if (
        phase.id === "intelligentCrawlerMesh" ||
        phase.id === "tieredEpistemicFilter"
      ) {
        return hasQuery;
      }
      if (phase.id === "deepComprehensionEngine") {
        return hasQuery || hasAttachments;
      }
      if (phase.id === "decisionIntelligenceLayer") {
        return needsDecisionLayer;
      }
      return true;
    })
    .map((phase) => {
      const priority =
        phase.id === "cognitiveCommandLayer"
          ? 1
          : phase.id === "activeSafetyAndEthics"
            ? 0.92
            : plan.scope.id === "replication_study" &&
                phase.id === "deepComprehensionEngine"
              ? 0.9
              : phase.id === "dialecticalSynthesisEngine"
                ? 0.88
                : 0.72;
      return {
        ...phase,
        priority,
      };
    });

  const activeIds = new Set(nodes.map((node) => node.id));
  const filteredNodes = nodes.map((node) => ({
    ...node,
    dependsOn: node.dependsOn.filter((dependency) => activeIds.has(dependency)),
  }));

  const edges = filteredNodes.flatMap((node) =>
    node.dependsOn.map((dependency) => ({
      from: dependency,
      to: node.id,
    })),
  );

  return {
    nodes: filteredNodes,
    edges,
  };
};

const buildParetoProfile = (depthPreference = DEFAULT_DEPTH) => {
  if (depthPreference === "speed") {
    return {
      mode: "speed",
      speed: 0.86,
      depth: 0.44,
      explanation:
        "Pareto front biased toward latency and rapid checkpoint delivery.",
    };
  }
  if (depthPreference === "deep") {
    return {
      mode: "deep",
      speed: 0.36,
      depth: 0.9,
      explanation:
        "Pareto front biased toward coverage, critique cycles, and extraction depth.",
    };
  }

  return {
    mode: "balanced",
    speed: 0.64,
    depth: 0.68,
    explanation:
      "Pareto front balanced for depth without overextending the graph.",
  };
};

const buildSafetyChecks = (query, domain) => {
  const normalized = normalizeText(query);
  const checks = [
    {
      id: "dual_use_flagging",
      label: "Dual-use flagging",
      active:
        /\b(biosecurity|surveillance|weapon|exploit|malware|pathogen)\b/i.test(
          normalized,
        ),
    },
    {
      id: "funding_conflict_detector",
      label: "Funding conflict detector",
      active:
        /(\b(drug|biomedical|policy|regulation|safety|surveillance)\b)/i.test(
          normalized,
        ),
    },
    {
      id: "predatory_journal_filter",
      label: "Predatory journal filter",
      active: true,
    },
    {
      id: "statistical_manipulation_detector",
      label: "Statistical manipulation detector",
      active:
        domain.id === "biomedical" ||
        domain.id === "social_science" ||
        domain.id === "cs_ml",
    },
  ];

  return {
    checks,
    activeCount: checks.filter((check) => check.active).length,
  };
};

const buildSubagents = (plan) => {
  const hasQuery = Boolean(plan.query);
  const attachmentCount = Number(plan.attachments || 0);
  const searchLaneCount = Math.max(1, plan.searchQueries.length || 1);
  const counterHypotheses = plan.queryMatrix.counterHypotheses.length;
  const criticCount = 3;
  const verifierCount = 5;
  const debateWorkers = plan.scope.id === "contested_topic" ? 2 : 1;

  const descriptors = [
    createSubagentDescriptor("cognitiveCommandLayer", {
      detail: `${plan.intentConfidence.ambiguousAxes.length || 0} low-confidence axis${plan.intentConfidence.ambiguousAxes.length === 1 ? "" : "es"}`,
    }),
    createSubagentDescriptor("domainDetector", {
      detail: `${plan.domain.label} / ${plan.domain.taxonomy}`,
    }),
    createSubagentDescriptor("queryVersionController", {
      detail: `${plan.queryMatrix.versions.length} logged revisions`,
    }),
    createSubagentDescriptor("adaptiveDeliveryHub", {
      detail: plan.outputMode.label,
    }),
    createSubagentDescriptor("activeSafetyAndEthics", {
      detail: `${plan.safety.activeCount} active safety check${plan.safety.activeCount === 1 ? "" : "s"}`,
    }),
    createSubagentDescriptor("causalRiskAnalyzer", {
      detail: plan.safety.activeCount
        ? "risk propagation watch"
        : "latent risk watch",
    }),
    createSubagentDescriptor("internalConsistencyCritic", {
      detail: `${criticCount} tribunal critic${criticCount === 1 ? "" : "s"}`,
    }),
    createSubagentDescriptor("claimVerifier", {
      detail: `${verifierCount} verifier lane${verifierCount === 1 ? "" : "s"} active`,
    }),
    createSubagentDescriptor("citationVerifier", {
      detail: "claim-to-citation integrity",
    }),
    createSubagentDescriptor("contradictionVerifier", {
      detail: "counterevidence surfacing",
    }),
    createSubagentDescriptor("uncertaintyVerifier", {
      detail: "confidence calibration",
    }),
    createSubagentDescriptor("taskFocusVerifier", {
      detail: `${Math.max(1, plan.taskFocus?.anchorTerms?.length || 0)} focus anchor${Math.max(1, plan.taskFocus?.anchorTerms?.length || 0) === 1 ? "" : "s"}`,
    }),
    createSubagentDescriptor("coverageAuditor", {
      detail: plan.scope.label,
    }),
    createSubagentDescriptor("userGoalAlignmentCritic", {
      detail: plan.outputMode.label,
    }),
    ...([
      "decision_brief",
      "policy_recommendation",
      "engineering_action_plan",
    ].includes(plan.outputMode.id)
      ? [
          createSubagentDescriptor("decisionIntelligenceLayer", {
            detail: plan.outputMode.label,
          }),
        ]
      : []),
  ];

  if (hasQuery) {
    descriptors.push(
      createSubagentDescriptor("devilsAdvocateDecomposer", {
        detail: `${counterHypotheses || 1} counter-hypothesis lane${counterHypotheses === 1 ? "" : "s"}`,
      }),
      ...(plan.scholarlyHarvest?.active
        ? [
            createSubagentDescriptor("scholarlySourceHarvester", {
              detail: `${plan.scholarlyHarvest.lanes.length} scholarly lane${plan.scholarlyHarvest.lanes.length === 1 ? "" : "s"}; academic index + .edu sweep`,
            }),
          ]
        : []),
      createSubagentDescriptor("forwardCitationTracer", {
        count: plan.depthPreference === "deep" ? 2 : 1,
        detail: `depth ${plan.depthPreference === "deep" ? 2 : 1} citation tracing`,
      }),
      createSubagentDescriptor("authorNetworkMapper", {
        detail: "co-authorship watch",
      }),
      createSubagentDescriptor("temporalTrendAnalyzer", {
        detail: `${plan.domain.recencyHalfLifeYears}-year recency horizon`,
      }),
      createSubagentDescriptor("temporalRelevanceDecay", {
        detail: `${plan.domain.recencyHalfLifeYears}-year half-life`,
      }),
      createSubagentDescriptor("retractedPaperGuard", {
        detail: "quarantine and audit log",
      }),
      createSubagentDescriptor("sampleSizeFilter", {
        detail: "confidence weighting",
      }),
      createSubagentDescriptor("thesisAgent", {
        count: debateWorkers,
        detail: "dominant-view argument",
      }),
      createSubagentDescriptor("antithesisAgent", {
        count: debateWorkers,
        detail: "counter-evidence argument",
      }),
      createSubagentDescriptor("synthesisMediator", {
        detail: "uncertainty-aware reconciliation",
      }),
      createSubagentDescriptor("narrativeArchitect", {
        detail: plan.outputMode.label,
      }),
      createSubagentDescriptor("quantitativeSynthesizer", {
        detail: "lightweight meta-analysis",
      }),
      createSubagentDescriptor("evidencePyramidBuilder", {
        detail: "evidence-type weighting",
      }),
      createSubagentDescriptor("evolvingNarrativeTracker", {
        detail: "consensus shift tracking",
      }),
    );
  }

  if (hasQuery || attachmentCount) {
    descriptors.push(
      createSubagentDescriptor("statisticalClaimExtractor", {
        detail: attachmentCount
          ? `${pluralize(attachmentCount, "attachment")} + fetched evidence`
          : "structured claim schema",
      }),
      createSubagentDescriptor("codeRepoAnalyzer", {
        detail: hasQuery
          ? "repo-linked paper checks"
          : "attachment code review",
      }),
      createSubagentDescriptor("supplementaryMaterialParser", {
        detail: "supplementary appendix sweep",
      }),
      createSubagentDescriptor("conceptEntityLinker", {
        detail: `${plan.domain.taxonomy} + knowledge graph`,
      }),
      createSubagentDescriptor("statisticalVerifier", {
        detail: "sandboxed recomputation hooks",
      }),
    );
  }

  return descriptors.filter(Boolean).map((descriptor) => ({
    ...descriptor,
    taskContract: createSubagentTaskContract(plan, descriptor),
  }));
};

export const compileResearchPlan = (options = {}) => {
  const query = normalizeText(options.query);
  const attachments = Math.max(0, Number(options.attachments) || 0);
  const depthPreference = ["speed", "balanced", "deep"].includes(
    options.depthPreference,
  )
    ? options.depthPreference
    : DEFAULT_DEPTH;
  const domain = detectResearchDomain(query);
  const scope = detectResearchScope(query);
  const outputMode = detectResearchOutputMode(query, options.forcedOutputMode);
  const intentConfidence = detectIntentConfidence(
    query,
    options.forcedOutputMode,
  );
  const queryMatrix = buildResearchQueryMatrix(query, {
    domain,
    scope,
    outputMode,
    depthPreference,
  });
  const continuity = inferSessionContinuity(
    query,
    options.savedSessions || [],
    {
      excludeId: options.currentSessionId,
    },
  );
  const pareto = buildParetoProfile(depthPreference);
  const safety = buildSafetyChecks(query, domain);
  const scholarlyHarvest = {
    active: Boolean(
      queryMatrix.scholarlyHarvestActive ||
      (queryMatrix.scholarlyDiscoveryLanes || []).length,
    ),
    lanes: queryMatrix.scholarlyDiscoveryLanes || [],
    providerBias: queryMatrix.scholarlyProviderBias || [],
  };
  const basePlan = {
    frameworkVersion: RESEARCH_FRAMEWORK_VERSION,
    query,
    attachments,
    depthPreference,
    domain,
    scope,
    outputMode,
    intentConfidence,
    queryMatrix,
    scholarlyHarvest,
    continuity,
    pareto,
    safety,
  };
  const taskFocus = buildTaskFocusContract(query, basePlan);

  const searchQueries = buildDynamicSearchQueries(basePlan, {
    maxQueries: options.maxQueries || 6,
  });
  const dag = buildDag({
    ...basePlan,
    searchQueries,
  });
  const plan = {
    ...basePlan,
    taskFocus,
    dag,
    searchQueries,
    refinementBudget: Math.max(
      1,
      Number(options.refinementBudget) || REFINEMENT_BUDGET,
    ),
  };

  return {
    ...plan,
    subagents: buildSubagents(plan),
  };
};

export const summarizeDag = (dag = {}) =>
  (Array.isArray(dag.nodes) ? dag.nodes : [])
    .map((node) => node.label)
    .join(" -> ");

export const buildConvergenceMetrics = ({
  iterations = 1,
  refinementBudget = REFINEMENT_BUDGET,
  coverageScore = 0.7,
  contradictionScore = 0.22,
  verificationScore = null,
  stabilityScore = null,
} = {}) => {
  const resolvedIterations = Math.max(1, Number(iterations) || 1);
  const resolvedCoverage = clamp01(coverageScore);
  const resolvedContradiction = clamp01(contradictionScore);
  const resolvedVerification =
    verificationScore == null ? null : clamp01(verificationScore);
  const resolvedStability =
    stabilityScore == null
      ? clamp01(
          0.58 +
            resolvedIterations * 0.1 +
            resolvedCoverage * 0.16 -
            resolvedContradiction * 0.2 +
            (resolvedVerification == null ? 0 : resolvedVerification * 0.16),
        )
      : clamp01(stabilityScore);
  const evidenceCoverageDelta = Number((1 - resolvedCoverage).toFixed(2));
  const residualUncertainty = Number(
    Math.max(
      0.08,
      1 - resolvedStability + resolvedContradiction * 0.32,
    ).toFixed(2),
  );
  const exhaustedBudget =
    resolvedIterations >=
    Math.max(1, Number(refinementBudget) || REFINEMENT_BUDGET);

  return {
    iterations: resolvedIterations,
    stability_score: Number(resolvedStability.toFixed(2)),
    evidence_coverage_delta: evidenceCoverageDelta,
    residual_uncertainty: residualUncertainty,
    stop_condition: exhaustedBudget
      ? "refinement_budget_exhausted"
      : resolvedStability < 0.95 && evidenceCoverageDelta > 0.03
        ? "residual_disagreement"
        : "stability_reached",
  };
};

export const buildTribunalSummary = ({
  coverageScore = 0.7,
  contradictionScore = 0.2,
  alignmentScore = 0.75,
  verifiers = {},
  iterations = 1,
  refinementBudget = REFINEMENT_BUDGET,
} = {}) => {
  const resolvedCoverage = clamp01(coverageScore);
  const resolvedContradiction = clamp01(contradictionScore);
  const resolvedAlignment = clamp01(alignmentScore);
  const normalizedVerifiers = {
    claim_support: clamp01(verifiers.claim_support),
    citation_integrity: clamp01(verifiers.citation_integrity),
    contradiction_handling: clamp01(verifiers.contradiction_handling),
    uncertainty_calibration: clamp01(verifiers.uncertainty_calibration),
    task_focus: clamp01(verifiers.task_focus),
  };
  const dimensionScores = [
    ["coverage", resolvedCoverage],
    ["internal_consistency", 1 - resolvedContradiction],
    ["user_goal_alignment", resolvedAlignment],
    ["claim_support", normalizedVerifiers.claim_support ?? 1],
    ["citation_integrity", normalizedVerifiers.citation_integrity ?? 1],
    ["contradiction_handling", normalizedVerifiers.contradiction_handling ?? 1],
    [
      "uncertainty_calibration",
      normalizedVerifiers.uncertainty_calibration ?? 1,
    ],
    ["task_focus", normalizedVerifiers.task_focus ?? 1],
  ];
  const [lowestDimension] = dimensionScores.reduce((lowest, current) =>
    current[1] < lowest[1] ? current : lowest,
  );

  return {
    refinement_budget: Math.max(
      1,
      Number(refinementBudget) || REFINEMENT_BUDGET,
    ),
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
