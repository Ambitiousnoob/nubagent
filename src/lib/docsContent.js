export const API_REFERENCE_SECTIONS = [
  {
    id: "chat-api",
    title: "Chat Runtime",
    paths: ["/api/chat"],
    methods: ["GET", "POST"],
    summary:
      "Primary answer-generation endpoint. Accepts plain chat messages and forwards them to Gemini Flash.",
    keyPoints: [
      "Use GET for capability metadata and POST for completions.",
      "This route is intentionally non-streaming and non-agentic.",
      "Use it when you want a narrow JSON chat contract backed by Gemini.",
    ],
    requestShape: `{
  "system": "Be concise.",
  "messages": [{ "role": "user", "content": "Say hello." }]
}`,
    responseShape: `{
  "ok": true,
  "model": "gemini-3-flash-preview",
  "output_text": "...",
  "usage": { "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0 }
}`,
    implementationFiles: ["api/chat.js", "lib/gemini-chat.js"],
  },
];

export const API_CREATION_STEPS = [
  {
    title: "Pick the boundary first",
    body: "Treat `api/chat.js` as the single public API contract and keep new behavior behind that boundary unless you intentionally want to re-expand the surface.",
    bullets: [
      "Prefer extending `api/chat.js` over adding a new top-level endpoint.",
      "Keep provider-specific Gemini wiring in `lib/` so the public contract stays narrow.",
      "If you later add another endpoint, update both the docs and `vercel.json` in the same change.",
    ],
  },
  {
    title: "Implement both metadata and action paths",
    body: "Every public endpoint should answer GET with clear metadata and POST with the real behavior.",
    bullets: [
      "GET should describe the endpoint, available methods, and high-level capabilities.",
      "POST should validate the body, return explicit error codes, and normalize output.",
      "Shared alias endpoints should resolve the alias before branching into behavior-specific logic.",
    ],
  },
  {
    title: "Wire the route in Vercel",
    body: "Keep the single public route aligned in `vercel.json`.",
    bullets: [
      "Only `/api/chat` should rewrite into the serverless backend.",
      "Do not leave dead rewrites behind when an endpoint is removed.",
      "Keep the SPA fallback last so the docs site still serves correctly.",
    ],
  },
  {
    title: "Document and test the contract",
    body: "Add the contract to the docs and verify at least one happy path and one safe failure path.",
    bullets: [
      "Update `API.md` with request and response examples.",
      "Update `README.md` or web-facing docs when the endpoint is part of the product story.",
      "Add targeted tests for routing helpers, alias resolution, or payload handling where regressions are likely.",
    ],
  },
];

export const API_CREATION_CHECKLIST = [
  "Confirm the change belongs inside `/api/chat`.",
  "Add GET metadata for discoverability.",
  "Add POST validation and stable JSON or text responses.",
  "Keep `/api/chat` as the only serverless rewrite in `vercel.json`.",
  "Update `API.md` and any web-facing docs.",
  "Add targeted verification commands before shipping.",
];

export const BASIC_ENDPOINT_EXAMPLE = `const writeJson = (res, status, data) => {
  res.status(status);
  res.setHeader("Content-Type", "application/json; charset=utf-8");
  res.end(JSON.stringify(data));
};

module.exports = async (req, res) => {
  if (req.method === "GET") {
    return writeJson(res, 200, {
      ok: true,
      endpoint: "/api/example",
      methods: ["GET", "POST"],
    });
  }

  if (req.method !== "POST") {
    return writeJson(res, 405, { error: "Method not allowed" });
  }

  const body = await readBody(req);
  return writeJson(res, 200, { ok: true, echo: body });
};`;

export const ALIAS_ENDPOINT_EXAMPLE = `// vercel.json
{
  "source": "/api/chat",
  "destination": "/api/chat.js"
}

// Keep helper logic in lib/, but preserve one public route.
module.exports = async (req, res) => {
  if (req.method === "GET") {
    return writeJson(res, 200, { ok: true, endpoint: "/api/chat" });
  }

  return runChatRuntime(req, res);
};`;

export const RESEARCH_FRAMEWORK_V3_GAPS = [
  {
    gap: "Agents are stateless between handoffs",
    consequence: "Context loss mid-pipeline.",
  },
  {
    gap: "Synthesis is still one-pass even with debate",
    consequence: "Weak convergence on contested topics.",
  },
  {
    gap: "No code/data execution",
    consequence: "Can't verify quantitative claims.",
  },
  {
    gap: "Memory store is passive (queried, not proactive)",
    consequence: "Missed cross-session insights.",
  },
  {
    gap: "Critics score but do not intervene mid-synthesis",
    consequence: "Garbage-in, garbage-out at synthesis time.",
  },
  {
    gap: "Single narrative output mode",
    consequence: "Poor fit for diverse user goals.",
  },
];

export const RESEARCH_FRAMEWORK_V3_PHASES = [
  {
    id: "cognitive-command-layer",
    phase: "0",
    title: "Cognitive Command Layer (CCL)",
    summary:
      "Meta-orchestrator that compiles a dynamic DAG and resolves ambiguity by domain, scope, and output format.",
    capabilities: [
      "Intent Confidence Decomposition across domain/scope/output axes.",
      "Dynamic Pipeline Compiler with query-type specific graph pruning.",
      "Cost-Quality Pareto Front for live speed-vs-depth steering.",
      "Session Continuity Manager that injects prior run context into Phase 1.",
    ],
  },
  {
    id: "adversarial-query-forge",
    phase: "1",
    title: "Adversarial Query Forge",
    summary:
      "Query intelligence layer that builds disconfirming lanes and auditable query revisions.",
    capabilities: [
      "DevilsAdvocateDecomposer for counter-hypothesis generation.",
      "DomainDetector for ontology-aware strategy swaps (MeSH, ACM CCS, JEL).",
      "QueryVersionController with rollback-ready transformation logs.",
      "4D strategy matrix: keywords x semantic embeddings x citation seeds x ontology vocabulary.",
    ],
  },
  {
    id: "intelligent-crawler-mesh",
    phase: "2",
    title: "Intelligent Crawler Mesh",
    summary:
      "Adaptive source mesh with proactive citation intelligence and trend scanning.",
    capabilities: [
      "Expanded source set: IEEE/ACM/JSTOR, Unpaywall, GitHub, Papers With Code, Retraction Watch.",
      "ScholarlySourceHarvester for Google Scholar-style discovery, Semantic Scholar, OpenAlex, and university-domain sweeps.",
      "ForwardCitationTracer with depth-aware citation expansion.",
      "AuthorNetworkMapper for co-authorship and echo-chamber analysis.",
      "TemporalTrendAnalyzer for emerging vs. saturated topic velocity.",
    ],
  },
  {
    id: "tiered-epistemic-filter",
    phase: "3",
    title: "Tiered Epistemic Filter",
    summary:
      "Four-tier evidence gate with explicit auditability and confidence shaping.",
    capabilities: [
      "Tier verdicts: Core, Supporting, Peripheral, Discard.",
      "TemporalRelevanceDecay tuned per domain half-life.",
      "RetractedPaperGuard quarantine with retraction rationale logging.",
      "SampleSizeFilter for confidence weighting instead of blind discard.",
    ],
  },
  {
    id: "deep-comprehension-engine",
    phase: "4",
    title: "Deep Comprehension Engine",
    summary:
      "Multi-modal extraction swarm with statistical and code-linked verification hooks.",
    capabilities: [
      "StatisticalClaimExtractor into structured claim schema.",
      "CodeRepoAnalyzer for reproducibility scorecards.",
      "SupplementaryMaterialParser for appendix-level method recovery.",
      "ConceptEntityLinker + StatisticalVerifier for semantic and quantitative checks.",
    ],
  },
  {
    id: "dialectical-synthesis-engine",
    phase: "5",
    title: "Dialectical Synthesis Engine",
    summary:
      "Three-stage synthesis that argues with itself before producing narrative output.",
    capabilities: [
      "Stage A: Position Mapping over key debate axes.",
      "Stage B: ThesisAgent vs AntithesisAgent with SynthesisMediator reconciliation.",
      "Stage C: NarrativeArchitect mode selection (tutorial, controversy map, gap analysis, etc).",
      "QuantitativeSynthesizer, EvidencePyramidBuilder, and EvolvingNarrativeTracker integration.",
    ],
  },
  {
    id: "recursive-self-improvement-loop",
    phase: "6",
    title: "Recursive Self-Improvement Loop",
    summary:
      "Quality tribunal plus verifier swarm with targeted refinement cycles and persistent postmortem learning.",
    capabilities: [
      "InternalConsistencyCritic, CoverageAuditor, and UserGoalAlignmentCritic.",
      "ClaimVerifier, CitationVerifier, ContradictionVerifier, and UncertaintyVerifier as the anti-hallucination gate.",
      "Refinement budget with dimension-targeted reruns (default: 3 cycles).",
      "RunPostmortem storage in ResearchMemoryStore for future pre-patching.",
      "Phase-level score deltas for measurable quality gains across sessions.",
    ],
  },
  {
    id: "adaptive-delivery-hub",
    phase: "7",
    title: "Adaptive Delivery Hub",
    summary:
      "Delivery layer that streams checkpoints and supports downstream research workflows.",
    capabilities: [
      "Streaming checkpoints by phase (hypotheses, inventory, summaries, drafts, final).",
      "Output modes: Obsidian/Notion markdown, slide deck outline, dataset export, research API.",
      "Live user steering at checkpoints with graph recompilation.",
      "Flexible output mode mapping to explicit user goals.",
    ],
  },
  {
    id: "active-safety-and-ethics",
    phase: "X",
    title: "Active Safety & Ethics (Cross-Cutting)",
    summary:
      "Always-on safety module across all phases, not a single checkpoint.",
    capabilities: [
      "Dual-use flagging for biosecurity/surveillance/weapons content.",
      "FundingConflictDetector for contested-topic confidence adjustment.",
      "PredatoryJournalFilter with quarantine behavior.",
      "StatisticalManipulationDetector for p-hacking/HARKing risk signals.",
    ],
  },
];

export const RESEARCH_FRAMEWORK_V31_STRESS_POINTS = [
  {
    id: "execution-semantics",
    title: "DAG compilation without execution semantics",
    impact: "Stalls, race conditions, and inconsistent intermediate state.",
  },
  {
    id: "convergence-guarantees",
    title: "Dialectical engine lacks convergence guarantees",
    impact: "Debate quality can increase while answers fail to stabilize.",
  },
  {
    id: "flat-memory",
    title: "Memory remains structurally flat",
    impact: "Retrieval precision degrades as run volume grows.",
  },
  {
    id: "quant-verification",
    title: "Statistical verification is underpowered",
    impact: "Isolated checks pass while cross-paper quantitative truth drifts.",
  },
  {
    id: "post-hoc-critics",
    title: "Critics still run mostly post-hoc",
    impact: "Weak reasoning propagates before corrections land.",
  },
  {
    id: "uncertainty-schema",
    title: "No formal uncertainty representation",
    impact: "Uncertainty is narrative instead of computable.",
  },
  {
    id: "decision-mode",
    title: "Output modes are not decision modes",
    impact: "Strong reports, weak decision support.",
  },
];

export const RESEARCH_FRAMEWORK_V31_STABILIZATION = [
  {
    id: "ddr-runtime",
    title: "Deterministic DAG Runtime (DDR)",
    summary:
      "DependencyResolver + AsyncScheduler + CheckpointManager + FailureRecovery with typed failure classes.",
  },
  {
    id: "convergence-engine",
    title: "Convergence Engine",
    summary:
      "ASS, ECD, and URR metrics with formal stop conditions and residual uncertainty output.",
  },
  {
    id: "hierarchical-memory",
    title: "Hierarchical Knowledge Graph (HKG)",
    summary:
      "Episode, concept, and abstraction layers for cross-run generalization.",
  },
  {
    id: "cross-paper-quant",
    title: "Cross-Paper Statistical Engine",
    summary:
      "EffectSizeNormalizer, HeterogeneityAnalyzer, and AssumptionValidator for field-level recomputation.",
  },
  {
    id: "inline-constraints",
    title: "Inline Constraint System (ICS)",
    summary:
      "Critics emit enforceable constraints during synthesis, not after it.",
  },
  {
    id: "probabilistic-epistemic",
    title: "Probabilistic Epistemic Layer (PEL)",
    summary:
      "Structured uncertainty schema with propagation rules and contradiction penalties.",
  },
  {
    id: "decision-intelligence",
    title: "Decision Intelligence Layer (DIL)",
    summary:
      "Decision briefs, policy recommendations, and engineering action plans with risk profiles.",
  },
  {
    id: "human-loop-hooks",
    title: "Human-in-the-Loop Control Hooks",
    summary:
      "Formal runtime controls for depth changes, source exclusions, and mode forcing.",
  },
  {
    id: "causal-risk-analyzer",
    title: "Causal Risk Analyzer",
    summary: "Technique-to-misuse causal chains with risk propagation scoring.",
  },
];

export const RESEARCH_FRAMEWORK_V3_CAPABILITY_SUMMARY = [
  {
    dimension: "Pipeline structure",
    v2: "Fixed linear + one loop",
    v3: "Compiled DAG, fully dynamic",
  },
  {
    dimension: "Synthesis paradigm",
    v2: "Additive summarization",
    v3: "Dialectical with uncertainty quantification",
  },
  {
    dimension: "Evidence handling",
    v2: "Qualitative",
    v3: "Qualitative + lightweight meta-analysis",
  },
  {
    dimension: "Self-improvement",
    v2: "Post-run prompt meta-learning",
    v3: "Per-cycle targeted refinement + postmortem store",
  },
  {
    dimension: "Output modes",
    v2: "5 formats",
    v3: "8 formats + streaming + REST API",
  },
  {
    dimension: "Safety handling",
    v2: "Phase 3 bias guardrail",
    v3: "Cross-cutting safety and ethics layer",
  },
  {
    dimension: "Code verification",
    v2: "Passive reproducibility scoring",
    v3: "Active repo and statistical verification",
  },
  {
    dimension: "Query resilience",
    v2: "Scope expansion on zero results",
    v3: "Adversarial decomposition + counter-hypothesis tracking",
  },
  {
    dimension: "User interaction",
    v2: "Optional checkpoints",
    v3: "Live steering at every phase boundary",
  },
];

export const SUBAGENT_GROUPS = [
  {
    id: "orchestration",
    title: "Orchestration And Release",
    summary:
      "These agents decide scope, sequencing, verification gates, and release readiness across the whole product.",
    agents: [
      {
        name: "nub_scope_chief",
        scope:
          "Defines the smallest coherent work slice and scope boundary before changes begin.",
      },
      {
        name: "nub_product_guardian",
        scope:
          "Keeps changes aligned with the product goal of being a trustworthy information finder.",
      },
      {
        name: "nub_program_manager",
        scope: "Sequences multi-owner work and keeps dependencies crisp.",
      },
      {
        name: "nub_system_architect",
        scope:
          "Confirms placement and boundaries when a change crosses surfaces.",
      },
      {
        name: "nub_release_ops",
        scope:
          "Owns deploy readiness, environment assumptions, and rollout risk.",
      },
    ],
  },
  {
    id: "research",
    title: "Research And Verification",
    summary:
      "These agents help NubAgent choose, verify, and triage information before it becomes answer content.",
    agents: [
      {
        name: "nub_competitive_analyst",
        scope:
          "Compares external options such as models, providers, libraries, or products.",
      },
      {
        name: "nub_data_researcher",
        scope:
          "Handles quantitative evidence, metrics, datasets, and evidence-backed measurement questions.",
      },
      {
        name: "nub_docs_researcher",
        scope:
          "Verifies external API or framework behavior from primary documentation.",
      },
      {
        name: "nub_research_analyst",
        scope:
          "Owns broader technical investigations when no narrower decision-support agent fits.",
      },
      {
        name: "nub_search_specialist",
        scope:
          "Finds the highest-signal files or external references before deeper analysis starts.",
      },
    ],
  },
  {
    id: "verifiers",
    title: "The Verifiers",
    summary:
      "These agents form a multi-pass hallucination gate so externally grounded answers are checked for support, citation integrity, contradiction, and calibration before finalization.",
    agents: [
      {
        name: "nub_answer_verification_orchestrator",
        scope:
          "Coordinates the verifier swarm and blocks answer finalization until the required verification lanes are complete.",
      },
      {
        name: "nub_fetched_info_verifier",
        scope:
          "Checks whether fetched pages, excerpts, and evidence blocks actually support the facts being claimed.",
      },
      {
        name: "nub_claim_verifier",
        scope:
          "Verifies that each material factual claim in a draft has explicit evidentiary support.",
      },
      {
        name: "nub_citation_verifier",
        scope:
          "Checks citation integrity, excerpt-to-claim alignment, and weak-source leakage into high-confidence narrative.",
      },
      {
        name: "nub_contradiction_verifier",
        scope:
          "Surfaces omitted counterevidence, disagreement, and false-consensus language before final answer release.",
      },
      {
        name: "nub_uncertainty_verifier",
        scope:
          "Calibrates confidence, caveats, and residual uncertainty so the final answer does not overclaim.",
      },
    ],
  },
  {
    id: "search-surface",
    title: "Search Product Surface",
    summary:
      "These owners cover the live search-first UX from intake through answer rendering and saved research sessions.",
    agents: [
      {
        name: "nub_search_intake_owner",
        scope:
          "Owns query entry, upload validation, and input-side SearchEngine behavior.",
      },
      {
        name: "nub_attachment_ingest_owner",
        scope:
          "Handles attachment parsing, upload digestion, and ingestion edge cases.",
      },
      {
        name: "nub_research_pipeline_owner",
        scope:
          "Owns planning, fan-out, ranking, fetch orchestration, evidence shaping, and synthesis flow.",
      },
      {
        name: "nub_answer_presentation_owner",
        scope: "Owns result cards, citations, tables, and answer rendering.",
      },
      {
        name: "nub_library_owner",
        scope:
          "Owns saved sessions, export and history behavior, and the library surface.",
      },
      {
        name: "nub_state_owner",
        scope:
          "Owns persisted UI state, namespaces, and remote or local state coordination.",
      },
    ],
  },
  {
    id: "shell",
    title: "Shell And Settings",
    summary:
      "These owners cover the application shell and shared interactive surfaces around the core research flow.",
    agents: [
      {
        name: "nub_app_shell_owner",
        scope:
          "Owns App layout, top-level navigation, view switching, and shell behavior.",
      },
      {
        name: "nub_settings_owner",
        scope:
          "Owns settings UI, provider preferences, and settings validation.",
      },
      {
        name: "nub_shared_ui_owner",
        scope:
          "Owns base UI primitives, theme providers, modal infrastructure, and shared styling tokens.",
      },
    ],
  },
  {
    id: "backend",
    title: "Backends And Tools",
    summary:
      "These owners cover public API boundaries, route contracts, provider logic, retrieval behavior, and persistence.",
    agents: [
      {
        name: "nub_chat_backend_owner",
        scope:
          "Owns `/api/chat`, model orchestration, and chat backend contracts.",
      },
      {
        name: "nub_search_backend_owner",
        scope:
          "Owns search-provider routing and source discovery logic used internally by the chat runtime.",
      },
      {
        name: "nub_content_backend_owner",
        scope:
          "Owns fetch, read, and crawl behavior used internally by the chat runtime.",
      },
      {
        name: "nub_search_tool_owner",
        scope:
          "Owns search tool implementation details and provider-specific logic.",
      },
      {
        name: "nub_fetch_tool_owner",
        scope: "Owns fetch and read tooling plus page extraction behavior.",
      },
      {
        name: "nub_image_tool_owner",
        scope:
          "Owns image search, image view, and image-related tool behavior.",
      },
      {
        name: "nub_memory_owner",
        scope:
          "Owns memory retrieval, namespace isolation, and persistence behind the chat boundary.",
      },
      {
        name: "nub_api_boundary_owner",
        scope:
          "Owns cross-endpoint request and response normalization plus boundary consistency.",
      },
    ],
  },
  {
    id: "quality",
    title: "Quality And Docs",
    summary:
      "These owners validate the result, keep repo docs aligned, and protect operator trust.",
    agents: [
      {
        name: "nub_test_engineer",
        scope:
          "Owns regression coverage, verification commands, and test-focused validation.",
      },
      {
        name: "nub_code_reviewer",
        scope:
          "Owns correctness, regression, maintainability, and evidence-path risk review.",
      },
      {
        name: "nub_docs_owner",
        scope:
          "Owns README, API docs, architecture notes, and operator-facing guidance.",
      },
      {
        name: "nub_docs_sync",
        scope: "Owns keeping docs aligned after behavior or contract changes.",
      },
    ],
  },
];
