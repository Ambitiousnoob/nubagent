const crypto = require("node:crypto");
const { getPool } = require("./db");

const RESEARCH_RUNS_TABLE = "research_runs";
const RESEARCH_CONTROLS_TABLE = "research_controls";
const RESEARCH_MEMORY_TABLE = "research_memory";
const RESEARCH_RESULT_LIMIT = 6;
const RESEARCH_MEMORY_CANDIDATE_LIMIT = 96;
const MEMORY_TEXT_LIMIT = 2400;
const RELATION_LIMIT = 16;
const GRAPH_NODE_LIMIT = 24;
const GRAPH_EDGE_LIMIT = 32;
const MEMORY_CONTEXT_RELATION_LIMIT = 3;
const unique = (values = []) => [...new Set(Array.isArray(values) ? values : [values])];
const RETENTION_BY_LAYER = Object.freeze({
  episode: { maxEntries: 80, maxAgeDays: 180, halfLifeHours: 24 * 21 },
  concept: { maxEntries: 180, maxAgeDays: 365, halfLifeHours: 24 * 75 },
  relation: { maxEntries: 240, maxAgeDays: 240, halfLifeHours: 24 * 60 },
  artifact: { maxEntries: 180, maxAgeDays: 240, halfLifeHours: 24 * 45 },
  abstraction: { maxEntries: 140, maxAgeDays: 540, halfLifeHours: 24 * 180 },
  postmortem: { maxEntries: 80, maxAgeDays: 365, halfLifeHours: 24 * 120 },
});
const STOP_WORDS = new Set([
  "the",
  "and",
  "for",
  "with",
  "that",
  "this",
  "from",
  "have",
  "your",
  "into",
  "about",
  "there",
  "their",
  "would",
  "could",
  "should",
  "after",
  "before",
  "where",
  "when",
  "what",
  "which",
  "while",
  "then",
  "than",
  "them",
  "they",
  "were",
  "been",
  "being",
  "also",
  "just",
  "over",
  "under",
  "through",
  "user",
  "assistant",
  "tool",
  "result",
  "results",
  "using",
  "used",
  "http",
  "https",
  "www",
  "com",
  "org",
  "net",
  "api",
  "json",
  "html",
  "text",
  "data",
  "file",
  "files",
  "reply",
  "said",
  "tell",
  "paper",
  "papers",
  "study",
  "studies",
  "research",
  "evidence",
  "analysis",
  "query",
  "question",
]);
const TERM_RE = /[a-z0-9_/-]{3,}/g;
const VALID_MEMORY_LAYERS = new Set([
  "episode",
  "concept",
  "relation",
  "artifact",
  "abstraction",
  "postmortem",
]);
const VALID_CONTROL_STATUSES = new Set([
  "pending",
  "claimed",
  "applied",
  "ignored",
]);

let initPromise = null;

const normalizeText = (value) =>
  String(value ?? "")
    .replace(/\u0000/g, "")
    .replace(/\r\n?/g, "\n")
    .trim();

const truncateText = (value, max = MEMORY_TEXT_LIMIT) => {
  const text = String(value ?? "");
  if (text.length <= max) return text;
  const suffix = ` ...[${text.length - max} chars omitted]`;
  if (max <= suffix.length + 8) return text.slice(0, max);
  return `${text.slice(0, max - suffix.length)}${suffix}`;
};

const clampInteger = (value, min, max, fallback) => {
  const number = Number(value);
  if (!Number.isFinite(number)) return fallback;
  return Math.min(max, Math.max(min, Math.floor(number)));
};

const stableHash = (value) =>
  crypto
    .createHash("sha256")
    .update(String(value || ""))
    .digest("hex");

const parseJsonObject = (value, fallback = null) => {
  if (value && typeof value === "object" && !Array.isArray(value)) return value;
  const text = normalizeText(value);
  if (!text) return fallback;

  try {
    return JSON.parse(text);
  } catch {
    return fallback;
  }
};

const serializeJson = (value, fallback = {}) =>
  JSON.stringify(value && typeof value === "object" ? value : fallback);

const normalizeScopeKey = (value) => {
  const normalized = normalizeText(value);
  return normalized.slice(0, 191);
};

const normalizeLayer = (value) => {
  const normalized = normalizeText(value).toLowerCase();
  return VALID_MEMORY_LAYERS.has(normalized) ? normalized : "episode";
};

const normalizeIdentifier = (value, max = 96) =>
  normalizeText(value)
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "")
    .slice(0, max);

const extractTerms = (value) =>
  Array.from(
    new Set(
      (normalizeText(value).toLowerCase().match(TERM_RE) || [])
        .filter((term) => !STOP_WORDS.has(term))
        .slice(0, 96),
    ),
  );

const buildTermSet = (value) => new Set(extractTerms(value));

const flattenValues = (value) => {
  if (Array.isArray(value)) return value.flatMap((item) => flattenValues(item));
  if (value === null || value === undefined) return [];
  return [value];
};

const normalizeStringArray = (values = [], limit = 24) =>
  Array.from(
    new Set(
      flattenValues(values)
        .flatMap((value) =>
          typeof value === "string"
            ? [value]
            : value && typeof value === "object"
              ? [
                  value.label,
                  value.name,
                  value.title,
                  value.target,
                  value.targetLabel,
                  value.type,
                  value.relation,
                  value.summary,
                  value.description,
                ]
              : [],
        )
        .map((value) => normalizeText(value))
        .filter(Boolean),
    ),
  ).slice(0, limit);

const buildSqlTimestamp = (date) =>
  date instanceof Date && !Number.isNaN(date.getTime())
    ? date.toISOString().slice(0, 19).replace("T", " ")
    : null;

const clampScore = (value, fallback = 0) => {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return fallback;
  return Math.max(0, Math.min(1, numeric));
};

const getRetentionPolicy = (layer) =>
  RETENTION_BY_LAYER[normalizeLayer(layer)] || RETENTION_BY_LAYER.episode;

const buildFreshnessScore = (layer, updatedAt) => {
  const policy = getRetentionPolicy(layer);
  const timestamp = updatedAt ? new Date(updatedAt).getTime() : Date.now();
  const ageHours = Math.max(0, (Date.now() - timestamp) / 36e5);
  return Math.exp(
    (-Math.log(2) * ageHours) / Math.max(1, policy.halfLifeHours || 24 * 30),
  );
};

const normalizeRelations = (relations = [], limit = RELATION_LIMIT) =>
  flattenValues(relations)
    .map((relation) => {
      if (typeof relation === "string") {
        const target = normalizeText(relation);
        return target ? { type: "related_to", target } : null;
      }

      if (!relation || typeof relation !== "object") return null;
      const target = normalizeText(
        relation.target ||
          relation.targetLabel ||
          relation.label ||
          relation.name ||
          relation.concept ||
          relation.node ||
          "",
      );
      const type = normalizeText(
        relation.type ||
          relation.relation ||
          relation.kind ||
          relation.edgeType ||
          "related_to",
      );
      if (!target) return null;
      return {
        type: normalizeIdentifier(type, 48) || "related_to",
        target,
        weight: clampScore(relation.weight ?? relation.score, 0.5),
        evidence: normalizeText(relation.evidence || relation.reason || ""),
      };
    })
    .filter(Boolean)
    .slice(0, limit);

const buildRelationDigest = (relations = []) =>
  normalizeRelations(relations)
    .map((relation) =>
      [relation.type, relation.target, relation.evidence]
        .filter(Boolean)
        .join(" "),
    )
    .join(" ");

const normalizeArtifacts = (artifacts = [], limit = RELATION_LIMIT) =>
  flattenValues(artifacts)
    .map((artifact) => {
      if (typeof artifact === "string") {
        const label = normalizeText(artifact);
        return label ? { type: "artifact", label } : null;
      }
      if (!artifact || typeof artifact !== "object") return null;
      const label = normalizeText(
        artifact.label ||
          artifact.name ||
          artifact.title ||
          artifact.id ||
          artifact.value ||
          "",
      );
      if (!label) return null;
      return {
        type: normalizeText(artifact.type || artifact.kind || "artifact")
          .replace(/\s+/g, "_")
          .toLowerCase(),
        label,
        value: truncateText(
          normalizeText(artifact.value || artifact.url || artifact.note || ""),
          320,
        ),
      };
    })
    .filter(Boolean)
    .slice(0, limit);

const buildArtifactDigest = (artifacts = []) =>
  normalizeArtifacts(artifacts)
    .map((artifact) =>
      [artifact.type, artifact.label, artifact.value].filter(Boolean).join(" "),
    )
    .join(" ");

const buildCompactJson = (value, max = 1200) =>
  truncateText(JSON.stringify(value ?? null, null, 2), max);

const matchConceptLabels = (value = "", labels = [], limit = 8) => {
  const normalized = normalizeText(value).toLowerCase();
  if (!normalized) return [];
  return labels
    .filter((label) => normalized.includes(label.toLowerCase()))
    .slice(0, limit);
};

const normalizeMemoryMetadata = (metadata = {}, entry = {}) => {
  const source = metadata && typeof metadata === "object" ? metadata : {};
  const title = normalizeText(entry.title || entry.name || "");
  const relations = normalizeRelations(
    [source.relations, source.edges, source.links, source.graphEdges],
    RELATION_LIMIT,
  );

  return {
    ...source,
    runId: normalizeText(source.runId || ""),
    query: truncateText(normalizeText(source.query || ""), 320),
    domain: truncateText(normalizeText(source.domain || ""), 96),
    scope: truncateText(normalizeText(source.scope || ""), 96),
    outputMode: truncateText(normalizeText(source.outputMode || ""), 96),
    ontology: truncateText(normalizeText(source.ontology || ""), 96),
    canonicalId: truncateText(normalizeText(source.canonicalId || ""), 160),
    patternType: truncateText(
      normalizeText(source.patternType || source.abstractionKind || ""),
      96,
    ),
    concepts: normalizeStringArray(
      [
        source.concepts,
        source.relatedConcepts,
        source.linkedConcepts,
        entry.layer === "concept" ? title : "",
      ],
      24,
    ),
    relatedConcepts: normalizeStringArray(
      [
        source.relatedConcepts,
        source.concepts,
        relations.map((relation) => relation.target),
      ],
      24,
    ),
    aliases: normalizeStringArray([source.aliases, source.synonyms], 12),
    tags: normalizeStringArray(
      [
        source.tags,
        source.domain,
        source.scope,
        source.outputMode,
        source.patternType,
        source.abstractionKind,
        source.ontology,
      ],
      24,
    ),
    claims: normalizeStringArray(
      [source.claims, source.hypotheses, source.counterHypotheses],
      16,
    ),
    artifacts: normalizeArtifacts(
      [
        source.artifacts,
        source.outputs,
        source.exports,
        source.sources,
        source.repositories,
        source.checkpoints,
      ],
      RELATION_LIMIT,
    ),
    promptPatches: normalizeStringArray(
      source.promptPatches || source.prompt_patches_applied || [],
      12,
    ),
    relations,
    artifactType: truncateText(
      normalizeText(source.artifactType || source.kind || ""),
      96,
    ),
    sourceConcept: truncateText(
      normalizeText(source.sourceConcept || source.source || ""),
      160,
    ),
    targetConcept: truncateText(
      normalizeText(source.targetConcept || source.target || ""),
      160,
    ),
    relationType: truncateText(
      normalizeIdentifier(
        source.relationType || source.edgeType || source.type || "",
        48,
      ),
      48,
    ),
    confidence: clampScore(source.confidence, 0),
    evidenceWeight: clampScore(
      source.evidenceWeight ?? source.evidence_weight,
      0,
    ),
    contradictionScore: clampScore(
      source.contradictionScore ?? source.contradiction_score,
      0,
    ),
    sensitivity: clampScore(source.sensitivity, 0),
    scoreDelta: Number((source.scoreDelta ?? source.final_score_delta) || 0),
    retentionPolicy: getRetentionPolicy(entry.layer),
  };
};

const summarizeStructuredCollection = (values = [], limit = 5) =>
  flattenValues(values)
    .map((value) => {
      if (typeof value === "string") return normalizeText(value);
      if (!value || typeof value !== "object") return "";
      const label = normalizeText(
        value.label ||
          value.name ||
          value.title ||
          value.topic ||
          value.id ||
          "",
      );
      const status = normalizeText(
        value.status || value.signal || value.trend || value.direction || "",
      );
      const numeric = [
        value.score,
        value.weight,
        value.count,
        value.velocity,
        value.value,
      ].find(
        (candidate) =>
          candidate !== undefined && candidate !== null && candidate !== "",
      );
      const summary = normalizeText(
        value.summary || value.description || value.note || value.reason || "",
      );
      return [
        label,
        status,
        numeric !== undefined ? String(numeric) : "",
        summary,
      ]
        .filter(Boolean)
        .join(" | ");
    })
    .filter(Boolean)
    .slice(0, limit)
    .join(" || ");

const inferConceptLabels = (structured = {}) => {
  const labels = new Map();

  for (const concept of Array.isArray(structured.concepts)
    ? structured.concepts
    : []) {
    const label = normalizeText(concept?.label || concept?.name);
    if (!label) continue;
    labels.set(label.toLowerCase(), label);
  }

  for (const claim of Array.isArray(structured.epistemicClaims)
    ? structured.epistemicClaims
    : []) {
    for (const concept of normalizeStringArray(
      claim?.concepts || claim?.relatedConcepts || [],
      12,
    )) {
      labels.set(concept.toLowerCase(), concept);
    }
  }

  return [...labels.values()];
};

const buildGraphArtifacts = (payload = {}) => {
  const structured = payload.structuredData || {};
  const concepts = Array.isArray(structured.concepts)
    ? structured.concepts
    : [];
  const conceptLabels = inferConceptLabels(structured);
  const labelLookup = new Map(
    conceptLabels.map((label) => [label.toLowerCase(), label]),
  );
  const edges = [];

  for (const concept of concepts) {
    const source = normalizeText(concept?.label || concept?.name);
    if (!source) continue;

    const relations = normalizeRelations([
      ...(Array.isArray(concept?.relations) ? concept.relations : []),
      ...(Array.isArray(concept?.edges) ? concept.edges : []),
      ...(Array.isArray(concept?.related) ? concept.related : []),
      ...(Array.isArray(concept?.links) ? concept.links : []),
      ...(Array.isArray(concept?.neighbors) ? concept.neighbors : []),
    ]);

    for (const relation of relations) {
      const canonicalTarget =
        labelLookup.get(relation.target.toLowerCase()) || relation.target;
      edges.push({
        source,
        target: canonicalTarget,
        type: relation.type,
        weight: relation.weight,
        evidence: relation.evidence,
      });
    }
  }

  for (const claim of Array.isArray(structured.epistemicClaims)
    ? structured.epistemicClaims
    : []) {
    const text = normalizeText(claim?.claim);
    if (!text) continue;
    const matchedLabels = conceptLabels.filter((label) =>
      text.toLowerCase().includes(label.toLowerCase()),
    );
    for (let index = 0; index < matchedLabels.length - 1; index += 1) {
      const source = matchedLabels[index];
      const target = matchedLabels[index + 1];
      edges.push({
        source,
        target,
        type: "co_claim",
        weight: clampScore(claim?.confidence, 0.55),
        evidence: truncateText(text, 280),
      });
    }
  }

  const relationMap = new Map();
  for (const edge of edges) {
    const key = `${edge.source.toLowerCase()}::${edge.type}::${edge.target.toLowerCase()}`;
    const existing = relationMap.get(key);
    if (!existing || (edge.weight || 0) > (existing.weight || 0)) {
      relationMap.set(key, edge);
    }
  }

  const dedupedEdges = [...relationMap.values()].slice(0, 120);
  const relationsBySource = new Map();
  const centrality = new Map();
  for (const edge of dedupedEdges) {
    const key = edge.source.toLowerCase();
    const bucket = relationsBySource.get(key) || [];
    bucket.push({
      type: edge.type,
      target: edge.target,
      weight: edge.weight,
      evidence: edge.evidence,
    });
    relationsBySource.set(key, bucket);

    centrality.set(
      edge.source,
      (centrality.get(edge.source) || 0) + (edge.weight || 0.5),
    );
    centrality.set(
      edge.target,
      (centrality.get(edge.target) || 0) + (edge.weight || 0.5),
    );
  }

  return {
    conceptLabels,
    edges: dedupedEdges,
    relationsBySource,
    centralConcepts: [...centrality.entries()]
      .sort((left, right) => right[1] - left[1])
      .map(([label]) => label)
      .slice(0, GRAPH_NODE_LIMIT),
  };
};

const buildArtifactIndex = (payload = {}) => {
  const plan = payload.plan || {};
  const structured = payload.structuredData || {};
  const checkpoints =
    payload.checkpoints && typeof payload.checkpoints === "object"
      ? payload.checkpoints
      : {};
  const outputs =
    payload.outputs && typeof payload.outputs === "object"
      ? payload.outputs
      : payload.final?.exports && typeof payload.final.exports === "object"
        ? payload.final.exports
        : {};

  return normalizeArtifacts([
    {
      type: "domain",
      label: normalizeText(plan.domain?.label || plan.domain?.id),
    },
    {
      type: "scope",
      label: normalizeText(plan.scope?.label || plan.scope?.id),
    },
    {
      type: "output_mode",
      label: normalizeText(plan.outputMode?.label || plan.outputMode?.id),
    },
    ...(Array.isArray(structured.paperInventory)
      ? structured.paperInventory
      : []
    )
      .slice(0, 8)
      .map((source) => ({
        type: "source",
        label: normalizeText(source?.title || source?.url),
        value: normalizeText(
          source?.url || source?.providerLabel || source?.provider,
        ),
      })),
    ...(Array.isArray(structured.repositories) ? structured.repositories : [])
      .slice(0, 6)
      .map((repo) => ({
        type: "repository",
        label: normalizeText(repo?.url || repo?.name),
        value: normalizeText(repo?.notes || ""),
      })),
    ...(Array.isArray(structured.supplementary) ? structured.supplementary : [])
      .slice(0, 4)
      .map((item) => ({
        type: "supplementary",
        label: normalizeText(item?.title || item?.url),
        value: normalizeText(item?.url || item?.excerpt || ""),
      })),
    ...(Array.isArray(structured.concepts) ? structured.concepts : [])
      .slice(0, 8)
      .map((concept) => ({
        type: "concept",
        label: normalizeText(concept?.label || concept?.name),
        value: normalizeText(concept?.canonicalId || concept?.ontology || ""),
      })),
    ...Object.keys(checkpoints)
      .slice(0, 8)
      .map((checkpointId) => ({
        type: "checkpoint",
        label: checkpointId,
      })),
    ...Object.keys(outputs)
      .filter((key) => normalizeText(outputs[key]))
      .slice(0, 8)
      .map((key) => ({
        type: "export",
        label: key,
      })),
    payload.result?.decision?.decision
      ? {
          type: "decision",
          label: normalizeText(payload.result.decision.decision),
          value: normalizeText(payload.result.decision.expected_outcome || ""),
        }
      : null,
  ]);
};

const scoreMemoryEntry = (queryTerms, row = {}) => {
  if (!queryTerms.size) return 0;
  const haystack = [row.title, row.content, row.searchText]
    .filter(Boolean)
    .join(" ");
  const terms = buildTermSet(haystack);
  const metadata = normalizeMemoryMetadata(row.metadata, row);
  const graphTerms = buildTermSet(
    [
      ...(Array.isArray(metadata.concepts) ? metadata.concepts : []),
      ...(Array.isArray(metadata.relatedConcepts)
        ? metadata.relatedConcepts
        : []),
      ...(Array.isArray(metadata.aliases) ? metadata.aliases : []),
      ...(Array.isArray(metadata.tags) ? metadata.tags : []),
      buildRelationDigest(metadata.relations),
      buildArtifactDigest(metadata.artifacts),
    ]
      .filter(Boolean)
      .join(" "),
  );

  let overlap = 0;
  let graphOverlap = 0;
  for (const term of queryTerms) {
    if (terms.has(term)) {
      overlap += 1;
      continue;
    }
    if (graphTerms.has(term)) graphOverlap += 1;
  }
  if (!overlap && !graphOverlap) return 0;

  const updatedAt = row.updatedAt
    ? new Date(row.updatedAt).getTime()
    : Date.now();
  const ageHours = Math.max(0, (Date.now() - updatedAt) / 36e5);
  const freshness = buildFreshnessScore(row.layer, row.updatedAt);
  const coverageBoost =
    (overlap + graphOverlap * 0.75) / Math.max(1, queryTerms.size);
  const relationBoost = Math.min(
    0.45,
    normalizeRelations(metadata.relations || [], RELATION_LIMIT).length * 0.05,
  );
  const artifactBoost = Math.min(
    0.28,
    normalizeArtifacts(metadata.artifacts || [], RELATION_LIMIT).length * 0.04,
  );
  const queryBoost =
    metadata.query &&
    normalizeText(metadata.query)
      .toLowerCase()
      .includes([...queryTerms][0] || "")
      ? 0.18
      : 0;
  const exactTitleBoost = [...queryTerms].some((term) =>
    normalizeText(row.title).toLowerCase().includes(term),
  )
    ? 0.22
    : 0;
  const confidenceBoost =
    clampScore(metadata.confidence ?? metadata.evidenceWeight, 0) * 0.18;
  const stalenessPenalty =
    ageHours > (getRetentionPolicy(row.layer).halfLifeHours || 720) * 2
      ? 0.08
      : 0;
  const layerBoost =
    row.layer === "abstraction"
      ? 0.42
      : row.layer === "concept"
        ? 0.22
        : row.layer === "postmortem"
          ? 0.16
          : 0;
  const contradictionPenalty =
    clampScore(metadata.contradictionScore, 0) * 0.16;
  return (
    overlap +
    graphOverlap * 0.7 +
    coverageBoost +
    freshness +
    layerBoost +
    relationBoost +
    artifactBoost +
    queryBoost +
    exactTitleBoost +
    confidenceBoost -
    contradictionPenalty -
    stalenessPenalty
  );
};

const normalizeMemoryEntry = (entry = {}) => {
  const layer = normalizeLayer(entry.layer);
  const title = truncateText(
    normalizeText(entry.title || entry.name || ""),
    320,
  );
  const content = truncateText(
    normalizeText(entry.content || entry.summary || ""),
    MEMORY_TEXT_LIMIT,
  );
  if (!content) return null;

  const metadata = normalizeMemoryMetadata(entry.metadata, {
    layer,
    title,
    content,
  });
  const relations = metadata.relations || [];
  const artifacts = metadata.artifacts || [];
  const searchText = truncateText(
    [
      title,
      content,
      metadata.query,
      metadata.domain,
      metadata.scope,
      metadata.outputMode,
      Array.isArray(metadata.tags) ? metadata.tags.join(" ") : "",
      Array.isArray(metadata.concepts) ? metadata.concepts.join(" ") : "",
      Array.isArray(metadata.claims) ? metadata.claims.join(" ") : "",
      Array.isArray(metadata.relatedConcepts)
        ? metadata.relatedConcepts.join(" ")
        : "",
      Array.isArray(metadata.aliases) ? metadata.aliases.join(" ") : "",
      buildRelationDigest(relations),
      buildArtifactDigest(artifacts),
      normalizeText(metadata.artifactType || ""),
      normalizeText(metadata.sourceConcept || ""),
      normalizeText(metadata.targetConcept || ""),
      normalizeText(metadata.relationType || ""),
      normalizeText(metadata.patternType || ""),
      normalizeText(metadata.canonicalId || ""),
      normalizeText(metadata.ontology || ""),
    ]
      .filter(Boolean)
      .join(" "),
    4000,
  );

  const baseKey = normalizeText(
    entry.entryKey || entry.id || title || content.slice(0, 180),
  );
  return {
    layer,
    entryKey: stableHash(`${layer}\n${baseKey}\n${content}`),
    title,
    content,
    metadata: {
      ...metadata,
      artifacts,
      relations,
      relatedConcepts: normalizeStringArray(
        metadata.relatedConcepts || metadata.concepts || [],
        24,
      ),
      tags: normalizeStringArray(metadata.tags || [], 24),
      claims: normalizeStringArray(metadata.claims || [], 16),
      artifactType: normalizeText(metadata.artifactType || ""),
      sourceConcept: normalizeText(metadata.sourceConcept || ""),
      targetConcept: normalizeText(metadata.targetConcept || ""),
      relationType: normalizeText(metadata.relationType || ""),
    },
    searchText,
  };
};

const normalizeRunRecord = (record = {}) => {
  const runId = normalizeText(record.runId || record.id);
  if (!runId) return null;

  const payload =
    record.payload && typeof record.payload === "object"
      ? record.payload
      : record;
  return {
    runId,
    scopeKey: normalizeScopeKey(record.scopeKey || payload.scopeKey || ""),
    status:
      normalizeText(record.status || payload.status || "completed").slice(
        0,
        32,
      ) || "completed",
    query: truncateText(
      normalizeText(record.query || payload.query || ""),
      2000,
    ),
    title: truncateText(
      normalizeText(
        record.title || payload.title || payload.heading || payload.query || "",
      ),
      320,
    ),
    domain: truncateText(
      normalizeText(
        record.domain ||
          payload.plan?.domain?.id ||
          payload.researchMeta?.domain?.id ||
          "",
      ),
      96,
    ),
    outputMode: truncateText(
      normalizeText(
        record.outputMode ||
          payload.plan?.outputMode?.id ||
          payload.researchMeta?.outputMode?.id ||
          "",
      ),
      96,
    ),
    payload,
  };
};

const normalizeControlStatus = (value) => {
  const normalized = normalizeText(value).toLowerCase();
  return VALID_CONTROL_STATUSES.has(normalized) ? normalized : "pending";
};

const normalizeControlRecord = (runId, control = {}) => {
  const normalizedRunId = normalizeText(runId);
  if (!normalizedRunId) return null;

  const type = normalizeText(
    control.type || control.command || control.action || control.name || "",
  ).slice(0, 96);
  if (!type) return null;

  return {
    runId: normalizedRunId,
    status: normalizeControlStatus(control.status),
    type,
    payload:
      control && typeof control === "object" ? control : { value: control },
  };
};

const coerceUpdatedAt = (value) => {
  const date = value ? new Date(value) : null;
  return date && !Number.isNaN(date.getTime()) ? date.toISOString() : null;
};

const asObject = (value) =>
  value && typeof value === "object" && !Array.isArray(value) ? value : {};

const pickArray = (value, fallback = []) =>
  Array.isArray(value) ? value : fallback;

const resolveRunMemoryPayload = (run = {}) => {
  const payload =
    run.payload &&
    typeof run.payload === "object" &&
    !Array.isArray(run.payload)
      ? run.payload
      : asObject(run);
  const plan = asObject(payload.plan || run.plan);
  const state = asObject(payload.state || run.state);
  const extraction = asObject(
    payload.extraction || state.extraction || run.extraction,
  );
  const result = asObject(payload.result || run.result);
  const final = asObject(payload.final);
  const researchMeta = asObject(payload.researchMeta || run.researchMeta);
  const structuredSource = asObject(payload.structuredData);

  const structured = {
    paperInventory: pickArray(
      structuredSource.paperInventory,
      pickArray(state.tieredSources),
    ),
    deepComprehension: pickArray(
      structuredSource.deepComprehension,
      pickArray(state.evidenceEntries),
    ),
    claims: pickArray(structuredSource.claims, pickArray(extraction.claims)),
    repositories: pickArray(
      structuredSource.repositories,
      pickArray(extraction.repositories),
    ),
    supplementary: pickArray(
      structuredSource.supplementary,
      pickArray(extraction.supplementary),
    ),
    concepts: pickArray(
      structuredSource.concepts,
      pickArray(extraction.concepts),
    ),
    abstractions: pickArray(
      structuredSource.abstractions,
      pickArray(extraction.abstractions),
    ),
    evidencePyramid: pickArray(
      structuredSource.evidencePyramid,
      pickArray(extraction.evidencePyramid),
    ),
    metaAnalysis:
      structuredSource.metaAnalysis || extraction.metaAnalysis || null,
    statisticalVerification:
      structuredSource.statisticalVerification ||
      extraction.statisticalVerification ||
      null,
    citationIntelligence:
      structuredSource.citationIntelligence ||
      state.sourceMesh?.citationIntelligence ||
      null,
    authorNetwork:
      structuredSource.authorNetwork || state.sourceMesh?.authorNetwork || null,
    temporalTrends:
      structuredSource.temporalTrends ||
      state.sourceMesh?.temporalTrends ||
      null,
    epistemicClaims: pickArray(
      structuredSource.epistemicClaims,
      pickArray(state.claimLedger, pickArray(final.claimLedger)),
    ),
    verifierSummary:
      structuredSource.verifierSummary ||
      payload.verifierSummary ||
      run.verifierSummary ||
      state.verifierSummary ||
      final.verifierSummary ||
      null,
    decision:
      structuredSource.decision ||
      result.decision ||
      state.decisionLayer ||
      final.decision ||
      null,
    safety:
      structuredSource.safety ||
      payload.safety ||
      run.safety ||
      extraction.safety ||
      state.safety ||
      null,
    tribunal:
      structuredSource.tribunal ||
      payload.tribunal ||
      run.tribunal ||
      state.tribunal ||
      final.tribunal ||
      null,
    convergence:
      structuredSource.convergence ||
      payload.convergence ||
      run.convergence ||
      state.convergence ||
      final.convergence ||
      null,
    postmortem:
      structuredSource.postmortem ||
      payload.postmortem ||
      run.postmortem ||
      state.postmortem ||
      null,
  };

  return {
    payload,
    plan,
    state,
    extraction,
    result,
    final,
    structured,
    runId: normalizeText(payload.runId || run.runId || run.id || payload.id),
    query: normalizeText(payload.query || run.query),
    title: normalizeText(
      payload.title ||
        run.title ||
        result.heading ||
        final.heading ||
        payload.query ||
        run.query,
    ),
    domain: normalizeText(
      plan.domain?.label ||
        plan.domain?.id ||
        researchMeta.domain?.label ||
        researchMeta.domain?.id ||
        run.domain?.label ||
        run.domain?.id ||
        run.domain ||
        "",
    ),
    domainId: normalizeText(
      plan.domain?.id ||
        researchMeta.domain?.id ||
        run.domain?.id ||
        run.domain ||
        "",
    ),
    scope: normalizeText(
      plan.scope?.label ||
        plan.scope?.id ||
        researchMeta.scope?.label ||
        researchMeta.scope?.id ||
        run.scope?.label ||
        run.scope?.id ||
        run.scope ||
        "",
    ),
    scopeId: normalizeText(
      plan.scope?.id ||
        researchMeta.scope?.id ||
        run.scope?.id ||
        run.scope ||
        "",
    ),
    outputMode: normalizeText(
      plan.outputMode?.label ||
        plan.outputMode?.id ||
        researchMeta.outputMode?.label ||
        researchMeta.outputMode?.id ||
        run.outputMode?.label ||
        run.outputMode?.id ||
        run.outputMode ||
        "",
    ),
    outputModeId: normalizeText(
      plan.outputMode?.id ||
        researchMeta.outputMode?.id ||
        run.outputMode?.id ||
        run.outputMode ||
        "",
    ),
    status: normalizeText(payload.status || run.status || ""),
    tribunal: asObject(structured.tribunal),
    convergence: asObject(structured.convergence),
    postmortem: asObject(structured.postmortem),
    verifierSummary: asObject(structured.verifierSummary),
    decision: asObject(structured.decision),
    safety: asObject(structured.safety),
    heading: normalizeText(
      final.heading ||
        result.heading ||
        payload.title ||
        run.title ||
        payload.query ||
        run.query,
    ),
    answer: normalizeText(
      final.answer || final.markdown || result.finalText || result.answer || "",
    ),
  };
};

const buildRunSummaryText = (payload = {}) => {
  const view = resolveRunMemoryPayload(payload);
  const lines = [
    view.query,
    view.heading || view.title,
    view.answer,
    view.decision?.decision
      ? `Decision: ${normalizeText(view.decision.decision)}`
      : "",
  ].filter(Boolean);
  return truncateText(lines.join("\n\n"), MEMORY_TEXT_LIMIT);
};

const hydrateRunRecord = (row = {}) => {
  const payload = parseJsonObject(row.payload, {});
  const runId = normalizeText(
    row.run_id || row.runId || payload.runId || payload.id,
  );
  if (!runId) return null;

  return {
    ...(payload && typeof payload === "object" ? payload : {}),
    id: normalizeText(payload.id || runId) || runId,
    runId,
    scopeKey: normalizeScopeKey(
      row.scope_key || row.scopeKey || payload.scopeKey || "",
    ),
    status:
      normalizeText(row.status || payload.status || "completed") || "completed",
    query: normalizeText(row.query_text || row.query || payload.query || ""),
    title: normalizeText(
      row.title ||
        payload.title ||
        payload.result?.heading ||
        payload.final?.heading ||
        payload.query ||
        "",
    ),
    domain: normalizeText(
      row.domain_id ||
        row.domain ||
        payload.plan?.domain?.id ||
        payload.researchMeta?.domain?.id ||
        "",
    ),
    outputMode: normalizeText(
      row.output_mode ||
        row.outputMode ||
        payload.plan?.outputMode?.id ||
        payload.researchMeta?.outputMode?.id ||
        "",
    ),
    payload,
    createdAt: coerceUpdatedAt(
      row.created_at || row.createdAt || payload.createdAt,
    ),
    updatedAt: coerceUpdatedAt(
      row.updated_at || row.updatedAt || payload.updatedAt,
    ),
  };
};

const normalizeMemoryResults = (rows = [], queryTerms = new Set()) =>
  (Array.isArray(rows) ? rows : [])
    .map((row) => {
      const metadata = normalizeMemoryMetadata(
        parseJsonObject(row.metadata_json, {}),
        { layer: row.layer, title: row.title },
      );
      const normalized = {
        layer: normalizeLayer(row.layer),
        title: normalizeText(row.title),
        content: normalizeText(row.content),
        metadata,
        searchText: normalizeText(row.search_text),
        updatedAt: coerceUpdatedAt(row.updated_at),
      };
      return {
        ...normalized,
        score: scoreMemoryEntry(queryTerms, normalized),
      };
    })
    .filter((row) => row.content);

const dedupeRows = (rows = [], limit = RESEARCH_RESULT_LIMIT) => {
  const output = [];
  const seen = new Set();
  for (const row of Array.isArray(rows) ? rows : []) {
    const key = `${row.layer}:${row.title}:${row.content}`;
    if (!row?.content || seen.has(key)) continue;
    seen.add(key);
    output.push(row);
    if (output.length >= limit) break;
  }
  return output;
};

const buildMemoryGraphSnapshot = (results = []) => {
  const nodeMap = new Map();
  const edgeMap = new Map();

  const addNode = (label, kind = "concept", weight = 1) => {
    const normalized = normalizeText(label);
    if (!normalized) return;
    const key = normalized.toLowerCase();
    const existing = nodeMap.get(key) || {
      label: normalized,
      kind,
      weight: 0,
      mentions: 0,
    };
    existing.kind = existing.kind === "concept" ? "concept" : kind;
    existing.weight = Number((existing.weight + weight).toFixed(2));
    existing.mentions += 1;
    nodeMap.set(key, existing);
  };

  const addEdge = (source, relation = {}) => {
    const normalizedSource = normalizeText(source || "");
    const target = normalizeText(relation.target || "");
    if (!target) return;
    const type =
      normalizeIdentifier(relation.type || "related_to", 48) || "related_to";
    const key = `${normalizedSource.toLowerCase()}::${type}::${target.toLowerCase()}`;
    const existing = edgeMap.get(key) || {
      source: normalizedSource,
      target,
      type,
      weight: 0,
      evidence: normalizeText(relation.evidence || ""),
    };
    existing.weight = Number(
      (existing.weight + clampScore(relation.weight, 0.5)).toFixed(2),
    );
    if (!existing.evidence)
      existing.evidence = normalizeText(relation.evidence || "");
    edgeMap.set(key, existing);
  };

  for (const row of Array.isArray(results) ? results : []) {
    const title = normalizeText(row?.title || "");
    const layer = normalizeLayer(row?.layer);
    const metadata = normalizeMemoryMetadata(row?.metadata, row);
    if (
      layer === "relation" &&
      metadata.sourceConcept &&
      metadata.targetConcept
    ) {
      addNode(metadata.sourceConcept, "concept", 1.2);
      addNode(metadata.targetConcept, "concept", 1.1);
      addEdge(metadata.sourceConcept, {
        type: metadata.relationType || "related_to",
        target: metadata.targetConcept,
        weight: metadata.confidence || 0.6,
        evidence: row?.content,
      });
      continue;
    }
    if (title)
      addNode(
        title,
        layer === "abstraction"
          ? "abstraction"
          : layer === "postmortem"
            ? "postmortem"
            : "concept",
        1.1,
      );
    for (const concept of metadata.relatedConcepts || []) {
      addNode(concept, "concept", 0.7);
    }
    for (const relation of metadata.relations || []) {
      addEdge(title, relation);
    }
  }

  const nodes = [...nodeMap.values()]
    .sort((left, right) => {
      if (right.weight !== left.weight) return right.weight - left.weight;
      return right.mentions - left.mentions;
    })
    .slice(0, GRAPH_NODE_LIMIT);
  const edges = [...edgeMap.values()]
    .sort((left, right) => right.weight - left.weight)
    .slice(0, GRAPH_EDGE_LIMIT);

  return {
    nodes,
    edges,
    centralConcepts: nodes
      .filter((node) => node.kind === "concept")
      .slice(0, 12)
      .map((node) => node.label),
  };
};

const buildResearchMemoryEntriesFromRun = (run = {}) => {
  const view = resolveRunMemoryPayload(run);
  const payload = view.payload;
  const plan = view.plan || {};
  const final = {
    ...(view.final || {}),
    heading:
      view.heading ||
      view.final?.heading ||
      view.title ||
      view.query ||
      "Research run",
    answer: view.answer || view.final?.answer || "",
  };
  const structured = view.structured || {};
  const tribunal = view.tribunal || {};
  const postmortem = view.postmortem || {};
  const convergence = view.convergence || {};
  const graphArtifacts = buildGraphArtifacts({ structuredData: structured });
  const artifactIndex = buildArtifactIndex({
    ...payload,
    runId: view.runId,
    query: view.query,
    status: view.status,
    plan,
    final,
    result: {
      ...(view.result || {}),
      decision: view.decision,
    },
    structuredData: structured,
    tribunal,
    postmortem,
    convergence,
  });
  const baseTags = [
    view.domainId,
    view.scopeId,
    view.outputModeId,
    view.status,
  ].filter(Boolean);
  const baseMetadata = {
    runId: view.runId,
    query: view.query,
    domain: view.domain,
    scope: view.scope,
    outputMode: view.outputMode,
  };
  const selectArtifacts = (types = [], limit = 8) =>
    artifactIndex
      .filter((artifact) => !types.length || types.includes(artifact.type))
      .slice(0, limit);
  const buildRelationMetadata = (relatedConcepts = [], limit = 6) => {
    const normalizedConcepts = normalizeStringArray(relatedConcepts, 12);
    return graphArtifacts.edges
      .filter(
        (edge) =>
          normalizedConcepts.includes(edge.source) ||
          normalizedConcepts.includes(edge.target),
      )
      .slice(0, limit)
      .map((edge) => ({
        type: edge.type,
        target: edge.target,
        weight: edge.weight,
        evidence: edge.evidence,
      }));
  };

  const entries = [];
  const episodeContent = buildRunSummaryText(payload);
  if (episodeContent) {
    entries.push({
      layer: "episode",
      title: final.heading || view.title || view.query || "Research run",
      content: episodeContent,
      metadata: {
        ...baseMetadata,
        tags: baseTags,
        artifacts: artifactIndex,
        concepts: graphArtifacts.conceptLabels.slice(0, 18),
        relatedConcepts: graphArtifacts.centralConcepts.slice(0, 12),
        relations: graphArtifacts.edges.slice(0, 8).map((edge) => ({
          type: edge.type,
          target: edge.target,
          weight: edge.weight,
          evidence: edge.evidence,
        })),
        hypotheses: Array.isArray(plan.queryMatrix?.counterHypotheses)
          ? plan.queryMatrix.counterHypotheses
          : [],
      },
    });
  }

  for (const concept of Array.isArray(structured.concepts)
    ? structured.concepts
    : []) {
    const label = normalizeText(concept?.label || concept?.name);
    if (!label) continue;
    const relations =
      graphArtifacts.relationsBySource.get(label.toLowerCase()) || [];
    entries.push({
      layer: "concept",
      title: label,
      content: normalizeText(
        concept?.summary ||
          concept?.description ||
          [
            label,
            concept?.ontology,
            concept?.canonicalId,
            relations.length
              ? `relations=${relations.map((relation) => `${relation.type}:${relation.target}`).join(", ")}`
              : "",
          ]
            .filter(Boolean)
            .join(" | "),
      ),
      metadata: {
        ...baseMetadata,
        ontology: concept?.ontology || "",
        canonicalId: concept?.canonicalId || "",
        concepts: [label],
        artifacts: artifactIndex
          .filter((artifact) =>
            ["concept", "source", "repository"].includes(artifact.type),
          )
          .slice(0, 8),
        relatedConcepts: relations.map((relation) => relation.target),
        relations,
        aliases: normalizeStringArray(
          concept?.aliases || concept?.synonyms || [],
          12,
        ),
        tags: Array.isArray(concept?.tags) ? concept.tags : [],
      },
    });
  }

  for (const claim of Array.isArray(structured.epistemicClaims)
    ? structured.epistemicClaims
    : []) {
    const text = normalizeText(claim?.claim);
    if (!text) continue;
    const relatedConcepts = graphArtifacts.conceptLabels
      .filter((label) => text.toLowerCase().includes(label.toLowerCase()))
      .slice(0, 8);
    entries.push({
      layer: "concept",
      title: truncateText(text, 240),
      content: [
        text,
        `confidence=${claim?.confidence ?? "n/a"}`,
        `evidence_weight=${claim?.evidence_weight ?? "n/a"}`,
        `contradiction_score=${claim?.contradiction_score ?? "n/a"}`,
        relatedConcepts.length ? `concepts=${relatedConcepts.join(", ")}` : "",
      ].join(" | "),
      metadata: {
        ...baseMetadata,
        claims: [text],
        artifacts: artifactIndex
          .filter((artifact) =>
            ["source", "concept", "decision"].includes(artifact.type),
          )
          .slice(0, 8),
        relatedConcepts,
        relations: relatedConcepts.map((target) => ({
          type:
            clampScore(claim?.contradiction_score, 0.2) >= 0.35
              ? "contests"
              : "supports",
          target,
          weight: clampScore(claim?.confidence ?? claim?.evidence_weight, 0.58),
          evidence: truncateText(text, 240),
        })),
        confidence: clampScore(claim?.confidence, 0.5),
        evidenceWeight: clampScore(claim?.evidence_weight, 0.5),
        contradictionScore: clampScore(claim?.contradiction_score, 0.2),
        tags: Array.isArray(claim?.tags) ? claim.tags : [],
      },
    });
  }

  for (const edge of graphArtifacts.edges.slice(0, 28)) {
    entries.push({
      layer: "relation",
      title: truncateText(`${edge.source} ${edge.type} ${edge.target}`, 240),
      content: truncateText(
        edge.evidence || `${edge.source} ${edge.type} ${edge.target}`,
        MEMORY_TEXT_LIMIT,
      ),
      metadata: {
        ...baseMetadata,
        artifactType: "graph_relation",
        sourceConcept: edge.source,
        targetConcept: edge.target,
        relationType: edge.type,
        confidence: clampScore(edge.weight, 0.55),
        concepts: [edge.source, edge.target],
        relatedConcepts: [edge.source, edge.target],
        relations: [
          {
            type: edge.type,
            target: edge.target,
            weight: edge.weight,
            evidence: edge.evidence,
          },
        ],
        artifacts: selectArtifacts(["concept", "source", "repository"], 6),
        tags: [...baseTags, "relation_edge"],
      },
    });
  }

  const artifactCandidates = [
    ...artifactIndex.map((artifact) => ({
      type: artifact.type,
      title: `${artifact.type.replace(/_/g, " ")}: ${artifact.label}`,
      content: [artifact.label, artifact.value].filter(Boolean).join(" | "),
    })),
    structured.metaAnalysis
      ? {
          type: "meta_analysis",
          title: `Meta-analysis: ${view.query || view.runId || "run"}`,
          content: [
            structured.metaAnalysis.combined_effect_size !== undefined
              ? `combined_effect_size=${structured.metaAnalysis.combined_effect_size}`
              : "",
            structured.metaAnalysis.i_squared !== undefined
              ? `i_squared=${structured.metaAnalysis.i_squared}`
              : "",
            normalizeText(structured.metaAnalysis.model || ""),
            summarizeStructuredCollection(
              structured.metaAnalysis.assumption_conflicts || [],
              4,
            ),
          ]
            .filter(Boolean)
            .join(" | "),
          evidenceWeight: clampScore(
            structured.metaAnalysis.i_squared !== undefined
              ? 1 -
                  Math.min(
                    1,
                    Number(structured.metaAnalysis.i_squared || 0) / 100,
                  )
              : 0.6,
            0.6,
          ),
        }
      : null,
    structured.statisticalVerification
      ? {
          type: "statistical_verification",
          title: `Statistical verification: ${view.query || view.runId || "run"}`,
          content: buildCompactJson(structured.statisticalVerification, 1200),
        }
      : null,
    structured.citationIntelligence
      ? {
          type: "citation_intelligence",
          title: `Citation intelligence: ${view.query || view.runId || "run"}`,
          content: buildCompactJson(structured.citationIntelligence, 1200),
        }
      : null,
    structured.authorNetwork
      ? {
          type: "author_network",
          title: `Author network: ${view.query || view.runId || "run"}`,
          content: buildCompactJson(structured.authorNetwork, 1200),
        }
      : null,
    structured.temporalTrends
      ? {
          type: "temporal_trends",
          title: `Temporal trends: ${view.query || view.runId || "run"}`,
          content: buildCompactJson(structured.temporalTrends, 1200),
        }
      : null,
    Object.keys(view.decision || {}).length
      ? {
          type: "decision",
          title: `Decision: ${normalizeText(view.decision.decision || view.query || view.runId || "run")}`,
          content: buildCompactJson(view.decision, 1000),
          confidence: clampScore(view.decision.confidence, 0.7),
        }
      : null,
    Object.keys(view.verifierSummary || {}).length
      ? {
          type: "verifier_summary",
          title: `Verifier summary: ${view.query || view.runId || "run"}`,
          content: buildCompactJson(view.verifierSummary, 1000),
          promptPatches: normalizeStringArray(
            [
              view.verifierSummary.rewriteBrief,
              view.verifierSummary.targetedDimension,
            ],
            6,
          ),
        }
      : null,
    Object.keys(tribunal || {}).length
      ? {
          type: "tribunal",
          title: `Tribunal: ${view.query || view.runId || "run"}`,
          content: buildCompactJson(tribunal, 1000),
          promptPatches: normalizeStringArray(
            [tribunal.rewrite_brief, tribunal.targeted_dimension],
            6,
          ),
        }
      : null,
    Object.keys(convergence || {}).length
      ? {
          type: "convergence",
          title: `Convergence: ${view.query || view.runId || "run"}`,
          content: buildCompactJson(convergence, 800),
        }
      : null,
    Object.keys(view.safety || {}).length
      ? {
          type: "safety",
          title: `Safety: ${view.query || view.runId || "run"}`,
          content: buildCompactJson(view.safety, 1000),
        }
      : null,
  ].filter(
    (artifact) =>
      normalizeText(artifact?.title || "") &&
      normalizeText(artifact?.content || ""),
  );

  for (const artifact of artifactCandidates.slice(0, 24)) {
    const relatedConcepts = normalizeStringArray(
      [
        matchConceptLabels(
          `${artifact.title}\n${artifact.content}`,
          graphArtifacts.conceptLabels,
          10,
        ),
      ],
      10,
    );
    entries.push({
      layer: "artifact",
      title: truncateText(artifact.title, 240),
      content: truncateText(artifact.content, MEMORY_TEXT_LIMIT),
      metadata: {
        ...baseMetadata,
        artifactType: artifact.type,
        confidence: clampScore(artifact.confidence, 0),
        evidenceWeight: clampScore(artifact.evidenceWeight, 0),
        concepts: relatedConcepts,
        relatedConcepts,
        relations: buildRelationMetadata(relatedConcepts, 6),
        artifacts: [
          {
            type: artifact.type,
            label: artifact.title,
            value: truncateText(artifact.content, 320),
          },
        ],
        promptPatches: normalizeStringArray(artifact.promptPatches || [], 6),
        tags: [...baseTags, artifact.type],
      },
    });
  }

  const derivedAbstractions = [
    structured?.metaAnalysis && typeof structured.metaAnalysis === "object"
      ? {
          text: [
            structured.metaAnalysis.combined_effect_size !== undefined
              ? `Combined effect size ${structured.metaAnalysis.combined_effect_size}.`
              : "",
            structured.metaAnalysis.i_squared !== undefined
              ? `Heterogeneity I2 ${structured.metaAnalysis.i_squared}.`
              : "",
            normalizeText(structured.metaAnalysis.model || ""),
            summarizeStructuredCollection(
              structured.metaAnalysis.assumption_conflicts || [],
              4,
            ),
          ]
            .filter(Boolean)
            .join(" "),
          patternType: "meta_analysis_pattern",
        }
      : null,
    Array.isArray(structured.evidencePyramid) &&
    structured.evidencePyramid.length
      ? {
          text: `Evidence pyramid: ${summarizeStructuredCollection(structured.evidencePyramid, 5)}`,
          patternType: "evidence_pyramid_pattern",
        }
      : null,
    structured.citationIntelligence
      ? {
          text: `Citation intelligence: ${summarizeStructuredCollection(
            structured.citationIntelligence?.topPapers ||
              structured.citationIntelligence?.traces ||
              structured.citationIntelligence,
            5,
          )}`,
          patternType: "citation_intelligence_pattern",
        }
      : null,
    structured.authorNetwork
      ? {
          text: `Author network: ${summarizeStructuredCollection(
            structured.authorNetwork?.clusters ||
              structured.authorNetwork?.authors ||
              structured.authorNetwork,
            5,
          )}`,
          patternType: "author_network_pattern",
        }
      : null,
    structured.temporalTrends
      ? {
          text: `Temporal trends: ${summarizeStructuredCollection(
            structured.temporalTrends?.trends ||
              structured.temporalTrends?.topics ||
              structured.temporalTrends,
            5,
          )}`,
          patternType: "temporal_trend_pattern",
        }
      : null,
    structured.statisticalVerification
      ? {
          text: `Statistical verification: ${summarizeStructuredCollection(
            [
              structured.statisticalVerification.summary,
              ...(Array.isArray(
                structured.statisticalVerification.assumption_conflicts,
              )
                ? structured.statisticalVerification.assumption_conflicts
                : []),
            ],
            5,
          )}`,
          patternType: "statistical_verification_pattern",
        }
      : null,
  ].filter((item) => normalizeText(item?.text || ""));

  const abstractionPatterns = [
    ...(Array.isArray(postmortem.prompt_patches_applied)
      ? postmortem.prompt_patches_applied
      : []),
    ...(Array.isArray(postmortem.phases_that_degraded_score)
      ? postmortem.phases_that_degraded_score
      : []),
    ...(Array.isArray(structured.abstractions) ? structured.abstractions : []),
    ...derivedAbstractions,
    ...graphArtifacts.edges
      .slice(0, 10)
      .map((edge) => `${edge.source} ${edge.type} ${edge.target}`),
    convergence?.residual_uncertainty >= 0.4
      ? `High residual uncertainty persisted (${convergence.residual_uncertainty}).`
      : "",
    tribunal?.targeted_dimension
      ? `Tribunal repeatedly targeted ${tribunal.targeted_dimension}.`
      : "",
  ]
    .map((item) =>
      typeof item === "string"
        ? { text: normalizeText(item), patternType: "" }
        : {
            ...item,
            text: normalizeText(item?.text || item?.summary || item?.label),
            patternType: normalizeText(item?.patternType || ""),
          },
    )
    .filter((item) => item.text);

  for (const patternEntry of abstractionPatterns.slice(0, 12)) {
    const pattern = normalizeText(
      patternEntry?.text || patternEntry?.summary || patternEntry?.label,
    );
    if (!pattern) continue;
    const normalizedPattern = pattern.toLowerCase();
    entries.push({
      layer: "abstraction",
      title: truncateText(pattern, 240),
      content: pattern,
      metadata: {
        ...baseMetadata,
        patternType:
          normalizeText(patternEntry?.patternType || "") ||
          (normalizedPattern.includes("uncertainty")
            ? "uncertainty_pattern"
            : normalizedPattern.includes("targeted")
              ? "tribunal_pattern"
              : normalizedPattern.includes("postmortem")
                ? "postmortem_pattern"
                : "research_pattern"),
        relatedConcepts: graphArtifacts.centralConcepts
          .filter((label) => normalizedPattern.includes(label.toLowerCase()))
          .slice(0, 8),
        artifacts: artifactIndex
          .filter((artifact) =>
            ["checkpoint", "export", "decision", "domain", "scope"].includes(
              artifact.type,
            ),
          )
          .slice(0, 8),
        relations: graphArtifacts.edges
          .filter(
            (edge) =>
              normalizedPattern.includes(edge.source.toLowerCase()) ||
              normalizedPattern.includes(edge.target.toLowerCase()),
          )
          .slice(0, 6)
          .map((edge) => ({
            type: edge.type,
            target: edge.target,
            weight: edge.weight,
            evidence: edge.evidence,
          })),
        tags: [
          plan.domain?.id,
          plan.scope?.id,
          tribunal?.targeted_dimension || "",
          normalizeText(patternEntry?.patternType || ""),
          "research_pattern",
        ].filter(Boolean),
      },
    });
  }

  if (Object.keys(postmortem || {}).length) {
    entries.push({
      layer: "postmortem",
      title: truncateText(
        `Postmortem: ${view.query || view.runId || "run"}`,
        240,
      ),
      content: truncateText(JSON.stringify(postmortem), MEMORY_TEXT_LIMIT),
      metadata: {
        ...baseMetadata,
        artifacts: artifactIndex
          .filter((artifact) => artifact.type !== "concept")
          .slice(0, 10),
        promptPatches: Array.isArray(postmortem.prompt_patches_applied)
          ? postmortem.prompt_patches_applied
          : [],
        scoreDelta: Number(postmortem.final_score_delta || 0),
        relations: graphArtifacts.edges.slice(0, 6).map((edge) => ({
          type: edge.type,
          target: edge.target,
          weight: edge.weight,
          evidence: edge.evidence,
        })),
        tags: [plan.domain?.id, plan.scope?.id, "postmortem"].filter(Boolean),
      },
    });
  }

  return entries.map((entry) => normalizeMemoryEntry(entry)).filter(Boolean);
};

const pruneResearchMemoryScope = async (scopeKey) => {
  const normalizedScopeKey = normalizeScopeKey(scopeKey);
  if (!normalizedScopeKey) return;

  await ensureResearchTables();
  const db = getPool();

  for (const [layer, retention] of Object.entries(RETENTION_BY_LAYER)) {
    const maxEntries = clampInteger(retention.maxEntries, 8, 500, 120);
    const maxAgeDays = clampInteger(retention.maxAgeDays, 7, 3650, 365);
    const cutoff = buildSqlTimestamp(
      new Date(Date.now() - maxAgeDays * 24 * 60 * 60 * 1000),
    );

    if (cutoff) {
      await db.query(
        `
                    DELETE FROM ${RESEARCH_MEMORY_TABLE}
                    WHERE scope_key = ? AND layer = ? AND updated_at < ?
                `,
        [normalizedScopeKey, layer, cutoff],
      );
    }

    await db.query(
      `
                DELETE FROM ${RESEARCH_MEMORY_TABLE}
                WHERE scope_key = ? AND layer = ? AND id NOT IN (
                    SELECT id FROM (
                        SELECT id
                        FROM ${RESEARCH_MEMORY_TABLE}
                        WHERE scope_key = ? AND layer = ?
                        ORDER BY updated_at DESC, id DESC
                        LIMIT ?
                    ) retained
                )
            `,
      [normalizedScopeKey, layer, normalizedScopeKey, layer, maxEntries],
    );
  }
};

const ensureResearchTables = async () => {
  if (initPromise) return initPromise;

  initPromise = (async () => {
    const db = getPool();
    await db.query(`
            CREATE TABLE IF NOT EXISTS ${RESEARCH_RUNS_TABLE} (
                run_id VARCHAR(96) NOT NULL PRIMARY KEY,
                scope_key VARCHAR(191) NULL,
                status VARCHAR(32) NOT NULL,
                query_text TEXT NOT NULL,
                title VARCHAR(320) NOT NULL,
                domain_id VARCHAR(96) NOT NULL,
                output_mode VARCHAR(96) NOT NULL,
                payload LONGTEXT NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                KEY idx_research_runs_scope (scope_key, updated_at),
                KEY idx_research_runs_status (status, updated_at)
            ) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin
        `);

    await db.query(`
            CREATE TABLE IF NOT EXISTS ${RESEARCH_CONTROLS_TABLE} (
                id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
                run_id VARCHAR(96) NOT NULL,
                status VARCHAR(24) NOT NULL DEFAULT 'pending',
                control_type VARCHAR(96) NOT NULL,
                payload LONGTEXT NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                KEY idx_research_controls_run (run_id, status, created_at)
            ) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin
        `);

    await db.query(`
            CREATE TABLE IF NOT EXISTS ${RESEARCH_MEMORY_TABLE} (
                id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
                scope_key VARCHAR(191) NOT NULL,
                layer VARCHAR(24) NOT NULL,
                entry_key CHAR(64) NOT NULL,
                title VARCHAR(320) NOT NULL,
                content TEXT NOT NULL,
                search_text TEXT NOT NULL,
                metadata_json LONGTEXT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                UNIQUE KEY uniq_research_memory (scope_key, layer, entry_key),
                KEY idx_research_memory_lookup (scope_key, layer, updated_at)
            ) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin
        `);
  })().catch((error) => {
    initPromise = null;
    throw error;
  });

  return initPromise;
};

const saveResearchRun = async (run = {}) => {
  const normalized = normalizeRunRecord(run);
  if (!normalized) return null;

  await ensureResearchTables();
  const db = getPool();
  await db.query(
    `
            INSERT INTO ${RESEARCH_RUNS_TABLE} (
                run_id,
                scope_key,
                status,
                query_text,
                title,
                domain_id,
                output_mode,
                payload
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON DUPLICATE KEY UPDATE
                scope_key = VALUES(scope_key),
                status = VALUES(status),
                query_text = VALUES(query_text),
                title = VALUES(title),
                domain_id = VALUES(domain_id),
                output_mode = VALUES(output_mode),
                payload = VALUES(payload),
                updated_at = CURRENT_TIMESTAMP
        `,
    [
      normalized.runId,
      normalized.scopeKey || null,
      normalized.status,
      normalized.query,
      normalized.title,
      normalized.domain,
      normalized.outputMode,
      serializeJson(normalized.payload),
    ],
  );

  return loadResearchRun(normalized.runId);
};

const loadResearchRun = async (runId) => {
  const normalizedRunId = normalizeText(runId);
  if (!normalizedRunId) return null;

  await ensureResearchTables();
  const db = getPool();
  const [rows] = await db.query(
    `
            SELECT run_id, scope_key, status, query_text, title, domain_id, output_mode, payload, created_at, updated_at
            FROM ${RESEARCH_RUNS_TABLE}
            WHERE run_id = ?
            LIMIT 1
        `,
    [normalizedRunId],
  );

  if (!Array.isArray(rows) || !rows.length) return null;
  return hydrateRunRecord(rows[0] || {});
};

const listResearchRuns = async (scopeKey, options = {}) => {
  const normalizedScopeKey = normalizeScopeKey(scopeKey);
  if (!normalizedScopeKey) return [];

  await ensureResearchTables();
  const db = getPool();
  const limit = clampInteger(options.limit, 1, 50, 20);
  const [rows] = await db.query(
    `
            SELECT run_id, scope_key, status, query_text, title, domain_id, output_mode, payload, created_at, updated_at
            FROM ${RESEARCH_RUNS_TABLE}
            WHERE scope_key = ?
            ORDER BY updated_at DESC
            LIMIT ?
        `,
    [normalizedScopeKey, limit],
  );

  return (Array.isArray(rows) ? rows : [])
    .map((row) => hydrateRunRecord(row))
    .filter(Boolean);
};

const queueResearchControl = async (runId, control = {}) => {
  const normalized = normalizeControlRecord(runId, control);
  if (!normalized) return null;

  await ensureResearchTables();
  const db = getPool();
  const [result] = await db.query(
    `
            INSERT INTO ${RESEARCH_CONTROLS_TABLE} (
                run_id,
                status,
                control_type,
                payload
            )
            VALUES (?, ?, ?, ?)
        `,
    [
      normalized.runId,
      normalized.status,
      normalized.type,
      serializeJson(normalized.payload),
    ],
  );

  return {
    id: Number(result?.insertId || 0),
    runId: normalized.runId,
    status: normalized.status,
    type: normalized.type,
    payload: normalized.payload,
  };
};

const pullPendingResearchControls = async (runId, options = {}) => {
  const normalizedRunId = normalizeText(runId);
  if (!normalizedRunId) return [];

  await ensureResearchTables();
  const db = getPool();
  const limit = clampInteger(options.limit, 1, 24, 8);
  const [rows] = await db.query(
    `
            SELECT id, run_id, status, control_type, payload, created_at, updated_at
            FROM ${RESEARCH_CONTROLS_TABLE}
            WHERE run_id = ? AND status = 'pending'
            ORDER BY created_at ASC, id ASC
            LIMIT ?
        `,
    [normalizedRunId, limit],
  );

  const pending = (Array.isArray(rows) ? rows : []).map((row) => ({
    id: Number(row.id || 0),
    runId: normalizeText(row.run_id),
    status: normalizeControlStatus(row.status),
    type: normalizeText(row.control_type),
    payload: parseJsonObject(row.payload, {}),
    createdAt: coerceUpdatedAt(row.created_at),
    updatedAt: coerceUpdatedAt(row.updated_at),
  }));

  if (!pending.length) return [];

  await db.query(
    `
            UPDATE ${RESEARCH_CONTROLS_TABLE}
            SET status = 'claimed', updated_at = CURRENT_TIMESTAMP
            WHERE id IN (${pending.map(() => "?").join(",")})
        `,
    pending.map((item) => item.id),
  );

  return pending.map((item) => ({
    ...item,
    status: "claimed",
  }));
};

const markResearchControlsStatus = async (
  controlIds = [],
  status = "applied",
) => {
  const normalizedStatus = normalizeControlStatus(status);
  const ids = unique(
    (Array.isArray(controlIds) ? controlIds : [])
      .map((value) => Number(value))
      .filter((value) => Number.isFinite(value) && value > 0),
  );
  if (!ids.length) return 0;

  await ensureResearchTables();
  const db = getPool();
  const [result] = await db.query(
    `
            UPDATE ${RESEARCH_CONTROLS_TABLE}
            SET status = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id IN (${ids.map(() => "?").join(",")})
        `,
    [normalizedStatus, ...ids],
  );
  return Number(result?.affectedRows || 0);
};

const markResearchControlsApplied = async (controlIds = []) =>
  markResearchControlsStatus(controlIds, "applied");
const markResearchControlsIgnored = async (controlIds = []) =>
  markResearchControlsStatus(controlIds, "ignored");

const saveResearchMemoryEntries = async (scopeKey, entries = []) => {
  const normalizedScopeKey = normalizeScopeKey(scopeKey);
  if (!normalizedScopeKey) return 0;

  const normalizedEntries = (Array.isArray(entries) ? entries : [])
    .map((entry) => normalizeMemoryEntry(entry))
    .filter(Boolean);
  if (!normalizedEntries.length) return 0;

  await ensureResearchTables();
  const db = getPool();
  let savedCount = 0;

  for (const entry of normalizedEntries) {
    await db.query(
      `
                INSERT INTO ${RESEARCH_MEMORY_TABLE} (
                    scope_key,
                    layer,
                    entry_key,
                    title,
                    content,
                    search_text,
                    metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON DUPLICATE KEY UPDATE
                    title = VALUES(title),
                    content = VALUES(content),
                    search_text = VALUES(search_text),
                    metadata_json = VALUES(metadata_json),
                    updated_at = CURRENT_TIMESTAMP
            `,
      [
        normalizedScopeKey,
        entry.layer,
        entry.entryKey,
        entry.title,
        entry.content,
        entry.searchText,
        serializeJson(entry.metadata),
      ],
    );
    savedCount += 1;
  }

  await pruneResearchMemoryScope(normalizedScopeKey);
  return savedCount;
};

const searchResearchMemoryDetailed = async (scopeKey, query, options = {}) => {
  const normalizedScopeKey = normalizeScopeKey(scopeKey);
  const normalizedQuery = normalizeText(query);
  if (!normalizedScopeKey || !normalizedQuery) {
    return {
      results: [],
      meta: {
        strategy: "empty",
        layers: [],
      },
    };
  }

  await ensureResearchTables();
  const db = getPool();
  const limit = clampInteger(options.limit, 1, 20, RESEARCH_RESULT_LIMIT);
  const candidateLimit = clampInteger(
    options.candidateLimit || options.candidate_limit,
    8,
    200,
    RESEARCH_MEMORY_CANDIDATE_LIMIT,
  );
  const requestedLayers = Array.isArray(options.layers)
    ? options.layers.map((layer) => normalizeLayer(layer)).filter(Boolean)
    : [];
  const effectiveLayers = requestedLayers.length
    ? requestedLayers
    : [...VALID_MEMORY_LAYERS];
  const perLayerCandidateLimit = clampInteger(
    Math.ceil(candidateLimit / Math.max(1, effectiveLayers.length)),
    4,
    candidateLimit,
    Math.max(
      4,
      Math.ceil(candidateLimit / Math.max(1, effectiveLayers.length)),
    ),
  );
  const rows = (
    await Promise.all(
      effectiveLayers.map(async (layer) => {
        const [layerRows] = await db.query(
          `
                    SELECT layer, title, content, search_text, metadata_json, updated_at
                    FROM ${RESEARCH_MEMORY_TABLE}
                    WHERE scope_key = ? AND layer = ?
                    ORDER BY updated_at DESC
                    LIMIT ?
                `,
          [normalizedScopeKey, layer, perLayerCandidateLimit],
        );
        return Array.isArray(layerRows) ? layerRows : [];
      }),
    )
  ).flat();

  const queryTerms = buildTermSet(normalizedQuery);
  const candidates = normalizeMemoryResults(rows, queryTerms);
  const ranked = dedupeRows(
    [...candidates].sort((left, right) => {
      if (right.score !== left.score) return right.score - left.score;
      return (
        new Date(right.updatedAt || 0).getTime() -
        new Date(left.updatedAt || 0).getTime()
      );
    }),
    limit,
  );

  return {
    results: ranked,
    meta: {
      strategy: queryTerms.size ? "layered_lexical_graph" : "layered_recent",
      layers: requestedLayers,
      searched_layers: effectiveLayers,
      candidate_count: candidates.length,
      graph_edges: ranked.reduce(
        (count, row) =>
          count +
          normalizeRelations(row.metadata?.relations || [], RELATION_LIMIT)
            .length,
        0,
      ),
    },
  };
};

const formatResearchMemoryContext = (results = []) => {
  const rows = (Array.isArray(results) ? results : []).slice(0, 6);
  if (!rows.length) return "";

  return [
    "ResearchMemoryStore context:",
    ...rows.map((row, index) =>
      [
        `[${index + 1}] ${row.layer.toUpperCase()}${row.title ? ` — ${row.title}` : ""}`,
        row.content,
        normalizeRelations(
          row.metadata?.relations || [],
          MEMORY_CONTEXT_RELATION_LIMIT,
        ).length
          ? `Relations: ${normalizeRelations(
              row.metadata?.relations || [],
              MEMORY_CONTEXT_RELATION_LIMIT,
            )
              .map((relation) => `${relation.type}:${relation.target}`)
              .join(", ")}`
          : "",
        normalizeArtifacts(row.metadata?.artifacts || [], 3).length
          ? `Artifacts: ${normalizeArtifacts(row.metadata?.artifacts || [], 3)
              .map((artifact) => `${artifact.type}:${artifact.label}`)
              .join(", ")}`
          : "",
      ].join("\n"),
    ),
    "Use this context only when it materially helps the current research run.",
  ].join("\n\n");
};

const extractPromptPatches = (rows = []) => {
  const promptPatches = [];
  for (const row of Array.isArray(rows) ? rows : []) {
    if (row?.layer === "abstraction") {
      const abstraction = normalizeText(row.content || row.title);
      if (abstraction) promptPatches.push(abstraction);
      const metadataPatches = normalizeStringArray(
        row.metadata?.promptPatches || [],
        8,
      );
      promptPatches.push(...metadataPatches);
      continue;
    }

    if (row?.layer !== "postmortem") continue;
    const parsed = parseJsonObject(row.content, {});
    const patches = Array.isArray(parsed?.prompt_patches_applied)
      ? parsed.prompt_patches_applied
      : [];
    promptPatches.push(
      ...patches.map((value) => normalizeText(value)).filter(Boolean),
    );
  }

  return [...new Set(promptPatches)].slice(0, 12);
};

const findRelevantResearchContext = async (scopeKey, query, options = {}) => {
  const normalizedScopeKey = normalizeScopeKey(scopeKey);
  const normalizedQuery = normalizeText(query);
  if (!normalizedScopeKey || !normalizedQuery) {
    return {
      results: [],
      episodes: [],
      concepts: [],
      abstractions: [],
      postmortems: [],
      promptPatches: [],
      meta: { strategy: "empty", layers: [] },
    };
  }

  const layers =
    Array.isArray(options.layers) && options.layers.length
      ? options.layers
      : ["episode", "concept", "abstraction", "postmortem"];
  const search = await searchResearchMemoryDetailed(
    normalizedScopeKey,
    normalizedQuery,
    {
      limit: options.limit || 8,
      candidateLimit: options.candidateLimit || 128,
      layers,
    },
  );
  const results = Array.isArray(search.results) ? search.results : [];
  const graph = buildMemoryGraphSnapshot(results);

  return {
    results,
    meta: search.meta || {},
    episodes: results
      .filter((row) => row.layer === "episode")
      .map((row) => ({
        runId: normalizeText(row.metadata?.runId || ""),
        query: normalizeText(row.metadata?.query || ""),
        title: normalizeText(row.title),
        summary: normalizeText(row.content),
        updatedAt: row.updatedAt,
      })),
    concepts: results
      .filter((row) => row.layer === "concept")
      .map((row) => ({
        label: normalizeText(row.title),
        summary: normalizeText(row.content),
        ontology: normalizeText(row.metadata?.ontology || ""),
        canonicalId: normalizeText(row.metadata?.canonicalId || ""),
        relations: normalizeRelations(
          row.metadata?.relations || [],
          RELATION_LIMIT,
        ),
        relatedConcepts: normalizeStringArray(
          row.metadata?.relatedConcepts || [],
          16,
        ),
      })),
    abstractions: results
      .filter((row) => row.layer === "abstraction")
      .map((row) => ({
        label: normalizeText(row.title),
        summary: normalizeText(row.content),
        updatedAt: row.updatedAt,
        patternType: normalizeText(row.metadata?.patternType || ""),
      })),
    postmortems: results
      .filter((row) => row.layer === "postmortem")
      .map((row) => ({
        ...parseJsonObject(row.content, {}),
        title: normalizeText(row.title),
        updatedAt: row.updatedAt,
      })),
    graph,
    artifacts: dedupeRows(
      results.flatMap((row) =>
        normalizeArtifacts(row.metadata?.artifacts || [], RELATION_LIMIT).map(
          (artifact) => ({
            layer: row.layer,
            title: artifact.label,
            content: artifact.value || artifact.label,
            type: artifact.type,
          }),
        ),
      ),
      24,
    ).map((artifact) => ({
      type: normalizeText(artifact.type),
      label: normalizeText(artifact.title),
      value: normalizeText(artifact.content),
    })),
    promptPatches: extractPromptPatches(results),
  };
};

const formatResearchContext = (memoryContext = {}) => {
  if (!memoryContext || typeof memoryContext !== "object") return "";
  if (Array.isArray(memoryContext.results)) {
    const blocks = [formatResearchMemoryContext(memoryContext.results)];
    if (
      Array.isArray(memoryContext.graph?.centralConcepts) &&
      memoryContext.graph.centralConcepts.length
    ) {
      blocks.push(
        `Cross-run concepts: ${memoryContext.graph.centralConcepts.slice(0, 6).join(", ")}`,
      );
    }
    if (
      Array.isArray(memoryContext.promptPatches) &&
      memoryContext.promptPatches.length
    ) {
      blocks.push(
        `Carry forward prompt patches:\n${memoryContext.promptPatches
          .slice(0, 4)
          .map((patch) => `- ${patch}`)
          .join("\n")}`,
      );
    }
    return blocks.filter(Boolean).join("\n\n");
  }
  return "";
};

const buildRunPostmortem = (run = {}) => {
  const payload =
    run.payload && typeof run.payload === "object" ? run.payload : run;
  const plan = payload.plan || {};
  const tribunal = payload.tribunal || {};
  const postmortem =
    payload.postmortem && typeof payload.postmortem === "object"
      ? payload.postmortem
      : {};

  return {
    query_type: normalizeText(
      postmortem.query_type || plan.scope?.id || plan.scope?.label || "",
    ),
    domain: normalizeText(
      postmortem.domain || plan.domain?.id || plan.domain?.label || "",
    ),
    phases_that_degraded_score: Array.isArray(
      postmortem.phases_that_degraded_score,
    )
      ? postmortem.phases_that_degraded_score
      : [
          tribunal?.targeted_dimension
            ? `Recursive loop targeted ${tribunal.targeted_dimension}`
            : "",
        ].filter(Boolean),
    prompt_patches_applied: Array.isArray(postmortem.prompt_patches_applied)
      ? postmortem.prompt_patches_applied
      : [],
    final_score_delta: Number(postmortem.final_score_delta || 0),
  };
};

const persistResearchRunArtifacts = async (run = {}) => {
  const normalized = normalizeRunRecord(run);
  if (!normalized) return null;

  const savedRun = await saveResearchRun(normalized);
  const scopeKey = normalizeScopeKey(
    normalized.scopeKey || normalized.payload.scopeKey || "",
  );
  if (scopeKey) {
    const entries = buildResearchMemoryEntriesFromRun({
      ...normalized.payload,
      runId: normalized.runId,
    });
    if (entries.length) {
      await saveResearchMemoryEntries(scopeKey, entries);
    }
  }

  return savedRun;
};

const indexResearchRun = async (scopeKey, run = {}) =>
  saveResearchRun({
    ...(run && typeof run === "object" ? run : {}),
    scopeKey: normalizeScopeKey(scopeKey || run?.scopeKey || ""),
  });

const updateResearchMemoryFromRun = async (scopeKey, run = {}) =>
  persistResearchRunArtifacts({
    ...(run && typeof run === "object" ? run : {}),
    scopeKey: normalizeScopeKey(scopeKey || run?.scopeKey || ""),
  });

module.exports = {
  RESEARCH_RUNS_TABLE,
  RESEARCH_CONTROLS_TABLE,
  RESEARCH_MEMORY_TABLE,
  ensureResearchTables,
  normalizeRunRecord,
  normalizeMemoryEntry,
  normalizeControlRecord,
  saveResearchRun,
  loadResearchRun,
  listResearchRuns,
  queueResearchControl,
  pullPendingResearchControls,
  markResearchControlsApplied,
  markResearchControlsIgnored,
  saveResearchMemoryEntries,
  pruneResearchMemoryScope,
  searchResearchMemoryDetailed,
  formatResearchMemoryContext,
  findRelevantResearchContext,
  formatResearchContext,
  buildResearchMemoryEntriesFromRun,
  buildRunPostmortem,
  persistResearchRunArtifacts,
  indexResearchRun,
  updateResearchMemoryFromRun,
};
