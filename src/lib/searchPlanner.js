const OPERATOR_HEAVY_RE = /\b(site:|filetype:|intitle:|inurl:|after:|before:)\b/i;
const DOCS_RE = /\b(api|sdk|docs?|documentation|guide|install|setup|reference|spec|error|troubleshoot(?:ing)?)\b/i;
const RESEARCH_RE = /\b(rrl|related literature|literature|study|studies|research|paper|papers|journal|benchmark|evaluation|meta-analysis|peer reviewed)\b/i;
const CURRENT_RE = /\b(latest|recent|today|current|new|newest|breaking|updated?|this week|this month|this year|202\d|release|released|price|stock)\b/i;
const COMPARISON_RE = /\b(vs|versus|compare|comparison|best|top|alternative|alternatives)\b/i;
const QUESTION_RE = /^(who|what|when|where|why|how)\b/i;

const normalizeVariant = (value) => String(value || "").replace(/\s+/g, " ").trim();

export function detectQuerySignals(query) {
  const base = normalizeVariant(query);
  return {
    operatorHeavy: OPERATOR_HEAVY_RE.test(base),
    docsIntent: DOCS_RE.test(base),
    researchIntent: RESEARCH_RE.test(base),
    currentIntent: CURRENT_RE.test(base),
    comparisonIntent: COMPARISON_RE.test(base),
    questionIntent: QUESTION_RE.test(base),
  };
}

export function buildSearchQueries(query, options = {}) {
  const base = normalizeVariant(query);
  const maxQueries = Math.max(1, Number(options.maxQueries) || 4);
  if (!base) return [];
  const benchmarkHeavy = /\b(benchmark|benchmarks|evaluation|evaluate|performance)\b/i.test(base);

  const {
    operatorHeavy,
    docsIntent,
    researchIntent,
    currentIntent,
    comparisonIntent,
    questionIntent,
  } = detectQuerySignals(base);

  const variants = [base];
  const primaryVariants = [];
  const secondaryVariants = [];
  const pushPrimary = (...nextVariants) => {
    primaryVariants.push(...nextVariants.map((variant) => normalizeVariant(variant)).filter(Boolean));
  };
  const pushSecondary = (...nextVariants) => {
    secondaryVariants.push(...nextVariants.map((variant) => normalizeVariant(variant)).filter(Boolean));
  };

  if (currentIntent) {
    pushPrimary(`${base} latest`, `${base} official source`);
    pushSecondary(`${base} updated`);
  }

  if (docsIntent) {
    pushPrimary(`${base} official documentation`);
    pushSecondary(`${base} reference`, `${base} troubleshooting`);
  }

  if (researchIntent) {
    if (benchmarkHeavy) {
      pushPrimary(`${base} benchmark analysis`, `${base} peer reviewed research`);
    } else {
      pushPrimary(`${base} peer reviewed research`);
    }
    pushSecondary(`${base} evidence`, `${base} benchmark analysis`);
  }

  if (comparisonIntent) {
    pushPrimary(`${base} benchmark analysis`);
    pushSecondary(`${base} official source`, `${base} evidence`);
  }

  if (questionIntent) {
    pushPrimary(`${base} expert analysis`);
    pushSecondary(`${base} official source`, `${base} evidence`);
  }

  if (operatorHeavy) {
    pushPrimary(`${base} evidence`);
    pushSecondary(`${base} analysis`, `${base} latest`);
  }

  if (
    !operatorHeavy
    && !docsIntent
    && !researchIntent
    && !comparisonIntent
    && !currentIntent
    && !questionIntent
  ) {
    pushPrimary(`${base} overview`, `${base} official source`);
    pushSecondary(`${base} evidence`);
  }

  variants.push(...primaryVariants, ...secondaryVariants);

  return [...new Set(variants.map(normalizeVariant).filter(Boolean))].slice(0, maxQueries);
}
