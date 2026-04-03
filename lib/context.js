const MAX_TURN_SNIPPET_CHARS = 160;
const DEFAULT_MEMORY_CONTEXT_ITEMS = 6;
const DEFAULT_MEMORY_CONTEXT_CHARS = 700;
const DEFAULT_SUMMARY_CHARS = 600;

function normalizeText(text) {
  return typeof text === "string" ? text.replace(/\s+/g, " ").trim() : "";
}

function trimToLength(text, maxChars) {
  if (!text || text.length <= maxChars) {
    return text;
  }

  return `${text.slice(0, Math.max(0, maxChars - 3)).trim()}...`;
}

function getTurnText(turn) {
  if (!Array.isArray(turn?.parts)) {
    return "";
  }

  return normalizeText(
    turn.parts
      .map((part) => (typeof part?.text === "string" ? part.text : ""))
      .filter(Boolean)
      .join(" "),
  );
}

function formatMemoryLabel(entry) {
  return entry?.kind === "preference" ? "Preference" : "Fact";
}

function cleanCapturedValue(value) {
  return trimToLength(normalizeText(value).replace(/[.]+$/g, ""), 100);
}

function dedupeMemoryEntries(entries) {
  const seen = new Set();
  const results = [];

  for (const entry of entries) {
    const content = normalizeText(entry?.content);
    const kind = entry?.kind === "preference" ? "preference" : "fact";
    const key = `${kind}:${content.toLowerCase()}`;

    if (!content || seen.has(key)) {
      continue;
    }

    seen.add(key);
    results.push({
      content,
      kind,
    });
  }

  return results;
}

export function classifyMemoryKind(text) {
  const normalized = normalizeText(text).toLowerCase();

  if (
    /\b(?:favorite|prefer|like|love|dislike|hate|usually|always)\b/.test(
      normalized,
    )
  ) {
    return "preference";
  }

  return "fact";
}

export function extractMemoryCandidates(text) {
  const normalized = normalizeText(text);

  if (!normalized) {
    return [];
  }

  const candidates = [];
  const patterns = [
    {
      regex: /\bmy name is ([^.!?\n]{1,60})/i,
      build: (match) => ({
        content: `Your name is ${cleanCapturedValue(match[1])}.`,
        kind: "fact",
      }),
    },
    {
      regex: /\bcall me ([^.!?\n]{1,60})/i,
      build: (match) => ({
        content: `You like to be called ${cleanCapturedValue(match[1])}.`,
        kind: "preference",
      }),
    },
    {
      regex: /\bi live in ([^.!?\n]{1,80})/i,
      build: (match) => ({
        content: `You live in ${cleanCapturedValue(match[1])}.`,
        kind: "fact",
      }),
    },
    {
      regex: /\bi am from ([^.!?\n]{1,80})/i,
      build: (match) => ({
        content: `You are from ${cleanCapturedValue(match[1])}.`,
        kind: "fact",
      }),
    },
    {
      regex: /\bmy favorite ([a-z][a-z\s]{1,30}) is ([^.!?\n]{1,80})/i,
      build: (match) => ({
        content: `Your favorite ${cleanCapturedValue(match[1])} is ${cleanCapturedValue(match[2])}.`,
        kind: "preference",
      }),
    },
    {
      regex: /\bi (?:really )?(like|love|prefer) ([^.!?\n]{3,100})/i,
      build: (match) => ({
        content: `You ${match[1].toLowerCase()} ${cleanCapturedValue(match[2])}.`,
        kind: "preference",
      }),
    },
    {
      regex: /\bmy pronouns are ([^.!?\n]{1,40})/i,
      build: (match) => ({
        content: `Your pronouns are ${cleanCapturedValue(match[1])}.`,
        kind: "fact",
      }),
    },
    {
      regex: /\bi work as (?:an? )?([^.!?\n]{1,80})/i,
      build: (match) => ({
        content: `You work as ${cleanCapturedValue(match[1])}.`,
        kind: "fact",
      }),
    },
    {
      regex: /\b(?:please )?remember(?: that)? ([^.!?\n]{5,120})/i,
      build: (match) => ({
        content: trimToLength(cleanCapturedValue(match[1]), 110),
        kind: classifyMemoryKind(match[1]),
      }),
    },
  ];

  for (const pattern of patterns) {
    const match = normalized.match(pattern.regex);

    if (!match) {
      continue;
    }

    candidates.push(pattern.build(match));
  }

  return dedupeMemoryEntries(candidates);
}

export function buildMemoryContext(
  memoryEntries,
  {
    maxItems = DEFAULT_MEMORY_CONTEXT_ITEMS,
    maxChars = DEFAULT_MEMORY_CONTEXT_CHARS,
  } = {},
) {
  const selectedEntries = Array.isArray(memoryEntries)
    ? memoryEntries.slice(0, maxItems)
    : [];

  if (selectedEntries.length === 0) {
    return "";
  }

  const lines = selectedEntries.map((entry) => {
    return `- ${formatMemoryLabel(entry)}: ${normalizeText(entry.content)}`;
  });

  return trimToLength(
    `Remembered user details:\n${lines.join("\n")}`,
    maxChars,
  );
}

export function buildRollingConversationSummary(
  history,
  { previousSummary = "", maxChars = DEFAULT_SUMMARY_CHARS } = {},
) {
  const turns = Array.isArray(history) ? history : [];
  const recentLines = turns
    .slice(-6)
    .map((turn) => {
      const text = trimToLength(getTurnText(turn), MAX_TURN_SNIPPET_CHARS);

      if (!text) {
        return "";
      }

      return `${turn.role === "model" ? "Assistant" : "User"}: ${text}`;
    })
    .filter(Boolean);

  const sections = [];
  const trimmedPrevious = trimToLength(normalizeText(previousSummary), 240);

  if (trimmedPrevious) {
    sections.push(`Earlier context: ${trimmedPrevious}`);
  }

  if (recentLines.length > 0) {
    sections.push(`Latest turns:\n${recentLines.join("\n")}`);
  }

  return trimToLength(sections.join("\n\n"), maxChars);
}

export function buildPromptWithPersistentContext({
  prompt,
  imageContext = "",
  conversationSummary = "",
  memoryEntries = [],
  sharedLocation = null,
  maxContextChars = 1400,
  maxMemoryItems = DEFAULT_MEMORY_CONTEXT_ITEMS,
  maxMemoryChars = DEFAULT_MEMORY_CONTEXT_CHARS,
}) {
  const normalizedPrompt = normalizeText(prompt);
  const sections = [];
  const trimmedSummary = trimToLength(
    normalizeText(conversationSummary),
    Math.max(200, Math.floor(maxContextChars * 0.35)),
  );

  if (trimmedSummary) {
    sections.push(`Conversation summary:\n${trimmedSummary}`);
  }

  const memoryContext = buildMemoryContext(memoryEntries, {
    maxItems: maxMemoryItems,
    maxChars: Math.min(maxMemoryChars, Math.floor(maxContextChars * 0.45)),
  });

  if (memoryContext) {
    sections.push(memoryContext);
  }

  if (
    Number.isFinite(sharedLocation?.latitude) &&
    Number.isFinite(sharedLocation?.longitude)
  ) {
    sections.push(
      `Latest shared location:\nLatitude: ${sharedLocation.latitude}\nLongitude: ${sharedLocation.longitude}\nUse this as the user's current location when the request depends on where they are.`,
    );
  }

  const trimmedImageContext = trimToLength(
    normalizeText(imageContext),
    Math.max(200, Math.floor(maxContextChars * 0.35)),
  );

  if (trimmedImageContext) {
    sections.push(
      "Use the following image context only if the user's message appears to refer to the previously sent image.\n\n" +
        `Image context:\n${trimmedImageContext}`,
    );
  }

  const boundedContext = trimToLength(sections.join("\n\n"), maxContextChars);

  if (!boundedContext) {
    return normalizedPrompt;
  }

  return `${boundedContext}\n\nUser message:\n${normalizedPrompt}`;
}
