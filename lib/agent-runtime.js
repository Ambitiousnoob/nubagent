const crypto = require("node:crypto");
const { performance } = require("node:perf_hooks");
const { Tiktoken } = require("js-tiktoken/lite");
const o200kBase = require("js-tiktoken/ranks/o200k_base");

const {
    clamp,
    normalizeUrl,
    extractTitle,
    extractMetaDescription,
    extractCanonical,
    stripHtml,
    extractLinks,
    extractHeadings,
    extractReadableHtml,
    fetchTextResource,
} = require("./web");

const AVAILABLE_MODELS = [
    { id: "stepfun/step-3.5-flash:free", label: "Step 3.5 Flash" },
    { id: "nvidia/nemotron-3-super-120b-a12b:free", label: "Nemotron 3 Super 120B" },
];
const AVAILABLE_MODEL_IDS = new Set(AVAILABLE_MODELS.map((model) => model.id));
const MAX_ITERATIONS = 20;
const MAX_RETRIES = 3;
const MAX_TOOL_CALLS_PER_ITERATION = 3;
const MAX_VERIFICATION_CYCLES = 1;
const MAX_OBSERVATION_CHARS = 2400;
const MAX_COMPLETION_TOKENS = 4096;
const MAX_SUBAGENT_TOKENS = 900;
const MAX_VERIFIER_TOKENS = 1200;
const MAX_CONTEXT_CHARS = 120000;
const MIN_CONTEXT_CHARS = 24000;
const MAX_CONTEXT_MESSAGE_CHARS = 16000;
const MAX_OLDER_CONTEXT_MESSAGE_CHARS = 8000;
const MAX_CONTEXT_MESSAGES = 14;
const WORKING_MEMORY_MESSAGES = 10;
const DEFAULT_PROMPT_TOKEN_LIMIT = 100000;
const DEFAULT_COMPLETION_TOKEN_RESERVE = 6000;
const IMAGE_ATTACHMENT_TOKEN_COST = 1200;
const TOOL_FIREWALL_CHUNK_CHARS = 1800;
const TOOL_FIREWALL_MAX_CHUNKS = 5;
const EPISODIC_SUMMARY_MAX_CHARS = 1400;
const CONTEXT_LENGTH_ERROR_RE = /(context_length_exceeded|max(?:imum)? context length|requested \d+ tokens)/i;
const tokenizer = new Tiktoken(o200kBase);

const createId = () => (
    typeof crypto.randomUUID === "function"
        ? crypto.randomUUID()
        : `${Date.now()}-${Math.random().toString(16).slice(2)}`
);

const normalizeTextBlock = (value) => String(value ?? "").replace(/\u0000/g, "").replace(/\r\n?/g, "\n").trim();

const truncateText = (value, max = MAX_OBSERVATION_CHARS) => {
    const text = String(value ?? "");
    if (text.length <= max) return text;
    const suffix = `\n...[truncated ${text.length - max} chars]`;
    if (max <= suffix.length + 8) return text.slice(0, max);
    return `${text.slice(0, max - suffix.length)}${suffix}`;
};

const roughTokenEstimate = (value) => {
    const text = String(value ?? "");
    if (!text) return 0;
    return Math.max(1, Math.ceil(text.length / 4));
};

const countTextTokens = async (value) => {
    const text = String(value ?? "");
    if (!text) return 0;
    try {
        return tokenizer.encode(text).length;
    } catch {
        return roughTokenEstimate(text);
    }
};

const countContentTokens = async (content) => {
    if (typeof content === "string") return countTextTokens(content);
    if (!Array.isArray(content)) return 0;
    let total = 0;
    for (const part of content) {
        if (part?.type === "text") {
            total += await countTextTokens(part.text || "");
        } else if (part?.type === "image_url") {
            total += IMAGE_ATTACHMENT_TOKEN_COST;
        }
    }
    return total;
};

const countMessagesTokens = async (messages = []) => {
    let total = 2;
    for (const message of Array.isArray(messages) ? messages : []) {
        if (!message) continue;
        total += 4;
        total += await countContentTokens(message.content);
    }
    return total;
};

const getMessageContentText = (content) => {
    if (typeof content === "string") return normalizeTextBlock(content);
    if (!Array.isArray(content)) return "";
    return normalizeTextBlock(
        content
            .filter((part) => part?.type === "text" && typeof part.text === "string")
            .map((part) => part.text)
            .join("\n\n")
    );
};

const messageHasImages = (message) => (
    Array.isArray(message?.content) &&
    message.content.some((part) => part?.type === "image_url" && part?.image_url?.url)
);

const messageHasContent = (message) => {
    if (!message) return false;
    if (getMessageContentText(message.content)) return true;
    return messageHasImages(message);
};

const rebuildMessageContent = (content, text) => {
    if (typeof content === "string" || !Array.isArray(content)) return text;
    const images = content.filter((part) => part?.type === "image_url" && part?.image_url?.url);
    const parts = [];
    if (text) parts.push({ type: "text", text });
    parts.push(...images);
    return parts;
};

const stripAgentArtifacts = (value) => {
    const text = normalizeTextBlock(value);
    if (!text) return "";

    const withoutThoughts = text.replace(/<thought>[\s\S]*?<\/thought>/gi, "").trim();
    const withoutToolCalls = withoutThoughts.replace(/<tool_call>[\s\S]*?<\/tool_call>/gi, "").trim();
    if (withoutToolCalls) return withoutToolCalls;

    const toolNames = [...text.matchAll(/<function=([^>]+)>/gi)]
        .map((match) => match[1]?.trim())
        .filter(Boolean);
    if (toolNames.length) return `Tool calls executed: ${[...new Set(toolNames)].join(", ")}.`;

    return text.replace(/\s+/g, " ").trim();
};

const sanitizeAssistantHistoryText = (value, parsed = null) => {
    const finalAnswer = normalizeTextBlock(parsed?.final_answer || "");
    if (finalAnswer) return finalAnswer;
    return stripAgentArtifacts(getMessageContentText(value) || value);
};

const buildMessageDigest = (message) => {
    if (!message || message.role === "system") return "";
    if (message.error) return `Assistant error: ${truncateText(String(message.error), 220)}`;

    const text = message.role === "assistant"
        ? sanitizeAssistantHistoryText(message.content)
        : normalizeTextBlock(getMessageContentText(message.content));

    if (!text && messageHasImages(message)) {
        return `${message.role === "assistant" ? "Assistant" : "User"} referenced image input.`;
    }
    if (!text) return "";
    return `${message.role === "assistant" ? "Assistant" : "User"}: ${truncateText(text.replace(/\s+/g, " "), 220)}`;
};

const buildEpisodicSummary = (messages = []) => {
    const relevant = (Array.isArray(messages) ? messages : []).filter((message) => (
        message &&
        message.role !== "system" &&
        (messageHasContent(message) || message.error)
    ));
    if (relevant.length <= WORKING_MEMORY_MESSAGES) return "";
    const digests = relevant
        .slice(0, -WORKING_MEMORY_MESSAGES)
        .map(buildMessageDigest)
        .filter(Boolean);
    if (!digests.length) return "";
    return truncateText(`Previously:\n${digests.join("\n")}`, EPISODIC_SUMMARY_MAX_CHARS);
};

const buildWorkingMemoryMessages = (messages = []) => (
    (Array.isArray(messages) ? messages : [])
        .filter((message) => (
            message &&
            message.role !== "system" &&
            (messageHasContent(message) || message.error)
        ))
        .slice(-WORKING_MEMORY_MESSAGES)
        .map((message) => {
            let text = "";
            if (message.error) {
                text = `Previous error: ${message.error}`;
            } else if (message.role === "assistant") {
                text = sanitizeAssistantHistoryText(message.content);
            } else {
                text = normalizeTextBlock(getMessageContentText(message.content));
                if (!text && messageHasImages(message)) text = "Analyze the referenced image input.";
            }
            if (!text) return null;
            return {
                role: message.role === "assistant" ? "assistant" : "user",
                content: truncateText(text, MAX_CONTEXT_MESSAGE_CHARS),
            };
        })
        .filter(Boolean)
);

const buildContextWindow = (messages, { maxChars = MAX_CONTEXT_CHARS, latestUserMaxChars = MAX_CONTEXT_MESSAGE_CHARS } = {}) => {
    const relevant = (Array.isArray(messages) ? messages : []).filter((item) => (
        item &&
        item.role !== "system" &&
        messageHasContent(item)
    ));
    const windowed = relevant.slice(-MAX_CONTEXT_MESSAGES);
    const prepared = [];
    let trimmed = relevant.length > windowed.length;
    let remaining = maxChars;

    for (let index = windowed.length - 1; index >= 0; index -= 1) {
        const item = windowed[index];
        const isRecent = index >= windowed.length - 4;
        const isLatestUser = index === windowed.length - 1 && item.role === "user";
        const perMessageCap = isLatestUser
            ? latestUserMaxChars
            : isRecent
                ? MAX_CONTEXT_MESSAGE_CHARS
                : MAX_OLDER_CONTEXT_MESSAGE_CHARS;
        const rawText = getMessageContentText(item.content);
        const baseText = item.role === "assistant"
            ? stripAgentArtifacts(rawText)
            : normalizeTextBlock(rawText);

        if (!baseText) {
            if (messageHasImages(item)) {
                prepared.unshift({
                    role: item.role === "assistant" ? "assistant" : "user",
                    content: rebuildMessageContent(item.content, ""),
                });
                continue;
            }
            trimmed = true;
            continue;
        }

        let content = baseText;
        if (content.length > perMessageCap) {
            content = truncateText(content, perMessageCap);
            trimmed = true;
        }
        if (content.length > remaining) {
            if (remaining < 800) {
                trimmed = true;
                continue;
            }
            content = truncateText(content, remaining);
            trimmed = true;
        }

        prepared.unshift({
            role: item.role === "assistant" ? "assistant" : "user",
            content: rebuildMessageContent(item.content, content),
        });
        remaining -= content.length;
        if (remaining <= 0) break;
    }

    return { messages: prepared, trimmed };
};

const buildPreparedModelMessages = (messages, { maxChars = MAX_CONTEXT_CHARS, latestUserMaxChars = MAX_CONTEXT_MESSAGE_CHARS } = {}) => {
    const systemMessages = [];
    const conversational = [];

    for (const message of Array.isArray(messages) ? messages : []) {
        if (!message || !messageHasContent(message)) continue;
        if (message.role === "system") {
            systemMessages.push({ role: "system", content: getMessageContentText(message.content).trim() });
        } else {
            conversational.push(message);
        }
    }

    const window = buildContextWindow(conversational, { maxChars, latestUserMaxChars });
    return {
        messages: [...systemMessages, ...window.messages],
        trimmed: window.trimmed,
    };
};

const compactSupplementalSystemMessages = (messages = [], cap = 1500) => {
    let firstSystem = true;
    return (Array.isArray(messages) ? messages : [])
        .map((message) => {
            if (!message || message.role !== "system") return message;
            const text = getMessageContentText(message.content);
            if (!text) return null;
            const next = {
                role: "system",
                content: truncateText(text, firstSystem ? Math.max(cap * 2, 3200) : cap),
            };
            firstSystem = false;
            return next;
        })
        .filter(Boolean);
};

const prepareMessagesForDispatch = async (messages, {
    chain = [],
    maxTokens = MAX_COMPLETION_TOKENS,
    maxContextChars = MAX_CONTEXT_CHARS,
    latestUserMaxChars = MAX_CONTEXT_MESSAGE_CHARS,
} = {}) => {
    const primaryModel = Array.isArray(chain) && chain.length ? chain[0] : AVAILABLE_MODELS[0];
    const promptLimit = DEFAULT_PROMPT_TOKEN_LIMIT;
    const completionReserve = Math.max(maxTokens, DEFAULT_COMPLETION_TOKEN_RESERVE);
    const promptBudget = Math.max(3000, promptLimit - completionReserve);

    let sourceMessages = Array.isArray(messages) ? messages : [];
    let charBudget = maxContextChars;
    let latestBudget = latestUserMaxChars;
    let prepared = buildPreparedModelMessages(sourceMessages, {
        maxChars: charBudget,
        latestUserMaxChars: latestBudget,
    });
    let tokenCount = await countMessagesTokens(prepared.messages);
    let trimmed = prepared.trimmed;
    let attempts = 0;

    while (tokenCount > promptBudget && attempts < 7) {
        attempts += 1;
        trimmed = true;

        if (attempts === 4) {
            sourceMessages = compactSupplementalSystemMessages(sourceMessages, 1200);
        }

        charBudget = Math.max(Math.floor(maxContextChars * 0.18), Math.floor(charBudget * 0.72));
        latestBudget = Math.max(1200, Math.min(latestBudget, Math.floor(charBudget * 0.7)));
        prepared = buildPreparedModelMessages(sourceMessages, {
            maxChars: charBudget,
            latestUserMaxChars: latestBudget,
        });
        tokenCount = await countMessagesTokens(prepared.messages);
    }

    return {
        messages: prepared.messages,
        tokenCount,
        promptBudget,
        promptLimit,
        completionReserve,
        trimmed,
    };
};

const getXmlTagValue = (text, tag) => (
    text.match(new RegExp(`<${tag}>([\\s\\S]*?)<\\/${tag}>`, "i"))?.[1]?.trim() || ""
);

const getXmlSectionItems = (text, sectionTag) => {
    const section = getXmlTagValue(text, sectionTag);
    return [...section.matchAll(/<item>([\s\S]*?)<\/item>/gi)].map((match) => match[1].trim()).filter(Boolean);
};

const formatPlanForSystem = (plan) => {
    if (!plan) return "";
    const lines = ["Execution plan:"];
    if (plan.goal) lines.push(`Goal: ${plan.goal}`);
    if (plan.complexity) lines.push(`Complexity: ${plan.complexity}`);
    if (plan.objectives?.length) lines.push(`Objectives: ${plan.objectives.join(" | ")}`);
    if (plan.constraints?.length) lines.push(`Constraints: ${plan.constraints.join(" | ")}`);
    if (plan.success?.length) lines.push(`Success criteria: ${plan.success.join(" | ")}`);
    if (plan.tooling?.length) lines.push(`Preferred tooling: ${plan.tooling.join(" | ")}`);
    if (plan.notes) lines.push(`Manager notes: ${plan.notes}`);
    return lines.join("\n");
};

const formatMemoryContext = (memoryStore) => {
    const entries = Object.entries(memoryStore || {});
    if (!entries.length) return "";
    const lines = ["Session memory:"];
    entries.forEach(([key, value]) => lines.push(`- ${key}: ${value?.value || ""}`));
    lines.push("Use this only when relevant and do not invent extra memory.");
    return lines.join("\n");
};

const formatStepsForVerifier = (steps = []) => {
    const actionSteps = steps.filter((step) => step.type === "action").slice(-8);
    if (!actionSteps.length) return "No tool evidence was collected.";
    return actionSteps.map((step, index) => {
        const input = step.input && Object.keys(step.input).length ? truncateText(JSON.stringify(step.input), 320) : "{}";
        const observation = truncateText(String(step.observation || "").replace(/\s+/g, " ").trim(), 900);
        return `Step ${index + 1}: ${step.action}\nInput: ${input}\nObservation: ${observation}`;
    }).join("\n\n");
};

const formatObservationBundle = (actions, wasTrimmed = false) => {
    const lines = ["Tool results:"];
    actions.forEach((item, index) => {
        lines.push(`${index + 1}. ${item.action}`);
        if (item.input && Object.keys(item.input).length) {
            lines.push(`Input: ${truncateText(JSON.stringify(item.input), 320)}`);
        }
        const firewallNotes = [];
        if (item.firewallMeta?.wasJson) firewallNotes.push("JSON pruned");
        if (item.firewallMeta?.wasHtml) firewallNotes.push("HTML cleaned");
        if (item.firewallMeta?.chunkCount > 1) firewallNotes.push(`showing chunk 1 of ${item.firewallMeta.chunkCount}`);
        if (firewallNotes.length) lines.push(`Firewall: ${firewallNotes.join(" | ")}`);
        lines.push(`Observation:\n${truncateText(item.observation, MAX_OBSERVATION_CHARS)}`);
        if (index < actions.length - 1) lines.push("---");
    });
    if (wasTrimmed) {
        lines.push("");
        lines.push(`Only the first ${MAX_TOOL_CALLS_PER_ITERATION} tool calls were executed from this batch.`);
    }
    lines.push("");
    lines.push("Continue. Use more tools if needed, or provide the final answer if the task is complete.");
    return lines.join("\n");
};

const buildPlannerPrompt = (desc) => `You are Execution Planner, a specialist sub-agent.

Your only job is to create a concise execution blueprint for the next agent pass.

AVAILABLE TOOLS:
${desc}

Respond using this exact XML format:
<plan>
<goal>one sentence goal</goal>
<complexity>low|medium|high</complexity>
<objectives>
<item>objective 1</item>
</objectives>
<constraints>
<item>constraint 1</item>
</constraints>
<success>
<item>success criterion 1</item>
</success>
<tooling>
<item>tool_name: why it helps</item>
</tooling>
<notes>short execution note</notes>
</plan>

Rules:
- Do not solve the user request.
- Keep each list short and high signal.
- Mention evidence requirements when facts or web content matter.
- Prefer tool-supported plans over pure reasoning.`;

const parsePlannerResponse = (text) => ({
    goal: getXmlTagValue(text, "goal"),
    complexity: getXmlTagValue(text, "complexity") || "medium",
    objectives: getXmlSectionItems(text, "objectives"),
    constraints: getXmlSectionItems(text, "constraints"),
    success: getXmlSectionItems(text, "success"),
    tooling: getXmlSectionItems(text, "tooling"),
    notes: getXmlTagValue(text, "notes"),
});

const buildVerifierPrompt = ({ query, plan, steps, answer }) => `You are Answer Verifier, a specialist sub-agent.

Your only job is to decide whether the draft answer is ready for the user.

USER REQUEST:
${query}

EXECUTION PLAN:
${plan ? formatPlanForSystem(plan) : "No structured plan was available."}

RECENT TOOL EVIDENCE:
${formatStepsForVerifier(steps)}

DRAFT ANSWER:
${answer}

Respond using this exact XML format:
<verification>
<status>pass|revise</status>
<summary>one sentence</summary>
<checks>
<item>what is already good</item>
</checks>
<missing>
<item>what still needs work</item>
</missing>
<next_action>single best next action</next_action>
</verification>

Rules:
- Use "pass" only if the draft is complete, grounded, and directly answers the request.
- Use "revise" if there is a material gap, unsupported claim, or missing execution step.
- Keep the response concise and execution-focused.`;

const parseVerifierResponse = (text) => {
    const status = (getXmlTagValue(text, "status") || "pass").toLowerCase();
    return {
        status: status === "revise" ? "revise" : "pass",
        summary: getXmlTagValue(text, "summary"),
        checks: getXmlSectionItems(text, "checks"),
        missing: getXmlSectionItems(text, "missing"),
        nextAction: getXmlTagValue(text, "next_action"),
    };
};

const formatVerifierFeedback = (report) => {
    const lines = [
        `Verifier status: ${report.status}`,
        `Summary: ${report.summary || "No summary provided."}`,
    ];
    if (report.checks?.length) lines.push(`Checks: ${report.checks.join(" | ")}`);
    if (report.missing?.length) lines.push(`Missing: ${report.missing.join(" | ")}`);
    if (report.nextAction) lines.push(`Next action: ${report.nextAction}`);
    return lines.join("\n");
};

const buildRouterPrompt = (desc) => `You are Tool Router, a specialist sub-agent.

Your only job is to choose the best starting tool for the request.

AVAILABLE TOOLS:
${desc}

Respond using this exact XML format:
<analysis>short reasoning</analysis>
<route>
<primary>tool_name_or_none</primary>
<secondary>comma,separated,backup,tools</secondary>
<reason>one sentence</reason>
</route>

Rules:
- Pick one primary tool or "none".
- Prefer web_read for a single page, web_crawl for multi-page docs/sites, web_fetch only as a lightweight fallback.
- Prefer specialized tools over generic ones.
- Do not answer the user request directly.
- Keep the analysis short.`;

const parseRouterResponse = (text) => {
    const analysis = text.match(/<analysis>([\s\S]*?)<\/analysis>/)?.[1]?.trim() || "";
    const primary = text.match(/<primary>([\s\S]*?)<\/primary>/)?.[1]?.trim() || "none";
    const secondaryRaw = text.match(/<secondary>([\s\S]*?)<\/secondary>/)?.[1]?.trim() || "";
    const reason = text.match(/<reason>([\s\S]*?)<\/reason>/)?.[1]?.trim() || "";
    const secondary = secondaryRaw ? secondaryRaw.split(",").map((item) => item.trim()).filter(Boolean) : [];
    return { analysis, primary, secondary, reason };
};

const buildSystemPrompt = (count, desc) => `You are an exceptionally capable autonomous AI agent.

PHILOSOPHY:
- Think deeply before acting. Break complex problems into subtasks.
- Use multiple tools in sequence when needed.
- When several tool calls are independent, batch up to ${MAX_TOOL_CALLS_PER_ITERATION} <tool_call> blocks in one response.
- Cross-verify important information using different tools.
- Be precise, comprehensive, and honest about uncertainty.
- Respect the manager plan, router guidance, memory context, and verifier feedback when they are provided.
- Preserve evidence from web tools and include source URLs when that helps the user.
- When you have enough information, synthesize a clear final answer.

AVAILABLE TOOLS (${count} total):
${desc}

STRICT XML FORMAT - you MUST use the following XML tags:

To think about what to do next:
<thought>
I need to calculate 25 * 4, I will use the calculator tool.
</thought>

To use a tool (must follow a <thought>):
<tool_call>
<function=tool_name>
<parameter=param_name>value</parameter>
</function>
</tool_call>

To give your final answer to the user (must follow a <thought>):
<thought>
I have the result. I will tell the user.
</thought>
The answer is 100.

RULES:
1. ALWAYS write a <thought> before taking an action or answering.
2. Use tools for any factual, computational, or data-retrieval need.
3. If a tool errors, adapt: try different params or tools.
4. Before finalizing, make sure the critical objectives are satisfied or explicitly state what is still missing.
5. Final answers should be complete, structured, and use markdown.
6. Do NOT use JSON for tool calls, ONLY use the exact <tool_call> XML format.`;

const parseXMLAgentResponse = (text) => {
    const response = { thought: "", action: null, action_input: {}, actions: [], final_answer: "" };
    const thoughtMatches = [...String(text).matchAll(/<thought>([\s\S]*?)(?:<\/thought>|$)/g)];
    if (thoughtMatches.length) response.thought = thoughtMatches[thoughtMatches.length - 1][1].trim();

    const toolCallMatches = [...String(text).matchAll(/<tool_call>([\s\S]*?)(?:<\/tool_call>|$)/g)];
    if (toolCallMatches.length) {
        toolCallMatches.forEach((toolCallMatch) => {
            const fnMatch = toolCallMatch[1].match(/<function=([^>]+)>([\s\S]*?)(?:<\/function>|$)/);
            if (!fnMatch) return;
            const actionName = fnMatch[1].trim();
            const paramsStr = fnMatch[2];
            const actionInput = {};
            const paramRegex = /<parameter=([^>]+)>([\s\S]*?)<\/parameter>/g;
            let match;
            while ((match = paramRegex.exec(paramsStr)) !== null) {
                let value = match[2].trim();
                try {
                    value = JSON.parse(value);
                } catch {
                    // Keep string value.
                }
                actionInput[match[1].trim()] = value;
            }
            response.actions.push({ name: actionName, input: actionInput });
        });
        if (response.actions.length) {
            response.action = response.actions[0].name;
            response.action_input = response.actions[0].input;
        }
        return response;
    }

    const finalSplit = String(text).split(/<\/thought>/);
    if (finalSplit.length > 1) {
        response.final_answer = finalSplit.slice(1).join("</thought>").trim();
    } else if (!thoughtMatches.length && String(text).trim()) {
        response.final_answer = String(text).trim();
    }
    return response;
};

const isContextLengthError = (error) => CONTEXT_LENGTH_ERROR_RE.test(String(error?.message || error || ""));

const splitTextIntoChunks = (value, maxChars = TOOL_FIREWALL_CHUNK_CHARS, overlap = 120) => {
    const text = normalizeTextBlock(value);
    if (!text) return [];
    if (text.length <= maxChars) return [text];

    const chunks = [];
    let start = 0;
    while (start < text.length) {
        let end = Math.min(text.length, start + maxChars);
        if (end < text.length) {
            const window = text.slice(start, end);
            const preferredBreak = Math.max(
                window.lastIndexOf("\n\n"),
                window.lastIndexOf("\n"),
                window.lastIndexOf(". "),
                window.lastIndexOf(" ")
            );
            if (preferredBreak > Math.floor(maxChars * 0.55)) {
                end = start + preferredBreak + 1;
            }
        }
        const chunk = normalizeTextBlock(text.slice(start, end));
        if (chunk) chunks.push(chunk);
        if (end >= text.length) break;
        start = Math.max(end - overlap, start + 1);
    }
    return chunks;
};

const stripHtmlNoise = (value) => normalizeTextBlock(
    String(value ?? "")
        .replace(/<script[\s\S]*?<\/script>/gi, " ")
        .replace(/<style[\s\S]*?<\/style>/gi, " ")
        .replace(/<nav[\s\S]*?<\/nav>/gi, " ")
        .replace(/<header[\s\S]*?<\/header>/gi, " ")
        .replace(/<footer[\s\S]*?<\/footer>/gi, " ")
        .replace(/<aside[\s\S]*?<\/aside>/gi, " ")
        .replace(/<[^>]+>/g, " ")
);

const pruneJsonValue = (value, depth = 0) => {
    if (value == null) return null;
    if (typeof value === "string") {
        const text = normalizeTextBlock(value);
        return text ? truncateText(text, 700) : null;
    }
    if (typeof value === "number" || typeof value === "boolean") return value;
    if (Array.isArray(value)) {
        const cleaned = value.map((item) => pruneJsonValue(item, depth + 1)).filter((item) => item != null);
        if (!cleaned.length) return null;
        if (cleaned.length > 10) return [...cleaned.slice(0, 10), `...[${cleaned.length - 10} more items]`];
        return cleaned;
    }
    if (typeof value === "object") {
        const entries = Object.entries(value)
            .filter(([key]) => !/^(raw|html|headers|metadata|analytics|tracking|debug|trace)$/i.test(key))
            .map(([key, item]) => [key, pruneJsonValue(item, depth + 1)])
            .filter(([, item]) => item != null);
        if (!entries.length) return null;
        return Object.fromEntries((depth > 1 ? entries.slice(0, 12) : entries.slice(0, 20)));
    }
    return truncateText(String(value), 700);
};

const formatToolChunkLabel = (actionName, chunkIndex, totalChunks) => `${actionName} chunk ${chunkIndex} of ${totalChunks}`;

const firewallToolOutput = (actionName, rawOutput) => {
    let cleanedValue = rawOutput;
    let wasJson = false;
    let wasHtml = false;

    if (cleanedValue != null && typeof cleanedValue === "object") {
        cleanedValue = pruneJsonValue(cleanedValue);
        wasJson = true;
    }

    if (typeof cleanedValue === "string") {
        const trimmed = cleanedValue.trim();
        if (!wasJson && (trimmed.startsWith("{") || trimmed.startsWith("["))) {
            try {
                cleanedValue = pruneJsonValue(JSON.parse(trimmed));
                wasJson = true;
            } catch {
                cleanedValue = trimmed;
            }
        }
    }

    let cleanedText = "";
    if (typeof cleanedValue === "string") {
        if (/<[a-z][\s\S]*>/i.test(cleanedValue)) {
            cleanedText = stripHtmlNoise(cleanedValue);
            wasHtml = true;
        } else {
            cleanedText = normalizeTextBlock(cleanedValue);
        }
    } else {
        try {
            cleanedText = JSON.stringify(cleanedValue, null, 2);
        } catch {
            cleanedText = String(cleanedValue ?? "");
        }
    }

    cleanedText = normalizeTextBlock(cleanedText);
    if (!cleanedText) cleanedText = "Tool returned no readable content.";

    const allChunks = splitTextIntoChunks(cleanedText, TOOL_FIREWALL_CHUNK_CHARS, 120);
    const chunkCount = allChunks.length || 1;
    const storedChunks = (allChunks.length ? allChunks : [cleanedText]).slice(0, TOOL_FIREWALL_MAX_CHUNKS);
    const omittedChunkCount = Math.max(0, chunkCount - storedChunks.length);
    let agentText = storedChunks[0] || cleanedText;

    if (chunkCount > 1) {
        agentText = `${formatToolChunkLabel(actionName, 1, chunkCount)}\n${agentText}\n\n[Firewall note: only chunk 1 of ${chunkCount} is shown to the agent. ${Math.max(0, storedChunks.length - 1)} more sanitized chunk(s) were kept for context${omittedChunkCount ? `; ${omittedChunkCount} chunk(s) were dropped.` : "."}]`;
    }

    return {
        agentText,
        chunkCount,
        tokenEstimate: roughTokenEstimate(cleanedText),
        wasJson,
        wasHtml,
    };
};

const formatReadReport = (payload, { includeLinks = true } = {}) => {
    const lines = [
        payload.title || payload.finalUrl,
        `URL: ${payload.finalUrl}`,
        `Content type: ${payload.contentType}`,
    ];
    if (payload.canonicalUrl && payload.canonicalUrl !== payload.finalUrl) lines.push(`Canonical: ${payload.canonicalUrl}`);
    if (payload.description) lines.push(`Description: ${payload.description}`);
    lines.push(`Words: ${payload.wordCount}`);
    if (payload.via) lines.push(`Fetched via: ${payload.via}`);
    if (payload.headings?.length) {
        lines.push("");
        lines.push("Headings:");
        payload.headings.forEach((item) => lines.push(`- H${item.level}: ${item.text}`));
    }
    lines.push("");
    lines.push("Content:");
    lines.push(payload.content || "No readable content extracted.");
    if (payload.contentTruncated) {
        lines.push("");
        lines.push("[Content truncated]");
    }
    if (includeLinks && payload.links?.length) {
        lines.push("");
        lines.push("Links:");
        payload.links.forEach((link) => lines.push(`- ${link}`));
    }
    return lines.join("\n").trim();
};

const formatCrawlReport = (payload) => {
    if (!payload?.pages?.length) {
        const errorText = payload?.errors?.length
            ? `\nErrors:\n${payload.errors.map((item) => `- ${item.url}: ${item.error}`).join("\n")}`
            : "";
        return `No readable pages were crawled from ${payload?.startUrl || "the requested URL"}.${errorText}`;
    }

    const lines = [
        `Crawl start: ${payload.startUrl}`,
        `Pages read: ${payload.pages.length}/${payload.maxPages} | Max depth: ${payload.maxDepth} | Same origin: ${payload.sameOrigin ? "yes" : "no"}`,
        `Discovered URLs: ${payload.discoveredCount}${payload.errors?.length ? ` | Errors: ${payload.errors.length}` : ""}`,
        "",
    ];

    payload.pages.forEach((page, index) => {
        lines.push(`${index + 1}. ${page.title || page.url}`);
        lines.push(`URL: ${page.url}`);
        lines.push(`Depth: ${page.depth} | Words: ${page.words} | Links found: ${page.linksDiscovered}${page.via ? ` | Via: ${page.via}` : ""}`);
        if (page.queuedLinks?.length) lines.push(`Next links: ${page.queuedLinks.slice(0, 5).join(", ")}`);
        lines.push(page.content || "(No readable text extracted)");
        if (page.contentTruncated) lines.push("[Content truncated]");
        lines.push("");
    });

    if (payload.errors?.length) {
        lines.push("Errors:");
        payload.errors.forEach((item) => lines.push(`- ${item.url}: ${item.error}`));
    }

    return lines.join("\n").trim();
};

const readWebPayload = async ({ url, maxChars = 3200, mode = "article", includeLinks = true, timeoutMs = 12000 }) => {
    const sourceUrl = normalizeUrl(url);
    if (!sourceUrl) throw new Error("A valid absolute http(s) url is required.");

    const resource = await fetchTextResource(sourceUrl, clamp(Number(timeoutMs) || 12000, 2000, 30000));
    const finalUrl = normalizeUrl(resource.finalUrl) || sourceUrl;
    let title = finalUrl;
    let description = "";
    let canonicalUrl = "";
    let headings = [];
    let links = [];
    let contentSource = resource.text.trim();

    if (resource.isHtml) {
        title = extractTitle(resource.text) || finalUrl;
        description = extractMetaDescription(resource.text);
        canonicalUrl = extractCanonical(resource.text, finalUrl) || "";
        headings = extractHeadings(resource.text);
        links = includeLinks ? extractLinks(resource.text, finalUrl).slice(0, 20) : [];

        if (mode === "outline") {
            contentSource = [
                description ? `Description: ${description}` : "",
                ...headings.map((item) => `${"#".repeat(Math.min(item.level, 3))} ${item.text}`),
            ].filter(Boolean).join("\n");
        } else {
            const readableHtml = mode === "full" ? resource.text : extractReadableHtml(resource.text);
            contentSource = stripHtml(readableHtml);
        }
    } else if (/application\/json/i.test(resource.contentType)) {
        title = resource.titleHint || finalUrl.split("/").pop() || finalUrl;
        try {
            contentSource = JSON.stringify(JSON.parse(resource.text), null, 2);
        } catch {
            contentSource = resource.text.trim();
        }
    } else {
        title = resource.titleHint || finalUrl.split("/").pop() || finalUrl;
        contentSource = resource.text.trim();
    }

    const content = contentSource.slice(0, clamp(Number(maxChars) || 3200, 200, 12000));
    return {
        sourceUrl,
        finalUrl,
        title,
        description,
        canonicalUrl,
        contentType: resource.contentType || "text/plain",
        mode,
        headings,
        links,
        content,
        contentTruncated: contentSource.length > content.length,
        wordCount: contentSource ? contentSource.split(/\s+/).filter(Boolean).length : 0,
        via: resource.via || "direct",
    };
};

const normalizePatterns = (patterns) => (
    Array.isArray(patterns)
        ? patterns.slice(0, 12).map((value) => String(value).trim().toLowerCase()).filter(Boolean)
        : []
);

const matchesPatterns = (value, patterns) => {
    if (!patterns.length) return true;
    const normalizedValue = String(value).toLowerCase();
    return patterns.some((pattern) => normalizedValue.includes(pattern));
};

const crawlWebPayload = async ({
    url,
    maxPages = 6,
    maxDepth = 1,
    sameOrigin = true,
    maxCharsPerPage = 1800,
    includePatterns = [],
    excludePatterns = [],
    timeoutMs = 12000,
}) => {
    const startUrl = normalizeUrl(url);
    if (!startUrl) throw new Error("A valid absolute http(s) url is required.");

    const queue = [{ url: startUrl, depth: 0 }];
    const visited = new Set();
    const discovered = new Set([startUrl]);
    const pages = [];
    const errors = [];
    const seedOrigin = new URL(startUrl).origin;
    const cappedInclude = normalizePatterns(includePatterns);
    const cappedExclude = normalizePatterns(excludePatterns);
    const cappedPages = clamp(Number(maxPages) || 6, 1, 12);
    const cappedDepth = clamp(Number(maxDepth) || 1, 0, 3);
    const cappedChars = clamp(Number(maxCharsPerPage) || 1800, 200, 5000);
    const cappedTimeout = clamp(Number(timeoutMs) || 12000, 2000, 30000);

    while (queue.length && pages.length < cappedPages) {
        const current = queue.shift();
        if (!current || visited.has(current.url)) continue;
        visited.add(current.url);

        try {
            const page = await fetchTextResource(current.url, cappedTimeout);
            const resolvedUrl = normalizeUrl(page.finalUrl) || current.url;
            const title = page.isHtml ? (extractTitle(page.text) || resolvedUrl) : (page.titleHint || resolvedUrl.split("/").pop() || resolvedUrl);
            const text = page.isHtml ? stripHtml(extractReadableHtml(page.text)) : page.text.trim();
            const words = text ? text.split(/\s+/).filter(Boolean).length : 0;
            const links = page.isHtml ? extractLinks(page.text, resolvedUrl) : [];
            const queuedLinks = [];

            if (current.depth < cappedDepth) {
                for (const link of links) {
                    if (visited.has(link) || discovered.has(link)) continue;
                    if (sameOrigin !== false && new URL(link).origin !== seedOrigin) continue;
                    if (cappedInclude.length && !matchesPatterns(link, cappedInclude)) continue;
                    if (cappedExclude.length && matchesPatterns(link, cappedExclude)) continue;
                    discovered.add(link);
                    queue.push({ url: link, depth: current.depth + 1 });
                    queuedLinks.push(link);
                }
            }

            pages.push({
                url: resolvedUrl,
                title,
                depth: current.depth,
                words,
                linksDiscovered: links.length,
                queuedLinks: queuedLinks.slice(0, 10),
                content: text.slice(0, cappedChars),
                contentTruncated: text.length > cappedChars,
                via: page.via || "direct",
            });
        } catch (error) {
            errors.push({
                url: current.url,
                depth: current.depth,
                error: error?.name === "AbortError" ? `Timed out after ${cappedTimeout}ms` : (error?.message || "Unknown fetch error"),
            });
        }
    }

    return {
        startUrl,
        maxPages: cappedPages,
        maxDepth: cappedDepth,
        sameOrigin: sameOrigin !== false,
        discoveredCount: discovered.size,
        pages,
        errors,
    };
};

const createTools = ({ getMemoryStore, setMemoryStore }) => {
    const timerStore = {};
    return {
        calculator: {
            params: "{ expression: string }",
            description: "Evaluate math expressions including complex arithmetic, percentages, powers",
            execute: async ({ expression }) => {
                try {
                    const safe = String(expression || "").replace(/[^0-9+\-*/().,\s%^eE]/g, "");
                    return `= ${Function('"use strict"; return (' + safe + ")")()}`;
                } catch (error) {
                    return `Math error: ${error.message}`;
                }
            },
        },
        unit_convert: {
            params: "{ value: number, from: string, to: string }",
            description: "Convert between common units",
            execute: async ({ value, from, to }) => {
                const conversions = { km_miles: 0.621371, miles_km: 1.60934, kg_lbs: 2.20462, lbs_kg: 0.453592, m_ft: 3.28084, ft_m: 0.3048 };
                const key = `${String(from || "").toLowerCase()}_${String(to || "").toLowerCase()}`;
                if (key === "celsius_fahrenheit") return `${value}°C = ${(value * 9 / 5 + 32).toFixed(2)}°F`;
                if (key === "fahrenheit_celsius") return `${value}°F = ${((value - 32) * 5 / 9).toFixed(2)}°C`;
                return conversions[key] ? `${value} ${from} = ${(value * conversions[key]).toFixed(4)} ${to}` : `Unknown: ${from} -> ${to}`;
            },
        },
        datetime: {
            params: "{ action?: 'now'|'diff', date1?: string, date2?: string, tz?: string }",
            description: "Get current date/time or calculate date differences",
            execute: async ({ action = "now", date1, date2, tz = "UTC" } = {}) => {
                if (action === "diff" && date1 && date2) {
                    const days = Math.floor(Math.abs(new Date(date2) - new Date(date1)) / 86400000);
                    return `Difference: ${days} days / ${Math.floor(days / 7)} weeks / ${Math.floor(days / 30.44)} months`;
                }
                const now = new Date();
                const formatter = new Intl.DateTimeFormat("en-US", {
                    timeZone: tz,
                    hour12: false,
                    year: "numeric",
                    month: "2-digit",
                    day: "2-digit",
                    hour: "2-digit",
                    minute: "2-digit",
                    second: "2-digit",
                });
                let formatted;
                try {
                    formatted = formatter.format(now);
                } catch {
                    formatted = new Intl.DateTimeFormat("en-US", { timeZone: "UTC", hour12: false, year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", second: "2-digit" }).format(now);
                    tz = "UTC (fallback)";
                }
                return `Now (${tz}): ${formatted}\nISO (UTC): ${now.toISOString()}`;
            },
        },
        text_analysis: {
            params: "{ text: string }",
            description: "Analyze text: word count, readability, frequency",
            execute: async ({ text }) => {
                if (!text) return "No text provided";
                const words = String(text).trim().split(/\s+/).filter(Boolean);
                const sentences = String(text).split(/[.!?]+/).filter(Boolean);
                const freq = {};
                words.forEach((word) => {
                    const key = word.toLowerCase().replace(/[^a-z]/g, "");
                    if (key.length > 3) freq[key] = (freq[key] || 0) + 1;
                });
                const top = Object.entries(freq).sort((a, b) => b[1] - a[1]).slice(0, 5).map(([word, count]) => `${word}(${count})`).join(", ");
                const readability = Math.max(0, Math.min(100, 206.835 - 1.015 * (words.length / Math.max(sentences.length, 1)) - 84.6 * (words.reduce((sum, word) => sum + word.length, 0) / Math.max(words.length, 1) / 4)));
                return JSON.stringify({
                    words: words.length,
                    sentences: sentences.length,
                    chars: String(text).length,
                    readability: readability > 70 ? "Easy" : readability > 50 ? "Standard" : "Complex",
                    topWords: top,
                }, null, 2);
            },
        },
        memory_write: {
            params: "{ key: string, value: string }",
            description: "Store information during this server-side run",
            execute: async ({ key, value }) => {
                const next = { ...getMemoryStore(), [key]: { value, timestamp: new Date().toISOString() } };
                setMemoryStore(next);
                return `Stored "${key}".`;
            },
        },
        memory_read: {
            params: "{ key?: string }",
            description: "Retrieve stored memory by key, or list all stored keys",
            execute: async ({ key } = {}) => {
                const memoryStore = getMemoryStore();
                if (key) {
                    const entry = memoryStore[key];
                    return entry ? `"${key}": "${entry.value}"` : `Key "${key}" not found.`;
                }
                return Object.keys(memoryStore).length
                    ? JSON.stringify(Object.fromEntries(Object.entries(memoryStore).map(([k, v]) => [k, v.value])), null, 2)
                    : "Memory empty.";
            },
        },
        memory_clear: {
            params: "{ key?: string }",
            description: "Clear specific memory key or all stored memory",
            execute: async ({ key } = {}) => {
                if (key) {
                    const next = { ...getMemoryStore() };
                    delete next[key];
                    setMemoryStore(next);
                    return `Deleted "${key}".`;
                }
                setMemoryStore({});
                return "Cleared all memory.";
            },
        },
        code_run: {
            params: "{ language: string, code: string }",
            description: "Execute Python, Javascript, C++, or Node code via Piston",
            execute: async ({ language, code }) => {
                if (!language || !code) return "Missing language or code";
                try {
                    const response = await fetch("https://emkc.org/api/v2/piston/execute", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ language, version: "*", files: [{ content: code }] }),
                    });
                    const payload = await response.json();
                    if (payload.message) return `Piston API Error: ${payload.message}`;
                    const output = payload.run?.output || payload.compile?.output || "No output";
                    const codeStatus = payload.run?.code !== undefined ? payload.run.code : (payload.compile?.code !== undefined ? payload.compile.code : -1);
                    return `Exit code: ${codeStatus}\nOutput:\n${output}`;
                } catch (error) {
                    return `Execution error: ${error.message}`;
                }
            },
        },
        base64: {
            params: "{ action: 'encode'|'decode', text: string }",
            description: "Encode or decode base64 strings",
            execute: async ({ action, text }) => {
                try {
                    if (action === "encode") return Buffer.from(String(text || ""), "utf8").toString("base64");
                    if (action === "decode") return Buffer.from(String(text || ""), "base64").toString("utf8");
                    return "action must be encode/decode";
                } catch (error) {
                    return `Error: ${error.message}`;
                }
            },
        },
        json_format: {
            params: "{ action: 'format'|'validate'|'keys'|'query', json: string, path?: string }",
            description: "Parse, validate, format, or query JSON data",
            execute: async ({ action, json, path }) => {
                try {
                    const parsed = JSON.parse(json);
                    if (action === "validate") return "Valid JSON ✓";
                    if (action === "keys") return `Keys: ${Object.keys(parsed).join(", ")}`;
                    if (action === "query" && path) {
                        const value = String(path).split(".").reduce((obj, key) => obj?.[key], parsed);
                        return value !== undefined ? JSON.stringify(value, null, 2) : `Path "${path}" not found`;
                    }
                    return JSON.stringify(parsed, null, 2);
                } catch (error) {
                    return `Invalid JSON: ${error.message}`;
                }
            },
        },
        random: {
            params: "{ type?: 'number'|'uuid'|'pick', min?: number, max?: number, choices?: string[], count?: number }",
            description: "Generate random numbers, UUIDs, or pick from a list",
            execute: async ({ type = "number", min = 1, max = 100, choices, count = 1 } = {}) => {
                if (type === "uuid") return Array.from({ length: count }, () => createId()).join("\n");
                if (type === "pick" && Array.isArray(choices) && choices.length) {
                    return `Picked: ${[...choices].sort(() => Math.random() - 0.5).slice(0, count).join(", ")}`;
                }
                return `Random (${min}-${max}): ${Array.from({ length: count }, () => Math.floor(Math.random() * (max - min + 1)) + min).join(", ")}`;
            },
        },
        weather: {
            params: "{ city: string, lat?: number, lon?: number }",
            description: "Get current weather for any city",
            execute: async ({ city, lat, lon }) => {
                try {
                    let latitude = lat;
                    let longitude = lon;
                    if (!latitude || !longitude) {
                        const geocode = await (await fetch(`https://geocoding-api.open-meteo.com/v1/search?name=${encodeURIComponent(city)}&count=1`)).json();
                        if (!geocode.results?.length) return `City "${city}" not found`;
                        latitude = geocode.results[0].latitude;
                        longitude = geocode.results[0].longitude;
                    }
                    const weather = await (await fetch(`https://api.open-meteo.com/v1/forecast?latitude=${latitude}&longitude=${longitude}&current=temperature_2m,relative_humidity_2m,wind_speed_10m,weathercode&temperature_unit=celsius`)).json();
                    const current = weather.current;
                    const codes = { 0: "Clear", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast", 45: "Fog", 51: "Drizzle", 61: "Rain", 71: "Snow", 80: "Showers", 95: "Thunderstorm" };
                    return `${city}: ${current.temperature_2m}°C, ${codes[current.weathercode] || `Code ${current.weathercode}`}\nHumidity: ${current.relative_humidity_2m}%\nWind: ${current.wind_speed_10m} km/h`;
                } catch (error) {
                    return `Weather error: ${error.message}`;
                }
            },
        },
        wikipedia: {
            params: "{ query: string }",
            description: "Search Wikipedia for factual information on any topic",
            execute: async ({ query }) => {
                try {
                    const response = await fetch(`https://en.wikipedia.org/api/rest_v1/page/summary/${encodeURIComponent(String(query || "").replace(/ /g, "_"))}`);
                    if (!response.ok) {
                        const suggestions = await (await fetch(`https://en.wikipedia.org/w/api.php?action=opensearch&search=${encodeURIComponent(query)}&limit=3&format=json&origin=*`)).json();
                        return suggestions[1]?.length ? `Suggestions: ${suggestions[1].join(", ")}` : `No results for "${query}"`;
                    }
                    const payload = await response.json();
                    return `**${payload.title}**\n${payload.extract}\n${payload.content_urls?.desktop?.page || ""}`;
                } catch (error) {
                    return `Wikipedia error: ${error.message}`;
                }
            },
        },
        summarize: {
            params: "{ text: string, bullets?: number }",
            description: "Summarize long text into key bullet points",
            execute: async ({ text, bullets = 5 }) => {
                if (!text) return "No text provided";
                const sentences = String(text).match(/[^.!?]+[.!?]+/g) || [String(text)];
                return sentences
                    .filter((_, index) => index % Math.max(1, Math.floor(sentences.length / bullets)) === 0)
                    .slice(0, bullets)
                    .map((item) => `• ${item.trim()}`)
                    .join("\n");
            },
        },
        color: {
            params: "{ color: string, to?: 'hex'|'rgb'|'hsl' }",
            description: "Convert between color formats: hex, rgb, hsl",
            execute: async ({ color, to = "rgb" }) => {
                try {
                    let r;
                    let g;
                    let b;
                    if (String(color).startsWith("#")) {
                        const hex = String(color).slice(1);
                        r = parseInt(hex.slice(0, 2), 16);
                        g = parseInt(hex.slice(2, 4), 16);
                        b = parseInt(hex.slice(4, 6), 16);
                    } else if (String(color).startsWith("rgb")) {
                        [r, g, b] = String(color).match(/\d+/g).map(Number);
                    } else {
                        return "Provide hex (#RRGGBB) or rgb(r,g,b)";
                    }
                    if (to === "rgb") return `rgb(${r}, ${g}, ${b})`;
                    if (to === "hex") return `#${[r, g, b].map((item) => item.toString(16).padStart(2, "0")).join("").toUpperCase()}`;
                    if (to === "hsl") {
                        let red = r / 255;
                        let green = g / 255;
                        let blue = b / 255;
                        const max = Math.max(red, green, blue);
                        const min = Math.min(red, green, blue);
                        const lightness = (max + min) / 2;
                        const delta = max - min;
                        let hue = 0;
                        let saturation = delta ? delta / (1 - Math.abs(2 * lightness - 1)) : 0;
                        if (delta) {
                            if (max === red) hue = ((green - blue) / delta) % 6;
                            else if (max === green) hue = (blue - red) / delta + 2;
                            else hue = (red - green) / delta + 4;
                            hue /= 6;
                        }
                        return `hsl(${Math.round(hue * 360)}, ${Math.round(saturation * 100)}%, ${Math.round(lightness * 100)}%)`;
                    }
                    return "Unsupported color conversion";
                } catch (error) {
                    return `Color error: ${error.message}`;
                }
            },
        },
        web_fetch: {
            params: "{ url: string, maxLength?: number, mode?: 'article'|'full'|'outline' }",
            description: "Read a single URL server-side and return cleaned text content",
            execute: async ({ url, maxLength = 3000, mode = "article" }) => {
                try {
                    const payload = await readWebPayload({ url, mode, maxChars: maxLength, includeLinks: false });
                    return `${payload.title || payload.finalUrl}\n${payload.finalUrl}\n\n${payload.content || "No readable content extracted."}${payload.contentTruncated ? "\n\n[Content truncated]" : ""}`;
                } catch (error) {
                    return `Fetch error: ${error.message}`;
                }
            },
        },
        web_read: {
            params: "{ url: string, maxChars?: number, mode?: 'article'|'full'|'outline', includeLinks?: boolean }",
            description: "Read a web page with metadata, headings, cleaned content, and optional links",
            execute: async ({ url, maxChars = 3200, mode = "article", includeLinks = true }) => {
                try {
                    return formatReadReport(await readWebPayload({ url, maxChars, mode, includeLinks }), { includeLinks });
                } catch (error) {
                    return `Read error: ${error.message}`;
                }
            },
        },
        web_crawl: {
            params: "{ url: string, maxPages?: number, maxDepth?: number, sameOrigin?: boolean, maxCharsPerPage?: number, includePatterns?: string[], excludePatterns?: string[] }",
            description: "Crawl a website or docs section across linked pages and return structured text",
            execute: async ({ url, maxPages = 6, maxDepth = 1, sameOrigin = true, maxCharsPerPage = 1800, includePatterns = [], excludePatterns = [] }) => {
                try {
                    return formatCrawlReport(await crawlWebPayload({ url, maxPages, maxDepth, sameOrigin, maxCharsPerPage, includePatterns, excludePatterns }));
                } catch (error) {
                    return `Crawler error: ${error.message}`;
                }
            },
        },
        regex: {
            params: "{ action: 'test'|'match'|'replace', text: string, pattern: string, flags?: string, replacement?: string }",
            description: "Test, match, or replace text using regex patterns",
            execute: async ({ action, text, pattern, flags = "g", replacement = "" }) => {
                try {
                    const regex = new RegExp(pattern, flags);
                    if (action === "test") return regex.test(text) ? "Match found ✓" : "No match ✗";
                    if (action === "match") return JSON.stringify(String(text).match(regex), null, 2) || "No matches";
                    if (action === "replace") return String(text).replace(regex, replacement);
                    return "action must be test/match/replace";
                } catch (error) {
                    return `Regex error: ${error.message}`;
                }
            },
        },
        hash: {
            params: "{ text: string }",
            description: "Generate SHA-256 hash of text",
            execute: async ({ text }) => `SHA-256: ${crypto.createHash("sha256").update(String(text || ""), "utf8").digest("hex")}`,
        },
        string_transform: {
            params: "{ text: string, action: 'reverse'|'slug'|'camel'|'upper'|'lower'|'count' }",
            description: "Transform strings",
            execute: async ({ text, action }) => {
                const value = String(text || "");
                if (action === "reverse") return value.split("").reverse().join("");
                if (action === "slug") return value.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
                if (action === "camel") return value.toLowerCase().replace(/[^a-z0-9]+(.)/g, (_, chr) => chr.toUpperCase());
                if (action === "upper") return value.toUpperCase();
                if (action === "lower") return value.toLowerCase();
                if (action === "count") return `Length: ${value.length} chars, ${value.split(/\s+/).filter(Boolean).length} words`;
                return "action must be reverse/slug/camel/upper/lower/count";
            },
        },
        timer: {
            params: "{ action: 'start'|'stop', name?: string }",
            description: "Start or stop a named timer to measure elapsed time",
            execute: async ({ action, name = "default" }) => {
                if (action === "start") {
                    timerStore[name] = Date.now();
                    return `Timer "${name}" started.`;
                }
                if (action === "stop") {
                    if (!timerStore[name]) return `Timer "${name}" not started.`;
                    const duration = Date.now() - timerStore[name];
                    delete timerStore[name];
                    return `Timer "${name}": ${duration}ms (${(duration / 1000).toFixed(2)}s)`;
                }
                return "action must be start/stop";
            },
        },
    };
};

const buildToolDescription = (tools) => Object.entries(tools)
    .map(([name, tool]) => `• ${name}\n  Params: ${tool.params}\n  Desc: ${tool.description}`)
    .join("\n\n");

const resolveModelChain = (requestedModel, fallbackModels) => {
    const chain = [];
    const addModel = (modelId) => {
        if (!AVAILABLE_MODEL_IDS.has(modelId)) return;
        if (chain.some((model) => model.id === modelId)) return;
        const model = AVAILABLE_MODELS.find((entry) => entry.id === modelId);
        if (model) chain.push(model);
    };

    addModel(typeof requestedModel === "string" ? requestedModel.trim() : "");
    if (Array.isArray(fallbackModels)) {
        fallbackModels.forEach((modelId) => addModel(String(modelId || "").trim()));
    }
    AVAILABLE_MODELS.forEach((model) => addModel(model.id));
    return chain.length ? chain : [AVAILABLE_MODELS[0]];
};

const getLatestUserQuery = (messages = []) => {
    const users = (Array.isArray(messages) ? messages : []).filter((message) => message?.role === "user");
    if (!users.length) return "";
    const latest = users[users.length - 1];
    const text = getMessageContentText(latest.content);
    if (text) return text;
    if (messageHasImages(latest)) return "Analyze the provided image input.";
    return "";
};

const addUsage = (total, usage) => {
    const next = usage && typeof usage === "object" ? usage : {};
    return {
        prompt_tokens: (total.prompt_tokens || 0) + (Number(next.prompt_tokens) || 0),
        completion_tokens: (total.completion_tokens || 0) + (Number(next.completion_tokens) || 0),
        total_tokens: (total.total_tokens || 0) + (Number(next.total_tokens) || 0),
    };
};

const shouldFallbackToNextModel = (error) => {
    const message = String(error?.message || "");
    return message.includes("402") || message.includes("429") || message === "rate_limited";
};

const sanitizeStepForHistory = (step = {}) => {
    if (!step || typeof step !== "object") return step;
    const { thought, ...rest } = step;
    if (rest.feedback && typeof rest.feedback === "object") {
        rest.feedback = {
            status: rest.feedback.status,
            summary: rest.feedback.summary,
            checks: rest.feedback.checks,
            missing: rest.feedback.missing,
            nextAction: rest.feedback.nextAction,
        };
    }
    return rest;
};

const sanitizeStepsForHistory = (steps = []) => (Array.isArray(steps) ? steps.map(sanitizeStepForHistory) : []);

const runServerAgent = async ({ body = {}, messages = [], callModel }) => {
    let memoryStore = body.memory && typeof body.memory === "object" ? body.memory : {};
    const getMemoryStore = () => memoryStore;
    const setMemoryStore = (next) => { memoryStore = next && typeof next === "object" ? next : {}; };
    const tools = createTools({ getMemoryStore, setMemoryStore });
    const toolDesc = buildToolDescription(tools);
    const chain = resolveModelChain(body.model, body.fallbackModels);
    const query = getLatestUserQuery(messages) || normalizeTextBlock(body.prompt || body.input || body.message || "");
    const historyMessages = buildWorkingMemoryMessages(messages);
    const episodicSummary = buildEpisodicSummary(messages);
    const memoryContext = formatMemoryContext(memoryStore);
    const steps = [];
    let totalToolCalls = 0;
    let verificationCycles = 0;
    let totalUsage = { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 };
    let routerDecision = { primary: "none", secondary: [], reason: "" };
    let managerPlan = null;
    let verificationReport = null;
    let contextRecoveryAttempts = 0;
    let liveContextBudget = MAX_CONTEXT_CHARS;
    let liveLatestUserBudget = MAX_CONTEXT_MESSAGE_CHARS;
    let finalModelUsed = chain[0]?.id || AVAILABLE_MODELS[0].id;

    const completeText = async (candidateMessages, { maxTokens, maxContextChars = liveContextBudget, latestUserMaxChars = liveLatestUserBudget } = {}) => {
        const prepared = await prepareMessagesForDispatch(candidateMessages, {
            chain,
            maxTokens,
            maxContextChars,
            latestUserMaxChars,
        });
        let lastError = null;
        for (const model of chain) {
            try {
                const result = await callModel({
                    messages: prepared.messages,
                    model: model.id,
                    maxTokens,
                });
                totalUsage = addUsage(totalUsage, result.usage);
                finalModelUsed = result.model || model.id;
                return { text: result.text, prepared };
            } catch (error) {
                lastError = error;
                if (isContextLengthError(error)) throw error;
                if (shouldFallbackToNextModel(error)) continue;
                throw error;
            }
        }
        throw lastError || new Error("All models failed");
    };

    try {
        try {
            const routerMessages = [{ role: "system", content: buildRouterPrompt(toolDesc) }];
            if (memoryContext) routerMessages.push({ role: "system", content: memoryContext });
            if (episodicSummary) routerMessages.push({ role: "system", content: episodicSummary });
            historyMessages.slice(-6).forEach((message) => routerMessages.push({ role: message.role, content: message.content }));
            routerMessages.push({ role: "user", content: query });
            const routerText = (await completeText(routerMessages, {
                maxTokens: MAX_SUBAGENT_TOKENS,
                maxContextChars: Math.min(MAX_CONTEXT_CHARS, 70000),
            })).text;
            routerDecision = parseRouterResponse(routerText);
        } catch {
            routerDecision = { primary: "none", secondary: [], reason: "" };
        }

        try {
            const plannerMessages = [{ role: "system", content: buildPlannerPrompt(toolDesc) }];
            if (memoryContext) plannerMessages.push({ role: "system", content: memoryContext });
            if (episodicSummary) plannerMessages.push({ role: "system", content: episodicSummary });
            if (routerDecision.primary && routerDecision.primary !== "none") {
                plannerMessages.push({
                    role: "system",
                    content: `Router preference:\nPrimary tool: ${routerDecision.primary}\nBackup tools: ${routerDecision.secondary.join(", ") || "none"}\nReason: ${routerDecision.reason || "No reason provided."}`,
                });
            }
            historyMessages.slice(-6).forEach((message) => plannerMessages.push({ role: message.role, content: message.content }));
            plannerMessages.push({ role: "user", content: query });
            const plannerText = (await completeText(plannerMessages, {
                maxTokens: MAX_SUBAGENT_TOKENS,
                maxContextChars: Math.min(MAX_CONTEXT_CHARS, 80000),
            })).text;
            managerPlan = parsePlannerResponse(plannerText);
        } catch {
            managerPlan = null;
        }

        const llmMessages = [{ role: "system", content: buildSystemPrompt(Object.keys(tools).length, toolDesc) }];
        if (memoryContext) llmMessages.push({ role: "system", content: memoryContext });
        if (episodicSummary) llmMessages.push({ role: "system", content: episodicSummary });
        if (routerDecision.primary && routerDecision.primary !== "none") {
            llmMessages.push({
                role: "system",
                content: `Tool Router guidance:\nPrimary tool: ${routerDecision.primary}\nBackup tools: ${routerDecision.secondary.join(", ") || "none"}\nReason: ${routerDecision.reason || "No reason provided."}\nStart by considering the router's primary tool unless the evidence clearly points elsewhere.`,
            });
        }
        if (managerPlan) llmMessages.push({ role: "system", content: formatPlanForSystem(managerPlan) });
        historyMessages.forEach((message) => llmMessages.push({ role: message.role, content: message.content }));
        llmMessages.push({ role: "user", content: query });

        for (let iteration = 0; iteration < MAX_ITERATIONS; iteration += 1) {
            let fullResponseText = "";
            try {
                fullResponseText = (await completeText(llmMessages, {
                    maxTokens: MAX_COMPLETION_TOKENS,
                    maxContextChars: liveContextBudget,
                    latestUserMaxChars: liveLatestUserBudget,
                })).text;
                contextRecoveryAttempts = 0;
            } catch (error) {
                if (isContextLengthError(error)) {
                    contextRecoveryAttempts += 1;
                    if (contextRecoveryAttempts > 2) {
                        throw new Error("The request is still too large after context compaction. Shorten the prompt or provide less history.");
                    }
                    liveContextBudget = contextRecoveryAttempts === 1
                        ? Math.max(MIN_CONTEXT_CHARS, Math.floor(liveContextBudget * 0.55))
                        : Math.max(MIN_CONTEXT_CHARS, Math.floor(liveContextBudget * 0.4));
                    liveLatestUserBudget = Math.max(1200, Math.min(liveLatestUserBudget, liveContextBudget));
                    iteration -= 1;
                    continue;
                }
                throw error;
            }

            const parsed = parseXMLAgentResponse(fullResponseText);
            const assistantHistoryText = sanitizeAssistantHistoryText(fullResponseText, parsed);
            const stamp = { iteration: iteration + 1, ts: new Date().toISOString() };
            const parsedActions = parsed.actions?.length ? parsed.actions : (parsed.action ? [{ name: parsed.action, input: parsed.action_input || {} }] : []);

            if (parsed.final_answer && parsedActions.length === 0) {
                try {
                    const verifierText = (await completeText([
                        { role: "system", content: buildVerifierPrompt({ query, plan: managerPlan, steps, answer: parsed.final_answer }) },
                        { role: "user", content: query },
                        { role: "assistant", content: parsed.final_answer },
                    ], {
                        maxTokens: MAX_VERIFIER_TOKENS,
                        maxContextChars: Math.min(MAX_CONTEXT_CHARS, 80000),
                    })).text;
                    verificationReport = parseVerifierResponse(verifierText);
                    steps.push({ type: "verification", verdict: verificationReport.status, feedback: verificationReport, ...stamp });
                    if (verificationReport.status === "revise" && verificationCycles < MAX_VERIFICATION_CYCLES) {
                        verificationCycles += 1;
                        llmMessages.push({ role: "assistant", content: assistantHistoryText });
                        llmMessages.push({ role: "user", content: `Verifier feedback:\n${formatVerifierFeedback(verificationReport)}\n\nRevise the answer. Use more tools if needed before finalizing.` });
                        continue;
                    }
                } catch {
                    verificationReport = null;
                }

                return {
                    id: `agent-${Date.now()}`,
                    model: finalModelUsed,
                    created: Math.floor(Date.now() / 1000),
                    output_text: assistantHistoryText,
                    finish_reason: "stop",
                    usage: totalUsage,
                    agentic: true,
                    steps: sanitizeStepsForHistory(steps),
                    strategy: {
                        router: routerDecision,
                        plan: managerPlan,
                        verification: verificationReport,
                        metrics: {
                            iterations: iteration + 1,
                            toolCalls: totalToolCalls,
                            verificationPasses: verificationReport?.status === "pass" ? 1 : 0,
                            verificationRevisions: verificationReport?.status === "revise" ? 1 : 0,
                        },
                    },
                };
            }

            if (parsedActions.length > 0) {
                const actions = parsedActions.slice(0, MAX_TOOL_CALLS_PER_ITERATION);
                const observations = [];

                for (let actionIndex = 0; actionIndex < actions.length; actionIndex += 1) {
                    const actionCall = actions[actionIndex];
                    const tool = tools[actionCall.name];
                    let observation;
                    const startedAt = performance.now();
                    try {
                        observation = tool ? await tool.execute(actionCall.input || {}) : `Unknown tool: "${actionCall.name}"`;
                    } catch (error) {
                        observation = `Tool error: ${error.message}`;
                    }
                    const durationMs = Math.round(performance.now() - startedAt);
                    const firewall = firewallToolOutput(actionCall.name, observation);

                    totalToolCalls += 1;
                    steps.push({
                        type: "action",
                        action: actionCall.name,
                        input: actionCall.input,
                        observation: firewall.agentText,
                        firewallMeta: {
                            chunkCount: firewall.chunkCount,
                            tokenEstimate: firewall.tokenEstimate,
                            wasJson: firewall.wasJson,
                            wasHtml: firewall.wasHtml,
                        },
                        batchIndex: actionIndex + 1,
                        batchSize: actions.length,
                        durationMs,
                        ...stamp,
                    });

                    observations.push({
                        action: actionCall.name,
                        input: actionCall.input || {},
                        observation: firewall.agentText,
                        firewallMeta: {
                            chunkCount: firewall.chunkCount,
                            tokenEstimate: firewall.tokenEstimate,
                            wasJson: firewall.wasJson,
                            wasHtml: firewall.wasHtml,
                        },
                    });
                }

                llmMessages.push({ role: "assistant", content: assistantHistoryText });
                llmMessages.push({ role: "user", content: formatObservationBundle(observations, parsedActions.length > actions.length) });
                continue;
            }

            return {
                id: `agent-${Date.now()}`,
                model: finalModelUsed,
                created: Math.floor(Date.now() / 1000),
                output_text: assistantHistoryText || parsed.thought || "Execution ended without a final answer.",
                finish_reason: "stop",
                usage: totalUsage,
                agentic: true,
                steps: sanitizeStepsForHistory(steps),
                strategy: {
                    router: routerDecision,
                    plan: managerPlan,
                    verification: verificationReport,
                    metrics: {
                        iterations: iteration + 1,
                        toolCalls: totalToolCalls,
                        verificationPasses: 0,
                        verificationRevisions: verificationCycles,
                    },
                },
            };
        }

        throw new Error("Iteration limit reached before a verified final answer was produced.");
    } catch (error) {
        return {
            id: `agent-${Date.now()}`,
            model: finalModelUsed,
            created: Math.floor(Date.now() / 1000),
            output_text: "",
            finish_reason: "error",
            usage: totalUsage,
            agentic: true,
            error: error?.message || "Agent request failed.",
            steps: sanitizeStepsForHistory(steps),
            strategy: {
                router: routerDecision,
                plan: managerPlan,
                verification: verificationReport,
                metrics: {
                    iterations: steps.length ? Math.max(...steps.map((step) => Number(step.iteration) || 0)) : 0,
                    toolCalls: totalToolCalls,
                    verificationPasses: verificationReport?.status === "pass" ? 1 : 0,
                    verificationRevisions: verificationReport?.status === "revise" ? 1 : 0,
                },
            },
        };
    }
};

module.exports = {
    AVAILABLE_MODELS,
    MAX_COMPLETION_TOKENS,
    MAX_SUBAGENT_TOKENS,
    MAX_VERIFIER_TOKENS,
    resolveModelChain,
    runServerAgent,
};
