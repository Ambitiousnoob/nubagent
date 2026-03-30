import { withClientStateKey } from "./clientStateKey.js";

const RESEARCH_API = "/api/research";

const normalizeText = (value) => String(value ?? "").trim();

const readErrorPayload = async (response) => {
  try {
    const contentType = response.headers.get("content-type") || "";
    if (/json/i.test(contentType)) {
      const data = await response.json();
      return normalizeText(data?.error || data?.message || response.statusText);
    }
    return normalizeText(await response.text()) || response.statusText;
  } catch {
    return response.statusText || "Request failed.";
  }
};

const parseSseChunkLines = (chunk) => chunk.split(/\r?\n/).filter(Boolean);

const parseSseStream = async (response, onEvent) => {
  const reader = response.body?.getReader?.();
  if (!reader) return null;

  const decoder = new globalThis.TextDecoder();
  let buffer = "";
  let eventType = "message";
  let eventData = "";
  let finalEvent = null;
  const events = [];

  const flushEvent = () => {
    if (!eventData) return;
    if (eventData === "[DONE]") {
      eventType = "message";
      eventData = "";
      return;
    }
    let parsed;
    try {
      parsed = JSON.parse(eventData);
    } catch {
      parsed = { raw: eventData };
    }
    const payload =
      parsed && typeof parsed === "object" && !Array.isArray(parsed)
        ? { ...parsed, type: parsed.type || eventType || "message" }
        : { type: eventType || "message", data: parsed };
    events.push(payload);
    onEvent?.(payload);
    if (payload.type === "final") finalEvent = payload;
    eventType = "message";
    eventData = "";
  };

  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });

    let separatorIndex = buffer.indexOf("\n\n");
    while (separatorIndex >= 0) {
      const block = buffer.slice(0, separatorIndex);
      buffer = buffer.slice(separatorIndex + 2);

      for (const line of parseSseChunkLines(block)) {
        if (line.startsWith("event:")) {
          eventType = line.slice(6).trim() || "message";
        } else if (line.startsWith("data:")) {
          eventData += line.slice(5).trim();
        }
      }

      flushEvent();
      separatorIndex = buffer.indexOf("\n\n");
    }

    if (done) {
      if (buffer.trim()) {
        for (const line of parseSseChunkLines(buffer)) {
          if (line.startsWith("event:")) {
            eventType = line.slice(6).trim() || "message";
          } else if (line.startsWith("data:")) {
            eventData += line.slice(5).trim();
          }
        }
        flushEvent();
      }
      break;
    }
  }

  return {
    events,
    final: finalEvent,
  };
};

export async function runResearchRequest(payload, { signal, onEvent } = {}) {
  const requestPayload = withClientStateKey(payload || {});
  const response = await fetch(RESEARCH_API, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    credentials: "include",
    cache: "no-store",
    signal,
    body: JSON.stringify(requestPayload),
  });

  if (!response.ok) {
    throw new Error(`${response.status} ${await readErrorPayload(response)}`);
  }

  const contentType = response.headers.get("content-type") || "";
  if (/text\/event-stream/i.test(contentType)) {
    const streamed = await parseSseStream(response, onEvent);
    if (streamed?.final) {
      return {
        ok: true,
        ...streamed.final,
        events: streamed.events,
      };
    }
    return {
      ok: true,
      events: streamed?.events || [],
    };
  }

  return response.json();
}

export async function invokeResearchRuntime(payload, signal, onEvent) {
  return runResearchRequest(payload, {
    signal,
    onEvent,
  });
}

export async function sendResearchControl(runId, control, { signal } = {}) {
  const requestPayload = withClientStateKey({
    action: "control",
    runId,
    control,
  });
  const response = await fetch(RESEARCH_API, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    credentials: "include",
    cache: "no-store",
    signal,
    body: JSON.stringify(requestPayload),
  });

  if (!response.ok) {
    throw new Error(`${response.status} ${await readErrorPayload(response)}`);
  }

  return response.json();
}

export async function fetchResearchExport(runId, format, { signal } = {}) {
  const requestPayload = withClientStateKey({
    action: "export",
    runId,
    format,
  });
  const response = await fetch(RESEARCH_API, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    credentials: "include",
    cache: "no-store",
    signal,
    body: JSON.stringify(requestPayload),
  });

  if (!response.ok) {
    throw new Error(`${response.status} ${await readErrorPayload(response)}`);
  }

  const contentType = response.headers.get("content-type") || "";
  if (/json/i.test(contentType)) {
    return response.json();
  }

  return {
    contentType,
    body: await response.text(),
  };
}
