import {
  buildRollingConversationSummary,
  classifyMemoryKind,
} from "./context.js";
import {
  buildLocationCaptureReply,
  buildLocationCaptureUrl,
  resolveLocationCaptureBaseUrl,
} from "./location-capture.js";

const HELP_REPLY = `Commands:
!help - show commands and capabilities
!reset - clear active chat context
!summary - show the current rolling summary
!memory - list remembered details
!memory add <text> - save a detail for later
!location - send a secure link to capture your current location
!location show - show the latest saved location
!location clear - delete the saved location
!forget <phrase> - delete matching memory
!forget all - delete all saved memory
!privacy - explain what NubAgent stores

NubAgent works best with text, links, supported images, and local search questions.`;

const PRIVACY_REPLY = `NubAgent stores recent chat turns, the latest image context, the latest shared location captured from Messenger pins or !location links, a rolling conversation summary, and saved long-term memory in Postgres.

Use !reset to clear active chat context, !forget all to delete saved long-term memory, and !location clear to delete the saved location.`;

const BARE_COMMANDS = new Set([
  "help",
  "reset",
  "summary",
  "memory",
  "location",
  "privacy",
]);
const COMMAND_ALIASES = new Map([
  ["help", "help"],
  ["start", "help"],
  ["reset", "reset"],
  ["summary", "summary"],
  ["memory", "memory"],
  ["location", "location"],
  ["forget", "forget"],
  ["privacy", "privacy"],
]);

function normalizeText(text) {
  return typeof text === "string" ? text.trim() : "";
}

function formatMemoryReply(memoryEntries) {
  if (!Array.isArray(memoryEntries) || memoryEntries.length === 0) {
    return "I am not holding any saved long-term memory for you right now.";
  }

  const lines = memoryEntries.map((entry) => {
    const label = entry.kind === "preference" ? "Preference" : "Fact";
    return `- ${label}: ${entry.content}`;
  });

  return `Saved memory:\n${lines.join("\n")}`;
}

function formatSummaryReply(summary) {
  const normalizedSummary = normalizeText(summary);

  if (!normalizedSummary) {
    return "I do not have enough conversation context to summarize yet.";
  }

  return `Current conversation summary:\n${normalizedSummary}`;
}

function formatLocationReply(location) {
  if (!location) {
    return "I do not have a saved location for you right now. Use !location to generate a secure capture link, or share a Messenger location pin, and I will use it for Google Maps grounded prompts.";
  }

  return `Saved location:\nLatitude: ${location.latitude}\nLongitude: ${location.longitude}\nUse !location clear if you want me to forget it.`;
}

function buildResetReply(result) {
  return `I cleared the active chat context.${
    result?.deletedMessageCount
      ? ` Removed ${result.deletedMessageCount} stored turn${result.deletedMessageCount === 1 ? "" : "s"}.`
      : ""
  } Long-term memory stays until you use !forget.`;
}

export function parseCommand(input) {
  const normalized = normalizeText(input);

  if (!normalized) {
    return null;
  }

  if (/^Postback payload:\s*NUBAGENT_GET_STARTED$/i.test(normalized)) {
    return {
      name: "help",
      args: "",
      raw: normalized,
    };
  }

  const bangMode = normalized.startsWith("!");
  const commandText = bangMode ? normalized.slice(1).trim() : normalized;
  const [nameToken = "", ...rest] = commandText.split(/\s+/);
  const commandName = COMMAND_ALIASES.get(nameToken.toLowerCase());

  if (!commandName) {
    return null;
  }

  if (!bangMode && !BARE_COMMANDS.has(commandText.toLowerCase())) {
    return null;
  }

  return {
    name: commandName,
    args: rest.join(" ").trim(),
    raw: normalized,
  };
}

export async function handleCommand({
  command,
  senderId,
  store,
  config,
  baseUrl,
}) {
  switch (command.name) {
    case "help":
      return { reply: HELP_REPLY };

    case "privacy":
      return { reply: PRIVACY_REPLY };

    case "reset": {
      const result = await store.clearConversationState(senderId);
      return {
        reply: buildResetReply(result),
      };
    }

    case "summary": {
      const summary =
        (await store.getConversationSummary(senderId)) ||
        buildRollingConversationSummary(
          await store.getSummarySourceHistory(senderId),
        );

      return {
        reply: formatSummaryReply(summary),
      };
    }

    case "memory": {
      const normalizedArgs = normalizeText(command.args);

      if (!normalizedArgs || /^list$/i.test(normalizedArgs)) {
        return {
          reply: formatMemoryReply(await store.listUserMemory(senderId)),
        };
      }

      const memoryText = normalizedArgs.replace(/^add\s+/i, "");

      if (!memoryText) {
        return {
          reply:
            "Use !memory to list saved details or !memory add <text> to save one.",
        };
      }

      const savedMemory = await store.saveMemory({
        senderId,
        content: memoryText,
        kind: classifyMemoryKind(memoryText),
        source: "command",
      });

      return {
        reply: `Saved memory: ${savedMemory.content}`,
      };
    }

    case "location": {
      const normalizedArgs = normalizeText(command.args);

      if (/^clear$/i.test(normalizedArgs)) {
        const result = await store.clearLatestLocation(senderId);

        return {
          reply: result.deleted
            ? "I cleared your saved location."
            : "I did not have a saved location to clear.",
        };
      }

      if (/^(show|saved|status)$/i.test(normalizedArgs)) {
        return {
          reply: formatLocationReply(await store.getLatestLocation(senderId)),
        };
      }

      const captureToken = await store.createLocationCaptureToken(senderId, {
        ttlMinutes: config?.locationCaptureTtlMinutes,
      });
      const captureUrl = buildLocationCaptureUrl({
        baseUrl: resolveLocationCaptureBaseUrl({ baseUrl, config }),
        token: captureToken.token,
      });

      return {
        reply: buildLocationCaptureReply({
          url: captureUrl,
          expiresInMinutes: config?.locationCaptureTtlMinutes,
        }),
      };
    }

    case "forget": {
      const normalizedArgs = normalizeText(command.args);

      if (!normalizedArgs) {
        return {
          reply:
            "Use !forget <phrase> to delete a matching memory, or !forget all to clear every saved memory.",
        };
      }

      const result = await store.deleteMemory(senderId, normalizedArgs);

      if (result.deletedCount === 0) {
        return {
          reply: "I could not find any saved memory matching that request.",
        };
      }

      return {
        reply:
          normalizedArgs.toLowerCase() === "all"
            ? `Deleted ${result.deletedCount} saved memor${result.deletedCount === 1 ? "y" : "ies"}.`
            : `Deleted ${result.deletedCount} saved memor${result.deletedCount === 1 ? "y" : "ies"} that matched "${normalizedArgs}".`,
      };
    }

    default:
      return null;
  }
}
