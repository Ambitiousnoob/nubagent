import { randomBytes } from "node:crypto";
import pg from "pg";

const { Pool } = pg;

export const MAX_PROMPT_TURNS = 12;
export const STORED_HISTORY_LIMIT = MAX_PROMPT_TURNS - 1;
const MESSAGE_TABLE = "messenger_messages";
const IMAGE_CONTEXT_TABLE = "messenger_image_context";
const LOCATION_TABLE = "messenger_user_location";
const LOCATION_CAPTURE_TABLE = "messenger_location_capture";
const MEMORY_TABLE = "messenger_user_memory";
const SUMMARY_TABLE = "messenger_conversation_summary";
const EVENT_TABLE = "messenger_event_log";
const FAILED_OUTBOUND_TABLE = "messenger_failed_outbound";
const SUMMARY_HISTORY_TURNS = 8;
const DEFAULT_RELEVANT_MEMORY_LIMIT = 4;
const MIN_TOKEN_LENGTH = 3;

let sharedConversationStorePromise;

function createPool(connectionString) {
  return new Pool({
    connectionString,
  });
}

function normalizeText(text) {
  return typeof text === "string" ? text.trim() : "";
}

function normalizeMemoryText(text) {
  return normalizeText(text).toLowerCase().replace(/\s+/g, " ");
}

function tokenizeText(text) {
  return normalizeMemoryText(text)
    .split(/[^a-z0-9]+/i)
    .filter((token) => token.length >= MIN_TOKEN_LENGTH);
}

function mapMemoryRow(row) {
  return {
    id: row.id,
    content: row.content,
    kind: row.kind,
    source: row.source,
    updatedAt: row.updated_at,
  };
}

function scoreMemoryRow(row, promptTokens, normalizedPrompt) {
  if (promptTokens.length === 0) {
    return row.kind === "preference" ? 1 : 0;
  }

  let score = 0;

  if (normalizedPrompt && normalizedPrompt.includes(row.normalized_content)) {
    score += 10;
  }

  const rowTokens = new Set(tokenizeText(row.normalized_content));

  for (const token of promptTokens) {
    if (rowTokens.has(token)) {
      score += 3;
    }
  }

  if (row.kind === "preference") {
    score += 1;
  }

  return score;
}

export function mapRowsToConversationHistory(rows) {
  return [...rows].reverse().map((row) => ({
    role: row.role,
    parts: [{ text: row.content }],
  }));
}

async function ensureSchema(pool) {
  await pool.query(`
    CREATE TABLE IF NOT EXISTS ${MESSAGE_TABLE} (
      id BIGSERIAL PRIMARY KEY,
      sender_psid TEXT NOT NULL,
      role TEXT NOT NULL CHECK (role IN ('user', 'model')),
      content TEXT NOT NULL,
      source_event_id TEXT UNIQUE,
      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
  `);

  await pool.query(`
    CREATE INDEX IF NOT EXISTS ${MESSAGE_TABLE}_sender_psid_created_at_idx
      ON ${MESSAGE_TABLE} (sender_psid, created_at DESC, id DESC)
  `);

  await pool.query(`
    CREATE TABLE IF NOT EXISTS ${IMAGE_CONTEXT_TABLE} (
      sender_psid TEXT PRIMARY KEY,
      summary TEXT NOT NULL,
      source_event_id TEXT,
      updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
  `);

  await pool.query(`
    CREATE TABLE IF NOT EXISTS ${LOCATION_TABLE} (
      sender_psid TEXT PRIMARY KEY,
      latitude DOUBLE PRECISION NOT NULL,
      longitude DOUBLE PRECISION NOT NULL,
      source_event_id TEXT,
      updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
  `);

  await pool.query(`
    CREATE TABLE IF NOT EXISTS ${LOCATION_CAPTURE_TABLE} (
      token TEXT PRIMARY KEY,
      sender_psid TEXT NOT NULL,
      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      expires_at TIMESTAMPTZ NOT NULL,
      used_at TIMESTAMPTZ
    )
  `);

  await pool.query(`
    CREATE INDEX IF NOT EXISTS ${LOCATION_CAPTURE_TABLE}_sender_psid_created_at_idx
      ON ${LOCATION_CAPTURE_TABLE} (sender_psid, created_at DESC)
  `);

  await pool.query(`
    CREATE INDEX IF NOT EXISTS ${LOCATION_CAPTURE_TABLE}_expires_at_idx
      ON ${LOCATION_CAPTURE_TABLE} (expires_at)
  `);

  await pool.query(`
    CREATE TABLE IF NOT EXISTS ${MEMORY_TABLE} (
      id BIGSERIAL PRIMARY KEY,
      sender_psid TEXT NOT NULL,
      normalized_content TEXT NOT NULL,
      content TEXT NOT NULL,
      kind TEXT NOT NULL CHECK (kind IN ('fact', 'preference')),
      source TEXT NOT NULL CHECK (source IN ('auto', 'command')),
      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      last_used_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      UNIQUE (sender_psid, normalized_content)
    )
  `);

  await pool.query(`
    CREATE INDEX IF NOT EXISTS ${MEMORY_TABLE}_sender_psid_last_used_idx
      ON ${MEMORY_TABLE} (sender_psid, last_used_at DESC, updated_at DESC)
  `);

  await pool.query(`
    CREATE TABLE IF NOT EXISTS ${SUMMARY_TABLE} (
      sender_psid TEXT PRIMARY KEY,
      summary TEXT NOT NULL,
      updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
  `);

  await pool.query(`
    CREATE TABLE IF NOT EXISTS ${EVENT_TABLE} (
      sender_psid TEXT NOT NULL,
      source_event_id TEXT NOT NULL,
      event_type TEXT NOT NULL,
      status TEXT NOT NULL CHECK (status IN ('processing', 'completed', 'failed')),
      stage TEXT NOT NULL,
      inbound_saved BOOLEAN NOT NULL DEFAULT FALSE,
      reply_generated BOOLEAN NOT NULL DEFAULT FALSE,
      reply_sent BOOLEAN NOT NULL DEFAULT FALSE,
      model_turn_saved BOOLEAN NOT NULL DEFAULT FALSE,
      retry_count INTEGER NOT NULL DEFAULT 0,
      failure_message TEXT,
      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      PRIMARY KEY (sender_psid, source_event_id)
    )
  `);

  await pool.query(`
    CREATE INDEX IF NOT EXISTS ${EVENT_TABLE}_status_updated_at_idx
      ON ${EVENT_TABLE} (status, updated_at DESC)
  `);

  await pool.query(`
    CREATE TABLE IF NOT EXISTS ${FAILED_OUTBOUND_TABLE} (
      id BIGSERIAL PRIMARY KEY,
      sender_psid TEXT NOT NULL,
      source_event_id TEXT,
      stage TEXT NOT NULL,
      reply_text TEXT NOT NULL,
      error_message TEXT NOT NULL,
      retry_count INTEGER NOT NULL DEFAULT 0,
      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
  `);

  await pool.query(`
    CREATE INDEX IF NOT EXISTS ${FAILED_OUTBOUND_TABLE}_created_at_idx
      ON ${FAILED_OUTBOUND_TABLE} (created_at DESC)
  `);
}

export function createConversationStore({
  pool,
  connectionString,
  historyLimit = STORED_HISTORY_LIMIT,
} = {}) {
  if (!pool && !connectionString) {
    throw new Error("A pg pool or POSTGRES_URL is required.");
  }

  const activePool = pool ?? createPool(connectionString);
  const ownsPool = !pool;
  let schemaReadyPromise;

  function getSchemaReadyPromise() {
    if (!schemaReadyPromise) {
      schemaReadyPromise = ensureSchema(activePool);
    }

    return schemaReadyPromise;
  }

  return {
    async ensureSchema() {
      await getSchemaReadyPromise();
    },

    async beginEventProcessing({
      senderId,
      sourceEventId,
      eventType = "message",
    }) {
      if (!senderId) {
        throw new Error("senderId is required for event tracking.");
      }

      await getSchemaReadyPromise();

      if (!sourceEventId) {
        return {
          inserted: true,
          tracked: false,
        };
      }

      const result = await activePool.query(
        `
          INSERT INTO ${EVENT_TABLE} (
            sender_psid,
            source_event_id,
            event_type,
            status,
            stage
          )
          VALUES ($1, $2, $3, 'processing', 'received')
          ON CONFLICT (sender_psid, source_event_id) DO NOTHING
          RETURNING source_event_id
        `,
        [senderId, sourceEventId, eventType],
      );

      return {
        inserted: result.rowCount > 0,
        tracked: true,
      };
    },

    async updateEventProcessing({
      senderId,
      sourceEventId,
      status,
      stage,
      inboundSaved,
      replyGenerated,
      replySent,
      modelTurnSaved,
      retryCount,
      failureMessage,
    }) {
      if (!senderId || !sourceEventId) {
        return;
      }

      await getSchemaReadyPromise();

      const values = [senderId, sourceEventId];
      const assignments = [];

      if (status !== undefined) {
        values.push(status);
        assignments.push(`status = $${values.length}`);
      }

      if (stage !== undefined) {
        values.push(stage);
        assignments.push(`stage = $${values.length}`);
      }

      if (inboundSaved !== undefined) {
        values.push(Boolean(inboundSaved));
        assignments.push(`inbound_saved = $${values.length}`);
      }

      if (replyGenerated !== undefined) {
        values.push(Boolean(replyGenerated));
        assignments.push(`reply_generated = $${values.length}`);
      }

      if (replySent !== undefined) {
        values.push(Boolean(replySent));
        assignments.push(`reply_sent = $${values.length}`);
      }

      if (modelTurnSaved !== undefined) {
        values.push(Boolean(modelTurnSaved));
        assignments.push(`model_turn_saved = $${values.length}`);
      }

      if (retryCount !== undefined) {
        values.push(retryCount);
        assignments.push(`retry_count = $${values.length}`);
      }

      if (failureMessage !== undefined) {
        values.push(failureMessage);
        assignments.push(`failure_message = $${values.length}`);
      }

      if (assignments.length === 0) {
        return;
      }

      assignments.push("updated_at = NOW()");

      await activePool.query(
        `
          UPDATE ${EVENT_TABLE}
          SET ${assignments.join(", ")}
          WHERE sender_psid = $1 AND source_event_id = $2
        `,
        values,
      );
    },

    async recordFailedOutbound({
      senderId,
      sourceEventId = null,
      stage,
      replyText,
      errorMessage,
      retryCount = 0,
    }) {
      const normalizedReply = normalizeText(replyText);

      if (!senderId || !stage || !normalizedReply) {
        throw new Error(
          "senderId, stage, and replyText are required for failed outbound records.",
        );
      }

      await getSchemaReadyPromise();

      await activePool.query(
        `
          INSERT INTO ${FAILED_OUTBOUND_TABLE} (
            sender_psid,
            source_event_id,
            stage,
            reply_text,
            error_message,
            retry_count
          )
          VALUES ($1, $2, $3, $4, $5, $6)
        `,
        [
          senderId,
          sourceEventId,
          stage,
          normalizedReply,
          normalizeText(errorMessage) || "Unknown outbound error.",
          retryCount,
        ],
      );
    },

    async saveInboundTurn({ senderId, text, sourceEventId = null }) {
      const normalizedText = normalizeText(text);

      if (!senderId || !normalizedText) {
        throw new Error("senderId and text are required for inbound turns.");
      }

      await getSchemaReadyPromise();

      const result = await activePool.query(
        `
          INSERT INTO ${MESSAGE_TABLE} (sender_psid, role, content, source_event_id)
          VALUES ($1, 'user', $2, $3)
          ON CONFLICT (source_event_id) DO NOTHING
          RETURNING id
        `,
        [senderId, normalizedText, sourceEventId],
      );

      if (result.rowCount === 0) {
        return {
          inserted: false,
          messageId: null,
        };
      }

      return {
        inserted: true,
        messageId: result.rows[0].id,
      };
    },

    async getConversationHistory(
      senderId,
      { excludeMessageId = null, limit = historyLimit } = {},
    ) {
      if (!senderId) {
        return [];
      }

      await getSchemaReadyPromise();

      const result = await activePool.query(
        `
          SELECT id, role, content
          FROM ${MESSAGE_TABLE}
          WHERE sender_psid = $1
            AND ($2::bigint IS NULL OR id <> $2)
          ORDER BY created_at DESC, id DESC
          LIMIT $3
        `,
        [senderId, excludeMessageId, limit],
      );

      return mapRowsToConversationHistory(result.rows);
    },

    async saveModelTurn({ senderId, text }) {
      const normalizedText = normalizeText(text);

      if (!senderId || !normalizedText) {
        throw new Error("senderId and text are required for model turns.");
      }

      await getSchemaReadyPromise();

      await activePool.query(
        `
          INSERT INTO ${MESSAGE_TABLE} (sender_psid, role, content, source_event_id)
          VALUES ($1, 'model', $2, NULL)
        `,
        [senderId, normalizedText],
      );
    },

    async clearConversationState(senderId, { clearMemory = false } = {}) {
      if (!senderId) {
        throw new Error("senderId is required to clear conversation state.");
      }

      await getSchemaReadyPromise();

      const deletedMessages = await activePool.query(
        `
          DELETE FROM ${MESSAGE_TABLE}
          WHERE sender_psid = $1
        `,
        [senderId],
      );
      const deletedImageContext = await activePool.query(
        `
          DELETE FROM ${IMAGE_CONTEXT_TABLE}
          WHERE sender_psid = $1
        `,
        [senderId],
      );
      const deletedSummary = await activePool.query(
        `
          DELETE FROM ${SUMMARY_TABLE}
          WHERE sender_psid = $1
        `,
        [senderId],
      );
      let deletedMemoryCount = 0;

      if (clearMemory) {
        const deletedMemory = await activePool.query(
          `
            DELETE FROM ${MEMORY_TABLE}
            WHERE sender_psid = $1
          `,
          [senderId],
        );
        deletedMemoryCount = deletedMemory.rowCount;
      }

      return {
        deletedMessageCount: deletedMessages.rowCount,
        deletedImageContext: deletedImageContext.rowCount > 0,
        deletedSummary: deletedSummary.rowCount > 0,
        deletedMemoryCount,
      };
    },

    async saveLatestImageContext({ senderId, summary, sourceEventId = null }) {
      const normalizedSummary = normalizeText(summary);

      if (!senderId || !normalizedSummary) {
        throw new Error(
          "senderId and summary are required for image context storage.",
        );
      }

      await getSchemaReadyPromise();

      await activePool.query(
        `
          INSERT INTO ${IMAGE_CONTEXT_TABLE} (sender_psid, summary, source_event_id)
          VALUES ($1, $2, $3)
          ON CONFLICT (sender_psid) DO UPDATE
          SET summary = EXCLUDED.summary,
              source_event_id = EXCLUDED.source_event_id,
              updated_at = NOW()
        `,
        [senderId, normalizedSummary, sourceEventId],
      );
    },

    async saveLatestLocation({
      senderId,
      latitude,
      longitude,
      sourceEventId = null,
    }) {
      if (
        !senderId ||
        !Number.isFinite(latitude) ||
        !Number.isFinite(longitude)
      ) {
        throw new Error(
          "senderId, latitude, and longitude are required for location storage.",
        );
      }

      await getSchemaReadyPromise();

      const result = await activePool.query(
        `
          INSERT INTO ${LOCATION_TABLE} (
            sender_psid,
            latitude,
            longitude,
            source_event_id
          )
          VALUES ($1, $2, $3, $4)
          ON CONFLICT (sender_psid) DO UPDATE
          SET latitude = EXCLUDED.latitude,
              longitude = EXCLUDED.longitude,
              source_event_id = EXCLUDED.source_event_id,
              updated_at = NOW()
          RETURNING latitude, longitude, source_event_id, updated_at
        `,
        [senderId, latitude, longitude, sourceEventId],
      );

      return {
        latitude: result.rows[0].latitude,
        longitude: result.rows[0].longitude,
        sourceEventId: result.rows[0].source_event_id,
        updatedAt: result.rows[0].updated_at,
      };
    },

    async createLocationCaptureToken(senderId, { ttlMinutes = 15 } = {}) {
      if (!senderId) {
        throw new Error("senderId is required to create a location link.");
      }

      await getSchemaReadyPromise();

      const normalizedTtl =
        Number.isFinite(ttlMinutes) && ttlMinutes > 0
          ? Math.floor(ttlMinutes)
          : 15;
      const token = randomBytes(24).toString("hex");
      const result = await activePool.query(
        `
          INSERT INTO ${LOCATION_CAPTURE_TABLE} (
            token,
            sender_psid,
            expires_at
          )
          VALUES ($1, $2, NOW() + ($3 * INTERVAL '1 minute'))
          RETURNING token, expires_at
        `,
        [token, senderId, normalizedTtl],
      );

      return {
        token: result.rows[0].token,
        expiresAt: result.rows[0].expires_at,
      };
    },

    async getLocationCaptureSession(token) {
      const normalizedToken = normalizeText(token);

      if (!normalizedToken) {
        return null;
      }

      await getSchemaReadyPromise();

      const result = await activePool.query(
        `
          SELECT
            sender_psid,
            created_at,
            expires_at,
            used_at,
            expires_at <= NOW() AS expired
          FROM ${LOCATION_CAPTURE_TABLE}
          WHERE token = $1
        `,
        [normalizedToken],
      );

      if (result.rowCount === 0) {
        return null;
      }

      const row = result.rows[0];

      return {
        senderId: row.sender_psid,
        createdAt: row.created_at,
        expiresAt: row.expires_at,
        usedAt: row.used_at,
        status: row.used_at ? "used" : row.expired ? "expired" : "ready",
      };
    },

    async saveLocationFromCapture({ token, latitude, longitude }) {
      const normalizedToken = normalizeText(token);

      if (
        !normalizedToken ||
        !Number.isFinite(latitude) ||
        !Number.isFinite(longitude)
      ) {
        throw new Error(
          "token, latitude, and longitude are required for captured location storage.",
        );
      }

      await getSchemaReadyPromise();

      const client = await activePool.connect();

      try {
        await client.query("BEGIN");

        const sessionResult = await client.query(
          `
            SELECT sender_psid, expires_at, used_at
            FROM ${LOCATION_CAPTURE_TABLE}
            WHERE token = $1
            FOR UPDATE
          `,
          [normalizedToken],
        );

        if (sessionResult.rowCount === 0) {
          await client.query("ROLLBACK");
          return {
            ok: false,
            status: "invalid",
          };
        }

        const session = sessionResult.rows[0];

        if (session.used_at) {
          await client.query("ROLLBACK");
          return {
            ok: false,
            status: "used",
          };
        }

        if (Number(new Date(session.expires_at)) <= Date.now()) {
          await client.query("ROLLBACK");
          return {
            ok: false,
            status: "expired",
          };
        }

        const locationResult = await client.query(
          `
            INSERT INTO ${LOCATION_TABLE} (
              sender_psid,
              latitude,
              longitude,
              source_event_id
            )
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (sender_psid) DO UPDATE
            SET latitude = EXCLUDED.latitude,
                longitude = EXCLUDED.longitude,
                source_event_id = EXCLUDED.source_event_id,
                updated_at = NOW()
            RETURNING latitude, longitude, source_event_id, updated_at
          `,
          [
            session.sender_psid,
            latitude,
            longitude,
            `capture:${normalizedToken.slice(0, 16)}`,
          ],
        );

        await client.query(
          `
            UPDATE ${LOCATION_CAPTURE_TABLE}
            SET used_at = NOW()
            WHERE token = $1
          `,
          [normalizedToken],
        );

        await client.query("COMMIT");

        return {
          ok: true,
          status: "saved",
          senderId: session.sender_psid,
          latitude: locationResult.rows[0].latitude,
          longitude: locationResult.rows[0].longitude,
          sourceEventId: locationResult.rows[0].source_event_id,
          updatedAt: locationResult.rows[0].updated_at,
        };
      } catch (error) {
        await client.query("ROLLBACK").catch(() => {});
        throw error;
      } finally {
        client.release();
      }
    },

    async getLatestLocation(senderId) {
      if (!senderId) {
        return null;
      }

      await getSchemaReadyPromise();

      const result = await activePool.query(
        `
          SELECT latitude, longitude, source_event_id, updated_at
          FROM ${LOCATION_TABLE}
          WHERE sender_psid = $1
        `,
        [senderId],
      );

      if (result.rowCount === 0) {
        return null;
      }

      return {
        latitude: result.rows[0].latitude,
        longitude: result.rows[0].longitude,
        sourceEventId: result.rows[0].source_event_id,
        updatedAt: result.rows[0].updated_at,
      };
    },

    async clearLatestLocation(senderId) {
      if (!senderId) {
        throw new Error("senderId is required to clear location.");
      }

      await getSchemaReadyPromise();

      const result = await activePool.query(
        `
          DELETE FROM ${LOCATION_TABLE}
          WHERE sender_psid = $1
        `,
        [senderId],
      );

      return {
        deleted: result.rowCount > 0,
      };
    },

    async getLatestImageContext(senderId) {
      if (!senderId) {
        return null;
      }

      await getSchemaReadyPromise();

      const result = await activePool.query(
        `
          SELECT summary
          FROM ${IMAGE_CONTEXT_TABLE}
          WHERE sender_psid = $1
        `,
        [senderId],
      );

      return result.rows[0]?.summary || null;
    },

    async saveConversationSummary({ senderId, summary }) {
      const normalizedSummary = normalizeText(summary);

      if (!senderId || !normalizedSummary) {
        throw new Error(
          "senderId and summary are required for conversation summaries.",
        );
      }

      await getSchemaReadyPromise();

      await activePool.query(
        `
          INSERT INTO ${SUMMARY_TABLE} (sender_psid, summary)
          VALUES ($1, $2)
          ON CONFLICT (sender_psid) DO UPDATE
          SET summary = EXCLUDED.summary,
              updated_at = NOW()
        `,
        [senderId, normalizedSummary],
      );
    },

    async getConversationSummary(senderId) {
      if (!senderId) {
        return "";
      }

      await getSchemaReadyPromise();

      const result = await activePool.query(
        `
          SELECT summary
          FROM ${SUMMARY_TABLE}
          WHERE sender_psid = $1
        `,
        [senderId],
      );

      return result.rows[0]?.summary || "";
    },

    async getSummarySourceHistory(senderId) {
      return this.getConversationHistory(senderId, {
        limit: SUMMARY_HISTORY_TURNS,
      });
    },

    async saveMemory({ senderId, content, kind = "fact", source = "auto" }) {
      const normalizedContent = normalizeMemoryText(content);
      const normalizedKind = kind === "preference" ? "preference" : "fact";
      const normalizedSource = source === "command" ? "command" : "auto";

      if (!senderId || !normalizedContent) {
        throw new Error("senderId and content are required for memory writes.");
      }

      await getSchemaReadyPromise();

      const result = await activePool.query(
        `
          INSERT INTO ${MEMORY_TABLE} (
            sender_psid,
            normalized_content,
            content,
            kind,
            source
          )
          VALUES ($1, $2, $3, $4, $5)
          ON CONFLICT (sender_psid, normalized_content) DO UPDATE
          SET content = EXCLUDED.content,
              kind = EXCLUDED.kind,
              source = EXCLUDED.source,
              updated_at = NOW(),
              last_used_at = NOW()
          RETURNING id, content, kind, source, updated_at
        `,
        [
          senderId,
          normalizedContent,
          normalizeText(content),
          normalizedKind,
          normalizedSource,
        ],
      );

      return mapMemoryRow(result.rows[0]);
    },

    async listUserMemory(senderId, { limit = 10 } = {}) {
      if (!senderId) {
        return [];
      }

      await getSchemaReadyPromise();

      const result = await activePool.query(
        `
          SELECT id, content, kind, source, updated_at
          FROM ${MEMORY_TABLE}
          WHERE sender_psid = $1
          ORDER BY last_used_at DESC, updated_at DESC, id DESC
          LIMIT $2
        `,
        [senderId, limit],
      );

      return result.rows.map(mapMemoryRow);
    },

    async findRelevantMemory(
      senderId,
      prompt,
      { limit = DEFAULT_RELEVANT_MEMORY_LIMIT } = {},
    ) {
      if (!senderId) {
        return [];
      }

      await getSchemaReadyPromise();

      const result = await activePool.query(
        `
          SELECT id, normalized_content, content, kind, source, updated_at
          FROM ${MEMORY_TABLE}
          WHERE sender_psid = $1
          ORDER BY last_used_at DESC, updated_at DESC, id DESC
          LIMIT 25
        `,
        [senderId],
      );

      if (result.rows.length === 0) {
        return [];
      }

      const normalizedPrompt = normalizeMemoryText(prompt);
      const promptTokens = tokenizeText(prompt);
      const ranked = result.rows
        .map((row) => ({
          row,
          score: scoreMemoryRow(row, promptTokens, normalizedPrompt),
        }))
        .sort((left, right) => {
          return (
            right.score - left.score ||
            Number(new Date(right.row.updated_at)) -
              Number(new Date(left.row.updated_at))
          );
        });

      const selected = ranked
        .filter((entry) => entry.score > 0)
        .slice(0, limit)
        .map((entry) => entry.row);

      const fallbackSelected =
        selected.length > 0
          ? selected
          : ranked.slice(0, Math.min(limit, 2)).map((entry) => entry.row);

      if (fallbackSelected.length > 0) {
        const ids = fallbackSelected.map((row) => row.id);

        await activePool.query(
          `
            UPDATE ${MEMORY_TABLE}
            SET last_used_at = NOW()
            WHERE id = ANY($1::bigint[])
          `,
          [ids],
        );
      }

      return fallbackSelected.map(mapMemoryRow);
    },

    async deleteMemory(senderId, query) {
      if (!senderId) {
        throw new Error("senderId is required for memory deletion.");
      }

      await getSchemaReadyPromise();

      const normalizedQuery = normalizeMemoryText(query);

      if (!normalizedQuery || normalizedQuery === "all") {
        const result = await activePool.query(
          `
            DELETE FROM ${MEMORY_TABLE}
            WHERE sender_psid = $1
            RETURNING content
          `,
          [senderId],
        );

        return {
          deletedCount: result.rowCount,
          deletedItems: result.rows.map((row) => row.content),
        };
      }

      const result = await activePool.query(
        `
          DELETE FROM ${MEMORY_TABLE}
          WHERE sender_psid = $1
            AND normalized_content LIKE '%' || $2 || '%'
          RETURNING content
        `,
        [senderId, normalizedQuery],
      );

      return {
        deletedCount: result.rowCount,
        deletedItems: result.rows.map((row) => row.content),
      };
    },

    async getOperationalSnapshot({ lookbackHours = 24 } = {}) {
      await getSchemaReadyPromise();

      const events = await activePool.query(
        `
          SELECT
            COUNT(*) FILTER (WHERE status = 'completed') AS completed_count,
            COUNT(*) FILTER (WHERE status = 'failed') AS failed_count,
            COUNT(*) FILTER (WHERE status = 'processing') AS processing_count,
            MAX(updated_at) FILTER (WHERE status = 'failed') AS last_failed_at
          FROM ${EVENT_TABLE}
          WHERE updated_at >= NOW() - ($1 * INTERVAL '1 hour')
        `,
        [lookbackHours],
      );
      const failedOutbound = await activePool.query(
        `
          SELECT
            COUNT(*) AS failed_outbound_count,
            MAX(created_at) AS last_failed_outbound_at
          FROM ${FAILED_OUTBOUND_TABLE}
          WHERE created_at >= NOW() - ($1 * INTERVAL '1 hour')
        `,
        [lookbackHours],
      );
      const memoryCount = await activePool.query(
        `
          SELECT COUNT(*) AS memory_count
          FROM ${MEMORY_TABLE}
        `,
      );

      return {
        completedCount: Number(events.rows[0]?.completed_count || 0),
        failedCount: Number(events.rows[0]?.failed_count || 0),
        processingCount: Number(events.rows[0]?.processing_count || 0),
        lastFailedAt: events.rows[0]?.last_failed_at || null,
        failedOutboundCount: Number(
          failedOutbound.rows[0]?.failed_outbound_count || 0,
        ),
        lastFailedOutboundAt:
          failedOutbound.rows[0]?.last_failed_outbound_at || null,
        memoryCount: Number(memoryCount.rows[0]?.memory_count || 0),
      };
    },

    async cleanupOperationalData({ retentionDays = 14 } = {}) {
      await getSchemaReadyPromise();

      const deletedEvents = await activePool.query(
        `
          DELETE FROM ${EVENT_TABLE}
          WHERE updated_at < NOW() - ($1 * INTERVAL '1 day')
        `,
        [retentionDays],
      );
      const deletedFailedOutbound = await activePool.query(
        `
          DELETE FROM ${FAILED_OUTBOUND_TABLE}
          WHERE created_at < NOW() - ($1 * INTERVAL '1 day')
        `,
        [retentionDays],
      );
      const deletedLocationCaptureTokens = await activePool.query(
        `
          DELETE FROM ${LOCATION_CAPTURE_TABLE}
          WHERE used_at IS NOT NULL OR expires_at < NOW()
        `,
      );

      return {
        deletedEventCount: deletedEvents.rowCount,
        deletedFailedOutboundCount: deletedFailedOutbound.rowCount,
        deletedLocationCaptureTokenCount: deletedLocationCaptureTokens.rowCount,
      };
    },

    async close() {
      if (ownsPool) {
        await activePool.end();
      }
    },
  };
}

export function getConversationStore(config) {
  if (!sharedConversationStorePromise) {
    sharedConversationStorePromise = Promise.resolve(
      createConversationStore({
        connectionString: config.postgresUrl,
      }),
    );
  }

  return sharedConversationStorePromise;
}

export async function resetConversationStoreForTesting() {
  if (!sharedConversationStorePromise) {
    return;
  }

  const store = await sharedConversationStorePromise;
  await store.close();
  sharedConversationStorePromise = undefined;
}
