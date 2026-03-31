import pg from "pg";

const { Pool } = pg;

export const MAX_PROMPT_TURNS = 12;
export const STORED_HISTORY_LIMIT = MAX_PROMPT_TURNS - 1;
const MESSAGE_TABLE = "messenger_messages";
const IMAGE_CONTEXT_TABLE = "messenger_image_context";

let sharedConversationStorePromise;

function createPool(connectionString) {
  return new Pool({
    connectionString,
  });
}

function normalizeText(text) {
  return typeof text === "string" ? text.trim() : "";
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

    async getConversationHistory(senderId, { excludeMessageId = null } = {}) {
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
        [senderId, excludeMessageId, historyLimit],
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
