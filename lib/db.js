const path = require("node:path");
const dotenv = require("dotenv");
const mysql = require("mysql2/promise");

dotenv.config();
dotenv.config({ path: path.join(__dirname, "..", ".env") });

let pool = null;
let initPromise = null;

const getDatabaseUrl = () => {
    const value = String(process.env.DATABASE_URL || "").trim();
    if (!value) {
        throw new Error("DATABASE_URL is not configured.");
    }
    return value;
};

const parseDatabaseUrl = () => {
    const parsed = new URL(getDatabaseUrl());
    if (parsed.protocol !== "mysql:") {
        throw new Error("DATABASE_URL must use the mysql:// scheme.");
    }

    return {
        host: parsed.hostname,
        port: Number(parsed.port || 3306),
        user: decodeURIComponent(parsed.username),
        password: decodeURIComponent(parsed.password),
        database: parsed.pathname.replace(/^\//, ""),
    };
};

const getPool = () => {
    if (pool) return pool;

    const connection = parseDatabaseUrl();
    pool = mysql.createPool({
        ...connection,
        waitForConnections: true,
        connectionLimit: 4,
        maxIdle: 4,
        idleTimeout: 60000,
        queueLimit: 0,
        enableKeepAlive: true,
        keepAliveInitialDelay: 0,
        ssl: {
            minVersion: "TLSv1.2",
            rejectUnauthorized: true,
        },
    });

    return pool;
};

const ensureStateTable = async () => {
    if (initPromise) return initPromise;

    initPromise = (async () => {
        const db = getPool();
        await db.query(`
            CREATE TABLE IF NOT EXISTS app_state (
                state_key VARCHAR(191) PRIMARY KEY,
                payload LONGTEXT NOT NULL,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            )
        `);
    })().catch((error) => {
        initPromise = null;
        throw error;
    });

    return initPromise;
};

const normalizeStateKey = (value) => {
    const key = String(value || "default").trim();
    return key || "default";
};

const loadAppState = async (stateKey = "default") => {
    await ensureStateTable();
    const db = getPool();
    const [rows] = await db.query(
        "SELECT payload, updated_at FROM app_state WHERE state_key = ? LIMIT 1",
        [normalizeStateKey(stateKey)],
    );

    if (!Array.isArray(rows) || !rows.length) return null;

    const row = rows[0] || {};
    let payload = row.payload;
    if (typeof payload === "string") {
        payload = payload.trim() ? JSON.parse(payload) : null;
    }

    return {
        state: payload,
        updatedAt: row.updated_at ? new Date(row.updated_at).toISOString() : null,
    };
};

const saveAppState = async (stateKey = "default", state = {}) => {
    await ensureStateTable();
    const db = getPool();
    const payload = JSON.stringify(state || {});
    await db.query(
        `
            INSERT INTO app_state (state_key, payload)
            VALUES (?, ?)
            ON DUPLICATE KEY UPDATE
                payload = VALUES(payload),
                updated_at = CURRENT_TIMESTAMP
        `,
        [normalizeStateKey(stateKey), payload],
    );

    return loadAppState(stateKey);
};

module.exports = {
    getPool,
    ensureStateTable,
    loadAppState,
    saveAppState,
};
