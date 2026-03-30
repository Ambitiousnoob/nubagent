/**
 * Library: localStorage-based persistence for research sessions.
 * Stores saved sessions with query, messages, sources, and metadata.
 */

const STORAGE_KEY = 'nubagent-library';
const MAX_SAVED_SESSIONS = 50;
const nowTimestamp = () => Date.now();

const toTimestamp = (value, fallback = nowTimestamp()) => {
    if (typeof value === 'number' && Number.isFinite(value)) return value;
    if (typeof value === 'string') {
        const parsed = Date.parse(value);
        if (Number.isFinite(parsed)) return parsed;
    }
    return fallback;
};

const normalizeSavedSession = (session) => {
    if (!session || typeof session !== 'object') return null;
    const createdAt = toTimestamp(session.createdAt);

    return {
        ...session,
        createdAt,
        updatedAt: toTimestamp(session.updatedAt, createdAt),
    };
};

export const getSessionTimestamp = (value, fallback = 0) => {
    const timestamp = toTimestamp(value, fallback);
    return Number.isFinite(timestamp) ? timestamp : fallback;
};

/**
 * @typedef {Object} SavedSession
 * @property {string} id - Session ID
 * @property {string} query - Original search query
 * @property {string} heading - Answer heading/title
 * @property {string} body - Answer body content
 * @property {Array} sources - Array of source objects
 * @property {number} createdAt - Unix timestamp in milliseconds
 * @property {number} updatedAt - Unix timestamp in milliseconds
 * @property {object} researchMeta - Research metadata
 */

/**
 * Get all saved sessions from localStorage.
 * @returns {SavedSession[]}
 */
export function getSavedSessions() {
    try {
        const raw = localStorage.getItem(STORAGE_KEY);
        if (!raw) return [];
        const parsed = JSON.parse(raw);
        return Array.isArray(parsed)
            ? parsed.map((session) => normalizeSavedSession(session)).filter(Boolean)
            : [];
    } catch {
        return [];
    }
}

/**
 * Save a session to the library.
 * @param {SavedSession} session
 * @returns {SavedSession}
 */
export function saveSession(session) {
    const sessions = getSavedSessions();
    const now = nowTimestamp();
    const normalizedSession = normalizeSavedSession(session) || session;

    const existingIndex = sessions.findIndex(s => s.id === session.id);
    const sessionToSave = {
        ...normalizedSession,
        createdAt: existingIndex >= 0 ? sessions[existingIndex].createdAt : getSessionTimestamp(normalizedSession?.createdAt, now),
        updatedAt: now,
    };

    if (existingIndex >= 0) {
        sessions[existingIndex] = sessionToSave;
    } else {
        sessions.unshift(sessionToSave);
        // Trim to max sessions
        if (sessions.length > MAX_SAVED_SESSIONS) {
            sessions.length = MAX_SAVED_SESSIONS;
        }
    }

    localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions));
    return sessionToSave;
}

export function buildSessionShareUrl(sessionId, locationLike = typeof window !== 'undefined' ? window.location : null) {
    if (!sessionId || !locationLike?.origin) return '';

    const url = new URL('/', locationLike.origin);
    url.searchParams.set('session', sessionId);
    return url.toString();
}

export function promptToCopySessionUrl(url) {
    if (!url || typeof window === 'undefined' || typeof window.prompt !== 'function') return false;
    window.prompt('Copy this link:', url);
    return true;
}

export function getSharedSessionIdFromLocation(locationLike = typeof window !== 'undefined' ? window.location : null) {
    if (!locationLike?.search) return '';
    return new URLSearchParams(locationLike.search).get('session')?.trim() || '';
}

export function buildChatSessionFromSaved(session) {
    if (!session?.id) return null;

    return {
        id: session.id,
        query: session.query || '',
        messages: [
            {
                id: `${session.id}-user`,
                role: 'user',
                text: session.query || '',
                attachments: Array.isArray(session.attachments) ? session.attachments : [],
            },
            {
                id: `${session.id}-bot`,
                role: 'bot',
                heading: session.heading || '',
                body: session.body || '',
                sources: Array.isArray(session.sources) ? session.sources : [],
                showPlanning: false,
                showSearching: false,
                searchDone: true,
                researchMeta: session.researchMeta || null,
            },
        ],
    };
}

/**
 * Get a single session by ID.
 * @param {string} id
 * @returns {SavedSession|null}
 */
export function getSessionById(id) {
    const sessions = getSavedSessions();
    return sessions.find(s => s.id === id) || null;
}

/**
 * Delete a session by ID.
 * @param {string} id
 * @returns {boolean} - True if deleted
 */
export function deleteSession(id) {
    const sessions = getSavedSessions();
    const filtered = sessions.filter(s => s.id !== id);
    if (filtered.length === sessions.length) return false;
    localStorage.setItem(STORAGE_KEY, JSON.stringify(filtered));
    return true;
}

/**
 * Clear all sessions.
 */
export function clearAllSessions() {
    localStorage.removeItem(STORAGE_KEY);
}

/**
 * Search sessions by query text.
 * @param {string} searchTerm
 * @returns {SavedSession[]}
 */
export function searchSessions(searchTerm) {
    const sessions = getSavedSessions();
    if (!searchTerm.trim()) return sessions;

    const term = searchTerm.toLowerCase();
    return sessions.filter(session =>
        session.query?.toLowerCase().includes(term) ||
        session.heading?.toLowerCase().includes(term) ||
        session.body?.toLowerCase().includes(term) ||
        session.attachments?.some((attachment) => attachment?.name?.toLowerCase().includes(term))
    );
}

/**
 * Get session count.
 * @returns {number}
 */
export function getSessionCount() {
    return getSavedSessions().length;
}

/**
 * Format date for display.
 * @param {string} isoString
 * @returns {string}
 */
export function formatSessionDate(isoString) {
    const date = new Date(isoString);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

    if (diffDays === 0) {
        return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    } else if (diffDays === 1) {
        return 'Yesterday';
    } else if (diffDays < 7) {
        return date.toLocaleDateString([], { weekday: 'short' });
    } else {
        return date.toLocaleDateString([], { month: 'short', day: 'numeric' });
    }
}
