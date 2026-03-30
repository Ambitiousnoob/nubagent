/**
 * Library: localStorage-based persistence for research sessions.
 * Stores saved sessions with query, messages, sources, and metadata.
 */

const STORAGE_KEY = 'nubagent-library';
const MAX_SAVED_SESSIONS = 50;

/**
 * @typedef {Object} SavedSession
 * @property {string} id - Session ID
 * @property {string} query - Original search query
 * @property {string} heading - Answer heading/title
 * @property {string} body - Answer body content
 * @property {Array} sources - Array of source objects
 * @property {string} createdAt - ISO timestamp
 * @property {string} updatedAt - ISO timestamp
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
        return Array.isArray(parsed) ? parsed : [];
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
    const now = new Date().toISOString();

    const existingIndex = sessions.findIndex(s => s.id === session.id);
    const sessionToSave = {
        ...session,
        createdAt: existingIndex >= 0 ? sessions[existingIndex].createdAt : now,
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
        session.body?.toLowerCase().includes(term)
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
