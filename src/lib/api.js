/**
 * API Client
 * Fetch wrappers for all API endpoints with error handling
 */

const API_BASE = '';

/**
 * Custom error class for API errors
 */
export class ApiError extends Error {
  constructor(message, status, data) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.data = data;
  }
}

/**
 * Generic fetch wrapper with error handling
 * @param {string} endpoint - API endpoint
 * @param {object} options - Fetch options
 * @returns {Promise<any>} Response data
 */
async function fetchApi(endpoint, options = {}) {
  const url = `${API_BASE}${endpoint}`;
  const { timeout = 30000, ...fetchOptions } = options;

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);

  try {
    const response = await fetch(url, {
      ...fetchOptions,
      signal: controller.signal,
      headers: {
        'Content-Type': 'application/json',
        ...fetchOptions.headers,
      },
      credentials: 'include',
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      let errorData;
      try {
        errorData = await response.json();
      } catch {
        errorData = { message: response.statusText };
      }
      throw new ApiError(
        errorData.message || `HTTP ${response.status}`,
        response.status,
        errorData
      );
    }

    // Handle no-content responses
    if (response.status === 204) {
      return null;
    }

    return response.json();
  } catch (error) {
    clearTimeout(timeoutId);

    if (error.name === 'AbortError') {
      throw new ApiError('Request timeout', 408, { message: 'Request timed out' });
    }

    if (error instanceof ApiError) {
      throw error;
    }

    throw new ApiError(
      error.message || 'Network error',
      0,
      { message: error.message }
    );
  }
}

/**
 * POST request wrapper
 */
export async function post(endpoint, data, options = {}) {
  return fetchApi(endpoint, {
    method: 'POST',
    body: JSON.stringify(data),
    ...options,
  });
}

/**
 * GET request wrapper
 */
export async function get(endpoint, options = {}) {
  return fetchApi(endpoint, {
    method: 'GET',
    ...options,
  });
}

/**
 * PUT request wrapper
 */
export async function put(endpoint, data, options = {}) {
  return fetchApi(endpoint, {
    method: 'PUT',
    body: JSON.stringify(data),
    ...options,
  });
}

/**
 * DELETE request wrapper
 */
export async function del(endpoint, options = {}) {
  return fetchApi(endpoint, {
    method: 'DELETE',
    ...options,
  });
}

/**
 * Chat API endpoints
 */
export const chatApi = {
  send: (payload, signal) =>
    fetchApi('/api/chat', {
      method: 'POST',
      body: JSON.stringify(payload),
      signal,
      timeout: 120000, // 2 minutes for chat
    }),

  health: () => get('/api/chat'),
};

/**
 * Search API endpoints
 */
export const searchApi = {
  search: (query, options = {}) =>
    post('/api/search', { query, ...options }, { timeout: 60000 }),
};

/**
 * Fetch API endpoints
 * Uses consolidated /api/content endpoint
 */
export const fetchApi_client = {
  fetch: (url) => post('/api/content?action=fetch', { url }, { timeout: 30000 }),
};

/**
 * Memory/State API endpoints
 */
export const memoryApi = {
  getState: (key) => get(`/api/state?key=${encodeURIComponent(key)}`),
  setState: (key, value) => post('/api/state', { key, value }),
  deleteState: (key) => del(`/api/state?key=${encodeURIComponent(key)}`),
};

/**
 * Library API endpoints
 */
export const libraryApi = {
  getSessions: () => get('/api/library'),
  getSession: (id) => get(`/api/library/${id}`),
  saveSession: (session) => post('/api/library', session),
  updateSession: (id, updates) => put(`/api/library/${id}`, updates),
  deleteSession: (id) => del(`/api/library/${id}`),
};

/**
 * Health check endpoint
 * Uses consolidated /api/utils endpoint
 */
export const healthApi = {
  check: () => get('/api/utils?action=health'),
};

/**
 * Analytics endpoint
 * Uses consolidated /api/utils endpoint
 */
export const analyticsApi = {
  track: (event, properties) => post('/api/utils?action=analytics', { event, properties }),
  getStats: () => get('/api/utils?action=analytics'),
};

/**
 * Export endpoint
 * Uses consolidated /api/utils endpoint
 */
export const exportApi = {
  export: (format, sessionIds) =>
    post('/api/utils?action=export', { format, sessionIds }, { timeout: 60000 }),
};

export default {
  fetch: fetchApi,
  post,
  get,
  put,
  del,
  chat: chatApi,
  search: searchApi,
  fetchUrl: fetchApi_client,
  memory: memoryApi,
  library: libraryApi,
  health: healthApi,
  analytics: analyticsApi,
  export: exportApi,
};
