import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { chatApi, searchApi, healthApi, analyticsApi, exportApi } from './api.js';

/**
 * Custom hooks for React Query data fetching
 */

/**
 * Hook for health check
 */
export function useHealthCheck() {
  return useQuery({
    queryKey: ['health'],
    queryFn: healthApi.check,
    retry: false,
    refetchInterval: 30000, // Check every 30 seconds
  });
}

/**
 * Hook for sending chat messages
 */
export function useChat() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (payload) => chatApi.send(payload),
    onSuccess: () => {
      // Invalidate health check to refresh status
      queryClient.invalidateQueries(['health']);
    },
  });
}

/**
 * Hook for search
 */
export function useSearch() {
  return useMutation({
    mutationFn: ({ query, options }) => searchApi.search(query, options),
  });
}

/**
 * Hook for tracking analytics events
 */
export function useAnalytics() {
  return useMutation({
    mutationFn: ({ event, properties }) =>
      analyticsApi.track(event, properties),
  });
}

/**
 * Hook for exporting sessions
 */
export function useExport() {
  return useMutation({
    mutationFn: ({ format, sessionIds }) =>
      exportApi.export(format, sessionIds),
  });
}

/**
 * Hook for chat with streaming support
 * @param {function} onChunk - Callback for streaming chunks
 * @returns {object} Chat functions and state
 */
export function useChatStream(onChunk) {
  const queryClient = useQueryClient();

  const sendChat = async (payload, signal) => {
    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        signal,
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const error = await response.json().catch(() => ({
          message: response.statusText,
        }));
        throw new Error(error.message || 'Chat request failed');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6);
            if (data === '[DONE]') continue;

            try {
              const parsed = JSON.parse(data);
              if (parsed.choices?.[0]?.delta?.content) {
                onChunk?.(parsed.choices[0].delta.content);
              }
            } catch {
              // Ignore parse errors for partial JSON
            }
          }
        }
      }

      queryClient.invalidateQueries(['health']);
    } catch (error) {
      if (error.name === 'AbortError') {
        throw new Error('Request cancelled');
      }
      throw error;
    }
  };

  return { sendChat };
}

export default {
  useHealthCheck,
  useChat,
  useSearch,
  useAnalytics,
  useExport,
  useChatStream,
};
