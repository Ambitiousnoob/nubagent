import { create } from 'zustand';
import { persist } from 'zustand/middleware';

/**
 * Chat store for managing conversation state
 * Handles messages, loading states, and conversation metadata
 */
const createId = () => `${Date.now()}-${Math.random().toString(16).slice(2)}`;

export const useChatStore = create(
  persist(
    (set, get) => ({
      // State
      conversationId: null,
      messages: [],
      isLoading: false,
      error: null,
      attachments: [],

      // Actions
      setConversationId: (id) => set({ conversationId: id }),

      startNewConversation: () => {
        const newId = createId();
        set({
          conversationId: newId,
          messages: [],
          error: null,
          attachments: [],
        });
        return newId;
      },

      addMessage: (message) =>
        set((state) => ({
          messages: [
            ...state.messages,
            {
              id: message.id || createId(),
              role: message.role,
              content: message.content,
              timestamp: message.timestamp || Date.now(),
              attachments: message.attachments || [],
              metadata: message.metadata || {},
            },
          ],
        })),

      updateMessage: (id, updates) =>
        set((state) => ({
          messages: state.messages.map((msg) =>
            msg.id === id ? { ...msg, ...updates } : msg
          ),
        })),

      removeMessage: (id) =>
        set((state) => ({
          messages: state.messages.filter((msg) => msg.id !== id),
        })),

      setLoading: (loading) => set({ isLoading: loading }),

      setError: (error) => set({ error }),

      clearError: () => set({ error: null }),

      addAttachment: (attachment) =>
        set((state) => ({
          attachments: [...state.attachments, attachment],
        })),

      removeAttachment: (id) =>
        set((state) => ({
          attachments: state.attachments.filter((att) => att.id !== id),
        })),

      clearAttachments: () => set({ attachments: [] }),

      setMessages: (messages) => set({ messages }),

      clearConversation: () =>
        set({
          conversationId: null,
          messages: [],
          error: null,
          attachments: [],
        }),
    }),
    {
      name: 'nubagent-chat-storage',
      partialize: (state) => ({
        conversationId: state.conversationId,
        messages: state.messages,
      }),
    }
  )
);

export default useChatStore;
