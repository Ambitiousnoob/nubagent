# NubAgent Production Development Plan

## Executive Summary

**Project:** NubAgent - AI Research Assistant Website  
**Current State:** Functional MVP with basic chat, search, and library features  
**Target State:** Production-ready, polished AI research platform  
**Timeline:** 12-16 weeks (phased approach)  
**Team Size:** 2-4 developers recommended  

### Key Recommendations

1. **Architecture:** Maintain current Vercel serverless + React SPA architecture; add Redis for caching and rate limiting
2. **Priority Focus:** UI/UX polish, error handling, performance optimization, and observability
3. **Tech Additions:** TypeScript migration, React Query for state management, Tailwind CSS for styling
4. **Critical Gaps:** No rate limiting, minimal error handling, no analytics, no testing suite

---

## 1. Current State Analysis

### 1.1 What's Working ✅

| Component | Status | Notes |
|-----------|--------|-------|
| **Chat Interface** | ✅ Functional | Real-time conversation with tool execution |
| **Web Search** | ✅ Functional | Multi-backend (DuckDuckGo, Tavily, Serper, Jina, Brave) |
| **Web Fetch** | ✅ Functional | Jina AI reader with fallbacks |
| **Library** | ✅ Functional | localStorage-based session persistence |
| **Image Upload** | ✅ Functional | Client-side OCR with Tesseract.js |
| **API Endpoints** | ✅ Functional | 9 serverless functions deployed on Vercel |
| **Memory System** | ✅ Functional | Scoped memory with MySQL backend |
| **Research Mode** | ✅ Functional | RAG re-ranking and evidence synthesis |

### 1.2 What Needs Development ⚠️

| Area | Severity | Description |
|------|----------|-------------|
| **UI/UX Polish** | High | Dated design, inconsistent styling, no theme system |
| **Error Handling** | High | Minimal user feedback on failures |
| **Loading States** | Medium | Basic spinners, no progress indicators |
| **Rate Limiting** | Critical | No protection against abuse |
| **Response Caching** | Medium | Repeated queries hit APIs every time |
| **Analytics** | Medium | No usage tracking or error monitoring |
| **Testing** | Critical | No unit, integration, or E2E tests |
| **Documentation** | Medium | API docs exist, no user guides |
| **Accessibility** | High | No ARIA labels, keyboard navigation gaps |
| **Mobile Experience** | Medium | Responsive but not optimized |
| **PWA Support** | Low | No offline capability or install prompt |
| **Export Features** | Low | No conversation export |
| **Settings Panel** | High | No UI for API keys or preferences |
| **Code Highlighting** | Medium | Basic `<code>` tags, no syntax highlighting |
| **Markdown Rendering** | Medium | Custom renderer, limited features |
| **Toast Notifications** | Medium | No notification system |

---

## 2. Architecture Overview

### 2.1 Current Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Client (Browser)                         │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  React 18 SPA (Vite)                                     │    │
│  │  ├── App.jsx (Root)                                      │    │
│  │  ├── SearchEngine.jsx (Main UI - 1881 lines)             │    │
│  │  ├── Library.jsx (Session management)                    │    │
│  │  ├── ai.jsx (Chat logic - 4588 lines)                    │    │
│  │  ├── TerminalMessage.jsx (Status display)                │    │
│  │  └── src/lib/ (Helpers: RAG, research, library)          │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ HTTP/HTTPS
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Vercel Serverless (Edge)                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  API Routes (vercel.json rewrites)                       │    │
│  │  ├── /api/chat → chat.js (Gemini AI orchestration)       │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│  ┌───────────────────────────┴──────────────────────────────┐    │
│  │  api/tools/ (Tool handlers)                               │    │
│  │  ├── calculate.js (Math expressions)                      │    │
│  │  ├── web_search.js (Multi-backend search)                 │    │
│  │  ├── web_fetch.js (Content extraction)                    │    │
│  │  ├── search_images.js (Image search)                      │    │
│  │  └── view_image.js (Image analysis)                       │    │
│  └───────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ MySQL Protocol
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Database (MySQL/TiDB)                       │
│  ┌─────────────────┐  ┌─────────────────┐                       │
│  │ agent_memory    │  │ agent_state     │                       │
│  │ - id            │  │ - id            │                       │
│  │ - scope         │  │ - scope         │                       │
│  │ - content       │  │ - state         │                       │
│  │ - embedding     │  │ - updated_at    │                       │
│  │ - created_at    │  └─────────────────┘                       │
│  └─────────────────┘                                            │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Proposed Architecture Additions

```
┌─────────────────────────────────────────────────────────────────┐
│                    New Components (Phase 1-3)                    │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │  Redis Cache    │  │  Rate Limiter   │  │  Health Check   │  │
│  │  (Upstash)      │  │  Middleware     │  │  Endpoint       │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
│                                                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │  Analytics      │  │  Error Tracking │  │  Logging        │  │
│  │  (PostHog)      │  │  (Sentry)       │  │  (Axiom)        │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  New Frontend Components                                 │    │
│  │  ├── components/ (Reusable UI)                           │    │
│  │  │   ├── ui/ (Base: Button, Input, Card, Modal)          │    │
│  │  │   ├── chat/ (MessageList, MessageInput, TypingIndicator)│  │
│  │  │   ├── search/ (SearchResults, SourceCard, Filters)    │    │
│  │  │   ├── library/ (SessionCard, SessionList, Filters)    │    │
│  │  │   ├── settings/ (SettingsPanel, ApiKeyForm, ThemeToggle)││
│  │  │   └── shared/ (Toast, LoadingSpinner, ErrorBoundary)  │    │
│  │  ├── hooks/ (Custom React hooks)                         │    │
│  │  │   ├── useChat.ts (Chat state management)              │    │
│  │  │   ├── useSearch.ts (Search with caching)              │    │
│  │  │   ├── useLibrary.ts (Library CRUD)                    │    │
│  │  │   ├── useSettings.ts (Settings persistence)           │    │
│  │  │   └── useToast.ts (Toast notifications)               │    │
│  │  ├── stores/ (Zustand state stores)                      │    │
│  │  │   ├── chatStore.ts                                    │    │
│  │  │   ├── settingsStore.ts                                │    │
│  │  │   └── uiStore.ts                                      │    │
│  │  └── utils/ (Helpers)                                    │    │
│  │      ├── markdown.ts (Enhanced markdown renderer)        │    │
│  │      ├── syntax.ts (Code highlighting)                   │    │
│  │      └── export.ts (PDF/Markdown export)                 │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Component Hierarchy (New Structure)

### 3.1 Proposed File Structure

```
nubagent/
├── src/
│   ├── main.jsx
│   ├── App.jsx
│   ├── index.css → styles/globals.css
│   │
│   ├── components/
│   │   ├── ui/
│   │   │   ├── Button.jsx
│   │   │   ├── Input.jsx
│   │   │   ├── Card.jsx
│   │   │   ├── Modal.jsx
│   │   │   ├── Dropdown.jsx
│   │   │   ├── Toggle.jsx
│   │   │   ├── Badge.jsx
│   │   │   └── Avatar.jsx
│   │   │
│   │   ├── chat/
│   │   │   ├── ChatContainer.jsx
│   │   │   ├── MessageList.jsx
│   │   │   ├── MessageInput.jsx
│   │   │   ├── MessageBubble.jsx
│   │   │   ├── TypingIndicator.jsx
│   │   │   ├── ToolCallDisplay.jsx
│   │   │   └── CodeBlock.jsx
│   │   │
│   │   ├── search/
│   │   │   ├── SearchBar.jsx
│   │   │   ├── SearchResults.jsx
│   │   │   ├── SourceCard.jsx
│   │   │   ├── SourceList.jsx
│   │   │   ├── SearchFilters.jsx
│   │   │   └── SearchSkeleton.jsx
│   │   │
│   │   ├── library/
│   │   │   ├── LibraryContainer.jsx
│   │   │   ├── SessionCard.jsx
│   │   │   ├── SessionList.jsx
│   │   │   ├── SessionDetail.jsx
│   │   │   ├── LibraryFilters.jsx
│   │   │   └── LibraryEmpty.jsx
│   │   │
│   │   ├── settings/
│   │   │   ├── SettingsPanel.jsx
│   │   │   ├── ApiKeyForm.jsx
│   │   │   ├── ModelSelector.jsx
│   │   │   ├── ThemeToggle.jsx
│   │   │   └── PreferencesForm.jsx
│   │   │
│   │   └── shared/
│   │       ├── Toast.jsx
│   │       ├── ToastContainer.jsx
│   │       ├── LoadingSpinner.jsx
│   │       ├── ErrorBoundary.jsx
│   │       ├── Header.jsx
│   │       ├── Sidebar.jsx
│   │       └── ImageUpload.jsx
│   │
│   ├── hooks/
│   │   ├── useChat.js
│   │   ├── useSearch.js
│   │   ├── useLibrary.js
│   │   ├── useSettings.js
│   │   ├── useToast.js
│   │   ├── useTheme.js
│   │   └── useDebounce.js
│   │
│   ├── stores/
│   │   ├── chatStore.js
│   │   ├── settingsStore.js
│   │   └── uiStore.js
│   │
│   ├── utils/
│   │   ├── markdown.js
│   │   ├── syntax.js
│   │   ├── export.js
│   │   ├── formatters.js
│   │   └── validators.js
│   │
│   ├── lib/
│   │   └── (existing helpers)
│   │
│   └── styles/
│       ├── globals.css
│       ├── components.css
│       ├── themes/
│       │   ├── light.css
│       │   └── dark.css
│       └── tokens.css
│
├── api/
│   ├── chat.js
│   ├── search.js
│   ├── web.js
│   ├── fetch.js
│   ├── read.js
│   ├── crawl.js
│   ├── memory.js
│   ├── state.js
│   ├── messenger.js
│   ├── health.js (NEW)
│   ├── analytics.js (NEW)
│   └── tools/
│       └── (existing tools)
│
├── public/
│   ├── manifest.json (NEW - PWA)
│   ├── sw.js (NEW - Service Worker)
│   └── icons/
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
│
└── (config files)
```

---

## 4. State Management Strategy

### 4.1 Current State Management Issues

- **Problem:** All state in `SearchEngine.jsx` and `ai.jsx` (1881 + 4588 lines)
- **Problem:** No centralized state management
- **Problem:** Heavy reliance on `useState` and `useEffect`
- **Problem:** No server state caching (repeated API calls)

### 4.2 Proposed Solution: Zustand + React Query

```javascript
// stores/chatStore.js (Zustand for UI state)
import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export const useChatStore = create(
  persist(
    (set, get) => ({
      // State
      conversations: [],
      currentConversationId: null,
      isTyping: false,
      attachments: [],
      
      // Actions
      addConversation: (conversation) => set((state) => ({
        conversations: [conversation, ...state.conversations]
      })),
      
      setCurrentConversation: (id) => set({ currentConversationId: id }),
      
      addMessage: (conversationId, message) => set((state) => ({
        conversations: state.conversations.map(conv =>
          conv.id === conversationId
            ? { ...conv, messages: [...conv.messages, message] }
            : conv
        )
      })),
      
      // ... more actions
    }),
    {
      name: 'nubagent-chat-storage',
      partialize: (state) => ({
        conversations: state.conversations,
        currentConversationId: state.currentConversationId
      })
    }
  )
);

// hooks/useChat.js (React Query for server state)
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { chatApi } from '../api/chat';

export function useChat() {
  const queryClient = useQueryClient();
  
  const sendMessage = useMutation({
    mutationFn: (payload) => chatApi.sendMessage(payload),
    onSuccess: (data) => {
      queryClient.invalidateQueries(['conversations']);
      // Handle response
    },
    onError: (error) => {
      // Handle error with toast
    }
  });
  
  return { sendMessage, isLoading: sendMessage.isPending };
}
```

### 4.3 State Flow Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                        Component Layer                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐           │
│  │ ChatView    │  │ LibraryView │  │ SettingsView│           │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘           │
│         │                │                │                   │
│         ▼                ▼                ▼                   │
│  ┌─────────────────────────────────────────────────────┐     │
│  │              Custom Hooks (useChat, etc.)            │     │
│  └──────┬──────────────────┬──────────────────┬────────┘     │
│         │                  │                  │               │
│         ▼                  ▼                  ▼               │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐         │
│  │ Zustand     │   │ React Query │   │ Local       │         │
│  │ (UI State)  │   │ (Server     │   │ Storage     │         │
│  │             │   │  State)     │   │ (Settings)  │         │
│  └─────────────┘   └──────┬──────┘   └─────────────┘         │
│                           │                                   │
│                           ▼                                   │
│                    ┌─────────────┐                           │
│                    │ API Layer   │                           │
│                    └──────┬──────┘                           │
└───────────────────────────┼───────────────────────────────────┘
                            │
                            ▼
                    ┌─────────────┐
                    │  Backend    │
                    └─────────────┘
```

---

## 5. API Contract Documentation

### 5.1 Public Endpoint

The current public backend surface is intentionally narrow:

- `GET /api/chat`
- `POST /api/chat`

Search, fetch, memory, state, research, and utility behavior now live behind the chat runtime instead of being exposed as separate public routes.

### 5.2 Chat Endpoint Enhancements

The chat endpoint needs:
- **Rate limiting headers:**
  ```
  X-RateLimit-Limit: 100
  X-RateLimit-Remaining: 95
  X-RateLimit-Reset: 1711800000
  ```
- **Standardized error format:**
  ```json
  {
    "error": {
      "code": "RATE_LIMIT_EXCEEDED",
      "message": "Too many requests. Please try again in 60 seconds.",
      "retry_after": 60
    }
  }
  ```
- **Request validation:**
  ```javascript
  // Add to all POST endpoints
  if (!body.query || typeof body.query !== 'string') {
    return res.status(400).json({
      error: {
        code: 'INVALID_REQUEST',
        message: 'Query parameter is required and must be a string'
      }
    });
  }
  ```

---

## 6. Feature Implementation Plan (Prioritized)

### Phase 1: Foundation (Weeks 1-3)
**Goal:** Establish core infrastructure and polish basic UX

| Priority | Feature | Effort | Dependencies |
|----------|---------|--------|--------------|
| P0 | Rate limiting middleware | 2 days | Redis/Upstash setup |
| P0 | Error boundary & error UI | 2 days | None |
| P0 | Toast notification system | 2 days | None |
| P1 | Loading states & skeletons | 3 days | None |
| P1 | Health check endpoint | 1 day | None |
| P1 | Request validation layer | 2 days | None |
| P2 | Response caching (Redis) | 3 days | Redis setup |
| P2 | Basic analytics (PostHog) | 2 days | None |

**Deliverables:**
- Rate-limited API endpoints (100 req/hour per IP)
- Global error boundary with retry UI
- Toast system for success/error messages
- Skeleton loaders for all async views
- Input validation on `/api/chat`
- Stable chat metadata and completion responses
- Internal tool orchestration behind the chat boundary

---

### Phase 2: UI/UX Overhaul (Weeks 4-7)
**Goal:** Modern, polished interface with professional design

| Priority | Feature | Effort | Dependencies |
|----------|---------|--------|--------------|
| P0 | Component library (base UI) | 5 days | None |
| P0 | Chat interface redesign | 4 days | Component library |
| P1 | Settings panel | 3 days | Component library |
| P1 | Dark/light theme system | 2 days | Component library |
| P1 | Improved search results display | 3 days | Component library |
| P2 | Code syntax highlighting | 2 days | None |
| P2 | Enhanced markdown rendering | 2 days | None |
| P2 | Image upload preview | 2 days | None |

**Deliverables:**
- Reusable component library (Button, Input, Card, Modal, etc.)
- Redesigned chat with message bubbles, avatars, timestamps
- Settings panel for API keys, model selection, preferences
- Theme toggle with system preference detection
- Rich search results with thumbnails, source badges, citations
- Syntax-highlighted code blocks with copy button
- Enhanced markdown (tables, task lists, math)
- Image upload with preview and OCR status

---

### Phase 3: Advanced Features (Weeks 8-11)
**Goal:** Power user features and productivity enhancements

| Priority | Feature | Effort | Dependencies |
|----------|---------|--------|--------------|
| P1 | Conversation export (PDF/MD) | 3 days | Markdown renderer |
| P1 | Keyboard shortcuts | 2 days | None |
| P1 | Library filters & search | 2 days | Component library |
| P2 | Session sharing (public links) | 4 days | Backend changes |
| P2 | Real-time collaboration | 5 days | WebSockets |
| P2 | PWA support | 3 days | Service worker |
| P3 | Voice input | 3 days | Web Speech API |
| P3 | Browser extension | 5 days | Extension APIs |

**Deliverables:**
- Export conversations as PDF or Markdown
- Keyboard shortcuts (Cmd+K, Cmd+Enter, Esc, etc.)
- Advanced library filtering (date, topic, source type)
- Shareable public links for sessions
- Real-time collaborative editing (optional)
- PWA with offline caching and install prompt
- Voice-to-text input for queries
- Chrome/Firefox extension for quick searches

---

### Phase 4: Optimization & Scale (Weeks 12-14)
**Goal:** Performance, scalability, and production readiness

| Priority | Feature | Effort | Dependencies |
|----------|---------|--------|--------------|
| P0 | TypeScript migration | 5 days | None |
| P0 | Unit test suite | 4 days | None |
| P1 | E2E test suite | 3 days | None |
| P1 | Performance optimization | 3 days | Profiling |
| P1 | SEO optimization | 2 days | None |
| P2 | CDN integration | 2 days | Vercel config |
| P2 | Database optimization | 2 days | None |
| P3 | Multi-region deployment | 3 days | Vercel Pro |

**Deliverables:**
- Full TypeScript migration with strict mode
- 80%+ unit test coverage
- E2E tests for critical user flows
- Bundle size < 100KB gzipped
- Lighthouse score > 90
- CDN caching for static assets
- Optimized database queries and indexes
- Multi-region deployment for lower latency

---

### Phase 5: Polish & Launch (Weeks 15-16)
**Goal:** Final polish, documentation, and launch preparation

| Priority | Feature | Effort | Dependencies |
|----------|---------|--------|--------------|
| P0 | Security audit | 3 days | External |
| P0 | Accessibility audit | 2 days | External |
| P1 | User documentation | 3 days | None |
| P1 | API documentation | 2 days | None |
| P1 | Onboarding flow | 2 days | None |
| P2 | Beta testing program | 3 days | None |
| P2 | Feedback system | 2 days | None |
| P3 | Marketing site | 5 days | None |

**Deliverables:**
- Third-party security audit report
- WCAG 2.1 AA compliance
- User guides and tutorials
- Updated API documentation
- First-time user onboarding
- Beta tester feedback incorporation
- In-app feedback widget
- Landing page with features and pricing

---

## 7. Technical Specifications

### 7.1 Rate Limiting Implementation

```javascript
// middleware/rate-limit.js
import { Ratelimit } from '@upstash/ratelimit';
import { Redis } from '@upstash/redis';

const redis = new Redis({
  url: process.env.UPSTASH_REDIS_REST_URL,
  token: process.env.UPSTASH_REDIS_REST_TOKEN,
});

const rateLimit = new Ratelimit({
  redis,
  limiter: Ratelimit.slidingWindow(100, '1 h'), // 100 requests per hour
  analytics: true,
  prefix: 'nubagent_ratelimit',
});

export async function checkRateLimit(identifier) {
  const { success, limit, reset, remaining } = await rateLimit.limit(identifier);
  
  return {
    success,
    headers: {
      'X-RateLimit-Limit': limit,
      'X-RateLimit-Remaining': remaining,
      'X-RateLimit-Reset': reset,
    },
  };
}

// Usage in API endpoints
export default async (req, res) => {
  const ip = req.headers['x-forwarded-for'] || 'anonymous';
  const { success, headers } = await checkRateLimit(ip);
  
  Object.entries(headers).forEach(([key, value]) => {
    res.setHeader(key, value);
  });
  
  if (!success) {
    return res.status(429).json({
      error: {
        code: 'RATE_LIMIT_EXCEEDED',
        message: 'Too many requests. Please try again later.',
        retry_after: Math.ceil((headers['X-RateLimit-Reset'] - Date.now()) / 1000),
      },
    });
  }
  
  // Continue with request...
};
```

### 7.2 Response Caching Strategy

```javascript
// lib/cache.js
import { Redis } from '@upstash/redis';

const redis = new Redis({
  url: process.env.UPSTASH_REDIS_REST_URL,
  token: process.env.UPSTASH_REDIS_REST_TOKEN,
});

const CACHE_TTL = {
  search: 300,      // 5 minutes
  fetch: 600,       // 10 minutes
  chat: 0,          // No caching for chat
};

export async function getCache(key) {
  try {
    const data = await redis.get(`cache:${key}`);
    return data ? JSON.parse(data) : null;
  } catch {
    return null;
  }
}

export async function setCache(key, value, ttl = CACHE_TTL.search) {
  try {
    await redis.setex(`cache:${key}`, ttl, JSON.stringify(value));
  } catch {
    // Cache write failures are non-fatal
  }
}

export async function invalidateCache(pattern) {
  try {
    const keys = await redis.keys(`cache:${pattern}`);
    if (keys.length) {
      await redis.del(...keys);
    }
  } catch {
    // Cache invalidation failures are non-fatal
  }
}

// Usage in search endpoint
const cacheKey = `search:${hash(query)}`;
const cached = await getCache(cacheKey);
if (cached) {
  return res.json(cached);
}

const result = await performSearch(query);
await setCache(cacheKey, result, CACHE_TTL.search);
return res.json(result);
```

### 7.3 Error Boundary Component

```jsx
// components/shared/ErrorBoundary.jsx
import React from 'react';
import { Button } from '../ui/Button';

export class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    // Log to error tracking service
    console.error('ErrorBoundary caught:', error, errorInfo);
  }

  handleRetry = () => {
    this.setState({ hasError: false, error: null });
  };

  render() {
    if (this.state.hasError) {
      return (
        <div className="error-boundary">
          <div className="error-boundary__icon">⚠️</div>
          <h2 className="error-boundary__title">Something went wrong</h2>
          <p className="error-boundary__message">
            {this.state.error?.message || 'An unexpected error occurred'}
          </p>
          <Button onClick={this.handleRetry} variant="primary">
            Try Again
          </Button>
        </div>
      );
    }

    return this.props.children;
  }
}
```

### 7.4 Toast Notification System

```jsx
// components/shared/Toast.jsx
import React, { useEffect } from 'react';

const TOAST_TYPES = {
  success: { icon: '✓', variant: 'success' },
  error: { icon: '✕', variant: 'error' },
  warning: { icon: '⚠', variant: 'warning' },
  info: { icon: 'ℹ', variant: 'info' },
};

export function Toast({ id, type, message, duration = 5000, onDismiss }) {
  useEffect(() => {
    const timer = setTimeout(() => {
      onDismiss(id);
    }, duration);
    return () => clearTimeout(timer);
  }, [id, duration, onDismiss]);

  const config = TOAST_TYPES[type] || TOAST_TYPES.info;

  return (
    <div className={`toast toast--${config.variant}`}>
      <span className="toast__icon">{config.icon}</span>
      <span className="toast__message">{message}</span>
      <button className="toast__dismiss" onClick={() => onDismiss(id)}>
        ×
      </button>
    </div>
  );
}

// hooks/useToast.js
import { useCallback } from 'react';
import { useUiStore } from '../stores/uiStore';

export function useToast() {
  const addToast = useUiStore((state) => state.addToast);
  const removeToast = useUiStore((state) => state.removeToast);

  const success = useCallback((message) => {
    addToast({ type: 'success', message });
  }, [addToast]);

  const error = useCallback((message) => {
    addToast({ type: 'error', message });
  }, [addToast]);

  const warning = useCallback((message) => {
    addToast({ type: 'warning', message });
  }, [addToast]);

  const info = useCallback((message) => {
    addToast({ type: 'info', message });
  }, [addToast]);

  return { success, error, warning, info, dismiss: removeToast };
}
```

---

## 8. Implementation Timeline

### Gantt Chart Overview

```
Week:     1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16
          │  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │
Phase 1   ██████████                                      Foundation
              │  │  │  │  │  │  │  │  │  │  │  │  │  │  │
Phase 2          ████████████████                          UI/UX
                      │  │  │  │  │  │  │  │  │  │  │  │  │
Phase 3                  ████████████████                  Advanced
                              │  │  │  │  │  │  │  │  │  │
Phase 4                          ████████████              Optimization
                                  │  │  │  │  │  │  │  │
Phase 5                              ████████              Launch
```

### Milestone Dates (Starting March 30, 2026)

| Milestone | Target Date | Deliverables |
|-----------|-------------|--------------|
| **M1: Foundation Complete** | April 20, 2026 | Rate limiting, error handling, caching, analytics |
| **M2: UI/UX Complete** | May 18, 2026 | New component library, redesigned chat, themes |
| **M3: Features Complete** | June 15, 2026 | Export, shortcuts, PWA, advanced library |
| **M4: Optimization Complete** | June 29, 2026 | TypeScript, tests, performance optimized |
| **M5: Launch Ready** | July 13, 2026 | Audits passed, docs complete, beta tested |

---

## 9. Risk Assessment

### Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Vercel serverless cold starts | Medium | Medium | Implement keep-alive pings, consider dedicated instances |
| Redis cache consistency | Low | Medium | Use Upstash managed Redis with replication |
| API rate limits from providers | High | High | Implement request queuing, fallback providers |
| Bundle size bloat | Medium | Medium | Regular bundle analysis, code splitting |
| TypeScript migration complexity | Medium | Low | Incremental migration, strict null checks |

### Business Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| API cost overruns | Medium | High | Usage monitoring, quota alerts, caching |
| User adoption | Medium | High | Beta testing, feedback loops, iteration |
| Competition | High | Medium | Focus on unique features (RAG, research mode) |
| Maintenance burden | Medium | Medium | Documentation, testing, automation |

### Operational Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Database failures | Low | High | Automated backups, read replicas |
| Security breaches | Medium | Critical | Regular audits, dependency updates |
| Downtime | Low | High | Health monitoring, alerting, runbooks |
| Data loss | Low | Critical | Point-in-time recovery, versioned backups |

---

## 10. Cost Estimates

### Infrastructure Costs (Monthly)

| Service | Tier | Cost | Notes |
|---------|------|------|-------|
| **Vercel** | Pro | $20 | Required for serverless functions |
| **Upstash Redis** | Pay-as-you-go | ~$10 | Rate limiting + caching |
| **MySQL/TiDB** | Basic | $0-25 | TiDB Serverless free tier or PlanetScale |
| **PostHog** | Free tier | $0 | Up to 1M events/month |
| **Sentry** | Team | $26 | Error tracking |
| **Axiom** | Basic | $49 | Log management |
| **Total** | | **$105-130/month** | |

### API Costs (Variable)

| Provider | Free Tier | Paid | Notes |
|----------|-----------|------|-------|
| **Google Gemini** | 60 req/min free | $0.000125-0.00025/1K tokens | Primary AI model |
| **Tavily** | 1,000 searches/month | $0.001/search | Search backend |
| **Serper** | 2,500 searches/month | $0.0004/search | Google dork support |
| **Jina AI** | 1M tokens/month | $0.0001/1K tokens | Reader API |
| **Estimated Total** | | **$50-200/month** | Depends on usage |

### Development Costs

| Role | Hours | Rate | Total |
|------|-------|------|-------|
| Senior Frontend Dev | 320 | $75/hr | $24,000 |
| Senior Backend Dev | 160 | $75/hr | $12,000 |
| QA Engineer | 80 | $60/hr | $4,800 |
| Designer (part-time) | 80 | $80/hr | $6,400 |
| **Total** | **640** | | **$47,200** |

**Total Project Cost:** ~$50,000-55,000 (one-time) + $150-350/month (ongoing)

---

## 11. Success Metrics

### Technical KPIs

| Metric | Current | Target | Measurement |
|--------|---------|--------|-------------|
| Lighthouse Performance | ~70 | >90 | Lighthouse CI |
| Bundle Size (gzipped) | 66 KB | <50 KB | Bundle analyzer |
| API Response Time (p95) | ~800ms | <500ms | Axiom logs |
| Error Rate | Unknown | <0.1% | Sentry |
| Uptime | Unknown | >99.9% | Uptime monitoring |
| Test Coverage | 0% | >80% | Vitest/Coveralls |

### User KPIs

| Metric | Target | Measurement |
|--------|--------|-------------|
| Daily Active Users | 100+ (Month 1) | PostHog |
| Session Duration | >5 minutes | PostHog |
| Retention (D7) | >40% | PostHog |
| NPS Score | >50 | Survey |
| Support Tickets | <5/week | Help desk |

### Business KPIs

| Metric | Target | Measurement |
|--------|--------|-------------|
| API Cost per User | <$0.50/month | Internal |
| Infrastructure Cost per User | <$1/month | Internal |
| Conversion (Free → Paid) | 5-10% | Stripe |
| MRR Growth | 20% MoM | Stripe |

---

## 12. Next Steps

### Immediate Actions (This Week)

1. **Set up development environment**
   - Create `.env.example` with all required variables
   - Document local development setup in README

2. **Initialize new dependencies**
   ```bash
   npm install zustand @tanstack/react-query
   npm install @sentry/react posthog-js
   npm install -D vitest @testing-library/react
   ```

3. **Create Upstash Redis instance**
   - Set up at https://upstash.com
   - Add `UPSTASH_REDIS_REST_URL` and `UPSTASH_REDIS_REST_TOKEN` to Vercel

4. **Set up monitoring**
   - Create Sentry project
   - Create PostHog project
   - Add DSN keys to environment

5. **Create project board**
   - Set up GitHub Projects or Linear
   - Create tickets for Phase 1 features

### Week 1 Sprint Goals

- [ ] Implement rate limiting middleware
- [ ] Add global error boundary
- [ ] Create toast notification system
- [ ] Add health check endpoint
- [ ] Set up Sentry error tracking
- [ ] Set up PostHog analytics

---

## Appendix A: Environment Variables

### Required

```bash
# AI Providers
GEMINI_API_KEY=your_gemini_api_key

# Database
DATABASE_URL=mysql://user:password@host:3306/database

# Redis (Upstash)
UPSTASH_REDIS_REST_URL=https://xxx.upstash.io
UPSTASH_REDIS_REST_TOKEN=your_token

# Monitoring
SENTRY_DSN=https://xxx@sentry.io/xxx
POSTHOG_API_KEY=phc_xxx
POSTHOG_HOST=https://app.posthog.com
```

### Optional

```bash
# Search Providers
TAVILY_API_KEY=your_tavily_api_key
SERPER_API_KEY=your_serper_api_key
JINA_API_KEY=your_jina_api_key
BRAVE_API_KEY=your_brave_api_key

# Memory
CEREBRAS_API_KEY=your_cerebras_api_key

# Messenger
MESSENGER_VERIFY_TOKEN=your_verify_token
PAGE_ACCESS_TOKEN=your_page_access_token
```

---

## Appendix B: Recommended Dependencies

### Production Dependencies

```json
{
  "dependencies": {
    "@google/genai": "^1.46.0",
    "@sentry/react": "^7.0.0",
    "@tanstack/react-query": "^5.0.0",
    "posthog-js": "^1.0.0",
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "tesseract.js": "^7.0.0",
    "zustand": "^4.5.0",
    "@upstash/ratelimit": "^1.0.0",
    "@upstash/redis": "^1.0.0",
    "react-markdown": "^9.0.0",
    "rehype-highlight": "^7.0.0",
    "remark-gfm": "^4.0.0",
    "react-hot-toast": "^2.4.0",
    "date-fns": "^3.0.0",
    "zod": "^3.22.0"
  }
}
```

### Development Dependencies

```json
{
  "devDependencies": {
    "@testing-library/react": "^14.0.0",
    "@testing-library/jest-dom": "^6.0.0",
    "@types/react": "^18.2.0",
    "@types/react-dom": "^18.2.0",
    "@vitejs/plugin-react": "^4.0.0",
    "vitest": "^1.0.0",
    "@vitest/coverage-v8": "^1.0.0",
    "typescript": "^5.0.0",
    "eslint": "^8.0.0",
    "prettier": "^3.0.0",
    "vite": "^5.0.0"
  }
}
```

---

## Appendix C: Testing Strategy

### Unit Tests

```javascript
// tests/unit/utils/formatters.test.js
import { describe, it, expect } from 'vitest';
import { formatBytes, formatDuration } from '../../../src/utils/formatters';

describe('formatBytes', () => {
  it('formats bytes correctly', () => {
    expect(formatBytes(0)).toBe('0 B');
    expect(formatBytes(1024)).toBe('1 KB');
    expect(formatBytes(1048576)).toBe('1 MB');
  });
});

describe('formatDuration', () => {
  it('formats duration correctly', () => {
    expect(formatDuration(1000)).toBe('1s');
    expect(formatDuration(60000)).toBe('1m');
    expect(formatDuration(3600000)).toBe('1h');
  });
});
```

### Integration Tests

```javascript
// tests/integration/api/chat.test.js
import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { startServer, stopServer } from '../../utils/server';

describe('GET /api/chat', () => {
  let server;
  
  beforeAll(async () => {
    server = await startServer();
  });
  
  afterAll(async () => {
    await stopServer(server);
  });
  
  it('returns endpoint metadata', async () => {
    const response = await fetch(`${server.url}/api/chat`);
    const data = await response.json();
    
    expect(response.status).toBe(200);
    expect(data).toHaveProperty('model');
    expect(data).toHaveProperty('tools');
  });
});
```

### E2E Tests

```javascript
// tests/e2e/chat.spec.js
import { test, expect } from '@playwright/test';

test.describe('Chat Flow', () => {
  test('sends a message and receives a response', async ({ page }) => {
    await page.goto('/');
    
    // Type message
    await page.getByTestId('chat-input').fill('What is quantum computing?');
    
    // Send message
    await page.getByTestId('send-button').click();
    
    // Wait for response
    await expect(page.getByTestId('message-list'))
      .toContainText('Quantum computing');
    
    // Check sources appear
    await expect(page.getByTestId('sources-section'))
      .toBeVisible();
  });
});
```

---

**Document Version:** 1.0  
**Last Updated:** March 30, 2026  
**Author:** Principal System Architect  
**Status:** Ready for Implementation
