# NubAgent Architecture Documentation

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                         CLIENT LAYER                                             │
│                                      (Browser / PWA)                                             │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │                                    React 18 SPA (Vite)                                   │   │
│  │  ┌───────────────────────────────────────────────────────────────────────────────────┐  │   │
│  │  │                              App.jsx (Root Component)                              │  │   │
│  │  └───────────────────────────────────────────────────────────────────────────────────┘  │   │
│  │                                       │                                                  │   │
│  │         ┌─────────────────────────────┼─────────────────────────────┐                   │   │
│  │         │                             │                             │                   │   │
│  │         ▼                             ▼                             ▼                   │   │
│  │  ┌─────────────┐             ┌─────────────┐               ┌─────────────┐             │   │
│  │  │   Header    │             │  Main View  │               │   Sidebar   │             │   │
│  │  │  Component  │             │  Router     │               │  Component  │             │   │
│  │  └─────────────┘             └──────┬──────┘               └─────────────┘             │   │
│  │                                     │                                                   │   │
│  │         ┌────────────────────────────┼────────────────────────────┐                     │   │
│  │         │                            │                            │                     │   │
│  │         ▼                            ▼                            ▼                     │   │
│  │  ┌─────────────┐            ┌─────────────┐              ┌─────────────┐               │   │
│  │  │  ChatView   │            │ LibraryView │              │ SettingsView│               │   │
│  │  │  (Primary)  │            │  (History)  │              │ (Config)    │               │   │
│  │  └──────┬──────┘            └──────┬──────┘              └──────┬──────┘               │   │
│  │         │                          │                            │                       │   │
│  │         └──────────────────────────┼────────────────────────────┘                       │   │
│  │                                    │                                                    │   │
│  │                                    ▼                                                    │   │
│  │  ┌─────────────────────────────────────────────────────────────────────────────────┐   │   │
│  │  │                           Component Library (Reusable)                           │   │   │
│  │  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │   │   │
│  │  │  │  Button  │ │   Input  │ │   Card   │ │   Modal  │ │  Toggle  │ │  Badge   │  │   │   │
│  │  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘  │   │   │
│  │  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │   │   │
│  │  │  │  Toast   │ │ Spinner  │ │  Avatar  │ │ Dropdown │ │   Tag    │ │ Skeleton │  │   │   │
│  │  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘  │   │   │
│  │  └─────────────────────────────────────────────────────────────────────────────────┘   │   │
│  │                                                                                         │   │
│  │  ┌─────────────────────────────────────────────────────────────────────────────────┐   │   │
│  │  │                            State Management Layer                                │   │   │
│  │  │  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐      │   │   │
│  │  │  │   Zustand Stores    │  │  React Query Cache  │  │   Local Storage     │      │   │   │
│  │  │  │  (UI State: chat,   │  │  (Server State:     │  │   (Settings: theme, │      │   │   │
│  │  │  │   settings, ui)     │  │   conversations,    │  │    api keys, prefs) │      │   │   │
│  │  │  │                     │  │   search results)   │  │                     │      │   │   │
│  │  │  └─────────────────────┘  └─────────────────────┘  └─────────────────────┘      │   │   │
│  │  └─────────────────────────────────────────────────────────────────────────────────┘   │   │
│  │                                                                                         │   │
│  │  ┌─────────────────────────────────────────────────────────────────────────────────┐   │   │
│  │  │                              Custom Hooks Layer                                  │   │   │
│  │  │  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐          │   │   │
│  │  │  │  useChat  │ │ useSearch │ │ useLibrary│ │ useSettings│ │  useToast │          │   │   │
│  │  │  └───────────┘ └───────────┘ └───────────┘ └───────────┘ └───────────┘          │   │   │
│  │  └─────────────────────────────────────────────────────────────────────────────────┘   │   │
│  │                                                                                         │   │
│  │  ┌─────────────────────────────────────────────────────────────────────────────────┐   │   │
│  │  │                            Utility Functions                                     │   │   │
│  │  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐            │   │   │
│  │  │  │  Markdown    │ │   Syntax     │ │    Export    │ │  Formatters  │            │   │   │
│  │  │  │  Renderer    │ │  Highlight   │ │  (PDF/MD)    │ │  (date, size)│            │   │   │
│  │  │  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘            │   │   │
│  │  └─────────────────────────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         │ HTTPS / JSON
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                        API GATEWAY                                              │
│                                     (Vercel Edge Network)                                       │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │                              Vercel Serverless Functions                                 │   │
│  │                                                                                          │   │
│  │  ┌────────────────────────────────────────────────────────────────────────────────┐     │   │
│  │  │                           Middleware Layer (New)                                │     │   │
│  │  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐           │     │   │
│  │  │  │    CORS      │ │     Rate     │ │    Request   │ │    Error     │           │     │   │
│  │  │  │    Handler   │ │   Limiting   │ │  Validation  │ │   Handling   │           │     │   │
│  │  │  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘           │     │   │
│  │  └────────────────────────────────────────────────────────────────────────────────┘     │   │
│  │                                                                                          │   │
│  │  ┌────────────────────────────────────────────────────────────────────────────────┐     │   │
│  │  │                            Public API Endpoints                                 │     │   │
│  │  │                                                                                │     │   │
│  │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │     │   │
│  │  │  │ POST /chat  │  │ POST /web   │  │ POST /fetch │  │ POST /read  │            │     │   │
│  │  │  │             │  │             │  │             │  │             │            │     │   │
│  │  │  │ AI Chat     │  │ Web         │  │ URL Content │  │ Single URL  │            │     │   │
│  │  │  │ w/ Tools    │  │ Research    │  │ Extraction  │  │ Reader      │            │     │   │
│  │  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘            │     │   │
│  │  │                                                                                │     │   │
│  │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │     │   │
│  │  │  │ POST /crawl │  │ POST /search│  │ POST /memory│  │ POST /state │            │     │   │
│  │  │  │             │  │             │  │             │  │             │            │     │   │
│  │  │  │ Site        │  │ Web Search  │  │ Scoped      │  │ App State   │            │     │   │
│  │  │  │ Crawler     │  │ (Dorking)   │  │ Memory CRUD │  │ Persistence │            │     │   │
│  │  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘            │     │   │
│  │  │                                                                                │     │   │
│  │  │  ┌─────────────┐  ┌─────────────┐                                              │     │   │
│  │  │  │ GET  /health│  │ POST /analytics│                                           │     │   │
│  │  │  │             │  │             │                                              │     │   │
│  │  │  │ Health      │  │ Usage       │                                              │     │   │
│  │  │  │ Check       │  │ Tracking    │                                              │     │   │
│  │  │  └─────────────┘  └─────────────┘                                              │     │   │
│  │  └────────────────────────────────────────────────────────────────────────────────┘     │   │
│  │                                                                                          │   │
│  │  ┌────────────────────────────────────────────────────────────────────────────────┐     │   │
│  │  │                         Tool Handlers (api/tools/)                              │     │   │
│  │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │     │   │
│  │  │  │  calculate  │  │  web_search │  │  web_fetch  │  │ search_images│           │     │   │
│  │  │  │             │  │             │  │             │  │             │            │     │   │
│  │  │  │  Math       │  │  Multi-     │  │  Content    │  │  Image      │            │     │   │
│  │  │  │  Expressions│  │  backend    │  │  Extraction │  │  Search     │            │     │   │
│  │  │  │             │  │  Search     │  │             │  │             │            │     │   │
│  │  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘            │     │   │
│  │  │                                                                                │     │   │
│  │  │  ┌─────────────┐                                                               │     │   │
│  │  │  │ view_image  │                                                               │     │   │
│  │  │  │             │                                                               │     │   │
│  │  │  │ Image       │                                                               │     │   │
│  │  │  │ Analysis    │                                                               │     │   │
│  │  │  └─────────────┘                                                               │     │   │
│  │  └────────────────────────────────────────────────────────────────────────────────┘     │   │
│  │                                                                                          │   │
│  │  ┌────────────────────────────────────────────────────────────────────────────────┐     │   │
│  │  │                          Shared Libraries (lib/)                                │     │   │
│  │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │     │   │
│  │  │  │  litehost-  │  │   chat-     │  │   api-key-  │  │    cache    │            │     │   │
│  │  │  │   chat.js   │  │  memory.js  │  │  memory.js  │  │    (Redis)  │            │     │   │
│  │  │  │             │  │             │  │             │  │             │            │     │   │
│  │  │  │  Gemini AI  │  │  Scoped     │  │  Persistent │  │  Response   │            │     │   │
│  │  │  │  Orchest.   │  │  Memory     │  │  Memory     │  │  Caching    │            │     │   │
│  │  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘            │     │   │
│  │  │                                                                                │     │   │
│  │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │     │   │
│  │  │  │    rag.js   │  │    web.js   │  │    db.js    │  │   security  │            │     │   │
│  │  │  │             │  │             │  │             │  │             │            │     │   │
│  │  │  │  RAG        │  │  Security   │  │  Database   │  │  Validation │            │     │   │
│  │  │  │  Ranking    │  │  (IP block) │  │  Connection │  │  & Sanitize │            │     │   │
│  │  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘            │     │   │
│  │  └────────────────────────────────────────────────────────────────────────────────┘     │   │
│  └─────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
                                         │
                    ┌────────────────────┼────────────────────┐
                    │                    │                    │
                    ▼                    ▼                    ▼
┌─────────────────────────┐ ┌─────────────────┐ ┌─────────────────────────┐
│   External AI APIs      │ │   Search APIs   │ │    Data Stores          │
├─────────────────────────┤ ├─────────────────┤ ├─────────────────────────┤
│ ┌─────────────────────┐ │ │ ┌─────────────┐ │ │ ┌─────────────────────┐ │
│ │ Google Gemini API   │ │ │ │   Tavily    │ │ │ │  MySQL/TiDB         │ │
│ │ (Primary AI Model)  │ │ │ │   Search    │ │ │ │  (Memory & State)   │ │
│ └─────────────────────┘ │ │ └─────────────┘ │ │ └─────────────────────┘ │
│                         │ │                 │ │                         │
│ ┌─────────────────────┐ │ │ ┌─────────────┐ │ │ ┌─────────────────────┐ │
│ │ Cerebras API        │ │ │ │   Serper    │ │ │ │  Upstash Redis      │ │
│ │ (Memory Reranking)  │ │ │ │   (Google)  │ │ │ │  (Cache & Rate      │ │
│ └─────────────────────┘ │ │ └─────────────┘ │ │ │   Limiting)         │ │
│                         │ │                 │ │ └─────────────────────┘ │
│ ┌─────────────────────┐ │ │ ┌─────────────┐ │ │                         │
│ │ Jina AI Reader      │ │ │ │  DuckDuckGo │ │ │ ┌─────────────────────┐ │
│ │ (Content Extract)   │ │ │ │   Search    │ │ │ │  Vercel Blob        │ │
│ └─────────────────────┘ │ │ └─────────────┘ │ │ │  (File Storage)     │ │
│                         │ │                 │ │ └─────────────────────┘ │
│ ┌─────────────────────┐ │ │ ┌─────────────┐ │ │                         │
│ │ Firecrawl           │ │ │ │   Brave     │ │ │                         │
│ │ (Web Scraper)       │ │ │ │   Search    │ │ │                         │
│ └─────────────────────┘ │ │ └─────────────┘ │ │                         │
└─────────────────────────┘ └─────────────────┘ └─────────────────────────┘
```

---

## Data Flow Diagrams

### Chat Request Flow

```
┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│  User    │     │  Frontend│     │  Vercel  │     │  Gemini  │     │  Tools   │
│          │     │  (React) │     │  (API)   │     │   AI     │     │          │
└────┬─────┘     └────┬─────┘     └────┬─────┘     └────┬─────┘     └────┬─────┘
     │                │                │                │                │
     │ 1. Type query  │                │                │                │
     │───────────────>│                │                │                │
     │                │                │                │                │
     │                │ 2. Validate    │                │                │
     │                │───────────────>│                │                │
     │                │                │                │                │
     │                │                │ 3. Rate limit  │                │
     │                │                │    check       │                │
     │                │                │                │                │
     │                │                │ 4. Load memory │                │
     │                │                │───────────────>│                │
     │                │                │                │                │
     │                │                │ 5. Send to AI  │                │
     │                │                │───────────────>│                │
     │                │                │                │                │
     │                │                │                │ 6. Tool call   │
     │                │                │                │───────────────>│
     │                │                │                │                │
     │                │                │                │ 7. Tool result │
     │                │                │                │<───────────────│
     │                │                │                │                │
     │                │                │ 8. AI response │                │
     │                │                │<───────────────│                │
     │                │                │                │                │
     │                │ 9. Save memory │                │                │
     │                │<───────────────│                │                │
     │                │                │                │                │
     │                │ 10. Display    │                │                │
     │<───────────────│                │                │                │
     │                │                │                │                │
```

### Search Request Flow (with Caching)

```
┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│  User    │     │  Frontend│     │  Vercel  │     │  Redis   │     │  Search  │
│          │     │  (React) │     │  (API)   │     │  Cache   │     │  APIs    │
└────┬─────┘     └────┬─────┘     └────┬─────┘     └────┬─────┘     └────┬─────┘
     │                │                │                │                │
     │ 1. Search      │                │                │                │
     │───────────────>│                │                │                │
     │                │                │                │                │
     │                │ 2. Check cache │                │                │
     │                │───────────────>│                │                │
     │                │                │                │                │
     │                │                │ 3. Cache miss  │                │
     │                │                │<───────────────│                │
     │                │                │                │                │
     │                │                │ 4. Query search│                │
     │                │                │───────────────────────────────>│
     │                │                │                │                │
     │                │                │ 5. Results     │                │
     │                │                │<───────────────────────────────│
     │                │                │                │                │
     │                │                │ 6. Store cache │                │
     │                │                │───────────────>│                │
     │                │                │                │                │
     │                │ 7. Return      │                │                │
     │                │<───────────────│                │                │
     │                │                │                │                │
     │ 8. Display     │                │                │                │
     │<───────────────│                │                │                │
     │                │                │                │                │
```

---

## Component Hierarchy Tree

```
App
├── ErrorBoundary
│   └── ToastContainer
│       └── [Toast]*
│
├── Header
│   ├── Logo
│   ├── ThemeToggle
│   └── Navigation
│       ├── ChatTab
│       ├── LibraryTab
│       └── SettingsTab
│
├── MainContent (Router)
│   │
│   ├── ChatView
│   │   ├── ChatContainer
│   │   │   ├── MessageList
│   │   │   │   └── MessageBubble*
│   │   │   │       ├── Avatar
│   │   │   │       ├── MessageContent
│   │   │   │       │   ├── MarkdownRenderer
│   │   │   │       │   │   └── CodeBlock (with syntax highlighting)
│   │   │   │       │   └── SourceCitations
│   │   │   │       │       └── SourceBadge*
│   │   │   │       ├── Timestamp
│   │   │   │       └── MessageActions
│   │   │   │           ├── CopyButton
│   │   │   │           └── RegenerateButton
│   │   │   │
│   │   │   ├── TypingIndicator
│   │   │   │
│   │   │   └── MessageInput
│   │   │       ├── TextArea
│   │   │       ├── AttachmentList
│   │   │       │   └── AttachmentPreview*
│   │   │       │       ├── ImageThumbnail
│   │   │       │       └── RemoveButton
│   │   │       ├── ImageUpload
│   │   │       ├── SendButton
│   │   │       └── StopButton
│   │   │
│   │   └── ResearchProgress
│   │       ├── PlanningCard
│   │       ├── SearchingCard
│   │       └── SourcePills
│   │
│   ├── LibraryView
│   │   ├── LibraryHeader
│   │   │   ├── Title
│   │   │   └── BackButton
│   │   │
│   │   ├── LibrarySearch
│   │   │   ├── SearchInput
│   │   │   └── ResultCount
│   │   │
│   │   ├── LibraryFilters
│   │   │   ├── DateFilter
│   │   │   ├── TopicFilter
│   │   │   └── SourceTypeFilter
│   │   │
│   │   ├── SessionList
│   │   │   └── SessionCard*
│   │   │       ├── SessionHeader
│   │   │       │   ├── DateBadge
│   │   │       │   └── SourceCount
│   │   │       ├── SessionPreview
│   │   │       │   ├── QueryTitle
│   │   │       │   └── AnswerSnippet
│   │   │       ├── SourcePreview
│   │   │       │   └── Favicon
│   │   │       └── SessionActions
│   │   │           ├── OpenButton
│   │   │           └── DeleteButton
│   │   │
│   │   └── SessionDetail (Modal)
│   │       ├── SessionContent
│   │       │   ├── QueryDisplay
│   │       │   ├── AnswerDisplay
│   │       │   └── SourcesList
│   │       └── SessionActions
│   │           ├── ExportButton
│   │           └── CloseButton
│   │
│   └── SettingsView
│       ├── SettingsHeader
│       │
│       ├── ApiKeySection
│       │   ├── ApiKeyForm
│       │   │   ├── GeminiKeyInput
│       │   │   ├── SearchApiKeyInput
│       │   │   └── SaveButton
│       │   └── KeyStatus
│       │       └── ValidityBadge
│       │
│       ├── ModelSection
│       │   └── ModelSelector
│       │       └── ModelOption*
│       │
│       ├── AppearanceSection
│       │   └── ThemeToggle
│       │       ├── LightOption
│       │       ├── DarkOption
│       │       └── SystemOption
│       │
│       └── PreferencesSection
│           ├── ResearchModeToggle
│           ├── AutoSaveToggle
│           └── AnalyticsToggle
│
├── Sidebar (Collapsible)
│   ├── ConversationHistory
│   │   └── ConversationItem*
│   │       ├── ConversationTitle
│   │       └── DeleteButton
│   │
│   └── NewChatButton
│
└── Footer
    ├── VersionInfo
    └── StatusIndicator
```

---

## State Management Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Application State                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                    UI State (Zustand)                       │ │
│  │  - Current view (chat/library/settings)                     │ │
│  │  - Sidebar open/close                                       │ │
│  │  - Modal states                                             │ │
│  │  - Toast notifications                                      │ │
│  │  - Loading states                                           │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                  Chat State (Zustand)                       │ │
│  │  - conversations[]                                          │ │
│  │    - id, title, createdAt                                   │ │
│  │    - messages[]                                             │ │
│  │      - role, content, timestamp, attachments                │ │
│  │  - currentConversationId                                    │ │
│  │  - isTyping                                                 │ │
│  │  - attachments[]                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │               Server State (React Query)                    │ │
│  │  - conversations (list)                                     │ │
│  │  - conversation (detail)                                    │ │
│  │  - search results                                           │ │
│  │  - library sessions                                         │ │
│  │  - Cache:                                                   │ │
│  │    - Query invalidation                                     │ │
│  │    - Refetch on window focus                                │ │
│  │    - Optimistic updates                                     │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              Settings State (Zustand + localStorage)        │ │
│  │  - theme (light/dark/system)                                │ │
│  │  - apiKeys (encrypted)                                      │ │
│  │  - modelSelection                                           │ │
│  │  - preferences                                              │ │
│  │    - researchMode                                           │ │
│  │    - autoSave                                               │ │
│  │    - analytics                                              │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Database Schema

### agent_memory Table

```sql
CREATE TABLE agent_memory (
  id VARCHAR(255) PRIMARY KEY,
  scope VARCHAR(255) NOT NULL,
  scope_type ENUM('api_key', 'state_key', 'anonymous') NOT NULL,
  content TEXT NOT NULL,
  embedding VECTOR(1536),  -- For semantic search
  role ENUM('user', 'assistant', 'system') NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  
  INDEX idx_scope (scope),
  INDEX idx_scope_type (scope_type),
  INDEX idx_created_at (created_at)
);
```

### agent_state Table

```sql
CREATE TABLE agent_state (
  id VARCHAR(255) PRIMARY KEY,
  scope VARCHAR(255) NOT NULL,
  scope_type ENUM('api_key', 'state_key', 'anonymous') NOT NULL,
  state JSON NOT NULL,
  version INT DEFAULT 1,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  
  INDEX idx_scope (scope),
  INDEX idx_scope_type (scope_type)
);
```

### agent_analytics Table (New)

```sql
CREATE TABLE agent_analytics (
  id VARCHAR(255) PRIMARY KEY,
  event_type VARCHAR(100) NOT NULL,
  session_id VARCHAR(255),
  user_id VARCHAR(255),
  properties JSON,
  metadata JSON,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  
  INDEX idx_event_type (event_type),
  INDEX idx_session_id (session_id),
  INDEX idx_created_at (created_at)
);
```

---

## API Endpoint Summary

| Endpoint | Method | Description | Auth | Rate Limit |
|----------|--------|-------------|------|------------|
| `/api/chat` | POST | AI chat with tools | Optional | 100/hr |
| `/api/chat` | GET | Endpoint metadata | None | 1000/hr |
| `/api/web` | POST | Combined web research | None | 60/hr |
| `/api/web` | GET | Endpoint metadata | None | 1000/hr |
| `/api/search` | POST | Web search | None | 60/hr |
| `/api/search` | GET | Endpoint metadata | None | 1000/hr |
| `/api/fetch` | POST | URL content extraction | None | 60/hr |
| `/api/read` | POST | Single URL reader | None | 60/hr |
| `/api/crawl` | POST | Site crawler | None | 20/hr |
| `/api/memory` | POST | Scoped memory CRUD | Required | 100/hr |
| `/api/memory` | GET | Endpoint metadata | None | 1000/hr |
| `/api/state` | POST | App state persistence | Optional | 100/hr |
| `/api/state` | GET | Load app state | Optional | 100/hr |
| `/api/health` | GET | Health check | None | Unlimited |
| `/api/analytics` | POST | Usage tracking | None | 1000/hr |
| `/api/cache` | DELETE | Invalidate cache | Admin | 10/hr |

---

## Security Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Security Layers                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Layer 1: Network Security                                       │
│  ├── HTTPS/TLS (Vercel managed)                                 │
│  ├── CORS headers (configured per endpoint)                     │
│  ├── IP blocking (private networks, known bad actors)           │
│  └── DDoS protection (Vercel Edge)                              │
│                                                                  │
│  Layer 2: Authentication & Authorization                         │
│  ├── API key validation (X-API-Key header)                      │
│  ├── Bearer token support (Authorization header)                │
│  ├── Scoped access (per-key isolation)                          │
│  └── Admin token for sensitive operations                       │
│                                                                  │
│  Layer 3: Input Validation                                       │
│  ├── JSON schema validation (Zod)                               │
│  ├── SQL injection prevention (parameterized queries)           │
│  ├── XSS prevention (output encoding)                           │
│  └── File upload validation (type, size limits)                 │
│                                                                  │
│  Layer 4: Rate Limiting                                          │
│  ├── Per-IP rate limiting (Redis)                               │
│  ├── Per-API-key rate limiting                                  │
│  ├── Endpoint-specific limits                                   │
│  └── Graceful degradation (429 with retry-after)                │
│                                                                  │
│  Layer 5: Data Protection                                        │
│  ├── API key encryption at rest                                 │
│  ├── Sensitive data masking in logs                             │
│  ├── Secure cookie flags (HttpOnly, Secure, SameSite)           │
│  └── Content Security Policy headers                            │
│                                                                  │
│  Layer 6: Monitoring & Response                                  │
│  ├── Error tracking (Sentry)                                    │
│  ├── Anomaly detection (unusual patterns)                       │
│  ├── Alert notifications (Slack, email)                         │
│  └── Incident response runbooks                                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Deployment Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Vercel Platform                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                   Edge Network (CDN)                      │  │
│  │  - Static assets (JS, CSS, images)                        │  │
│  │  - Cached API responses                                   │  │
│  │  - Global distribution (70+ regions)                      │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                   │
│                              ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │               Serverless Functions                        │  │
│  │  - Auto-scaling (0 to 1000+ instances)                    │  │
│  │  - Cold start optimization                                │  │
│  │  - Memory: 1024 MB default                                │  │
│  │  - Timeout: 60 seconds max                                │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                   │
│                              ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              External Services                            │  │
│  │  - Upstash Redis (global edge)                            │  │
│  │  - TiDB Serverless (global distributed)                   │  │
│  │  - Google Gemini API (regional)                           │  │
│  │  - Search providers (various)                             │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

**Document Version:** 1.0  
**Last Updated:** March 30, 2026  
**Status:** Architecture Approved
