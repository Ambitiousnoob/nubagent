# NubAgent Development Plan - Quick Reference

> Historical planning document. Treat the current runtime architecture in `src/`, `api/`, and `lib/` as the source of truth.

## 📊 Project Overview

| Aspect | Current | Target |
|--------|---------|--------|
| **Status** | MVP Functional | Production-Ready |
| **Timeline** | - | 16 weeks |
| **Team** | 1-2 devs | 2-4 devs |
| **Budget** | ~$50/month | ~$150-350/month + $50k dev |
| **Users** | Beta | 100+ DAU (Month 1) |

---

## 🎯 Priority Features (Top 10)

1. **Rate Limiting** - Prevent abuse (Critical)
2. **Error Handling** - User-friendly error UI (Critical)
3. **Toast Notifications** - Success/error feedback (High)
4. **Loading States** - Skeleton loaders (High)
5. **Settings Panel** - API key management (High)
6. **Theme System** - Dark/light mode (High)
7. **Response Caching** - Redis caching (Medium)
8. **Code Highlighting** - Syntax highlighting (Medium)
9. **Export Feature** - PDF/Markdown download (Medium)
10. **PWA Support** - Offline capability (Low)

---

## 📁 New File Structure

```
src/
├── components/
│   ├── ui/           # Base components (Button, Input, Card)
│   ├── chat/         # Chat-specific components
│   ├── search/       # Search components
│   ├── library/      # Library components
│   ├── settings/     # Settings components
│   └── shared/       # Shared components (Toast, Header)
├── hooks/            # Custom React hooks
├── stores/           # Zustand state stores
├── utils/            # Utility functions
└── styles/           # CSS and themes
```

---

## 🛠️ Tech Stack Additions

### New Dependencies

```bash
# State Management
npm install zustand

# UI & Styling
npm install react-markdown rehype-highlight remark-gfm
npm install react-hot-toast

# Monitoring
npm install @sentry/react posthog-js

# Infrastructure
npm install @upstash/ratelimit @upstash/redis

# Validation
npm install zod

# Dev Tools
npm install -D vitest @testing-library/react
npm install -D typescript @types/react
```

### New Services

| Service | Purpose | Cost |
|---------|---------|------|
| **Upstash Redis** | Rate limiting, caching | ~$10/mo |
| **Sentry** | Error tracking | $26/mo |
| **PostHog** | Analytics | Free (1M events) |
| **Vercel Pro** | Serverless functions | $20/mo |

---

## 📋 Phase Breakdown

### Phase 1: Foundation (Weeks 1-3)
- ✅ Rate limiting middleware
- ✅ Error boundaries
- ✅ Toast system
- ✅ Loading states
- ✅ Health check endpoint
- ✅ Request validation
- ✅ Response caching
- ✅ Basic analytics

**Deliverable:** Stable, monitored infrastructure

---

### Phase 2: UI/UX (Weeks 4-7)
- ✅ Component library
- ✅ Chat redesign
- ✅ Settings panel
- ✅ Theme system
- ✅ Search results UI
- ✅ Code highlighting
- ✅ Markdown rendering
- ✅ Image upload preview

**Deliverable:** Professional, polished interface

---

### Phase 3: Advanced Features (Weeks 8-11)
- ✅ Conversation export
- ✅ Keyboard shortcuts
- ✅ Library filters
- ✅ Session sharing
- ✅ PWA support
- ✅ Voice input (optional)

**Deliverable:** Power user features

---

### Phase 4: Optimization (Weeks 12-14)
- ✅ TypeScript migration
- ✅ Unit tests (80%+ coverage)
- ✅ E2E tests
- ✅ Performance optimization
- ✅ SEO improvements

**Deliverable:** Production-ready codebase

---

### Phase 5: Launch (Weeks 15-16)
- ✅ Security audit
- ✅ Accessibility audit (WCAG 2.1 AA)
- ✅ Documentation
- ✅ Onboarding flow
- ✅ Beta testing

**Deliverable:** Launch-ready product

---

## 🔑 Key API Endpoints

### Existing (Enhance)
| Endpoint | Purpose | Enhancement Needed |
|----------|---------|-------------------|
| `POST /api/chat` | AI chat | Rate limiting, validation |

---

## 📊 State Management

### Before
- Most live state in `SearchEngine.jsx`
- Shared settings/library state in Zustand stores
- No caching
- No separate server-state client cache layer

### After
```javascript
// Zustand for shared UI state
useSettingsStore → theme, apiKeys, preferences
useUIStore → toasts, modals, sidebar
useLibraryStore → saved sessions, filters, selection
```

---

## 🎨 Component Examples

### Button Component
```jsx
// components/ui/Button.jsx
export function Button({ 
  variant = 'primary',  // primary, secondary, ghost, danger
  size = 'md',         // sm, md, lg
  children,
  ...props 
}) {
  return (
    <button className={`btn btn--${variant} btn--${size}`} {...props}>
      {children}
    </button>
  );
}
```

### Toast Usage
```jsx
// In any component
const { success, error } = useToast();

const handleSave = async () => {
  try {
    await api.save();
    success('Settings saved successfully');
  } catch {
    error('Failed to save settings');
  }
};
```

### Chat Hook
```jsx
// hooks/useChat.js
const { sendMessage, isLoading, error } = useChat();

const handleSend = async (message) => {
  await sendMessage({
    role: 'user',
    content: message
  });
};
```

---

## 🔒 Security Checklist

- [ ] Rate limiting on all endpoints
- [ ] Input validation with Zod schemas
- [ ] SQL injection prevention (parameterized queries)
- [ ] XSS prevention (output encoding)
- [ ] CORS headers configured
- [ ] API keys encrypted at rest
- [ ] Sensitive data masked in logs
- [ ] HTTPS enforced
- [ ] Security audit completed

---

## 📈 Success Metrics

### Technical
| Metric | Current | Target |
|--------|---------|--------|
| Lighthouse Performance | ~70 | >90 |
| Bundle Size (gzipped) | 66 KB | <50 KB |
| API Response Time (p95) | ~800ms | <500ms |
| Error Rate | Unknown | <0.1% |
| Test Coverage | 0% | >80% |

### User
| Metric | Target |
|--------|--------|
| Daily Active Users | 100+ (Month 1) |
| Session Duration | >5 minutes |
| Retention (D7) | >40% |
| NPS Score | >50 |

---

## 🚀 Quick Start (Week 1)

### Day 1-2: Setup
```bash
# Install dependencies
npm install zustand
npm install @sentry/react posthog-js
npm install @upstash/ratelimit @upstash/redis

# Create Upstash Redis instance
# Add to Vercel environment:
UPSTASH_REDIS_REST_URL=...
UPSTASH_REDIS_REST_TOKEN=...
SENTRY_DSN=...
POSTHOG_API_KEY=...
```

### Day 3-4: Rate Limiting
```javascript
// middleware/rate-limit.js
import { Ratelimit } from '@upstash/ratelimit';

const ratelimit = new Ratelimit({
  redis: Redis.fromEnv(),
  limiter: Ratelimit.slidingWindow(100, '1 h'),
});

// Use in API endpoints
const { success } = await ratelimit.limit(ip);
if (!success) return res.status(429).json({ error: 'Rate limited' });
```

### Day 5: Error Boundary
```jsx
// components/shared/ErrorBoundary.jsx
export class ErrorBoundary extends React.Component {
  state = { hasError: false };
  
  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }
  
  render() {
    if (this.state.hasError) {
      return <ErrorFallback error={this.state.error} />;
    }
    return this.props.children;
  }
}
```

---

## 📞 Key Decisions Needed

1. **Redis Provider**: Upstash (recommended) vs self-hosted?
2. **Analytics**: PostHog (recommended) vs Mixpanel vs Amplitude?
3. **Error Tracking**: Sentry (recommended) vs Datadog vs LogRocket?
4. **Testing**: Vitest + Playwright (recommended) vs Jest + Cypress?
5. **TypeScript**: Full migration (recommended) vs gradual vs stay JavaScript?

---

## 🎯 Week 1 Sprint Goals

- [ ] Set up Upstash Redis
- [ ] Implement rate limiting middleware
- [ ] Add global error boundary
- [ ] Create toast notification system
- [ ] Add health check endpoint
- [ ] Set up Sentry error tracking
- [ ] Set up PostHog analytics
- [ ] Create `.env.example` file
- [ ] Update README with setup instructions

---

## 📚 Documentation to Create

1. **User Guide** - How to use NubAgent features
2. **API Reference** - Updated API.md with new endpoints
3. **Contributing Guide** - How to contribute to the project
4. **Deployment Guide** - How to deploy and configure
5. **Troubleshooting** - Common issues and solutions
6. **Changelog** - Version history and changes

---

## 🔗 Related Documents

- [`DEVELOPMENT_PLAN.md`](./DEVELOPMENT_PLAN.md) - Full detailed plan
- [`ARCHITECTURE.md`](./ARCHITECTURE.md) - System architecture
- [`README.md`](./README.md) - Project overview
- [`API.md`](./API.md) - API documentation

---

**Last Updated:** March 30, 2026  
**Version:** 1.0  
**Status:** Ready for Implementation
