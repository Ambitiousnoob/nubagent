# NubAgent

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Node.js](https://img.shields.io/badge/Node.js-18+-green.svg)](https://nodejs.org/)
[![React](https://img.shields.io/badge/React-18.3-blue.svg)](https://reactjs.org/)

**NubAgent** is a modern, AI-powered research assistant and chat interface. It combines advanced language models with web search capabilities to provide accurate, cited responses to your questions.

![NubAgent Screenshot](./docs/screenshot.png)

## Features

### 🤖 AI-Powered Chat
- Natural language conversations with advanced AI models
- Support for multiple AI providers (Gemini, OpenAI, Anthropic)
- Context-aware responses with conversation memory
- File upload support for images and documents

### 🔍 Web Research
- Real-time web search integration
- Multi-source fact verification
- Automatic citation generation
- Source credibility indicators

### 📚 Session Library
- Save and organize research sessions
- Search and filter through history
- Export sessions (JSON, Markdown, Text)
- Bulk operations support

### 🎨 Modern UI/UX
- Dark/Light theme support
- Responsive design for all devices
- PWA support for offline access
- Smooth animations and transitions

### 🔒 Privacy & Security
- Local storage for sensitive data
- Rate limiting protection
- Input sanitization
- CORS configuration

## Quick Start

### Prerequisites

- Node.js 18 or higher
- npm 9 or higher
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-org/nubagent.git
   cd nubagent
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. **Start development server**
   ```bash
   npm run dev
   ```

5. **Open in browser**
   ```
   http://localhost:5173
   ```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `NODE_ENV` | Environment mode | `development` |
| `PORT` | Server port | `3000` |
| `GEMINI_API_KEY` | Google Gemini API key | - |
| `OPENAI_API_KEY` | OpenAI API key | - |
| `ANTHROPIC_API_KEY` | Anthropic API key | - |
| `DATABASE_URL` | Database connection string | - |
| `SENTRY_DSN` | Sentry error tracking | - |
| `POSTHOG_API_KEY` | PostHog analytics | - |

See `.env.example` for all available options.

### API Keys

Get API keys from:
- **Google Gemini**: [makersuite.google.com](https://makersuite.google.com/app/apikey)
- **OpenAI**: [platform.openai.com](https://platform.openai.com/api-keys)
- **Anthropic**: [console.anthropic.com](https://console.anthropic.com/settings/keys)

## Usage

### Basic Chat

1. Type your question in the chat input
2. Press Enter or click Send
3. Wait for the AI response
4. Click citations to view sources

### Web Research

1. Enter a research query
2. NubAgent searches the web automatically
3. Review sources and citations
4. Save session to library

### Session Management

- **Save**: Sessions are auto-saved
- **Library**: Access via sidebar or `/library`
- **Search**: Filter sessions by query
- **Export**: Download as JSON, Markdown, or Text

## Architecture

### Frontend

- **React 18** - UI framework
- **Zustand** - State management
- **React Query** - Server state
- **Vite** - Build tool
- **Sonner** - Toast notifications

### Backend

- **Node.js** - Runtime
- **Express** - Web framework (via Vercel)
- **Google GenAI** - AI integration
- **MySQL** - Database (optional)

### Key Components

```
src/
├── components/
│   ├── Chat/          # Chat interface
│   ├── Search/        # Search results
│   ├── Library/       # Session library
│   ├── Settings/      # User settings
│   └── UI/            # Reusable components
├── store/             # Zustand stores
├── hooks/             # Custom hooks
└── lib/               # Utilities

api/
├── middleware/        # Express middleware
├── chat.js           # Chat endpoint
├── search.js         # Search endpoint
├── health.js         # Health check
├── analytics.js      # Event tracking
└── export.js         # Export functionality
```

## API Endpoints

### Chat

```http
POST /api/chat
Content-Type: application/json

{
  "model": "nub-agent",
  "messages": [{"role": "user", "content": "Hello!"}],
  "stream": false
}
```

### Search

```http
POST /api/search
Content-Type: application/json

{
  "query": "latest AI developments",
  "limit": 10
}
```

### Health

```http
GET /api/health
```

### Analytics

```http
POST /api/analytics
Content-Type: application/json

{
  "event": "chat_message_sent",
  "properties": {"length": 50}
}
```

### Export

```http
POST /api/export
Content-Type: application/json

{
  "format": "markdown",
  "sessionIds": ["session-1", "session-2"]
}
```

## Development

### Scripts

```bash
# Development
npm run dev          # Start dev server

# Production
npm run build        # Build for production
npm run preview      # Preview production build

# Code Quality
npm run lint:regex   # Run linting
```

### Project Structure

See [CONTRIBUTING.md](./CONTRIBUTING.md) for detailed development guidelines.

## PWA Support

NubAgent is a Progressive Web App:

1. **Install**: Click the install prompt in supported browsers
2. **Offline**: Basic functionality works offline
3. **Push Notifications**: Coming soon

## Testing

```bash
# Run tests
npm test

# Test coverage
npm run test:coverage
```

## Troubleshooting

### Common Issues

**API Key Errors**
- Verify keys in `.env`
- Check API provider status
- Ensure no trailing spaces

**Build Failures**
- Clear `node_modules`: `rm -rf node_modules && npm install`
- Clear cache: `npm run build -- --force`

**CORS Errors**
- Check `CORS_ORIGINS` in `.env`
- Ensure proper protocol (http/https)

## Security

- All API keys stored client-side only
- Rate limiting enabled by default
- Input sanitization on all endpoints
- HTTPS required in production

See [SECURITY.md](./SECURITY.md) for detailed security policy.

## Contributing

We welcome contributions! See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

### Ways to Contribute

- 🐛 Report bugs
- ✨ Suggest features
- 📝 Improve documentation
- 💻 Submit pull requests
- 🎨 Design improvements

## License

MIT License - see [LICENSE](./LICENSE) for details.

## Acknowledgments

- [Google GenAI](https://ai.google.dev/) for AI capabilities
- [React](https://reactjs.org/) for the UI framework
- [Vite](https://vitejs.dev/) for fast builds
- [Zustand](https://zustand-demo.pmnd.rs/) for state management
- All contributors and supporters

## Contact

- **Issues**: [GitHub Issues](https://github.com/your-org/nubagent/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/nubagent/discussions)

---

Built with ❤️ by the NubAgent Team
