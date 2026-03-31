# NubAgent

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Node.js](https://img.shields.io/badge/Node.js-18+-green.svg)](https://nodejs.org/)
[![React](https://img.shields.io/badge/React-18.3-blue.svg)](https://reactjs.org/)

**NubAgent** exposes a minimal Gemini-powered chat API. The public backend surface is intentionally small: one `POST /api/chat` endpoint backed by Gemini Flash.

![NubAgent Screenshot](./docs/screenshot.png)

## Features

### 🤖 Simple Chat API
- One public endpoint: `GET /api/chat` and `POST /api/chat`
- Gemini Flash-backed completions
- OpenAI-style message array support
- Small JSON response contract

### 🔒 Safer Request Handling
- JSON-only request parsing
- Body-size cap on `/api/chat`
- Explicit non-streaming behavior
- CORS headers for browser clients

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
| `GEMINI_CHAT_MODEL` | Optional Gemini model override | `gemini-3-flash-preview` |
| `GEMINI_CHAT_THINKING_LEVEL` | Optional thinking level override | `low` |
| `SENTRY_DSN` | Sentry error tracking | - |
| `POSTHOG_API_KEY` | PostHog analytics | - |

See `.env.example` for all available options.

### API Keys

Get API keys from:
- **Google Gemini**: [makersuite.google.com](https://makersuite.google.com/app/apikey)
## Usage

Send a JSON body to `/api/chat`:

```http
POST /api/chat
Content-Type: application/json

{
  "messages": [
    { "role": "user", "content": "Hello!" }
  ]
}
```

## Architecture

### Frontend

- **React 18** - UI framework
- **Zustand** - State management
- **React Query** - Server state
- **Vite** - Build tool
- **Sonner** - Toast notifications

### Backend

- **Node.js** - Serverless runtime
- **Gemini API** - Text generation backend
- **Vercel** - Serverless routing

### Key Components

```
api/
└── chat.js           # Public chat endpoint

lib/
├── gemini-chat.js    # Gemini request/response adapter
└── web.js            # Shared request parsing helpers
```

## API Endpoints

### Chat

```http
POST /api/chat
Content-Type: application/json

{
  "messages": [{"role": "user", "content": "Hello!"}]
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
