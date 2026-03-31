# NubAgent

Minimal repo with:

- a docs-only React frontend
- one public API route: `GET /api/chat` and `POST /api/chat`
- a simple Gemini-backed chat handler with no agentic runtime

## Setup

```bash
npm install
cp .env.example .env
npm run dev
```

Required env vars:

```bash
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_CHAT_MODEL=gemini-3-flash-preview
GEMINI_CHAT_THINKING_LEVEL=low
CHAT_BODY_LIMIT_BYTES=65536
```

## API

```http
GET /api/chat
POST /api/chat
Content-Type: application/json

{
  "system": "Be concise.",
  "messages": [
    { "role": "user", "content": "Hello!" }
  ]
}
```

The response is a non-streaming JSON chat completion with `output_text`, `choices`, and `usage`.

## Development

```bash
npm run dev
npm run lint
npm run build
```
- [Vite](https://vitejs.dev/) for fast builds
- [Zustand](https://zustand-demo.pmnd.rs/) for state management
- All contributors and supporters

## Contact

- **Issues**: [GitHub Issues](https://github.com/your-org/nubagent/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/nubagent/discussions)

---

Built with ❤️ by the NubAgent Team
