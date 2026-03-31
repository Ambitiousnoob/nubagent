# NubAgent

**Direct AI Access API** — Deploy your own Gemini chat endpoint with Facebook Messenger integration.

A single unified API for offline AI access: no proxies, no middlemen. Use the `/api/chat` endpoint directly or integrate with Facebook Messenger via `/api/webhook`.

## Features

- **Gemini Chat API** (`/api/chat`) — OpenAI-compatible chat completions endpoint
- **Facebook Messenger Integration** (`/api/webhook`) — Direct message handling with automatic AI responses
- **Direct Function Calls** — No HTTP hops; Messenger webhook calls Gemini directly
- **Production Ready** — Deployed to Vercel, load-balanced, auto-scaling

## Quick Start

```bash
git clone https://github.com/Ambitiousnoob/nubagent.git
cd nubagent
npm install
cp .env.example .env
npm run dev
```

### Test Locally

```bash
curl -X POST http://localhost:5173/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "system": "Be concise."
  }'
```

## Configuration

### Required Environment Variables

```bash
GEMINI_API_KEY=your_gemini_api_key_here
```

Get your key from [Google AI Studio](https://aistudio.google.com/app/apikey).

### Optional: Facebook Messenger

```bash
PAGE_ACCESS_TOKEN=your_facebook_page_token
VERIFY_TOKEN=your_webhook_verification_token
```

See [docs-complete.html](./docs-complete.html) for complete setup guide.

## API Endpoints

### POST /api/chat
Chat completions endpoint. Compatible with OpenAI API format.

**Request:**
```json
{
  "messages": [
    { "role": "user", "content": "What is Node.js?" }
  ],
  "system": "You are a helpful assistant.",
  "temperature": 0.7,
  "maxOutputTokens": 512,
  "thinkingLevel": "low"
}
```

**Response:**
```json
{
  "ok": true,
  "output_text": "Node.js is a JavaScript runtime...",
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "Node.js is a JavaScript runtime..."
      }
    }
  ],
  "usage": {
    "prompt_tokens": 15,
    "completion_tokens": 45,
    "total_tokens": 60
  }
}
```

### GET/POST /api/webhook
Facebook Messenger webhook. 
- **GET:** Webhook verification for Facebook
- **POST:** Receive and process messages

Users message your Facebook page → Webhook receives message → Calls Gemini directly → Sends response back to user.

## Deployment

Live at: **https://nubagent.vercel.app**

```bash
vercel --prod
```

Then add environment variables in Vercel dashboard → Settings → Environment Variables.

## Architecture

```
Facebook User → /api/webhook (direct call) → Gemini
External App  → /api/chat (REST) → Gemini
```

Both endpoints call Gemini directly. No intermediate proxies or API layers.

## Documentation

- **[docs-complete.html](./docs-complete.html)** — Complete zero-gap setup guide (22 steps, both paths)
- **[DEPLOYMENT.md](./DEPLOYMENT.md)** — Deployment status and verification
- **[MERGE_SUMMARY.md](./MERGE_SUMMARY.md)** — Architecture changes (merged .bot + nubagent)

## Development

```bash
npm run dev       # Start Vite dev server
npm run build     # Build for production
npm run lint      # Run ESLint
npm run format    # Format with Prettier
```

**Stack:**
- [Vite](https://vitejs.dev/) for blazing fast builds
- [React 18](https://react.dev/) for frontend
- [Zustand](https://zustand-demo.pmnd.rs/) for state
- [Vercel](https://vercel.com/) for deployment

## Support

- **GitHub:** [Ambitiousnoob/nubagent](https://github.com/Ambitiousnoob/nubagent)
- **Issues:** [GitHub Issues](https://github.com/Ambitiousnoob/nubagent/issues)
- **Docs:** [docs-complete.html](./docs-complete.html)

---

Built by [ambitiousnoob](https://github.com/ambitiousnoob) | Powered by [Google Gemini](https://ai.google.dev)
