# NubAgent

**Direct AI Access API** — Deploy your own Gemini chat endpoint with Facebook Messenger integration.

A single unified API for offline AI access: no proxies, no middlemen. Use the `/api/chat` endpoint directly or integrate with Facebook Messenger via `/api/webhook`.

## Features

- **Facebook Messenger Integration** (`/api/webhook`) — Direct message handling with automatic AI responses
- **Direct Gemini Calls** — No HTTP hops; Messenger webhook calls Gemini directly
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
# Gemini API (from Google AI Studio)
GEMINI_API_KEY=your_gemini_api_key_here

# Facebook Messenger (from Facebook Developer Dashboard)
PAGE_ACCESS_TOKEN=your_facebook_page_access_token
VERIFY_TOKEN=your_webhook_verification_token
```

Get your keys:
- **Gemini API Key:** [Google AI Studio](https://aistudio.google.com/app/apikey)
- **Facebook Tokens:** [Facebook Developer Dashboard](https://developers.facebook.com/)

See [docs-complete.html](./docs-complete.html) for complete setup guide.

## API Endpoint

### POST /api/webhook
Facebook Messenger webhook. 
- **GET:** Webhook verification for Facebook
- **POST:** Receive and process messages

**Flow:** User messages your Facebook page → Webhook receives message → Calls Gemini directly → Sends response back to user.

**Setup:**
1. Create Facebook App & Page
2. Add webhook URL: `https://your-deployment.vercel.app/api/webhook`
3. Set `VERIFY_TOKEN` to any random string (same one used in verification)
4. Set `PAGE_ACCESS_TOKEN` from your Facebook page
5. Subscribe to `messages` and `messaging_postbacks` events

## Deployment

### One-Click Deploy to Vercel

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2FAmbitiousnoob%2Fnubagent&env=GEMINI_API_KEY,PAGE_ACCESS_TOKEN,VERIFY_TOKEN&project-name=nubagent&repo-name=nubagent)

**Steps:**
1. Click the button above
2. Connect your GitHub account
3. Add environment variables when prompted:
   - `GEMINI_API_KEY` (required) — Get from [Google AI Studio](https://aistudio.google.com/app/apikey)
   - `PAGE_ACCESS_TOKEN` (optional) — For Facebook Messenger
   - `VERIFY_TOKEN` (optional) — For Facebook Messenger
4. Click "Deploy"
5. Done! Your API is live

### Or Deploy from CLI

```bash
git clone https://github.com/Ambitiousnoob/nubagent.git
cd nubagent
vercel --prod
```

Then add environment variables in Vercel dashboard → Settings → Environment Variables.

**Live Example:** https://nubagent.vercel.app

## Architecture

```
Facebook User → /api/webhook → Gemini → Response back to Facebook
```

Single endpoint that:
1. Verifies webhook with Facebook
2. Receives incoming messages
3. Calls Gemini directly (no intermediate hop)
4. Sends AI response back to user

## Documentation

- **[index.html](./index.html)** — Live API documentation with examples
- **GitHub:** [Ambitiousnoob/nubagent](https://github.com/Ambitiousnoob/nubagent)
- **Issues:** [GitHub Issues](https://github.com/Ambitiousnoob/nubagent/issues)

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
- **Live API:** https://nubagent.vercel.app

---

Built by [ambitiousnoob](https://github.com/ambitiousnoob) | Powered by [Google Gemini](https://ai.google.dev)
