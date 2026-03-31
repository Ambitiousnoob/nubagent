# NubAgent

NubAgent is a working Facebook Messenger to Gemini bridge for Vercel. It exposes a real `/api/webhook` endpoint, verifies the Messenger webhook handshake, forwards inbound text to Gemini, and sends the model reply back through the Messenger Send API.

## What is included

- A Vercel serverless function at `api/webhook.js`
- Direct Gemini REST integration through the official `generateContent` endpoint
- Messenger reply helpers for `typing_on`, `typing_off`, `mark_seen`, and text replies
- A small Vite frontend so the project builds cleanly and documents the deployment flow
- Postgres-backed conversation history keyed by sender PSID

## Architecture

```text
Messenger user
  -> Facebook webhook event
  -> Vercel function (/api/webhook)
  -> Gemini API
  -> Messenger Send API
  -> reply back to the same user
```

## Prerequisites

You need:

1. A Gemini API key from Google AI Studio
   You can provide one key or a comma-separated key pool for round-robin use.
2. An ordered Gemini model list in `GEMINI_CHAT_MODEL`
   Put the primary model first, followed by fallbacks.
3. A Facebook Page access token with Messenger permissions
4. A webhook verification token you choose yourself
5. A Postgres database reachable from Vercel

Optional but recommended:

1. `PAGE_ID` so replies use the current `/{page-id}/messages` path directly
2. `FACEBOOK_APP_SECRET` so the webhook can verify `X-Hub-Signature-256`

## Environment variables

Required:

| Variable | Description |
|----------|-------------|
| `GEMINI_API_KEY` | One Gemini API key or a comma-separated key pool |
| `GEMINI_CHAT_MODEL` | Required ordered model list, for example `model1,model2,model3` |
| `PAGE_ACCESS_TOKEN` | Facebook Page access token |
| `VERIFY_TOKEN` | Token used only for Facebook webhook verification |
| `POSTGRES_URL` | Postgres connection string for durable message history |

Optional:

| Variable | Default | Description |
|----------|---------|-------------|
| `PAGE_ID` | empty | Recommended. Page ID for the Send API path |
| `FACEBOOK_APP_SECRET` | empty | Enables request signature validation |
| `GRAPH_API_VERSION` | `v23.0` | Graph API version used for Messenger requests |
| `GEMINI_CHAT_THINKING_LEVEL` | `low` | `minimal`, `low`, `medium`, `high`, or `none` |
| `SYSTEM_PROMPT` | bundled default | System instruction passed to Gemini |

## Local development

Install dependencies:

```bash
npm install
```

Run the frontend locally:

```bash
npm run dev
```

Run checks:

```bash
npm run lint
npm run build
```

If you want to test the webhook locally against Facebook, run the project through Vercel's local runtime:

```bash
vercel dev
```

## Deploy to Vercel

### Option 1: One-click deploy

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2FAmbitiousnoob%2Fnubagent&env=GEMINI_API_KEY,PAGE_ACCESS_TOKEN,VERIFY_TOKEN,POSTGRES_URL,PAGE_ID,FACEBOOK_APP_SECRET,GRAPH_API_VERSION,GEMINI_CHAT_MODEL,GEMINI_CHAT_THINKING_LEVEL,SYSTEM_PROMPT&project-name=nubagent&repo-name=nubagent)

### Option 2: CLI deploy

```bash
git clone https://github.com/Ambitiousnoob/nubagent.git
cd nubagent
npm install
vercel --prod
```

## Configure the Facebook webhook

1. Deploy the project.
2. In Vercel, make sure login protection is not blocking `/api/webhook`.
3. In the Facebook developer dashboard, set the callback URL to:

```text
https://your-deployment.vercel.app/api/webhook
```

4. Use the same `VERIFY_TOKEN` value you set in Vercel.
5. Subscribe the app to `messages` and `messaging_postbacks`.

## Runtime behavior

- `GET /api/webhook` handles the Messenger verification challenge.
- `POST /api/webhook` accepts webhook events for `object: "page"`.
- Text messages are sent to Gemini and the reply is returned to Messenger.
- Postbacks are converted into a text prompt using the payload.
- Attachment-only messages currently receive a plain text fallback.
- Conversation history is stored in Postgres and only the most recent turns are sent to Gemini.
- Duplicate inbound events with the same Messenger event ID are acknowledged without generating a second reply.
- Gemini API keys are selected in round-robin order from the comma-separated `GEMINI_API_KEY` env value on each running server instance.
- `GEMINI_CHAT_MODEL` is treated as an ordered priority list, so the first model is preferred and later models are used as fallbacks when Gemini returns a load or availability-style backend error.

## Project structure

```text
api/
  webhook.js
lib/
  config.js
  gemini.js
  history.js
  messenger.js
src/
  App.jsx
  main.jsx
  styles.css
```
