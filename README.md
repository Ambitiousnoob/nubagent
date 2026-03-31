# NubAgent

NubAgent is a Facebook Messenger to Gemini bridge designed for Vercel. It exposes a real `/api/webhook` endpoint, verifies the Messenger handshake, forwards inbound text to Gemini, and sends the reply back through the Messenger Send API.

This README is written as an operator guide. It is intentionally step by step, and it only documents behavior that matches the current code in this repository.

<a href="https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2FAmbitiousnoob%2Fnubagent&env=GEMINI_API_KEY,PAGE_ACCESS_TOKEN,VERIFY_TOKEN,POSTGRES_URL,PAGE_ID,FACEBOOK_APP_SECRET,GRAPH_API_VERSION,GEMINI_ENABLE_GOOGLE_SEARCH,GEMINI_ENABLE_CODE_EXECUTION,GEMINI_ENABLE_URL_CONTEXT,GEMINI_CHAT_MODEL,GEMINI_CHAT_THINKING_LEVEL,OPTIONAL_INSTRUCTION,SYSTEM_PROMPT&project-name=nubagent&repo-name=nubagent"><img src="https://vercel.com/button" alt="Deploy with Vercel" /></a>

## What This Repo Actually Does

- Serves a Messenger webhook from `api/webhook.js`
- Calls Gemini through the REST `generateContent` endpoint
- Can optionally enable web grounding with Gemini Google Search or DuckDuckGo
- Can optionally enable Gemini URL Context for supported URLs in prompts
- Can optionally enable Gemini code execution for Python-backed reasoning
- Can pass supported Messenger image attachments through to Gemini as visual input
- For image-only messages, first inspects the image and then waits for the user's next instruction
- Sends `mark_seen`, `typing_on`, and `typing_off` sender actions in Messenger
- Stores conversation history in Postgres by sender PSID
- Uses round-robin Gemini API key selection when you provide multiple keys
- Uses ordered Gemini model fallback when the primary model returns a load or availability style backend failure

## What This Repo Does Not Do

- It does not process audio, video, file, or other non-image attachments as model input
- It does not provide an admin dashboard
- It does not ship database migrations; it creates its single table automatically on first use
- It does not persist any state outside Postgres

## Architecture

```text
Messenger user
  -> Facebook / Meta webhook event
  -> Vercel function (/api/webhook)
  -> Gemini API
  -> Messenger Send API
  -> reply back to the same user
```

## Official Quick Links

- [Google AI Studio](https://aistudio.google.com/)
- [Gemini API key guide](https://ai.google.dev/gemini-api/docs/api-key)
- [Gemini model list](https://ai.google.dev/models/gemini)
- [Gemini Grounding with Google Search](https://ai.google.dev/gemini-api/docs/google-search)
- [Gemini Code Execution](https://ai.google.dev/gemini-api/docs/code-execution)
- [Gemini URL Context](https://ai.google.dev/gemini-api/docs/url-context)
- [Gemini Image Understanding](https://ai.google.dev/gemini-api/docs/image-understanding)
- [Meta app dashboard](https://developers.facebook.com/apps/)
- [Meta Messenger app setup guide](https://developers.facebook.com/docs/messenger-platform/getting-started/app-setup)
- [Meta webhook setup guide](https://developers.facebook.com/docs/messenger-platform/getting-started/webhook-setup/)
- [Meta Messenger send messages guide](https://developers.facebook.com/docs/messenger-platform/send-messages)
- [Meta Page access tokens guide](https://developers.facebook.com/docs/pages/access-tokens/)
- [Meta Graph API Explorer](https://developers.facebook.com/tools/explorer/)
- [Meta Access Token Debugger](https://developers.facebook.com/tools/debug/accesstoken/)
- [Vercel deployment protection guide](https://vercel.com/docs/deployment-protection)
- [Vercel CLI docs](https://vercel.com/docs/cli)

## Before You Start

You need all of the following before this repo can work end to end:

1. A Google AI Studio project and at least one Gemini API key
2. At least one Gemini model ID that can generate text replies
3. A Facebook Page that will act as the Messenger identity
4. A Meta developer app connected to that Page and configured for Messenger
5. A Page access token that can send Messenger replies for that Page
6. A Postgres database reachable from your deployment
7. A public HTTPS URL for `/api/webhook`

Recommended extras:

1. `PAGE_ID`, so Messenger requests use `/{page-id}/messages` directly
2. `FACEBOOK_APP_SECRET`, so the webhook can verify `X-Hub-Signature-256`
3. `GEMINI_ENABLE_GOOGLE_SEARCH=true`, so Gemini can use web grounding
4. `GEMINI_ENABLE_CODE_EXECUTION=true`, so Gemini can run Python for calculation-heavy prompts
5. `GEMINI_ENABLE_URL_CONTEXT=true`, so Gemini can read supported URLs in prompts

## Step 1. Clone The Repo And Install Dependencies

```bash
git clone https://github.com/Ambitiousnoob/nubagent.git
cd nubagent
npm install
```

Useful local commands:

```bash
npm run dev
npm run lint
npm run build
npm run format
npm run format:check
```

Important:

- `npm run dev` starts the Vite frontend only
- `npm run dev` does not serve `api/webhook`
- Use `vercel dev` or `npx vercel dev` when you want to exercise the webhook locally

## Step 2. Create Gemini API Key(s)

1. Open [Google AI Studio](https://aistudio.google.com/).
2. If you want the official key instructions, open the [Gemini API key guide](https://ai.google.dev/gemini-api/docs/api-key).
3. Go to the API Keys page.
4. Create one Gemini API key, or create multiple keys if you want NubAgent to rotate through them.
5. Copy the key values.
6. Put them into `GEMINI_API_KEY` as a comma-separated list with no quotes.

Example:

```env
GEMINI_API_KEY=key1,key2,key3
```

How the repo uses that value:

- One key: NubAgent always uses that one key
- Multiple keys: NubAgent rotates keys in round-robin order per running server instance
- If you enable web grounding, Gemini still uses the same key rotation

## Step 3. Choose Gemini Model(s)

Use the official [Gemini model catalog](https://ai.google.dev/models/gemini).

1. Pick at least one Gemini model ID that returns text output.
2. Put the primary model first.
3. Put fallback models after it, in the exact order you want them tried.
4. Save the list in `GEMINI_CHAT_MODEL` as a comma-separated value.

Example:

```env
GEMINI_CHAT_MODEL=gemini-2.5-flash,gemini-2.5-pro
```

Notes:

- This repo requires `GEMINI_CHAT_MODEL`; there is no built-in default model
- If you paste model names with a `models/` prefix, the repo strips that prefix automatically
- The first model is the priority model
- Later models are only tried when Gemini returns a retryable load or availability style backend failure
- If you enable grounding, supported non-preview models use Gemini's Google Search tool and preview models use a DuckDuckGo search pass

## Step 4. Prepare Your Facebook Page And Meta App

You need a Page and a Meta developer app that can receive Messenger webhook events and send replies as that Page.

Required values:

1. The Page access token you will use as `PAGE_ACCESS_TOKEN`
2. A verification string you choose yourself for `VERIFY_TOKEN`

Use this practical checklist:

1. Create or choose the Facebook Page that should reply to users.
2. Create or choose the Meta developer app that owns the Messenger integration.
3. Add Messenger to the app if it is not already enabled.
4. Connect the app to the Page you want to message from.
5. Generate the Page access token for that Page.
6. Copy the Page ID.
7. Copy the App Secret if you want signed webhook verification enabled.
8. Choose your own `VERIFY_TOKEN` string. This can be any secret string you control.

Direct links:

- [Meta app dashboard](https://developers.facebook.com/apps/)
- [Messenger app setup guide](https://developers.facebook.com/docs/messenger-platform/getting-started/app-setup)
- [Page access token guide](https://developers.facebook.com/docs/pages/access-tokens/)
- [Graph API Explorer](https://developers.facebook.com/tools/explorer/)
- [Access Token Debugger](https://developers.facebook.com/tools/debug/accesstoken/)

About the token:

- This repo needs a valid Page access token, not an app token
- Before production use, inspect the token in Meta's Access Token Debugger and confirm it is the token you intend to run with
- Meta changes dashboard labels often; if a menu name has moved, use the current equivalent UI in the Meta dashboard

## Step 5. Create The Postgres Database

1. Provision a Postgres database that Vercel can reach.
2. Copy its connection string.
3. Put that connection string into `POSTGRES_URL`.

Example:

```env
POSTGRES_URL=postgresql://user:password@host:5432/database
```

What the repo does automatically:

- Creates the `messenger_messages` table on first use if it does not exist
- Creates an index for `sender_psid`, `created_at`, and `id`
- Stores inbound user turns and outbound model turns in that table

What you do not need to do:

- No manual migration step
- No schema file import
- No seed data

## Step 6. Create Your `.env` File

Copy the example file and fill it in:

```bash
cp .env.example .env
```

Minimum working example:

```env
GEMINI_API_KEY=key1,key2
GEMINI_CHAT_MODEL=gemini-2.5-flash,gemini-2.5-pro
PAGE_ACCESS_TOKEN=your_page_access_token
VERIFY_TOKEN=your_webhook_verify_token
POSTGRES_URL=postgresql://user:password@host:5432/database
```

Recommended full example:

```env
GEMINI_API_KEY=key1,key2
GEMINI_CHAT_MODEL=gemini-2.5-flash,gemini-2.5-pro
PAGE_ACCESS_TOKEN=your_page_access_token
VERIFY_TOKEN=your_webhook_verify_token
POSTGRES_URL=postgresql://user:password@host:5432/database
PAGE_ID=your_page_id
FACEBOOK_APP_SECRET=your_app_secret
GRAPH_API_VERSION=v23.0
GEMINI_ENABLE_GOOGLE_SEARCH=true
GEMINI_ENABLE_CODE_EXECUTION=true
GEMINI_ENABLE_URL_CONTEXT=true
GEMINI_CHAT_THINKING_LEVEL=low
OPTIONAL_INSTRUCTION=Prefer concise replies and include one concrete next step when useful.
SYSTEM_PROMPT=You are NubAgent, a concise and helpful assistant replying inside Facebook Messenger. Prefer precise, practical wording over marketing language.
```

## Step 7. Understand Every Environment Variable

### Required Variables

| Variable | Meaning |
|----------|---------|
| `GEMINI_API_KEY` | One Gemini API key or a comma-separated pool of keys |
| `GEMINI_CHAT_MODEL` | Required ordered model list, for example `model1,model2,model3` |
| `PAGE_ACCESS_TOKEN` | Valid Page access token for Messenger sends |
| `VERIFY_TOKEN` | Secret string used only for webhook verification |
| `POSTGRES_URL` | Postgres connection string for durable message history |

### Optional Variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `PAGE_ID` | empty | Recommended. If empty, Messenger requests use `me` |
| `FACEBOOK_APP_SECRET` | empty | Enables `X-Hub-Signature-256` verification for POST webhooks |
| `GRAPH_API_VERSION` | `v23.0` | Graph API version used for Messenger requests |
| `GEMINI_ENABLE_GOOGLE_SEARCH` | `false` | Enables web grounding for Gemini requests |
| `GEMINI_ENABLE_CODE_EXECUTION` | `false` | Enables Gemini's Python code execution tool |
| `GEMINI_ENABLE_URL_CONTEXT` | `false` | Enables Gemini URL Context on supported models |
| `GEMINI_CHAT_THINKING_LEVEL` | `low` | Accepts `none`, `off`, `minimal`, `low`, `medium`, or `high` |
| `OPTIONAL_INSTRUCTION` | empty | Lower-priority guidance injected below `SYSTEM_PROMPT` |
| `SYSTEM_PROMPT` | bundled default | System instruction passed to Gemini |

Important details:

- `GEMINI_ENABLE_GOOGLE_SEARCH=true` enables web grounding for Gemini requests
- Preview models use a DuckDuckGo search pass; supported non-preview models use Gemini grounding tools
- Gemini 1.5 grounding uses the legacy retrieval tool shape automatically; newer supported non-preview models use `google_search`
- `GEMINI_ENABLE_CODE_EXECUTION=true` adds Gemini's built-in Python code execution tool
- `GEMINI_ENABLE_URL_CONTEXT=true` adds Gemini's URL Context tool on supported models
- URL Context only helps when the user's prompt includes one or more URLs
- Supported Messenger image attachments are forwarded to Gemini automatically as long as they fit within the Gemini inline request size budget; non-image attachments are not
- `OPTIONAL_INSTRUCTION` is injected as a lower-priority user-context turn, so it does not outrank `SYSTEM_PROMPT`
- The latest real user message is still sent after `OPTIONAL_INSTRUCTION`
- Current repo history is stored as plain text only, so code execution is most reliable for single-turn reasoning in this bridge
- `GEMINI_CHAT_THINKING_LEVEL=none` and `GEMINI_CHAT_THINKING_LEVEL=off` both disable the field
- In the current code, `GEMINI_CHAT_THINKING_LEVEL` is only sent for Gemini 3+ model names
- If `FACEBOOK_APP_SECRET` is set, every incoming POST must include a valid signature or the webhook returns `403`
- If any required messaging env is missing, `POST /api/webhook` returns `500`
- If `VERIFY_TOKEN` is missing, `GET /api/webhook` returns `500`

## Step 8. Run The App Locally

### Frontend Only

If you only want the landing page:

```bash
npm run dev
```

### Webhook And Frontend Through Vercel

If you want the actual `/api/webhook` endpoint locally:

```bash
vercel dev
```

If you do not have the Vercel CLI installed globally, use:

```bash
npx vercel dev
```

If you need the CLI docs, open the [Vercel CLI documentation](https://vercel.com/docs/cli).

### Smoke Test The Verification Endpoint

After `vercel dev` starts, use the port it prints and test the handshake directly.

Example:

```bash
curl "http://localhost:3000/api/webhook?hub.mode=subscribe&hub.verify_token=your_webhook_verify_token&hub.challenge=test123"
```

Expected response body:

```text
test123
```

If that call fails:

1. Check that the `VERIFY_TOKEN` in your `.env` matches the URL value exactly.
2. Check that you started `vercel dev`, not only `npm run dev`.
3. Check that your env file is loaded in the same shell session or Vercel environment.

## Step 9. Deploy To Vercel

### Option 1. One-Click Deploy

<a href="https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2FAmbitiousnoob%2Fnubagent&env=GEMINI_API_KEY,PAGE_ACCESS_TOKEN,VERIFY_TOKEN,POSTGRES_URL,PAGE_ID,FACEBOOK_APP_SECRET,GRAPH_API_VERSION,GEMINI_ENABLE_GOOGLE_SEARCH,GEMINI_ENABLE_CODE_EXECUTION,GEMINI_ENABLE_URL_CONTEXT,GEMINI_CHAT_MODEL,GEMINI_CHAT_THINKING_LEVEL,OPTIONAL_INSTRUCTION,SYSTEM_PROMPT&project-name=nubagent&repo-name=nubagent"><img src="https://vercel.com/button" alt="Deploy with Vercel" /></a>

### Option 2. CLI Deploy

```bash
vercel --prod
```

Deploy checklist:

1. Import or link the repo in Vercel.
2. Add every required environment variable.
3. Add the recommended optional variables too.
4. Redeploy after changing env vars.
5. Decide which public URL you will give Meta for the webhook callback.

Important for Vercel:

- The webhook callback must be reachable without an interactive login wall
- The safest callback target is a public production URL
- Do not point Meta at a URL that is still protected by Deployment Protection

Official Vercel reference:

[Vercel Deployment Protection documentation](https://vercel.com/docs/deployment-protection)

## Step 10. Configure The Messenger Webhook In Meta

After deployment, wire your public webhook URL into your Meta app.

Use this checklist:

1. Copy your public callback URL:

```text
https://your-public-domain.example/api/webhook
```

2. Open the Messenger webhook settings for your app.
3. Paste that callback URL.
4. Paste the exact same `VERIFY_TOKEN` value you configured in Vercel.
5. Complete the webhook verification flow.
6. Subscribe the app to these webhook fields:
   - `messages`
   - `messaging_postbacks`
7. Save the webhook configuration.

Notes:

- Meta dashboard labels and menu paths can move over time; use the current equivalent UI if wording changes
- This repo only processes Messenger page events
- The callback path must stay `/api/webhook`

Helpful Meta docs:

- [Webhook setup guide](https://developers.facebook.com/docs/messenger-platform/getting-started/webhook-setup/)
- [Messenger app setup guide](https://developers.facebook.com/docs/messenger-platform/getting-started/app-setup)
- [Send messages guide](https://developers.facebook.com/docs/messenger-platform/send-messages)

## Step 11. Send A Real End-To-End Test Message

After deployment and webhook setup:

1. Open Messenger.
2. Message the connected Page from a real user account that can interact with the Page.
3. Send a plain text message.
4. Wait for the bot reply.

Expected behavior:

1. NubAgent marks the message as seen.
2. NubAgent turns typing on.
3. If you sent only an image, NubAgent inspects it and asks you to send the question or instruction next.
4. If you sent text, NubAgent sends your text, recent conversation history, and every supported image attachment that fits within the Gemini inline request size budget to Gemini.
5. NubAgent sends the reply back to Messenger.
6. NubAgent stores the reply in Postgres.
7. NubAgent turns typing off.

## Exact Runtime Behavior

These details come directly from the current code.

### HTTP Behavior

- `GET /api/webhook` handles the Messenger verification challenge
- `POST /api/webhook` only processes payloads where `object === "page"`
- Methods other than `GET` and `POST` return `405`

### Event Handling

- Echo messages are ignored
- Delivery receipts are ignored
- Read receipts are ignored
- Text messages are forwarded to Gemini
- Supported image attachments are fetched and sent to Gemini as inline image parts until the Gemini inline request size budget is full
- Image-only messages are summarized into stored image context, then the bot asks the user for the follow-up instruction
- Postback payloads are converted into `Postback payload: <payload>`
- Non-image attachments still receive a plain text fallback

### Persistence

- Inbound user turns are stored first
- Duplicate inbound events are detected by `message.mid` or `postback.mid`
- Duplicate events are acknowledged without generating a second reply
- Outbound model turns are only stored after the Messenger send succeeds
- The table name is `messenger_messages`

### Prompt Window

- The repo stores full history in Postgres
- The Gemini prompt uses up to 11 prior stored turns plus the current inbound user turn
- That means the effective prompt window is at most 12 turns total per request

### Gemini Key And Model Selection

- `GEMINI_API_KEY` accepts one key or many keys separated by commas
- Keys are selected in round-robin order per running server instance
- `GEMINI_CHAT_MODEL` accepts one model or many models separated by commas
- The first model is always attempted first
- Later models are only attempted when Gemini returns a retryable load or availability style backend failure
- `GEMINI_ENABLE_GOOGLE_SEARCH=true` enables web grounding
- Preview models use a DuckDuckGo search pass and prompt injection before the Gemini request
- Supported non-preview models send Gemini grounding tools in the request body
- `GEMINI_ENABLE_CODE_EXECUTION=true` adds Gemini's code execution tool to the request body
- `GEMINI_ENABLE_URL_CONTEXT=true` adds Gemini's URL Context tool to the request body on supported models
- If grounding is enabled and none of the configured models support either path, the request fails clearly

### Messenger Sending

- Replies are sent through `https://graph.facebook.com/{GRAPH_API_VERSION}/{PAGE_ID or me}/messages`
- Long text replies are split into chunks of at most 1800 characters

## Troubleshooting

### `GET /api/webhook` Returns `403`

Your `hub.verify_token` value does not match `VERIFY_TOKEN`.

### `GET /api/webhook` Returns `500`

`VERIFY_TOKEN` is missing from the runtime environment.

### `POST /api/webhook` Returns `500`

One or more required messaging env vars are missing:

- `GEMINI_API_KEY`
- `GEMINI_CHAT_MODEL`
- `PAGE_ACCESS_TOKEN`
- `VERIFY_TOKEN`
- `POSTGRES_URL`

### `POST /api/webhook` Returns `403`

If `FACEBOOK_APP_SECRET` is set, the incoming request signature is missing or invalid.

### Messenger User Gets The Attachment Fallback

The message was attachment-only, but the attachment was not a supported image or the image could not be fetched successfully.

### Messenger User Gets "I checked the image. Now send your question or instruction about it."

That is the expected flow for image-only messages. NubAgent inspects the image first, stores image context, and waits for the next Messenger message because Messenger image upload flow does not reliably pair the image with a same-turn text instruction.

### Messenger User Gets The Upstream Error Fallback

The webhook reached Gemini or another upstream dependency and failed during processing. Check:

1. Gemini API key validity
2. Gemini model name spelling
3. Postgres connectivity
4. Messenger Page access token validity

### No Reply Arrives In Messenger

Check all of the following:

1. The callback URL is public and points to `/api/webhook`
2. Meta webhook verification succeeded
3. The webhook is subscribed to `messages` and `messaging_postbacks`
4. Vercel Deployment Protection is not blocking the callback URL
5. The Page access token is valid for the Page you connected

## Project Structure

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
