# NubAgent

![NubAgent banner](assets/nubagent-banner.svg)

![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg)

NubAgent is a Facebook Messenger to Gemini bridge designed for Vercel. It exposes a real `/api/webhook` endpoint, verifies the Messenger handshake, forwards inbound text to Gemini, and sends the reply back through the Messenger Send API.

It is built for people who want direct access to stronger AI behavior from a simple Messenger chat, which makes it practical on low-end phones that cannot run local models well.

This README is written as an operator guide. It is intentionally step by step, and it only documents behavior that matches the current code in this repository.

<a href="https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2FAmbitiousnoob%2Fnubagent&env=GEMINI_API_KEY,PAGE_ACCESS_TOKEN,VERIFY_TOKEN,POSTGRES_URL,GEMINI_CHAT_MODEL,GEMINI_CHAT_THINKING_LEVEL,OPTIONAL_INSTRUCTION&project-name=nubagent&repo-name=nubagent"><img src="https://vercel.com/button" alt="Deploy with Vercel" /></a>

## Table of Contents

- [Access Without Regular Paid Data](#access-without-regular-paid-data)
- [Quick Visual Overview](#quick-visual-overview)
- [What This Repo Actually Does](#what-this-repo-actually-does)
- [What This Repo Does Not Do](#what-this-repo-does-not-do)
- [Architecture](#architecture)
- [Official Quick Links](#official-quick-links)
- [Before You Start](#before-you-start)
- [Step 1. Clone The Repo And Install Dependencies](#step-1-clone-the-repo-and-install-dependencies)
- [Step 2. Create Gemini API Keys](#step-2-create-gemini-api-keys)
- [Step 3. Choose Gemini Models](#step-3-choose-gemini-models)
- [Step 4. Prepare Your Facebook Page And Meta App](#step-4-prepare-your-facebook-page-and-meta-app)
- [Step 5. Create The Postgres Database](#step-5-create-the-postgres-database)
- [Step 6. Create Your .env File](#step-6-create-your-env-file)
- [Step 7. Understand Every Environment Variable](#step-7-understand-every-environment-variable)
- [Step 8. Run The App Locally](#step-8-run-the-app-locally)
- [Step 9. Deploy To Vercel](#step-9-deploy-to-vercel)
- [Step 10. Configure The Messenger Webhook In Meta](#step-10-configure-the-messenger-webhook-in-meta)
- [Step 11. Send A Real End-To-End Test Message](#step-11-send-a-real-end-to-end-test-message)
- [Exact Runtime Behavior](#exact-runtime-behavior)
- [Troubleshooting](#troubleshooting)
- [Project Structure](#project-structure)

## Access Without Regular Paid Data

This is not an offline system. Messenger still needs a working network path.

NubAgent can still be practical in cases like these:

- countries and carriers that support Facebook basic mode or Free Basics
- carriers that offer Facebook or Messenger access without regular data charges
- SIMs or plans where Facebook or Messenger traffic is sponsored or zero-rated
- low-end phones that can run Messenger but cannot run local AI models well

Check these official references before you rely on that access pattern:

- [Connect to Facebook without data charges](https://www.facebook.com/help/548059885266896/)
- [Devices you can use to connect to Facebook without data charges](https://www.facebook.com/help/320951141622544/)
- [Unable to connect to basic mode on Facebook](https://www.facebook.com/help/193971691069066/)
- [How to get Free Basics](https://www.facebook.com/help/1489856091251073/)

## Quick Visual Overview

![NubAgent flow](assets/nubagent-flow.svg)

If you want a GitHub social preview image, use `assets/social-preview.png` from this repository in the repo settings.

## What This Repo Actually Does

- Serves a Messenger webhook from `api/webhook.js`
- Calls Gemini through the REST `generateContent` endpoint
- Supports in-chat control commands like `!help`, `!reset`, `!summary`, `!memory`, `!location`, `!forget`, and `!privacy`
- Automatically attaches every enabled Gemini tool that the selected model supports
- Automatically enables web grounding on supported models
- Automatically enables Gemini URL Context on supported models and prefers a URL-capable configured model when the prompt includes URLs
- Automatically enables Gemini code execution on supported models
- Automatically enables Grounding with Google Maps on supported models and prefers a Maps-capable configured model for Maps-related prompts
- Saves per-user location from Messenger pins or `!location` capture links and uses the latest shared location for Google Maps grounded prompts such as `near me` or `where am I`
- Supports Gemini context caching through implicit cache hits and optional explicit `cachedContent` resources
- Can pass supported Messenger image attachments through to Gemini as visual input
- For image-only messages, first inspects the image, replies with a short visual summary, and then waits for the user's next instruction
- Sends `mark_seen`, `typing_on`, and `typing_off` sender actions in Messenger
- Repairs the Messenger `Get Started` state and the persistent menu item that links to NubAgent
- Stores conversation history, long-term memory, rolling summaries, and operational event records in Postgres by sender PSID
- Exposes a JSON health/status route at `/api/health`
- Uses round-robin Gemini API key selection when you provide multiple keys
- Uses ordered Gemini model fallback when the primary model returns a load or availability style backend failure

## What This Repo Does Not Do

- It does not process audio, video, file, or other non-image and non-location attachments as model input
- It does not provide an admin dashboard
- It does not ship database migrations; it creates its tables automatically on first use
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

Optional controls:

1. `GEMINI_CHAT_THINKING_LEVEL`, if you want to tune thinking on supported models
2. `OPTIONAL_INSTRUCTION`, if you want lower-priority project guidance added to each request
3. `SYSTEM_PROMPT`, if you want to replace the bundled Messenger-focused system instruction
4. Reliability and memory tuning env vars like `RELIABILITY_RETRY_LIMIT`, `RELIABILITY_RETRY_BASE_MS`, `MEMORY_MAX_ITEMS`, and `SUMMARY_MAX_CHARS`

## Step 1. Clone The Repo And Install Dependencies

```bash
git clone https://github.com/ambitiousnoob/nubagent.git
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

## Step 2. Create Gemini API Keys

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
- If the active model uses web grounding, Gemini still uses the same key rotation

## Step 3. Choose Gemini Models

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
- Search grounding, code execution, and URL Context turn on automatically when the active model supports them

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
6. Choose your own `VERIFY_TOKEN` string. This can be any secret string you control.

Direct links:

- [Meta app dashboard](https://developers.facebook.com/apps/)
- [Messenger app setup guide](https://developers.facebook.com/docs/messenger-platform/getting-started/app-setup)
- [Page access token guide](https://developers.facebook.com/docs/pages/access-tokens/)
- [Graph API Explorer](https://developers.facebook.com/tools/explorer/)
- [Access Token Debugger](https://developers.facebook.com/tools/debug/accesstoken/)

- This repo needs a valid Page access token, not an app token
- `PAGE_ID` is not required; Messenger sends use the `me` path by default
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
GEMINI_CHAT_THINKING_LEVEL=low
OPTIONAL_INSTRUCTION=Prefer concise replies and include one concrete next step when useful.
SYSTEM_PROMPT=You are NubAgent. Keep replies short, practical, and easy to read on mobile.
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
| `FACEBOOK_APP_SECRET` | Your Facebook App Secret, used to verify webhook signatures. |

### Optional Variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `CRON_SECRET` | empty | A secret passed in `Authorization: Bearer` to secure cron jobs. |
| `GRAPH_API_VERSION` | `v23.0` | The Facebook Graph API version to use. |
| `GEMINI_ENABLE_GOOGLE_SEARCH` | `true` | Enable or disable Google Search grounding. |
| `GEMINI_ENABLE_CODE_EXECUTION` | `true` | Enable or disable code execution. |
| `GEMINI_ENABLE_GOOGLE_MAPS` | `true` | Enable or disable Grounding with Google Maps for geo-specific prompts. |
| `GEMINI_ENABLE_URL_CONTEXT` | `true` | Enable or disable URL context. |
| `GEMINI_CACHED_CONTENT` | empty | Optional explicit cached content resource. Use either one `cachedContents/...` value for a single configured model or comma-separated `model=cachedContents/...` mappings for multi-model setups. |
| `GEMINI_GOOGLE_MAPS_LATITUDE` | empty | Optional default latitude used as Google Maps location context when a user has not shared a Messenger location pin. |
| `GEMINI_GOOGLE_MAPS_LONGITUDE` | empty | Optional default longitude used as Google Maps location context when a user has not shared a Messenger location pin. |
| `GEMINI_CHAT_THINKING_LEVEL` | `low` | Accepts `none`, `off`, `minimal`, `low`, `medium`, or `high` |
| `OPTIONAL_INSTRUCTION` | empty | Lower-priority guidance injected below the bundled system instruction |
| `SYSTEM_PROMPT` | bundled default | Replaces the built-in Messenger-focused system instruction |
| `APP_BASE_URL` | empty | Optional canonical base URL used when `!location` generates a browser capture link. If empty, NubAgent uses the current webhook request origin. |
| `LOCATION_CAPTURE_TTL_MINUTES` | `15` | Expiration window for one-time `!location` browser capture links. |

Important details:

- Search grounding, code execution, URL Context, and Google Maps grounding are attached automatically whenever the active model supports them. You can disable them with the `GEMINI_ENABLE_` flags.
- Implicit Gemini caching is automatic on supported models, and the repo can also attach an explicit `cachedContent` resource when `GEMINI_CACHED_CONTENT` is configured.
- `supportsGoogleSearch` is enabled for Gemini 1.5, 2.0-flash, 2.5 and 3 models.
- `supportsCodeExecution` is enabled for Gemini 1.5 text models, Gemini 2.0 Flash, Gemini 2.5, and Gemini 3 models.
- `supportsUrlContext` is enabled for Gemini 2.5 and 3 models.
- Grounding with Google Maps is enabled for Gemini 2.5 models and Gemini 2.0 Flash, but not Gemini 3 or Gemini 2.0 Flash Lite.
- Gemini explicit caches can only be used with the model they were created for, so multi-model configurations should use `model=cachedContents/...` mappings instead of a single fallback cache name.
- Capability questions are answered from the active request's enabled tool list, so the bot should not deny access to a tool that is currently attached
- Supported models use Gemini grounding tools directly; unsupported preview-only cases fall back to a DuckDuckGo search pass
- Gemini 1.5 grounding uses the legacy retrieval tool shape automatically; newer supported models use `google_search`
- URL Context only helps when the user's prompt includes one or more URLs, and the runtime prefers the first configured model that supports it
- Geo-specific prompts prefer the first configured model that supports Grounding with Google Maps
- Supported Messenger image attachments are forwarded to Gemini automatically as long as they fit within the Gemini inline request size budget
- Messenger location pins and `!location` browser captures are stored and reused as Google Maps grounding context
- When Google Maps grounding is used, Messenger replies append plain text Google Maps source links
- `OPTIONAL_INSTRUCTION` is injected as a lower-priority user-context turn, so it does not outrank the bundled system instruction
- The latest real user message is still sent after `OPTIONAL_INSTRUCTION`
- `SYSTEM_PROMPT` replaces the bundled system instruction when you set it
- Current repo history is stored as plain text only, so code execution is most reliable for single-turn reasoning in this bridge
- `GEMINI_CHAT_THINKING_LEVEL=none` and `GEMINI_CHAT_THINKING_LEVEL=off` both disable the field
- In the current code, `GEMINI_CHAT_THINKING_LEVEL` is only sent for Gemini 3+ model names
- If any required messaging env is missing, `POST /api/webhook` returns `500`
- If `VERIFY_TOKEN` is missing, `GET /api/webhook` returns `500`
- If `FACEBOOK_APP_SECRET` is set and the signature is invalid, `POST /api/webhook` returns `403`.

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

<a href="https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2FAmbitiousnoob%2Fnubagent&env=GEMINI_API_KEY,PAGE_ACCESS_TOKEN,VERIFY_TOKEN,POSTGRES_URL,GEMINI_CHAT_MODEL,GEMINI_CHAT_THINKING_LEVEL,OPTIONAL_INSTRUCTION&project-name=nubagent&repo-name=nubagent"><img src="https://vercel.com/button" alt="Deploy with Vercel" /></a>

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
3. In the "Callback URL" field, paste your public webhook URL.
4. In the "Verify token" field, paste the exact same `VERIFY_TOKEN` value you configured in Vercel.
5. Click "Verify and save".
6. In the "Webhook" section, select "Page" from the dropdown menu.
7. Click "Subscribe to events".
8. In the subscription fields, subscribe to `messages` and `messaging_postbacks`.
9. Turn off "Attach a client certificate to Webhook requests".
10. Save the webhook configuration.

Notes:

- Meta dashboard labels and menu paths can move over time; use the current equivalent UI if wording changes
- This repo only processes Messenger page events
- The callback path must stay `/api/webhook`

Helpful Meta docs:

- [Your app-specific webhook settings](https://developers.facebook.com/apps/1098695178551898/use_cases/customize/?use_case_enum=WEBHOOKS&business_id=4157540321144368&selected_tab=webhooks&product_route=webhooks)
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
3. If you sent only an image, NubAgent inspects it, replies with a short visual summary, and asks you to send the question or instruction next.
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
- Incoming webhook traffic opportunistically repairs the Messenger profile state in the background

### Event Handling

- Echo messages are ignored
- Delivery receipts are ignored
- Read receipts are ignored
- Text messages are forwarded to Gemini
- Supported image attachments are fetched and sent to Gemini as inline image parts until the Gemini inline request size budget is full
- Image-only messages are summarized into stored image context, then the bot sends that summary back to Messenger before waiting for the follow-up instruction
- Prompts with URLs prefer a configured Gemini model that supports URL Context
- Geo-specific prompts can use Grounding with Google Maps, and grounded replies include Google Maps source links
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
- Search grounding, code execution, and URL Context are enabled automatically when the active model supports them
- Every enabled supported Gemini tool is attached to each request for the selected model
- When `GEMINI_CACHED_CONTENT` is configured, the request attaches the matching explicit `cachedContent` resource and logs cache-hit usage metadata when Gemini reports cached tokens
- Supported models send Gemini grounding tools in the request body
- Unsupported preview-only cases use a DuckDuckGo search pass and prompt injection before the Gemini request

### Messenger Sending

- Replies are sent through Meta Graph API using the built-in default version and the `me` path
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

Webhook signature verification is enabled, and the incoming request signature is missing or invalid.

### Messenger User Gets The Attachment Fallback

The message was attachment-only, but the attachment was not a supported image or the image could not be fetched successfully.

### Messenger User Gets An Image Summary And Follow-Up Prompt

That is the expected flow for image-only messages. NubAgent inspects the image first, stores image context, sends a short visual summary back to Messenger, and then waits for the next Messenger message because Messenger image upload flow does not reliably pair the image with a same-turn text instruction.

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

### URL Context Does Not Seem To Work

Check all of the following:

1. The prompt includes one or more full `https://` or `http://` URLs
2. Your configured model list includes at least one Gemini 2.5 or Gemini 3 text model
3. `GEMINI_ENABLE_URL_CONTEXT` is not disabled

### Google Maps Grounding Does Not Seem To Work

Check all of the following:

1. The prompt is clearly geo-specific, such as asking for places, routes, or nearby recommendations
2. Your configured model list includes Gemini 2.5 Pro, Gemini 2.5 Flash, Gemini 2.5 Flash-Lite, or Gemini 2.0 Flash
3. `GEMINI_ENABLE_GOOGLE_MAPS` is not disabled
4. For `near me` style prompts, either the user has shared a Messenger location pin, opened a `!location` capture link, or `GEMINI_GOOGLE_MAPS_LATITUDE` and `GEMINI_GOOGLE_MAPS_LONGITUDE` are set as a fixed default location context

### Context Caching Does Not Seem To Work

Check all of the following:

1. For implicit caching, the repeated prefix is large enough and similar enough across nearby requests for Gemini to reuse it
2. For explicit caching, `GEMINI_CACHED_CONTENT` is set to a valid `cachedContents/...` resource name
3. For multi-model setups, the cache is mapped to the exact configured model name that created it
4. If you are using explicit caching, the cache has not expired

### Menu Options Or Quick Actions Does Not Show "Built with nubagent"

The repo now repairs Messenger profile state during normal webhook traffic and still supports `/api/maintenance` for explicit repair runs. If the button still does not appear, send a real message to the Page first so Messenger webhook traffic can trigger the background repair, then confirm the Page access token belongs to the same Page whose Messenger profile you expect to update.

## Project Structure

```text
api/
  location-capture.js
  maintenance.js
  webhook.js
lib/
  config.js
  duckduckgo.js
  gemini.js
  history.js
  location-capture.js
  messenger.js
  profile-state.js
  vision.js
src/
  App.jsx
  main.jsx
  styles.css
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request.

## License

This project is licensed under the MIT License.
