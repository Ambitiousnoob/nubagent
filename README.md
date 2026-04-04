# NubAgent

![NubAgent banner](assets/nubagent-banner.svg)

NubAgent is a Facebook Messenger bot that sends user messages to Gemini and sends the reply back through Messenger.

It is built for simple mobile use:
- Messenger in
- Gemini reply out
- optional images, location, memory, and commands

<a href="https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2FAmbitiousnoob%2Fnubagent&env=GEMINI_API_KEY,PAGE_ACCESS_TOKEN,VERIFY_TOKEN,POSTGRES_URL,GEMINI_CHAT_MODEL,OPTIONAL_INSTRUCTION&project-name=nubagent&repo-name=nubagent"><img src="https://vercel.com/button" alt="Deploy with Vercel" /></a>

## Main Features

- Messenger webhook at `/api/webhook`
- Gemini model fallback
- image understanding for supported image attachments
- Google Maps grounding for location-based prompts
- exact-address lookup from saved coordinates
- per-user memory and rolling summaries in Postgres
- `!thinking` command for per-user thinking level
- browser-based `!location` capture link
- health endpoint at `/api/health`

## Commands

- `!help`
- `!credits`
- `!thinking`
- `!thinking off|minimal|low|medium|high`
- `!thinking default`
- `!reset`
- `!summary`
- `!memory`
- `!memory add <text>`
- `!location`
- `!location show`
- `!location clear`
- `!forget <phrase>`
- `!forget all`
- `!privacy`

Any message that starts with `!` is treated as command mode. Unknown bang commands are not sent to Gemini.

## What You Need

- a Google AI Studio API key
- one or more Gemini model names
- a Facebook Page
- a Meta app with Messenger enabled
- a Page access token
- a `VERIFY_TOKEN`
- a Postgres database
- a Vercel project or another Node-compatible host

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/ambitiousnoob/nubagent.git
cd nubagent
npm install
```

### 2. Create `.env`

Copy the example file:

```bash
cp .env.example .env
```

Minimum working example:

```env
GEMINI_API_KEY=your_gemini_api_key
GEMINI_CHAT_MODEL=gemini-2.5-flash,gemini-2.5-pro
PAGE_ACCESS_TOKEN=your_page_access_token
VERIFY_TOKEN=your_verify_token
POSTGRES_URL=postgresql://user:password@host:5432/database
```

Useful optional settings:

```env
OPTIONAL_INSTRUCTION=
SYSTEM_PROMPT=
APP_BASE_URL=
GEMINI_CHAT_THINKING_LEVEL=low
GEMINI_ENABLE_GOOGLE_SEARCH=true
GEMINI_ENABLE_CODE_EXECUTION=true
GEMINI_ENABLE_GOOGLE_MAPS=true
GEMINI_ENABLE_URL_CONTEXT=true
GOOGLE_GEOCODING_API_KEY=
LOCATION_CAPTURE_TTL_MINUTES=15
```

Important:
- `GEMINI_CHAT_THINKING_LEVEL` is only the default
- users can override it in chat with `!thinking`
- `GOOGLE_GEOCODING_API_KEY` is optional but helps exact-address lookups

### 3. Run locally

```bash
npm run lint
npm run build
```

Local behavior:
- `npm run dev` runs the frontend only
- `npx vercel dev` is the right way to test the API routes locally

### 4. Deploy

Use the button above or:

```bash
npx vercel
```

For production:

```bash
npx vercel --prod
```

### 5. Configure the Messenger webhook

Use these values in the Meta dashboard:

- callback URL: `https://YOUR-DOMAIN/api/webhook`
- verify token: the same value as `VERIFY_TOKEN`

Then subscribe your Page to the Messenger events your bot needs.

### 6. Test it in Messenger

Try these in order:

1. `!help`
2. a normal text message
3. `!thinking medium`
4. `!location`
5. a nearby prompt like `cafes near me`
6. an exact-address prompt like `what is my exact address?`

## Important Environment Variables

| Variable | Required | Purpose |
| --- | --- | --- |
| `GEMINI_API_KEY` | yes | Gemini API key or comma-separated keys |
| `GEMINI_CHAT_MODEL` | yes | ordered Gemini model list |
| `PAGE_ACCESS_TOKEN` | yes | Messenger send token |
| `VERIFY_TOKEN` | yes | webhook verification token |
| `POSTGRES_URL` | yes | database for chat state |
| `OPTIONAL_INSTRUCTION` | no | extra low-priority guidance |
| `SYSTEM_PROMPT` | no | replaces the default bot instruction |
| `GEMINI_CHAT_THINKING_LEVEL` | no | bot-wide default thinking level |
| `APP_BASE_URL` | no | used for `!location` capture links |
| `GOOGLE_GEOCODING_API_KEY` | no | improves exact-address reverse geocoding |

## Routes

- `GET /api/webhook` - Meta webhook verification
- `POST /api/webhook` - Messenger event handler
- `GET /api/health` - JSON health/status
- `GET /api/location-capture` - browser location capture page
- `POST /api/location-capture` - saves browser location
- `GET /api/maintenance` - maintenance/profile repair endpoint

## How It Behaves

- text messages go to Gemini
- supported image attachments can be sent to Gemini
- location pins are saved for later Maps prompts
- `!location` generates a browser link that saves the user's location
- exact-address questions use reverse geocoding from saved coordinates
- recent chat, memory, summaries, location, and reliability records are stored in Postgres
- duplicate Messenger events are ignored safely

## Notes

- `!thinking` is per-user and overrides the default env value
- thinking settings are only sent on supported Gemini text models in the current runtime
- if a user has not shared a location, location-based prompts use the configured default Maps coordinates if present
- `.vercel/` and `test/` are ignored by Git

## Credits

- Developer: `Ambitiousnoob`
- Repository: `github.com/Ambitiousnoob/nubagent`
