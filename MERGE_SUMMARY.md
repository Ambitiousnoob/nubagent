# Merge Summary: nubagent + .bot

## What Changed

### 1. **New Webhook Handler**
- Added `api/webhook.js` - Facebook Messenger webhook that directly calls Gemini API
- **Key change:** No longer makes HTTP calls to external API. Instead:
  - Imports `runGeminiChat()` directly from `lib/gemini-chat.js`
  - Calls it in-process (no network overhead)
  - Constructs the chat payload from user message + system context (date/time/user name)

### 2. **Dependencies Updated**
- Added `@vercel/functions` to `package.json` (required for `waitUntil()` in webhook handler)
- All other dependencies remain the same

### 3. **Environment Variables**
- Updated `.env.example` with new Facebook Messenger settings:
  - `PAGE_ACCESS_TOKEN` - Facebook page access token
  - `VERIFY_TOKEN` - Webhook verification token
- Existing Gemini settings unchanged

### 4. **Deleted**
- `.bot/` directory removed completely
- All code merged into nubagent as a unified project

## Architecture

**Before:**
```
Facebook User → .bot webhook (HTTP call) → nubagent API → Gemini
```

**After:**
```
Facebook User → nubagent webhook (direct function call) → Gemini
```

## API Endpoints

### `/api/chat` (unchanged)
- Direct chat API for external clients
- HTTP POST endpoint
- Supports CORS
- Same request/response format as before

### `/api/webhook` (new)
- Facebook Messenger webhook handler
- GET: Webhook verification
- POST: Message receiving & processing
- Directly calls Gemini (no intermediate HTTP)

## How It Works

1. Facebook sends message to `/api/webhook`
2. Webhook handler:
   - Fetches user profile (name) from Facebook Graph API
   - Checks if message needs direct reply (date/time/name queries)
   - If direct reply: send immediately
   - If AI needed: constructs system message with context + user message
   - Calls `runGeminiChat()` directly from `lib/gemini-chat.js`
   - Formats response and sends back via Facebook Graph API

## Files Modified

- `api/webhook.js` - NEW (516 lines)
- `api/chat.js` - unchanged
- `lib/gemini-chat.js` - unchanged
- `lib/web.js` - unchanged
- `package.json` - added @vercel/functions
- `.env.example` - added Facebook tokens

## Testing

All code formatted and linted:
```bash
npm run lint      # ✓ ESLint passed
npm run format    # ✓ Prettier applied
npm run build     # vite build for frontend
npm run dev       # vite dev server
```

## Deployment

Deploy as usual to Vercel - both endpoints are now in the same project:
- Frontend: `src/` + `index.html` (React docs)
- APIs: `api/chat.js` + `api/webhook.js`

No changes to deployment process.
