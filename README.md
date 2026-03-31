# NubAgent

Access Google Gemini through Facebook Messenger—enabling AI access when direct internet connectivity is unavailable.

## Problem Statement

Organizations and individuals need reliable access to advanced AI models, but face significant constraints:

- **Device limitations**: Low-end devices cannot run local models
- **Connectivity requirements**: Existing AI apps require direct internet access
- **Cost and privacy concerns**: Third-party API wrappers introduce latency, cost, and data handling risks
- **Dependency complexity**: Middleman services create maintenance overhead and reduce control

## Solution

NubAgent enables direct access to Google Gemini through Facebook Messenger as a gateway. By deploying a Vercel serverless function that directly integrates with Gemini's API, users can leverage Messenger (universally available on mobile devices) as their access point to advanced AI capabilities—regardless of direct internet connectivity.

**Architecture**: User → Facebook Messenger → Vercel Function → Gemini API (direct, no intermediaries)

## Key Features

- **Direct Gemini Integration** — Vercel serverless function calls Google Gemini API directly (no third-party wrappers)
- **Facebook Messenger Gateway** — Interact with Gemini through ubiquitous messaging platform
- **Offline-Capable** — Access advanced AI via Messenger when direct internet is unavailable; backend maintains connection
- **Data Privacy** — Direct Vercel → Gemini pipeline ensures no data passes through intermediary services
- **Self-Hosted** — Deploy and control your own instance; your configuration, your data

## Getting Started

### Prerequisites

Obtain the following API credentials:

#### 1. Google Gemini API Key

Visit [Google AI Studio](https://aistudio.google.com/app/apikey) and create a new API key.

#### 2. Facebook Page Access Token

First, you need a Facebook page. If you don't have one, [create a Facebook page](https://m.facebook.com/help/104002523024878/?helpref=uf_share).

The initial token is valid for only 1 hour. You must extend it to get a permanent token. Follow these steps:

1. Go to [Facebook Graph API Explorer](https://developers.facebook.com/tools/explorer/)
2. In the dropdown labeled "User or Page", select your Facebook page
3. Generate a token if you don't have one (click "Generate Access Token")
4. Copy the token and visit [Facebook Access Token Debugger](https://developers.facebook.com/tools/debug/accesstoken/)
5. Paste the token into the debugger and click "Debug"
6. Scroll to the bottom and click "Extend Access Token"
7. This generates a permanent token; copy this extended token for your configuration

**Note**: The extended token is permanent and won't expire. Always use the extended token, not the original 1-hour token.

#### 3. Webhook Verification Token

Create a secure random string (e.g., using `openssl rand -hex 32` or any password generator). This token is used only for Facebook's webhook verification during setup.

### Deployment

#### Option 1: One-Click Deploy to Vercel

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2FAmbitiousnoob%2Fnubagent&env=GEMINI_API_KEY,PAGE_ACCESS_TOKEN,VERIFY_TOKEN&project-name=nubagent&repo-name=nubagent)

1. Click the deploy button above
2. Authenticate with GitHub
3. Enter required environment variables
4. Click "Deploy"
5. Note your Vercel deployment URL

#### Option 2: CLI Deployment

```bash
git clone https://github.com/Ambitiousnoob/nubagent.git
cd nubagent
npm install
vercel --prod
```

Then configure environment variables in the Vercel dashboard.

### Environment Variables

All variables are required:

| Variable | Description |
|----------|-------------|
| `GEMINI_API_KEY` | Google Gemini API authentication key (see Prerequisites above) |
| `PAGE_ACCESS_TOKEN` | **Permanent** Facebook page token for messaging (see Prerequisites above—use the extended token, not the original) |
| `VERIFY_TOKEN` | Webhook verification token (see Prerequisites above) |

⚠️ **Important**: Use the **permanent (extended) token** from the Access Token Debugger, not the original token. This prevents authentication failures in production.

### Configure Facebook Webhook

After deployment, complete the following steps in the Facebook Developer Dashboard:

1. Navigate to **Messenger** → **Settings**
2. **Critical**: Disable Vercel login protection to allow Facebook webhook verification:
   - Go to Vercel Project Settings → **Security**
   - Uncheck **"Login Protection"** for `/api/webhook`
   - This permits Facebook to verify your webhook endpoint
3. Configure webhook details:
   - **Callback URL**: `https://your-vercel-deployment.vercel.app/api/webhook`
   - **Verify Token**: The token you created in environment variables
4. Subscribe to event types:
   - `messages`
   - `messaging_postbacks`
5. Configuration complete—your bot is now ready to receive messages

## Architecture

```
User Message (Facebook Messenger)
         ↓
Facebook Platform
         ↓
Vercel Serverless Function (/api/webhook)
         ↓
Google Gemini API (Direct)
         ↓
Response Generation
         ↓
Vercel Function
         ↓
Facebook Messenger
         ↓
User Response
```

**Key principle**: Direct integration between Vercel and Gemini eliminates intermediary services, reducing latency, cost, and data exposure.

## Documentation & Resources

- **Repository**: [github.com/Ambitiousnoob/nubagent](https://github.com/Ambitiousnoob/nubagent)
- **Issue Tracker**: [GitHub Issues](https://github.com/Ambitiousnoob/nubagent/issues)
- **Google Gemini API**: [ai.google.dev](https://ai.google.dev)
- **Facebook Messenger Platform**: [developers.facebook.com](https://developers.facebook.com/)
- **Vercel Deployment**: [vercel.com/docs](https://vercel.com/docs)

## Implementation Guide

This repository is documentation-focused. To implement NubAgent, you will need:

**Core Requirements**:
- Vercel serverless function (Node.js, Python, or other supported runtime)
- [Google Generative AI SDK](https://ai.google.dev) integration
- [Facebook Messenger API](https://developers.facebook.com/docs/messenger-platform) webhook handler

**Typical Implementation Flow**:

```
1. POST /api/webhook receives Facebook message event
2. Extract message content and sender ID
3. Initialize Gemini client with API key
4. Call Gemini with user message
5. Retrieve and parse response
6. Send response to Facebook Messenger API
7. Return 200 success response to Facebook
```

**Supported Runtimes**:
- [Node.js on Vercel](https://vercel.com/docs/functions/serverless-functions)
- [Python on Vercel](https://vercel.com/docs/functions/serverless-functions/python)
- [Other Vercel-supported runtimes](https://vercel.com/docs/functions)

## Support

- **Questions?** [Open a GitHub Issue](https://github.com/Ambitiousnoob/nubagent/issues)
- **Gemini API Documentation**: [ai.google.dev](https://ai.google.dev)
- **Vercel Support**: [vercel.com/docs](https://vercel.com/docs)
- **Facebook Developer Docs**: [developers.facebook.com](https://developers.facebook.com/docs)
