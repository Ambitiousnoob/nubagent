# NubAgent

## The Problem

You want **access to cutting-edge AI** (Gemini) **even when you can't access it directly**.

Real talk:
- 🚫 **Your device is low-end.** Can't run local models like Llama or Mistral—no GPU.
- 🚫 **Gemini/GPT apps require internet.** If you're offline, you can't use them.
- 🚫 **API wrappers are expensive and slow.** You want direct, free access.
- 🚫 **You don't want middlemen.** Wrappers = they see your data.

## The Solution

**Access Gemini through Facebook Messenger.**

You don't need direct internet to chat with Gemini. Deploy NubAgent (your own Gemini bot) on Vercel. Connect it to Facebook Messenger. Now:
- Your friends can message your Facebook page
- Your bot (running on Vercel) calls Gemini
- Gemini's response comes back through Messenger
- **No middleman API.** Direct Vercel → Gemini.

**Offline AI access:** When you don't have direct internet, use Messenger (which your phone has) to chat with your Gemini bot. It's like having a personal AI assistant on Facebook.

## Features

- **Vercel + Gemini Direct** — Your bot on Vercel calls Gemini directly (no middleman APIs)
- **Facebook Messenger Access** — Chat with your Gemini bot through Messenger
- **Offline-Friendly** — Use Messenger when direct internet isn't available; bot still has internet to call Gemini
- **No Data Sharing** — Direct Vercel → Gemini. No wrappers, no third parties seeing your data
- **Your Own Deployment** — You own the bot. You control the key. Your data stays yours.

## Quick Start

This is a **documentation repository**. To use NubAgent:

1. **Get your API keys:**
   - Gemini API Key: [Google AI Studio](https://aistudio.google.com/app/apikey)
   - Facebook Page Token & Verify Token: [Facebook Developer Dashboard](https://developers.facebook.com/)

2. **Deploy to Vercel** using the button below

3. **Add your keys** in Vercel environment variables

4. **Set webhook URL** in Facebook Developer Dashboard to your Vercel deployment

That's it. Users message your Facebook page → Vercel calls Gemini → AI responds.

## Environment Variables

**All required:**

```bash
# Google Gemini
GEMINI_API_KEY=your_gemini_api_key_here

# Facebook Messenger
PAGE_ACCESS_TOKEN=your_facebook_page_access_token
VERIFY_TOKEN=your_webhook_verification_token
```

**Where to get them:**
- **GEMINI_API_KEY:** [Google AI Studio](https://aistudio.google.com/app/apikey) → Create API key
- **PAGE_ACCESS_TOKEN:** [Facebook Developer Dashboard](https://developers.facebook.com/) → Your Page → Settings → Messenger
- **VERIFY_TOKEN:** Any random string you create (same one used in webhook verification)

## Deploy

### One-Click Deploy to Vercel

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2FAmbitiousnoob%2Fnubagent&env=GEMINI_API_KEY,PAGE_ACCESS_TOKEN,VERIFY_TOKEN&project-name=nubagent&repo-name=nubagent)

**Steps:**
1. Click deploy button
2. Connect your GitHub account
3. Add all 3 environment variables (see above)
4. Click "Deploy"
5. Copy your Vercel URL (you'll need it for Facebook)

### After Deployment: Connect Facebook

1. Go to [Facebook Developer Dashboard](https://developers.facebook.com/)
2. Create/select your App and Page
3. Go to **Messenger** → **Settings**
4. Add webhook:
   - **Callback URL:** `https://your-vercel-deployment.vercel.app/api/webhook`
   - **Verify Token:** The token you set in env variables
5. Subscribe to **messages** and **messaging_postbacks**
6. Done! Users can now message your page

### Or Deploy from CLI

```bash
git clone https://github.com/Ambitiousnoob/nubagent.git
cd nubagent
npm install
vercel --prod
```

Then add environment variables in Vercel dashboard and configure Facebook webhook.

## How It Works

```
User messages your Facebook page
    ↓
Facebook sends POST to /api/webhook
    ↓
Your Vercel function receives it
    ↓
Vercel calls Google Gemini (with your key)
    ↓
Gemini generates response
    ↓
Vercel sends response back to Facebook
    ↓
User sees the AI response
```

**That's it.** Direct line from Facebook → Vercel → Gemini. No middlemen. Your data never goes through third-party APIs.

## Documentation

- **GitHub:** [Ambitiousnoob/nubagent](https://github.com/Ambitiousnoob/nubagent)
- **Issues & Questions:** [GitHub Issues](https://github.com/Ambitiousnoob/nubagent/issues)
- **Gemini API Docs:** [ai.google.dev](https://ai.google.dev)
- **Facebook Messenger:** [developers.facebook.com](https://developers.facebook.com/)

## Development

This is a documentation repo. To build NubAgent:

**Required:**
1. Vercel serverless function
2. [Google Generative AI SDK](https://ai.google.dev)
3. [Facebook Messenger API](https://developers.facebook.com/docs/messenger-platform)

**Implementation outline:**
```
POST /api/webhook (Facebook webhook)
  ↓
Verify webhook with Facebook
  ↓
Parse incoming message
  ↓
Call Gemini API with message
  ↓
Send response back to Facebook
```

**Stack Options:**
- [Node.js + Vercel](https://vercel.com/docs/functions/serverless-functions)
- [Python + Vercel](https://vercel.com/docs/functions/serverless-functions/python)
- [Any Vercel-supported runtime](https://vercel.com/docs/functions)

## Support

- **Questions?** [Open a GitHub Issue](https://github.com/Ambitiousnoob/nubagent/issues)
- **API Docs:** [Google Gemini API](https://ai.google.dev)
- **Vercel Help:** [vercel.com/docs](https://vercel.com/docs)

---

Built by [ambitiousnoob](https://github.com/ambitiousnoob) | Direct Gemini integration on Vercel
