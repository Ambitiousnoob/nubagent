# NubAgent

## The Problem

You want **offline AI access**—total control, no middlemen, no external dependencies.

But:
- 🚫 **Can't run local models.** Your device is too low-end. Llama 2? Mistral? They need GPU. You don't have one.
- 🚫 **Need real-world AI.** Local models aren't good enough. You need Gemini, GPT, Claude—the cutting edge.
- 🚫 **Can't use API wrappers.** They're expensive. They're slow. They see your data. They're another middleman.

**You're stuck:** Need powerful AI, but want to own the deployment.

## The Solution

**Deploy your own Gemini endpoint.**

NubAgent is a documentation guide for setting up **Vercel + Google Gemini direct integration**. Your deployment, your API key, your data. Vercel calls Gemini directly. No proxies. No middlemen. No wrappers.

That's offline AI access: *you* own the deployment.

## Features

- **Vercel → Gemini Direct** — Serverless function calls Gemini API directly
- **Facebook Messenger** — Optional integration for chat via Facebook
- **No Intermediaries** — Your key, your deployment, your data
- **Production Ready** — Auto-scaling, load-balanced, on Vercel

## Quick Start

This is a **documentation repository**. To use NubAgent:

1. **Clone the repo** (or fork it):
   ```bash
   git clone https://github.com/Ambitiousnoob/nubagent.git
   ```

2. **Deploy to Vercel** with the button below (or use CLI)

3. **Add your keys** in Vercel environment variables

4. **Connect Facebook** (optional) to your webhook endpoint

That's it. Vercel runs your Gemini integration serverless.

## Environment Variables

### Required

```bash
GEMINI_API_KEY=your_gemini_api_key_here
```

Get it from [Google AI Studio](https://aistudio.google.com/app/apikey).

### Optional: Facebook Messenger

```bash
PAGE_ACCESS_TOKEN=your_facebook_page_access_token
VERIFY_TOKEN=your_webhook_verification_token
```

Get tokens from [Facebook Developer Dashboard](https://developers.facebook.com/).

## Deployment

### One-Click Deploy to Vercel

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2FAmbitiousnoob%2Fnubagent&env=GEMINI_API_KEY,PAGE_ACCESS_TOKEN,VERIFY_TOKEN&project-name=nubagent&repo-name=nubagent)

**Steps:**
1. Click the deploy button above
2. Connect your GitHub account
3. Add your `GEMINI_API_KEY` (required)
4. Add `PAGE_ACCESS_TOKEN` and `VERIFY_TOKEN` (optional, for Messenger)
5. Click "Deploy"
6. Done! Your Gemini endpoint is live on Vercel

### Or Deploy from CLI

```bash
git clone https://github.com/Ambitiousnoob/nubagent.git
cd nubagent
npm install
vercel --prod
```

Then add environment variables in Vercel dashboard.

## Architecture

```
Your Vercel Deployment
    ↓ (with your GEMINI_API_KEY)
Calls Google Gemini directly
    ↓
Facebook Messenger (optional)
OR
Your own app/client
```

**That's it.** No proxies. No middlemen. Your key, your deployment, direct to Gemini.

## Documentation

- **GitHub:** [Ambitiousnoob/nubagent](https://github.com/Ambitiousnoob/nubagent)
- **Issues & Questions:** [GitHub Issues](https://github.com/Ambitiousnoob/nubagent/issues)
- **Gemini API Docs:** [ai.google.dev](https://ai.google.dev)
- **Facebook Messenger:** [developers.facebook.com](https://developers.facebook.com/)

## Development

This is a documentation repo. To implement NubAgent yourself:

1. Create a Vercel serverless function
2. Use the [Google Generative AI SDK](https://ai.google.dev)
3. Call `genai.Client().models.generate_content()` or equivalent
4. (Optional) Integrate with Facebook Messenger API

**Stack Options:**
- [Node.js + Vercel](https://vercel.com/docs/functions/serverless-functions)
- [Python + Vercel](https://vercel.com/docs/functions/serverless-functions/python)
- [Any runtime that Vercel supports](https://vercel.com/docs/functions)

## Support

- **Questions?** [Open a GitHub Issue](https://github.com/Ambitiousnoob/nubagent/issues)
- **API Docs:** [Google Gemini API](https://ai.google.dev)
- **Vercel Help:** [vercel.com/docs](https://vercel.com/docs)

---

Built by [ambitiousnoob](https://github.com/ambitiousnoob) | Direct Gemini integration on Vercel
