import {
  Bot,
  KeyRound,
  MessageSquareMore,
  ServerCog,
  ShieldCheck,
  Workflow,
} from "lucide-react";

const envBlock = `GEMINI_API_KEY=your_gemini_api_key,your_second_gemini_api_key
GEMINI_CHAT_MODEL=primary_model,fallback_model,another_fallback_model
PAGE_ACCESS_TOKEN=your_page_access_token
VERIFY_TOKEN=your_webhook_verification_token
POSTGRES_URL=postgresql://user:password@host:5432/database
PAGE_ID=your_page_id`;

const callbackBlock = `Callback URL
https://your-deployment.vercel.app/api/webhook

Subscribe to
messages
messaging_postbacks`;

const featureCards = [
  {
    icon: MessageSquareMore,
    title: "Messenger gateway",
    body: "Receives Facebook webhook events, marks messages seen, and sends plain text replies back to the same PSID.",
  },
  {
    icon: Bot,
    title: "Direct Gemini call",
    body: "Uses the Gemini REST API directly from the Vercel function instead of layering another service in the middle.",
  },
  {
    icon: ShieldCheck,
    title: "Webhook hardening",
    body: "Supports the verification challenge by default and optional `X-Hub-Signature-256` validation when an app secret is present.",
  },
];

const runtimeCards = [
  {
    icon: Workflow,
    label: "Flow",
    value: "Messenger -> /api/webhook -> Gemini -> Send API",
  },
  {
    icon: ServerCog,
    label: "Deployment",
    value: "Vercel static frontend plus serverless webhook",
  },
  {
    icon: KeyRound,
    label: "Secrets",
    value: "Gemini key pool, model, page token, verify token",
  },
];

export default function App() {
  return (
    <main className="page-shell">
      <section className="hero">
        <div className="hero-copy">
          <p className="eyebrow">Messenger to Gemini</p>
          <h1>
            NubAgent turns a Facebook Page into a direct Gemini chat surface.
          </h1>
          <p className="hero-text">
            The repository now includes a real webhook implementation, a Gemini
            request path, and a buildable frontend. Deploy it to Vercel, connect
            the Messenger webhook, and route inbound text directly to Gemini.
          </p>
          <div className="hero-pills">
            <span>Webhook verification</span>
            <span>Typing indicators</span>
            <span>Postgres-backed chat history</span>
            <span>Plain-text mobile replies</span>
          </div>
        </div>
        <div className="signal-panel">
          {runtimeCards.map(({ icon: Icon, label, value }) => (
            <article className="signal-card" key={label}>
              <Icon size={20} />
              <div>
                <p>{label}</p>
                <strong>{value}</strong>
              </div>
            </article>
          ))}
        </div>
      </section>

      <section className="card-grid">
        {featureCards.map(({ icon: Icon, title, body }) => (
          <article className="feature-card" key={title}>
            <div className="feature-icon">
              <Icon size={20} />
            </div>
            <h2>{title}</h2>
            <p>{body}</p>
          </article>
        ))}
      </section>

      <section className="setup-grid">
        <article className="setup-card">
          <h2>Required environment</h2>
          <p>
            `GEMINI_API_KEY` accepts one key or a comma-separated key pool. The
            server rotates through that list in round-robin order per request.
            `GEMINI_CHAT_MODEL` also accepts a comma-separated priority list, so
            the next model can take over when the primary one is overloaded.
            `PAGE_ID` is optional but recommended so the Send API can target the
            page path directly.
          </p>
          <pre>
            <code>{envBlock}</code>
          </pre>
        </article>

        <article className="setup-card">
          <h2>Facebook webhook setup</h2>
          <p>
            Point the Messenger webhook at the Vercel deployment and use the
            same verification token you configured in the environment.
          </p>
          <pre>
            <code>{callbackBlock}</code>
          </pre>
        </article>
      </section>
    </main>
  );
}
