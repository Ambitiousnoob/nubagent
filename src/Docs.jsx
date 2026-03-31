import React from "react";
import {
  ALIAS_ENDPOINT_EXAMPLE,
  API_CREATION_CHECKLIST,
  API_CREATION_STEPS,
  API_REFERENCE_SECTIONS,
  BASIC_ENDPOINT_EXAMPLE,
  ENVIRONMENT_VARIABLES,
  MIGRATION_NOTES,
  OPERATIONS_SECTIONS,
} from "./docsContent.js";

function CodeBlock({ children }) {
  return (
    <pre className="docs-page__code">
      <code>{children}</code>
    </pre>
  );
}

function EndpointCard({ endpoint }) {
  return (
    <article className="docs-card docs-card--endpoint" id={endpoint.id}>
      <div className="docs-card__eyebrow">
        <span>{endpoint.methods.join(" · ")}</span>
        <span>
          {endpoint.paths.length} path{endpoint.paths.length === 1 ? "" : "s"}
        </span>
      </div>
      <h3 className="docs-card__title">{endpoint.title}</h3>
      <p className="docs-card__body">{endpoint.summary}</p>

      <div className="docs-card__paths">
        {endpoint.paths.map((path) => (
          <code key={path}>{path}</code>
        ))}
      </div>

      <ul className="docs-card__list">
        {endpoint.keyPoints.map((point) => (
          <li key={point}>{point}</li>
        ))}
      </ul>

      <details className="docs-card__details">
        <summary>Request, response, and owning files</summary>
        <div className="docs-card__details-body">
          <div className="docs-card__details-group">
            <h4>Request shape</h4>
            <CodeBlock>{endpoint.requestShape}</CodeBlock>
          </div>
          <div className="docs-card__details-group">
            <h4>Response shape</h4>
            <CodeBlock>{endpoint.responseShape}</CodeBlock>
          </div>
          <div className="docs-card__details-group">
            <h4>Owning files</h4>
            <div className="docs-card__paths">
              {endpoint.implementationFiles.map((file) => (
                <code key={file}>{file}</code>
              ))}
            </div>
          </div>
        </div>
      </details>
    </article>
  );
}

function StepCard({ index, step }) {
  return (
    <article className="docs-step">
      <div className="docs-step__number">{index + 1}</div>
      <div className="docs-step__content">
        <h3 className="docs-step__title">{step.title}</h3>
        <p className="docs-step__body">{step.body}</p>
        <ul className="docs-step__list">
          {step.bullets.map((bullet) => (
            <li key={bullet}>{bullet}</li>
          ))}
        </ul>
      </div>
    </article>
  );
}

function NoteCard({ title, body, bullets = [] }) {
  return (
    <article className="docs-card">
      <h3 className="docs-card__title">{title}</h3>
      <p className="docs-card__body">{body}</p>
      {bullets.length > 0 && (
        <ul className="docs-card__list">
          {bullets.map((bullet) => (
            <li key={bullet}>{bullet}</li>
          ))}
        </ul>
      )}
    </article>
  );
}

export default function Docs() {
  const totalEndpoints = API_REFERENCE_SECTIONS.reduce(
    (count, endpoint) => count + endpoint.paths.length,
    0,
  );

  return (
    <div className="docs-page docs-page--refined">
      <div className="docs-page__hero">
        <div className="docs-page__hero-copy">
          <div className="docs-page__eyebrow">NubAgent Docs</div>
          <h1 className="docs-page__title">Simple Gemini chat API reference</h1>
          <p className="docs-page__lead">
            Operator-facing docs for the single public chat endpoint, the files
            that own it, and the operational rules around the current docs-only
            frontend.
          </p>
        </div>

        <div className="docs-page__stats">
          <div className="docs-stat">
            <span className="docs-stat__value">
              {API_REFERENCE_SECTIONS.length}
            </span>
            <span className="docs-stat__label">API section</span>
          </div>
          <div className="docs-stat">
            <span className="docs-stat__value">{totalEndpoints}</span>
            <span className="docs-stat__label">Public path</span>
          </div>
          <div className="docs-stat">
            <span className="docs-stat__value">JSON</span>
            <span className="docs-stat__label">Response mode</span>
          </div>
        </div>
      </div>

      <nav className="docs-page__jump-nav" aria-label="Docs sections">
        <a href="#api-reference">API Reference</a>
        <a href="#api-creation">API Playbook</a>
        <a href="#operations">Operations</a>
        <a href="#migration">Migration</a>
      </nav>

      <section className="docs-section" id="api-reference">
        <header className="docs-section__header">
          <div>
            <div className="docs-section__eyebrow">Public API</div>
            <h2 className="docs-section__title">API reference</h2>
          </div>
          <p className="docs-section__body">
            Use this section to inspect the live contract for the only public
            endpoint in the repo.
          </p>
        </header>

        <div className="docs-section__intro-grid">
          <div className="docs-section__intro-card">
            <span>Surface area</span>
            <strong>One public route: `/api/chat`.</strong>
          </div>
          <div className="docs-section__intro-card">
            <span>Behavior</span>
            <strong>
              Plain chat completion backed by Gemini, with no agentic runtime.
            </strong>
          </div>
        </div>

        <div className="docs-endpoints">
          {API_REFERENCE_SECTIONS.map((endpoint) => (
            <EndpointCard key={endpoint.id} endpoint={endpoint} />
          ))}
        </div>
      </section>

      <section className="docs-section" id="api-creation">
        <header className="docs-section__header">
          <div>
            <div className="docs-section__eyebrow">Implementation Guide</div>
            <h2 className="docs-section__title">API playbook</h2>
          </div>
          <p className="docs-section__body">
            This is the maintenance path for the current backend shape: one
            public route, one provider adapter, and one docs surface.
          </p>
        </header>

        <div className="docs-section__intro-grid">
          <div className="docs-section__intro-card">
            <span>Primary rule</span>
            <strong>Do not re-expand the public API accidentally.</strong>
          </div>
          <div className="docs-section__intro-card">
            <span>Outcome</span>
            <strong>
              Contract changes stay small, reviewable, and easy to verify.
            </strong>
          </div>
        </div>

        <div className="docs-steps">
          {API_CREATION_STEPS.map((step, index) => (
            <StepCard key={step.title} index={index} step={step} />
          ))}
        </div>

        <div className="docs-creation-grid">
          <NoteCard
            title="Creation checklist"
            body="Run through this list whenever `/api/chat` changes."
            bullets={API_CREATION_CHECKLIST}
          />

          <article className="docs-card">
            <h3 className="docs-card__title">Where API work usually lands</h3>
            <div className="docs-card__paths">
              <code>api/chat.js</code>
              <code>lib/gemini-chat.js</code>
              <code>lib/web.js</code>
              <code>vercel.json</code>
              <code>README.md</code>
            </div>
            <p className="docs-card__body">
              Keep externally visible behavior in `api/chat.js`, push Gemini
              mapping into `lib/gemini-chat.js`, and keep shared parsing or size
              guards in `lib/web.js`.
            </p>
          </article>
        </div>

        <div className="docs-examples">
          <article className="docs-card">
            <h3 className="docs-card__title">Minimal endpoint skeleton</h3>
            <CodeBlock>{BASIC_ENDPOINT_EXAMPLE}</CodeBlock>
          </article>

          <article className="docs-card">
            <h3 className="docs-card__title">Single-route wiring</h3>
            <CodeBlock>{ALIAS_ENDPOINT_EXAMPLE}</CodeBlock>
          </article>
        </div>
      </section>

      <section className="docs-section" id="operations">
        <header className="docs-section__header">
          <div>
            <div className="docs-section__eyebrow">Operations</div>
            <h2 className="docs-section__title">Runtime and config notes</h2>
          </div>
          <p className="docs-section__body">
            These are the practical rules that keep the endpoint predictable in
            deployment.
          </p>
        </header>

        <div className="docs-creation-grid">
          {OPERATIONS_SECTIONS.map((section) => (
            <NoteCard
              key={section.title}
              title={section.title}
              body={section.body}
              bullets={section.bullets}
            />
          ))}
        </div>

        <article className="docs-card">
          <h3 className="docs-card__title">Environment variables</h3>
          <div className="docs-table-wrap">
            <table className="docs-table">
              <thead>
                <tr>
                  <th>Variable</th>
                  <th>Required</th>
                  <th>Default</th>
                  <th>Description</th>
                </tr>
              </thead>
              <tbody>
                {ENVIRONMENT_VARIABLES.map((item) => (
                  <tr key={item.name}>
                    <td>
                      <code>{item.name}</code>
                    </td>
                    <td>{item.required}</td>
                    <td>{item.defaultValue}</td>
                    <td>{item.description}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </article>
      </section>

      <section className="docs-section" id="migration">
        <header className="docs-section__header">
          <div>
            <div className="docs-section__eyebrow">Migration</div>
            <h2 className="docs-section__title">What changed</h2>
          </div>
          <p className="docs-section__body">
            These notes capture the intentional shift away from the older,
            broader runtime.
          </p>
        </header>

        <div className="docs-creation-grid">
          {MIGRATION_NOTES.map((item) => (
            <NoteCard key={item.title} title={item.title} body={item.body} />
          ))}
        </div>
      </section>
    </div>
  );
}
