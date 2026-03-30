import React from 'react';
import {
  ALIAS_ENDPOINT_EXAMPLE,
  API_CREATION_CHECKLIST,
  API_CREATION_STEPS,
  API_REFERENCE_SECTIONS,
  BASIC_ENDPOINT_EXAMPLE,
  SUBAGENT_GROUPS,
} from './lib/docsContent.js';

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
        <span>{endpoint.methods.join(' · ')}</span>
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

function SubagentGroup({ group }) {
  return (
    <section className="docs-group" id={group.id}>
      <header className="docs-group__header">
        <div>
          <h3 className="docs-group__title">{group.title}</h3>
          <p className="docs-group__summary">{group.summary}</p>
        </div>
        <span className="docs-group__count">
          {group.agents.length} agent{group.agents.length === 1 ? '' : 's'}
        </span>
      </header>

      <div className="docs-subagents">
        {group.agents.map((agent) => (
          <article key={agent.name} className="docs-subagent">
            <div className="docs-subagent__name">{agent.name}</div>
            <p className="docs-subagent__scope">{agent.scope}</p>
          </article>
        ))}
      </div>
    </section>
  );
}

export default function Docs() {
  const totalEndpoints = API_REFERENCE_SECTIONS.reduce((count, endpoint) => count + endpoint.paths.length, 0);
  const totalSubagents = SUBAGENT_GROUPS.reduce((count, group) => count + group.agents.length, 0);

  return (
    <div className="docs-page">
      <div className="docs-page__hero">
        <div className="docs-page__hero-copy">
          <div className="docs-page__eyebrow">NubAgent Docs</div>
          <h1 className="docs-page__title">API reference, endpoint creation, and the full subagent map</h1>
          <p className="docs-page__lead">
            This is the web-facing operator guide for NubAgent. It pulls the important repo knowledge into one place:
            what the APIs do, how new endpoints should be created, and which subagents own each part of the product.
          </p>
        </div>

        <div className="docs-page__stats">
          <div className="docs-stat">
            <span className="docs-stat__value">{API_REFERENCE_SECTIONS.length}</span>
            <span className="docs-stat__label">API sections</span>
          </div>
          <div className="docs-stat">
            <span className="docs-stat__value">{totalEndpoints}</span>
            <span className="docs-stat__label">Public paths</span>
          </div>
          <div className="docs-stat">
            <span className="docs-stat__value">{totalSubagents}</span>
            <span className="docs-stat__label">Subagents</span>
          </div>
        </div>
      </div>

      <nav className="docs-page__jump-nav" aria-label="Docs sections">
        <a href="#api-reference">API Reference</a>
        <a href="#api-creation">API Creation</a>
        <a href="#subagent-catalog">Subagents</a>
      </nav>

      <section className="docs-section" id="api-reference">
        <header className="docs-section__header">
          <div>
            <div className="docs-section__eyebrow">Public API</div>
            <h2 className="docs-section__title">API reference</h2>
          </div>
          <p className="docs-section__body">
            The current runtime mixes dedicated endpoints with alias-based shared boundaries. Use these cards to see what
            exists, what each path is for, and which files own the implementation.
          </p>
        </header>

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
            <h2 className="docs-section__title">API creation playbook</h2>
          </div>
          <p className="docs-section__body">
            Use this when you need to add or reshape a public endpoint inside NubAgent. The main rule is to choose the
            right boundary first, then keep metadata, aliases, tests, and docs aligned.
          </p>
        </header>

        <div className="docs-steps">
          {API_CREATION_STEPS.map((step, index) => (
            <StepCard key={step.title} index={index} step={step} />
          ))}
        </div>

        <div className="docs-creation-grid">
          <article className="docs-card">
            <h3 className="docs-card__title">Creation checklist</h3>
            <ul className="docs-card__list">
              {API_CREATION_CHECKLIST.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          </article>

          <article className="docs-card">
            <h3 className="docs-card__title">Where new API work usually lands</h3>
            <div className="docs-card__paths">
              <code>api/&lt;name&gt;.js</code>
              <code>api/content.js</code>
              <code>api/utils-handler.cjs</code>
              <code>api/tools/&lt;tool&gt;.js</code>
              <code>vercel.json</code>
              <code>API.md</code>
            </div>
            <p className="docs-card__body">
              Use dedicated files for real product surfaces, `api/content.js` for retrieval aliases, and
              `api/utils-handler.cjs` for utility aliases that share one backend implementation.
            </p>
          </article>
        </div>

        <div className="docs-examples">
          <article className="docs-card">
            <h3 className="docs-card__title">Basic endpoint shape</h3>
            <CodeBlock>{BASIC_ENDPOINT_EXAMPLE}</CodeBlock>
          </article>

          <article className="docs-card">
            <h3 className="docs-card__title">Alias pattern</h3>
            <CodeBlock>{ALIAS_ENDPOINT_EXAMPLE}</CodeBlock>
          </article>
        </div>
      </section>

      <section className="docs-section" id="subagent-catalog">
        <header className="docs-section__header">
          <div>
            <div className="docs-section__eyebrow">Ownership Map</div>
            <h2 className="docs-section__title">NubAgent subagent catalog</h2>
          </div>
          <p className="docs-section__body">
            Every meaningful slice in NubAgent has a dedicated owner. This catalog is the web view of the local
            `nub_*` roster so you can see who owns orchestration, search, fetch, answer verification, docs, and release.
          </p>
        </header>

        <div className="docs-groups">
          {SUBAGENT_GROUPS.map((group) => (
            <SubagentGroup key={group.id} group={group} />
          ))}
        </div>
      </section>
    </div>
  );
}
