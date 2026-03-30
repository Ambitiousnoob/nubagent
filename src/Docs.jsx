import React from 'react';
import {
  ALIAS_ENDPOINT_EXAMPLE,
  API_CREATION_CHECKLIST,
  API_CREATION_STEPS,
  API_REFERENCE_SECTIONS,
  BASIC_ENDPOINT_EXAMPLE,
  RESEARCH_FRAMEWORK_V3_CAPABILITY_SUMMARY,
  RESEARCH_FRAMEWORK_V3_GAPS,
  RESEARCH_FRAMEWORK_V3_PHASES,
  RESEARCH_FRAMEWORK_V31_STABILIZATION,
  RESEARCH_FRAMEWORK_V31_STRESS_POINTS,
  SUBAGENT_GROUPS,
} from './lib/docsContent.js';
import './styles/knowledge-surfaces.css';

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
        <span>{endpoint.paths.length} path{endpoint.paths.length === 1 ? '' : 's'}</span>
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

function FrameworkPhaseCard({ phase }) {
  return (
    <article className="docs-card docs-card--framework" id={phase.id}>
      <div className="docs-card__eyebrow">
        <span>Phase {phase.phase}</span>
      </div>
      <h3 className="docs-card__title">{phase.title}</h3>
      <p className="docs-card__body">{phase.summary}</p>
      <ul className="docs-card__list">
        {phase.capabilities.map((item) => (
          <li key={item}>{item}</li>
        ))}
      </ul>
    </article>
  );
}

function ShortCard({ title, body, tone = 'default' }) {
  return (
    <article className={`docs-short-card docs-short-card--${tone}`}>
      <h3 className="docs-short-card__title">{title}</h3>
      <p className="docs-short-card__body">{body}</p>
    </article>
  );
}

export default function Docs() {
  const totalEndpoints = API_REFERENCE_SECTIONS.reduce((count, endpoint) => count + endpoint.paths.length, 0);
  const totalSubagents = SUBAGENT_GROUPS.reduce((count, group) => count + group.agents.length, 0);
  const frameworkPhases = RESEARCH_FRAMEWORK_V3_PHASES.filter((phase) => phase.phase !== 'X');
  const crossCuttingLayers = RESEARCH_FRAMEWORK_V3_PHASES.filter((phase) => phase.phase === 'X');

  return (
    <div className="docs-page docs-page--refined">
      <div className="docs-page__hero">
        <div className="docs-page__hero-copy">
          <div className="docs-page__eyebrow">NubAgent Docs</div>
          <h1 className="docs-page__title">Runtime, API, and ownership reference</h1>
          <p className="docs-page__lead">
            Operator-facing docs for the API surface, research runtime, and the owners behind each part of the system.
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
        <a href="#research-framework">Research Framework v3.0</a>
        <a href="#subagent-catalog">Subagents</a>
      </nav>

      <section className="docs-section" id="api-reference">
        <header className="docs-section__header">
          <div>
            <div className="docs-section__eyebrow">Public API</div>
            <h2 className="docs-section__title">API reference</h2>
          </div>
          <p className="docs-section__body">
            Use these cards to see which public paths exist, what they do, and which files own them.
          </p>
        </header>

        <div className="docs-section__intro-grid">
          <div className="docs-section__intro-card">
            <span>What this covers</span>
            <strong>Dedicated endpoints, shared aliases, and the current public contract.</strong>
          </div>
          <div className="docs-section__intro-card">
            <span>Use it for</span>
            <strong>Fast lookup of owning files, request shape, and response shape.</strong>
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
            <h2 className="docs-section__title">API creation playbook</h2>
          </div>
          <p className="docs-section__body">
            Use this when you need to add or reshape a public endpoint. Pick the right boundary first, then keep aliases, tests, and docs aligned.
          </p>
        </header>

        <div className="docs-section__intro-grid">
          <div className="docs-section__intro-card">
            <span>Primary rule</span>
            <strong>Choose the right boundary first.</strong>
          </div>
          <div className="docs-section__intro-card">
            <span>Outcome</span>
            <strong>New API work lands in a predictable place and stays easy to maintain.</strong>
          </div>
        </div>

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
              <code>lib/tools/&lt;tool&gt;.js</code>
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

      <section className="docs-section" id="research-framework">
        <header className="docs-section__header">
          <div>
            <div className="docs-section__eyebrow">Research Runtime</div>
            <h2 className="docs-section__title">Research Framework v3.0 — The Living Research Intelligence</h2>
          </div>
          <p className="docs-section__body">
            The research runtime is organized as a compiled graph with verification, synthesis, and delivery stages that can be steered as work progresses.
          </p>
        </header>

        <div className="docs-section__intro-grid">
          <div className="docs-section__intro-card">
            <span>Core shift</span>
            <strong>From a fixed pipeline to a steerable research runtime.</strong>
          </div>
          <div className="docs-section__intro-card">
            <span>Runtime shape</span>
            <strong>Compiled graph execution, recursive verification, and explicit checkpoints.</strong>
          </div>
        </div>

        <div className="docs-framework-grid">
          {frameworkPhases.map((phase) => (
            <FrameworkPhaseCard key={phase.id} phase={phase} />
          ))}
        </div>

        {crossCuttingLayers.length > 0 && (
          <div className="docs-framework-crosscutting">
            {crossCuttingLayers.map((layer) => (
              <FrameworkPhaseCard key={layer.id} phase={layer} />
            ))}
          </div>
        )}

        <article className="docs-card">
          <h3 className="docs-card__title">What v2.0 still gets wrong</h3>
          <div className="docs-table-wrap">
            <table className="docs-table">
              <thead>
                <tr>
                  <th>Gap</th>
                  <th>Consequence</th>
                </tr>
              </thead>
              <tbody>
                {RESEARCH_FRAMEWORK_V3_GAPS.map((row) => (
                  <tr key={row.gap}>
                    <td>{row.gap}</td>
                    <td>{row.consequence}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </article>

        <article className="docs-card">
          <h3 className="docs-card__title">v3.0 capability summary</h3>
          <div className="docs-table-wrap">
            <table className="docs-table">
              <thead>
                <tr>
                  <th>Dimension</th>
                  <th>v2.0</th>
                  <th>v3.0</th>
                </tr>
              </thead>
              <tbody>
                {RESEARCH_FRAMEWORK_V3_CAPABILITY_SUMMARY.map((row) => (
                  <tr key={row.dimension}>
                    <td>{row.dimension}</td>
                    <td>{row.v2}</td>
                    <td>{row.v3}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </article>

        <div className="docs-creation-grid">
          <article className="docs-card">
            <h3 className="docs-card__title">Critical stress points in v3.0</h3>
            <div className="docs-short-grid">
              {RESEARCH_FRAMEWORK_V31_STRESS_POINTS.map((item) => (
                <ShortCard key={item.id} title={item.title} body={item.impact} tone="warning" />
              ))}
            </div>
          </article>

          <article className="docs-card">
            <h3 className="docs-card__title">v3.1 stabilization layer</h3>
            <div className="docs-short-grid">
              {RESEARCH_FRAMEWORK_V31_STABILIZATION.map((item) => (
                <ShortCard key={item.id} title={item.title} body={item.summary} tone="success" />
              ))}
            </div>
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
            This is the web view of the local `nub_*` roster so you can see who owns orchestration, search, fetch, verification, docs, and release.
          </p>
        </header>

        <div className="docs-section__intro-grid">
          <div className="docs-section__intro-card">
            <span>Catalog purpose</span>
            <strong>Every meaningful surface has a named owner.</strong>
          </div>
          <div className="docs-section__intro-card">
            <span>Reading mode</span>
            <strong>Use these clusters as routing hints when work spans runtime, retrieval, verification, or docs.</strong>
          </div>
        </div>

        <div className="docs-groups">
          {SUBAGENT_GROUPS.map((group) => (
            <SubagentGroup key={group.id} group={group} />
          ))}
        </div>
      </section>
    </div>
  );
}
