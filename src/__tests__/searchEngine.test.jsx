import React from 'react';
import { describe, it, expect, vi } from 'vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import SearchEngine from '../SearchEngine.jsx';

describe('SearchEngine shell handoff', () => {
  it('loads a saved library session passed in from the app shell', async () => {
    const onSessionLoaded = vi.fn();

    render(
      <SearchEngine
        session={{
          id: 'session-1',
          query: 'What is NubAgent?',
          heading: 'NubAgent',
          body: 'A research assistant.',
          sources: [],
        }}
        onSessionLoaded={onSessionLoaded}
      />,
    );

    expect((await screen.findAllByText('A research assistant.')).length).toBeGreaterThan(0);
    expect((await screen.findAllByText('What is NubAgent?')).length).toBeGreaterThan(0);
    await waitFor(() => expect(onSessionLoaded).toHaveBeenCalledTimes(1));
  });

  it('returns to the landing view when the shell requests a new chat', async () => {
    const { rerender } = render(
      <SearchEngine
        session={{
          id: 'session-1',
          query: 'What is NubAgent?',
          heading: 'NubAgent',
          body: 'A research assistant.',
          sources: [],
        }}
        resetSignal={0}
      />,
    );

    expect((await screen.findAllByText('A research assistant.')).length).toBeGreaterThan(0);

    rerender(<SearchEngine session={null} resetSignal={1} />);
    expect(await screen.findByText('Ask anything. Get a clear answer with sources.')).toBeInTheDocument();
  });

  it('creates a library record for a typed query even if the search request fails', async () => {
    global.fetch.mockResolvedValue({
      ok: false,
      status: 500,
      statusText: 'Internal Server Error',
      json: async () => ({ error: 'search backend unavailable' }),
      text: async () => 'search backend unavailable',
    });

    render(<SearchEngine />);

    const input = screen.getByPlaceholderText('Ask a question or use site:, filetype:, or intitle: operators...');
    fireEvent.change(input, { target: { value: 'Will this save to library?' } });
    fireEvent.submit(input.closest('form'));

    await waitFor(() => {
      const storedSessions = JSON.parse(window.localStorage.getItem('nubagent-library') || '[]');
      expect(storedSessions).toHaveLength(1);
      expect(storedSessions[0].query).toBe('Will this save to library?');
      expect(storedSessions[0].heading).toBe('Error');
      expect(storedSessions[0].body).toBeTruthy();
    });
  });

  it('renders compiled research metadata and verifier ownership for a saved research session', async () => {
    render(
      <SearchEngine
        session={{
          id: 'session-research-1',
          query: 'Should we adopt retrieval caching?',
          heading: 'Decision brief',
          body: 'Recommendation: adopt retrieval caching with staged rollout.',
          sources: [
            {
              title: 'Caching study',
              url: 'https://example.com/caching-study',
              description: 'Benchmarks for retrieval caching.',
              providerCount: 2,
              queryHitCount: 2,
            },
          ],
          researchMeta: {
            frameworkVersion: '3.1',
            attachments: 1,
            rankedSites: 8,
            fetchedSites: 4,
            fetchPlanned: 5,
            fetchAttempts: 6,
            synthesisWorkers: 3,
            dagSummary: 'Cognitive Command Layer -> Dialectical Synthesis Engine -> Decision Intelligence Layer -> Adaptive Delivery Hub',
            domain: { label: 'CS + ML' },
            scope: { label: 'Decision Briefing' },
            outputMode: { label: 'Decision Brief' },
            pareto: { mode: 'deep', explanation: 'favor completeness over latency' },
            continuity: { active: true, overlap: 0.67 },
            intentConfidence: { ambiguousAxes: ['scope'] },
            safety: { activeCount: 2 },
            tribunal: {
              refinement_cycles: 2,
              refinement_budget: 3,
              targeted_dimension: 'coverage',
              critics: {
                internal_consistency: 0.91,
                coverage: 0.88,
                user_goal_alignment: 0.94,
              },
            },
            convergence: {
              stability_score: 0.92,
              evidence_coverage_delta: 0.03,
              residual_uncertainty: 0.18,
              stop_condition: 'budget_exhausted',
            },
            subagents: [
              {
                id: 'claimVerifier',
                label: 'Claim verifier',
                scope: 'claim support verification',
                count: 1,
                detail: '4 verifier lanes active',
              },
              {
                id: 'citationVerifier',
                label: 'Citation verifier',
                scope: 'citation integrity',
                count: 1,
                detail: 'claim-to-citation integrity',
              },
              {
                id: 'decisionIntelligenceLayer',
                label: 'Decision intelligence layer',
                scope: 'decision payload generation and risk shaping',
                count: 1,
                detail: 'Decision Brief',
              },
            ],
          },
        }}
      />,
    );

    expect(await screen.findByText('Scope & Coverage')).toBeInTheDocument();
    expect(screen.getByText('Research Framework v3.1 with compiled DAG orchestration')).toBeInTheDocument();
    expect(screen.getByText(/Linked prior session context into Phase 1 with 67% overlap/i)).toBeInTheDocument();
    expect(screen.getByText(/2\/3 cycles; targeted coverage; critics: consistency 0.91, coverage 0.88, alignment 0.94/i)).toBeInTheDocument();
    expect(screen.getByText(/stability 0.92; coverage delta 0.03; residual uncertainty 0.18; stop condition budget exhausted/i)).toBeInTheDocument();
    expect(screen.getByText('Subagent Ownership')).toBeInTheDocument();
    expect(screen.getByText('Claim verifier')).toBeInTheDocument();
    expect(screen.getByText('Citation verifier')).toBeInTheDocument();
    expect(screen.getByText('Decision intelligence layer')).toBeInTheDocument();
  });
});
