import React from 'react';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import SearchEngine from '../SearchEngine.jsx';
import Library from '../Library.jsx';
import { useSettingsStore } from '../store/useSettingsStore.js';

const LIBRARY_STORAGE_KEY = 'nubagent-library';

const createHeaders = (values = {}) => ({
  get(name) {
    return values[String(name || '').toLowerCase()] ?? null;
  },
});

const createSseResponse = (chunks = []) => {
  const encoder = new TextEncoder();
  let index = 0;

  return {
    ok: true,
    headers: createHeaders({ 'content-type': 'text/event-stream; charset=utf-8' }),
    body: {
      getReader() {
        return {
          read: async () => {
            if (index >= chunks.length) {
              return { done: true, value: undefined };
            }
            const value = encoder.encode(chunks[index]);
            index += 1;
            return { done: false, value };
          },
        };
      },
    },
  };
};

describe('SearchEngine library persistence', () => {
  const storage = new Map();

  beforeEach(() => {
    storage.clear();
    useSettingsStore.getState().clearAllSettings();

    window.localStorage.getItem.mockImplementation((key) => (
      storage.has(key) ? storage.get(key) : null
    ));
    window.localStorage.setItem.mockImplementation((key, value) => {
      storage.set(key, String(value));
    });
    window.localStorage.removeItem.mockImplementation((key) => {
      storage.delete(key);
    });
    window.localStorage.clear.mockImplementation(() => {
      storage.clear();
    });

    fetch.mockReset();
    window.history.replaceState({}, '', '/');
  });

  afterEach(() => {
    window.history.replaceState({}, '', '/');
  });

  it('creates a saved library session as soon as a search is submitted', async () => {
    fetch.mockImplementation(() => new Promise(() => {}));

    render(<SearchEngine />);

    const input = screen.getByLabelText('Message NubAgent');
    fireEvent.change(input, { target: { value: 'best laptop battery life' } });
    fireEvent.submit(input.closest('form'));

    await waitFor(() => {
      const raw = storage.get(LIBRARY_STORAGE_KEY);
      expect(raw).toBeTruthy();
      const sessions = JSON.parse(raw);
      expect(sessions).toHaveLength(1);
      expect(sessions[0].query).toBe('best laptop battery life');
    });
  });

  it('aborts the in-flight search when the search view unmounts', async () => {
    let requestSignal;
    fetch.mockImplementation((_, options = {}) => {
      requestSignal = options.signal;
      return new Promise(() => {});
    });

    const view = render(<SearchEngine />);

    const input = screen.getByLabelText('Message NubAgent');
    fireEvent.change(input, { target: { value: 'abort on unmount' } });
    fireEvent.submit(input.closest('form'));

    await waitFor(() => {
      expect(requestSignal).toBeDefined();
      expect(requestSignal.aborted).toBe(false);
    });

    view.unmount();

    expect(requestSignal.aborted).toBe(true);
  });

  it('keeps the saved library session when the search fails', async () => {
    fetch.mockRejectedValue(new Error('Search backend offline'));

    render(<SearchEngine />);

    const input = screen.getByLabelText('Message NubAgent');
    fireEvent.change(input, { target: { value: 'search failure case' } });
    fireEvent.submit(input.closest('form'));

    await waitFor(() => {
      const raw = storage.get(LIBRARY_STORAGE_KEY);
      expect(raw).toBeTruthy();
      const sessions = JSON.parse(raw);
      expect(sessions[0].query).toBe('search failure case');
      expect(sessions[0].heading).toBe('Error');
      expect(String(sessions[0].body || '')).not.toBe('');
    });
  });

  it('preserves streamed sources when the final runtime payload has none', async () => {
    fetch.mockResolvedValue(createSseResponse([
      `event: inventory\ndata: ${JSON.stringify({
        type: 'inventory',
        counts: { core: 1, supporting: 0, peripheral: 0 },
        sources: [
          {
            citationIndex: 1,
            title: 'Caching Paper',
            url: 'https://example.com/paper',
            tier: 'core',
          },
        ],
      })}\n\n`,
      `event: final\ndata: ${JSON.stringify({
        type: 'final',
        runId: 'research-run-1',
        final: {
          heading: 'Decision Draft',
          body: 'Caching improves latency.',
          markdown: '# Decision Draft\n\nCaching improves latency.',
          sources: [],
        },
        researchMeta: {
          outputMode: { label: 'State of the Field' },
        },
      })}\n\n`,
      'data: [DONE]\n\n',
    ]));

    render(<SearchEngine />);

    const input = screen.getByLabelText('Message NubAgent');
    fireEvent.change(input, { target: { value: 'preserve sources after final reveal' } });
    fireEvent.submit(input.closest('form'));

    await waitFor(() => {
      const raw = storage.get(LIBRARY_STORAGE_KEY);
      expect(raw).toBeTruthy();
      const sessions = JSON.parse(raw);
      expect(sessions[0].sources).toHaveLength(1);
      expect(sessions[0].sources[0]).toMatchObject({
        title: 'Caching Paper',
        url: 'https://example.com/paper',
      });
    });
  });

  it('forwards configured search provider keys with research runs', async () => {
    useSettingsStore.getState().setApiKey('tvly-preview-key-1234567890', 'tavily');
    fetch.mockResolvedValue(createSseResponse([
      `event: final\ndata: ${JSON.stringify({
        type: 'final',
        runId: 'research-run-2',
        final: {
          heading: 'Research Answer',
          body: 'Quantum computing threatens RSA.',
          markdown: '# Research Answer\n\nQuantum computing threatens RSA.',
          sources: [],
        },
        researchMeta: {},
      })}\n\n`,
      'data: [DONE]\n\n',
    ]));

    render(<SearchEngine />);

    const input = screen.getByLabelText('Message NubAgent');
    fireEvent.change(input, { target: { value: 'quantum computing and encryption' } });
    fireEvent.submit(input.closest('form'));

    await waitFor(() => {
      expect(fetch).toHaveBeenCalled();
    });

    const [, request] = fetch.mock.calls[0];
    const payload = JSON.parse(request.body);
    expect(payload.searchProviderKeys).toMatchObject({
      tavily: 'tvly-preview-key-1234567890',
    });
  });

  it('keeps a true no-sources final report empty instead of inventing source cards', async () => {
    fetch.mockResolvedValue(createSseResponse([
      `event: final\ndata: ${JSON.stringify({
        type: 'final',
        runId: 'research-run-3',
        final: {
          heading: 'No Sources Retrieved',
          body: 'I could not retrieve grounded sources for this run.',
          markdown: '# No Sources Retrieved\n\nI could not retrieve grounded sources for this run.',
          sources: [],
          sourceSelection: {
            mode: 'no_grounded_sources',
            totalTieredSources: 0,
          },
        },
        researchMeta: {
          outputMode: { label: 'State of the Field' },
          providerErrors: ['DuckDuckGo returned bot challenge'],
        },
      })}\n\n`,
      'data: [DONE]\n\n',
    ]));

    render(<SearchEngine />);

    const input = screen.getByLabelText('Message NubAgent');
    fireEvent.change(input, { target: { value: 'no sources final state' } });
    fireEvent.submit(input.closest('form'));

    expect(await screen.findByText(/No grounded sources were retrieved for this run/i)).toBeInTheDocument();
    expect(screen.queryByText('Caching Paper')).not.toBeInTheDocument();

    const raw = storage.get(LIBRARY_STORAGE_KEY);
    const sessions = JSON.parse(raw);
    expect(sessions[0].sources).toEqual([]);
  });

  it('shows the saved session inside the Library view after submit', async () => {
    fetch.mockImplementation(() => new Promise(() => {}));

    const searchView = render(<SearchEngine />);

    const input = screen.getByLabelText('Message NubAgent');
    fireEvent.change(input, { target: { value: 'library visibility check' } });
    fireEvent.submit(input.closest('form'));

    await waitFor(() => {
      const raw = storage.get(LIBRARY_STORAGE_KEY);
      expect(raw).toBeTruthy();
    });

    searchView.unmount();

    render(
      <Library
        onBack={() => {}}
        onViewSession={() => {}}
        onNewSearch={() => {}}
      />
    );

    expect(await screen.findByText('library visibility check')).toBeInTheDocument();
  });

  it('persists edits made from the Library session editor', async () => {
    const now = Date.now();
    storage.set(LIBRARY_STORAGE_KEY, JSON.stringify([
      {
        id: 'saved-1',
        query: 'original query',
        heading: 'Original heading',
        body: 'Original body',
        sources: [],
        attachments: [],
        createdAt: now,
        updatedAt: now,
      },
    ]));

    render(
      <Library
        onBack={() => {}}
        onViewSession={() => {}}
        onNewSearch={() => {}}
      />
    );

    expect(await screen.findByText('original query')).toBeInTheDocument();

    fireEvent.click(screen.getByTitle('Edit'));
    fireEvent.click(await screen.findByRole('button', { name: /edit session/i }));

    const queryInput = await screen.findByPlaceholderText('Search query');
    fireEvent.change(queryInput, { target: { value: 'updated query' } });
    fireEvent.change(screen.getByPlaceholderText('Answer heading'), { target: { value: 'Updated heading' } });
    fireEvent.change(screen.getByPlaceholderText('Saved answer text'), { target: { value: 'Updated body' } });
    fireEvent.click(screen.getByRole('button', { name: /save changes/i }));

    await waitFor(() => {
      const sessions = JSON.parse(storage.get(LIBRARY_STORAGE_KEY));
      expect(sessions[0]).toMatchObject({
        id: 'saved-1',
        query: 'updated query',
        heading: 'Updated heading',
        body: 'Updated body',
      });
    });

    expect(await screen.findByText('updated query')).toBeInTheDocument();
  });

  it('clears the shared session query param when the shell requests a fresh chat', () => {
    window.history.replaceState({}, '', '/?session=session-123');

    const { rerender } = render(<SearchEngine resetSignal={0} />);
    rerender(<SearchEngine resetSignal={1} />);

    expect(window.location.search).toBe('');
  });
});
