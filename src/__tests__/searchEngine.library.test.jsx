import React from 'react';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import SearchEngine from '../SearchEngine.jsx';
import Library from '../Library.jsx';

const LIBRARY_STORAGE_KEY = 'nubagent-library';

describe('SearchEngine library persistence', () => {
  const storage = new Map();

  beforeEach(() => {
    storage.clear();

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

    const input = screen.getByPlaceholderText(/ask a question or use site:/i);
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

    const input = screen.getByPlaceholderText(/ask a question or use site:/i);
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

    const input = screen.getByPlaceholderText(/ask a question or use site:/i);
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

  it('shows the saved session inside the Library view after submit', async () => {
    fetch.mockImplementation(() => new Promise(() => {}));

    const searchView = render(<SearchEngine />);

    const input = screen.getByPlaceholderText(/ask a question or use site:/i);
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

  it('clears the shared session query param when starting a fresh chat from the search view', () => {
    window.history.replaceState({}, '', '/?session=session-123');

    render(<SearchEngine />);

    fireEvent.click(screen.getByTitle('New search'));

    expect(window.location.search).toBe('');
  });
});
