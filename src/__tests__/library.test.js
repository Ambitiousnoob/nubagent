import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

describe('library persistence', () => {
  beforeEach(() => {
    vi.resetModules();
    localStorage.clear();
  });

  it('preserves attachments when a saved session is updated', async () => {
    const { saveSession, getSavedSessions } = await import('../lib/library.js');

    saveSession({
      id: 'session-1',
      query: 'Initial query',
      heading: 'Draft',
      body: '',
      attachments: [
        {
          id: 'attachment-1',
          name: 'notes.txt',
          kind: 'text',
          size: 42,
          textContent: 'hello world',
          truncated: false,
        },
      ],
    });

    saveSession({
      id: 'session-1',
      query: 'Updated query',
      heading: 'Final heading',
      body: 'Final body',
      attachments: [
        {
          id: 'attachment-1',
          name: 'notes.txt',
          kind: 'text',
          size: 42,
          textContent: 'hello world',
          truncated: false,
        },
      ],
      sources: [{ title: 'Example', url: 'https://example.com' }],
    });

    const [saved] = getSavedSessions();
    expect(saved.query).toBe('Updated query');
    expect(saved.heading).toBe('Final heading');
    expect(saved.attachments).toHaveLength(1);
    expect(saved.attachments[0].name).toBe('notes.txt');
    expect(saved.sources).toHaveLength(1);
  });
});

describe('library store filtering', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-03-30T12:00:00.000Z'));
    localStorage.clear();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('keeps newest sessions first and filters attached sessions by date range', async () => {
    vi.resetModules();
    const { saveSession } = await import('../lib/library.js');
    const { useLibraryStore } = await import('../store/useLibraryStore.js');

    saveSession({
      id: 'older',
      query: 'Older session',
      heading: 'Older',
      body: 'Older body',
      createdAt: new Date('2026-01-10T00:00:00.000Z').getTime(),
      updatedAt: new Date('2026-01-10T00:00:00.000Z').getTime(),
    });

    saveSession({
      id: 'newer',
      query: 'Newer session',
      heading: 'Newer',
      body: 'Newer body',
      createdAt: new Date('2026-03-28T00:00:00.000Z').getTime(),
      updatedAt: new Date('2026-03-28T00:00:00.000Z').getTime(),
      attachments: [
        {
          id: 'attachment-2',
          name: 'brief.md',
          kind: 'text',
          size: 20,
          textContent: 'brief',
          truncated: false,
        },
      ],
    });

    const loadPromise = useLibraryStore.getState().loadSessions();
    await vi.advanceTimersByTimeAsync(200);
    await loadPromise;

    expect(useLibraryStore.getState().getFilteredSessions().map((session) => session.id)).toEqual(['newer', 'older']);

    useLibraryStore.getState().setFilters({ dateRange: 'month' });
    expect(useLibraryStore.getState().getFilteredSessions().map((session) => session.id)).toEqual(['newer']);

    useLibraryStore.getState().setFilters({ dateRange: 'all', hasAttachments: true });
    expect(useLibraryStore.getState().getFilteredSessions().map((session) => session.id)).toEqual(['newer']);
  });
});

describe('library utilities', () => {
  it('builds share URLs with a session query parameter', async () => {
    const { buildSessionShareUrl } = await import('../lib/library.js');

    const url = buildSessionShareUrl('session-123', {
      origin: 'https://nubagent.vercel.app',
      pathname: '/app',
    });

    expect(url).toBe('https://nubagent.vercel.app/?session=session-123');
  });

  it('hydrates saved sessions into chat sessions with attachments and sources', async () => {
    const { buildChatSessionFromSaved } = await import('../lib/library.js');

    const hydrated = buildChatSessionFromSaved({
      id: 'session-1',
      query: 'Analyze the roadmap',
      heading: 'Roadmap summary',
      body: 'Key milestones are listed below.',
      attachments: [{ id: 'file-1', name: 'roadmap.pdf', kind: 'text' }],
      sources: [{ url: 'https://example.com/doc', title: 'Example doc' }],
      researchMeta: { searchCount: 3 },
    });

    expect(hydrated.id).toBe('session-1');
    expect(hydrated.query).toBe('Analyze the roadmap');
    expect(hydrated.messages[0]).toMatchObject({
      role: 'user',
      text: 'Analyze the roadmap',
      attachments: [{ id: 'file-1', name: 'roadmap.pdf', kind: 'text' }],
    });
    expect(hydrated.messages[1]).toMatchObject({
      role: 'bot',
      heading: 'Roadmap summary',
      body: 'Key milestones are listed below.',
      sources: [{ url: 'https://example.com/doc', title: 'Example doc' }],
      searchDone: true,
      researchMeta: { searchCount: 3 },
    });
  });
});
