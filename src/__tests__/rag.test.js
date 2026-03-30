import { describe, it, expect } from 'vitest';
import { Company } from '../lib/company.js';
import {
  canonicalizeSourceUrl,
  mergeSourcesByCanonicalUrl,
  rankEvidenceEntriesForQuery,
  rerankSourcesForQuery,
  selectSourcesForFetch,
} from '../lib/rag.js';

describe('rag utilities', () => {
  it('canonicalizes URLs by stripping tracking params and normalizing slashes', () => {
    expect(
      canonicalizeSourceUrl('https://Example.com/path///to/page/?utm_source=test&ref=abc&x=1#section')
    ).toBe('https://example.com/path/to/page?x=1');
  });

  it('exposes canonical source merging on the company facade', () => {
    expect(Company.rag.mergeSourcesByCanonicalUrl).toBe(mergeSourcesByCanonicalUrl);
  });

  it('merges duplicate sources while preserving provider and query provenance', () => {
    const merged = mergeSourcesByCanonicalUrl([
      {
        title: 'Official docs',
        url: 'https://docs.example.com/api?utm_source=search',
        description: 'Primary documentation.',
        source: 'duckduckgo',
        queryVariant: 'api docs',
      },
      {
        title: 'Official API docs',
        url: 'https://docs.example.com/api',
        description: 'Expanded API reference documentation.',
        source: 'brave',
        queryVariant: 'api reference',
      },
    ]);

    expect(merged).toHaveLength(1);
    expect(merged[0]?.providers).toEqual(['duckduckgo', 'brave']);
    expect(merged[0]?.providerCount).toBe(2);
    expect(merged[0]?.queryVariants).toEqual(['api docs', 'api reference']);
    expect(merged[0]?.queryHitCount).toBe(2);
  });

  it('prefers authoritative relevant sources over generic blogs', () => {
    const ranked = rerankSourcesForQuery('CRISPR base editing study', [
      {
        title: 'Base editing results from a personal lab blog',
        url: 'https://example-blog.com/posts/base-editing',
        description: 'Notes and opinions about editing tools.',
      },
      {
        title: 'Programmable base editing in genomic DNA',
        url: 'https://www.nature.com/articles/example',
        description: 'Peer-reviewed study describing programmable base editing outcomes.',
      },
    ]);

    expect(ranked[0]?.url).toContain('nature.com');
  });

  it('selects diverse fetch candidates and removes canonical duplicates', () => {
    const selected = selectSourcesForFetch('latest ai regulation 2026', [
      {
        title: 'Official update',
        url: 'https://www.fda.gov/ai/regulation?utm_source=newsletter',
        description: 'FDA update on regulation.',
        date: '2026-03-01',
      },
      {
        title: 'Official update duplicate',
        url: 'https://fda.gov/ai/regulation',
        description: 'Same source through a different URL variant.',
        date: '2026-03-01',
      },
      {
        title: 'Reuters coverage',
        url: 'https://www.reuters.com/world/us/ai-regulation-update',
        description: 'Reporting on the latest AI regulation changes.',
        date: '2026-03-02',
      },
      {
        title: 'Analysis from another news site',
        url: 'https://www.theguardian.com/technology/2026/mar/03/ai-regulation',
        description: 'Independent coverage of the regulation change.',
        date: '2026-03-03',
      },
      {
        title: 'Second FDA page',
        url: 'https://www.fda.gov/ai/faq',
        description: 'Follow-up FAQ from FDA.',
        date: '2026-03-02',
      },
    ], {
      limit: 3,
      perDomainLimit: 1,
    });

    expect(selected).toHaveLength(3);
    expect(new Set(selected.map((item) => item.url)).size).toBe(3);
    expect(new Set(selected.map((item) => new URL(item.url).hostname.replace(/^www\./, ''))).size).toBe(3);
  });

  it('promotes evidence with stronger content matches while limiting same-domain clustering', () => {
    const ranked = rankEvidenceEntriesForQuery('FDA AI regulation update', [
      {
        source: {
          title: 'Blog recap',
          url: 'https://example-blog.com/ai-regulation-recap',
          description: 'A recap of AI regulation news.',
        },
        content: 'This page mostly summarizes general AI news without much detail about the FDA announcement.',
      },
      {
        source: {
          title: 'FDA announcement',
          url: 'https://www.fda.gov/medical-devices/ai-update',
          description: 'Official FDA update on AI regulation.',
        },
        content: 'The FDA announced an AI regulation update covering safety review, submission expectations, and deployment oversight.',
      },
      {
        source: {
          title: 'Second FDA explainer',
          url: 'https://www.fda.gov/medical-devices/ai-faq',
          description: 'Follow-up FAQ on the same policy.',
        },
        content: 'This FAQ covers follow-up questions about the FDA AI regulation update and filing process.',
      },
      {
        source: {
          title: 'Reuters coverage',
          url: 'https://www.reuters.com/world/us/fda-ai-regulation-update',
          description: 'Reuters reporting on the FDA update.',
        },
        content: 'Reuters reports the FDA AI regulation update with timing, scope, and industry reaction.',
      },
    ], {
      maxPerDomain: 1,
    });

    expect(ranked[0]?.source?.url).toContain('fda.gov');
    expect(new URL(ranked[0]?.source?.url).hostname.replace(/^www\./, '')).not.toBe(
      new URL(ranked[1]?.source?.url).hostname.replace(/^www\./, '')
    );
  });
});
