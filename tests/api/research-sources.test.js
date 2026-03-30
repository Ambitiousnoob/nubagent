import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { loadCommonJsModule } from "../../src/__tests__/loadCommonJsModule.js";

const createJsonResponse = (payload, status = 200) => ({
  ok: status >= 200 && status < 300,
  status,
  json: async () => payload,
});

describe("research source retrieval recovery", () => {
  beforeEach(() => {
    vi.resetModules();
    global.fetch.mockReset();
  });

  afterEach(() => {
    delete process.env.IEEE_XPLORE_API_KEY;
    delete process.env.ACM_SEARCH_API_URL;
    delete process.env.JSTOR_SEARCH_API_URL;
    delete process.env.UNPAYWALL_EMAIL;
    delete process.env.RETRACTION_WATCH_API_URL;
  });

  it("recovers with a reduced sequential provider pass when the initial lane burst returns nothing", async () => {
    let scholarlyCallCount = 0;

    global.fetch.mockImplementation(async (url) => {
      const value = String(url || "");

      if (value.startsWith("https://api.openalex.org/works")) {
        scholarlyCallCount += 1;
        if (scholarlyCallCount <= 6) {
          return createJsonResponse({ error: "rate limited" }, 429);
        }
        return createJsonResponse({
          results: [
            {
              id: "https://openalex.org/W1",
              doi: "https://doi.org/10.1234/quantum-risk",
              display_name: "Quantum computing threats to public-key encryption",
              primary_location: {
                landing_page_url: "https://doi.org/10.1234/quantum-risk",
                source: {
                  display_name: "Journal of Quantum Security",
                  host_organization_name: "Example Press",
                },
              },
              type: "article",
              publication_date: "2024-01-02",
              cited_by_count: 42,
              authorships: [
                {
                  author: { display_name: "Alice Example" },
                },
              ],
              abstract_inverted_index: {
                Quantum: [0],
                computing: [1],
                threatens: [2],
                RSA: [3],
                encryption: [4],
              },
              locations: [],
            },
          ],
        });
      }

      if (value.startsWith("https://api.crossref.org/works")) {
        scholarlyCallCount += 1;
        if (scholarlyCallCount <= 6) {
          return createJsonResponse({ error: "rate limited" }, 429);
        }
        return createJsonResponse({
          message: {
            items: [
              {
                DOI: "10.5555/post-quantum",
                title: ["Post-quantum migration planning"],
                URL: "https://doi.org/10.5555/post-quantum",
                "container-title": ["Security Review"],
                publisher: "Example Publisher",
                author: [
                  {
                    given: "Bob",
                    family: "Example",
                  },
                ],
                created: { "date-time": "2023-06-01T00:00:00Z" },
                "is-referenced-by-count": 7,
              },
            ],
          },
        });
      }

      throw new Error(`Unexpected fetch: ${value}`);
    });

    const { searchResearchSources } = loadCommonJsModule(
      "/root/.bot/.downloads/nubagent/lib/research-sources.js",
      {
        "./tools/web_search": {
          handler: vi.fn(async () => JSON.stringify([])),
        },
      },
    );

    const result = await searchResearchSources({
      query: "how does quantum computing threaten encryption",
      plan: {
        depthPreference: "balanced",
        domain: { id: "general_research" },
        searchQueries: [
          "quantum computing rsa encryption threats",
          "post quantum cryptography risks",
        ],
      },
      maxResults: 12,
    });

    expect(scholarlyCallCount).toBeGreaterThan(6);
    expect(result.sources.length).toBeGreaterThan(0);
    expect(result.sources.some((source) => /quantum/i.test(source.title))).toBe(true);
    expect(result.providersUsed).toContain("openalex");
    expect(result.providerErrors.length).toBeGreaterThan(0);
  });

  it("surfaces web-search provider failures instead of silently swallowing them", async () => {
    global.fetch.mockImplementation(async (url) => {
      const value = String(url || "");

      if (value.startsWith("https://api.openalex.org/works")) {
        return createJsonResponse({ results: [] });
      }

      if (value.startsWith("https://api.crossref.org/works")) {
        return createJsonResponse({ message: { items: [] } });
      }

      throw new Error(`Unexpected fetch: ${value}`);
    });

    const { searchResearchSources } = loadCommonJsModule(
      "/root/.bot/.downloads/nubagent/lib/research-sources.js",
      {
        "./tools/web_search": {
          handler: vi.fn(async () => "Error: search failed (DuckDuckGo returned bot challenge)"),
        },
      },
    );

    const result = await searchResearchSources({
      query: "how does quantum computing threaten encryption",
      plan: {
        depthPreference: "balanced",
        domain: { id: "general_research" },
        searchQueries: ["quantum computing encryption"],
      },
      maxResults: 10,
    });

    expect(result.sources).toEqual([]);
    expect(result.providerErrors.some((message) => /DuckDuckGo returned bot challenge/i.test(message))).toBe(true);
  });

  it("carries scholarly lanes into the source mesh and labels them distinctly", async () => {
    global.fetch.mockImplementation(async (url) => {
      const value = String(url || "");

      if (value.startsWith("https://api.openalex.org/works")) {
        return createJsonResponse({ results: [] });
      }

      if (value.startsWith("https://api.crossref.org/works")) {
        return createJsonResponse({ message: { items: [] } });
      }

      throw new Error(`Unexpected fetch: ${value}`);
    });

    const { searchResearchSources } = loadCommonJsModule(
      "/root/.bot/.downloads/nubagent/lib/research-sources.js",
      {
        "./tools/web_search": {
          handler: vi.fn(async () => JSON.stringify([])),
        },
      },
    );

    const result = await searchResearchSources({
      query: "how does quantum computing threaten encryption",
      plan: {
        depthPreference: "balanced",
        domain: { id: "general_research" },
        searchQueries: ["quantum computing encryption"],
        queryMatrix: {
          versions: [],
          scholarlyDiscoveryLanes: ["how does quantum computing threaten encryption site:scholar.google.com"],
          scholarlyProviderBias: ["openalex", "crossref"],
        },
        scholarlyHarvest: {
          active: true,
        },
      },
      maxResults: 10,
    });

    expect(result.searchLanes).toEqual(expect.arrayContaining([
      expect.objectContaining({
        laneType: "scholarly",
        query: expect.stringContaining("site:scholar.google.com"),
      }),
    ]));
    expect(result.auditLog.scholarlyRouting).toMatchObject({
      active: true,
      laneCount: 1,
      providerBias: ["openalex", "crossref"],
    });
  });

  it("boosts academic providers and domains when scholarly harvesting is active", async () => {
    const { scoreTieredSources } = loadCommonJsModule(
      "/root/.bot/.downloads/nubagent/lib/research-sources.js",
      {
        "./tools/web_search": {
          handler: vi.fn(async () => JSON.stringify([])),
        },
      },
    );

    const ranked = scoreTieredSources(
      "how does quantum computing threaten encryption",
      [
        {
          title: "Quantum cryptography overview",
          url: "https://news.example.com/quantum-crypto",
          provider: "web_search",
          description: "quantum computing threaten encryption overview",
        },
        {
          title: "Quantum computing threats to public-key encryption",
          url: "https://openalex.org/W1",
          provider: "openalex",
          description: "quantum computing threaten encryption overview",
        },
        {
          title: "Post-quantum cryptography course notes",
          url: "https://crypto.stanford.edu/post-quantum-notes",
          provider: "web_search",
          description: "quantum computing threaten encryption overview",
        },
      ],
      {
        domain: { id: "general_research", recencyHalfLifeYears: 4 },
        queryMatrix: {
          scholarlyDiscoveryLanes: ["how does quantum computing threaten encryption site:scholar.google.com"],
          scholarlyProviderBias: ["openalex", "crossref"],
        },
        scholarlyHarvest: {
          active: true,
        },
      },
    );

    const byUrl = new Map(ranked.map((item) => [item.url, item]));
    expect(byUrl.get("https://openalex.org/W1")?.scoreComponents.scholarlyBoost).toBeGreaterThan(0);
    expect(byUrl.get("https://crypto.stanford.edu/post-quantum-notes")?.scoreComponents.scholarlyBoost).toBeGreaterThan(0);
    expect(byUrl.get("https://openalex.org/W1")?.scoreComponents.scholarlyBoost).toBeGreaterThan(
      byUrl.get("https://news.example.com/quantum-crypto")?.scoreComponents.scholarlyBoost ?? 0,
    );
  });
});
