import React from 'react';
import { ExternalLink, Calendar, Tag } from 'lucide-react';

/**
 * SearchResultCard Component
 * Individual search result card with rich display
 * 
 * @param {object} source - Source object with title, url, description, date
 * @param {number} index - Result index for citation
 * @param {function} onClick - Click handler
 */
export function SearchResultCard({ source, index, onClick }) {
  const displayIndex = Number.isInteger(Number(source?.citationIndex)) && Number(source?.citationIndex) > 0
    ? Number(source.citationIndex)
    : (index + 1);

  const getDomain = (url) => {
    try {
      return new URL(url).hostname.replace(/^www\./, '');
    } catch {
      return url;
    }
  };

  const getFavicon = (url) => {
    try {
      return `https://www.google.com/s2/favicons?domain=${new URL(url).origin}&sz=32`;
    } catch {
      return null;
    }
  };

  const getSourceCategory = (url) => {
    const domain = getDomain(url);
    if (!domain) return 'Web';
    if (/(\.gov|\.mil)\b/i.test(domain) || /(nist|nih|cisa|fda|who|un\.org|europa)/i.test(domain))
      return 'Government';
    if (/(\.edu)\b/i.test(domain) ||
      /(arxiv|nature|science|springer|ieee|acm|pubmed|doi\.org|nejm|jamanetwork|cell|mit\.edu)/i.test(domain))
      return 'Research';
    if (/(reuters|apnews|bbc|euronews|nytimes|washingtonpost|theguardian|wired|technologyreview|cyberscoop|techcrunch|theverge)/i.test(domain))
      return 'News';
    if (/(google|cloudflare|microsoft|openai|anthropic|aws|ibm|meta|github|docs\.)/i.test(domain))
      return 'Vendor';
    return 'Web';
  };

  const domain = getDomain(source?.url || '');
  const category = getSourceCategory(source?.url);
  const favicon = getFavicon(source?.url);

  const categoryColors = {
    Government: 'badge--official',
    Research: 'badge--research',
    News: 'badge--news',
    Vendor: 'badge--vendor',
    Web: 'badge--web',
  };

  return (
    <div
      className="search-result-card"
      onClick={() => onClick?.(source)}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => e.key === 'Enter' && onClick?.(source)}
    >
      <div className="search-result-card__header">
        <div className="search-result-card__source">
          {favicon && (
            <img
              src={favicon}
              alt=""
              className="search-result-card__favicon"
              onError={(e) => { e.currentTarget.style.display = 'none'; }}
            />
          )}
          <span className="search-result-card__domain">{domain}</span>
          <span className={`search-result-card__category ${categoryColors[category] || ''}`}>
            {category}
          </span>
        </div>
        <a
          href={source.url}
          target="_blank"
          rel="noopener noreferrer"
          className="search-result-card__external"
          onClick={(e) => e.stopPropagation()}
          aria-label={`Open ${domain} in new tab`}
        >
          <ExternalLink size={14} />
        </a>
      </div>

      <h3 className="search-result-card__title">
        <span className="search-result-card__index">{displayIndex}</span>
        {source.title || domain}
      </h3>

      {source.description && (
        <p className="search-result-card__description">
          {source.description.slice(0, 200)}
          {source.description.length > 200 ? '...' : ''}
        </p>
      )}

      <div className="search-result-card__footer">
        {source.date && (
          <span className="search-result-card__date">
            <Calendar size={12} />
            {new Date(source.date).toLocaleDateString()}
          </span>
        )}
        {source.tags?.length > 0 && (
          <div className="search-result-card__tags">
            {source.tags.slice(0, 3).map((tag, i) => (
              <span key={i} className="search-result-card__tag">
                <Tag size={10} />
                {tag}
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export default SearchResultCard;
