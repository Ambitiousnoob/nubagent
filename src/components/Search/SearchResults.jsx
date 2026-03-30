import React from 'react';
import { SearchResultCard } from './SearchResultCard.jsx';
import { ListSkeleton } from '../UI/Skeleton.jsx';

/**
 * SearchResults Component
 * Rich search results display with grid layout
 * 
 * @param {array} sources - Array of source objects
 * @param {function} onSourceClick - Source click handler
 * @param {boolean} isLoading - Loading state
 * @param {string} query - Original search query
 */
export function SearchResults({
  sources = [],
  onSourceClick,
  isLoading = false,
  query = '',
}) {
  if (isLoading) {
    return (
      <div className="search-results search-results--loading">
        <ListSkeleton count={5} />
      </div>
    );
  }

  if (sources.length === 0) {
    return null;
  }

  return (
    <div className="search-results">
      <div className="search-results__header">
        <h3 className="search-results__title">Sources</h3>
        <span className="search-results__count">{sources.length} result{sources.length !== 1 ? 's' : ''}</span>
      </div>
      <div className="search-results__grid">
        {sources.map((source, index) => (
          <SearchResultCard
            key={source.url || index}
            source={source}
            index={index}
            onClick={onSourceClick}
          />
        ))}
      </div>
    </div>
  );
}

/**
 * SearchResultsList Component
 * Compact list view of search results
 */
export function SearchResultsList({ sources = [], onSourceClick, isLoading = false }) {
  if (isLoading) {
    return (
      <div className="search-results-list search-results-list--loading">
        <ListSkeleton count={5} />
      </div>
    );
  }

  if (sources.length === 0) {
    return null;
  }

  return (
    <div className="search-results-list">
      {sources.map((source, index) => (
        <div
          key={source.url || index}
          className="search-results-list__item"
          onClick={() => onSourceClick?.(source)}
        >
          <span className="search-results-list__index">{index + 1}</span>
          <div className="search-results-list__content">
            <h4 className="search-results-list__title">{source.title}</h4>
            <p className="search-results-list__url">{source.url}</p>
          </div>
        </div>
      ))}
    </div>
  );
}

export default SearchResults;
