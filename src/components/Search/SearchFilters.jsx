import React from 'react';
import { Filter, X } from 'lucide-react';
import { Button } from '../UI/Button.jsx';

/**
 * SearchFilters Component
 * Filter controls for search results
 * 
 * @param {object} filters - Current filter values
 * @param {function} onFilterChange - Filter change handler
 * @param {function} onReset - Reset filters handler
 */
export function SearchFilters({
  filters = {},
  onFilterChange,
  onReset,
}) {
  const [isOpen, setIsOpen] = React.useState(false);

  const hasActiveFilters = Object.values(filters).some(
    (value) => value && value !== 'all' && value !== false
  );

  const handleFilterChange = (key, value) => {
    onFilterChange?.({ [key]: value });
  };

  const handleReset = () => {
    onReset?.();
  };

  return (
    <div className="search-filters">
      <div className="search-filters__bar">
        <button
          className={`search-filters__toggle ${isOpen ? 'search-filters__toggle--open' : ''}`}
          onClick={() => setIsOpen(!isOpen)}
        >
          <Filter size={16} />
          <span>Filters</span>
          {hasActiveFilters && <span className="search-filters__badge" />}
        </button>

        {hasActiveFilters && (
          <Button
            variant="ghost"
            size="sm"
            onClick={handleReset}
            className="search-filters__reset"
          >
            <X size={14} />
            Clear all
          </Button>
        )}
      </div>

      {isOpen && (
        <div className="search-filters__panel">
          <div className="search-filters__group">
            <label className="search-filters__label">Date Range</label>
            <select
              className="search-filters__select"
              value={filters.dateRange || 'all'}
              onChange={(e) => handleFilterChange('dateRange', e.target.value)}
            >
              <option value="all">Any time</option>
              <option value="day">Past 24 hours</option>
              <option value="week">Past week</option>
              <option value="month">Past month</option>
              <option value="year">Past year</option>
            </select>
          </div>

          <div className="search-filters__group">
            <label className="search-filters__label">Source Type</label>
            <select
              className="search-filters__select"
              value={filters.sourceType || 'all'}
              onChange={(e) => handleFilterChange('sourceType', e.target.value)}
            >
              <option value="all">All sources</option>
              <option value="government">Government</option>
              <option value="research">Research</option>
              <option value="news">News</option>
              <option value="vendor">Vendor</option>
            </select>
          </div>

          <div className="search-filters__group">
            <label className="search-filters__label">Sort By</label>
            <select
              className="search-filters__select"
              value={filters.sortBy || 'relevance'}
              onChange={(e) => handleFilterChange('sortBy', e.target.value)}
            >
              <option value="relevance">Relevance</option>
              <option value="date">Date</option>
              <option value="domain">Domain</option>
            </select>
          </div>
        </div>
      )}
    </div>
  );
}

export default SearchFilters;
