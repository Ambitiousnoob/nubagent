import React from 'react';
import { ArrowUpDown, CalendarRange, Filter, Paperclip, X } from 'lucide-react';
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
  const dateRangeLabel = {
    all: 'Any time',
    today: 'Past 24 hours',
    week: 'Past week',
    month: 'Past month',
    year: 'Past year',
  }[filters.dateRange || 'all'];
  const sortByLabel = {
    date: 'Date',
    title: 'Title',
    sources: 'Sources',
  }[filters.sortBy || 'date'];
  const sortOrderLabel = (filters.sortOrder || 'desc') === 'asc' ? 'Oldest first' : 'Newest first';

  const hasActiveFilters = (
    (filters.dateRange && filters.dateRange !== 'all')
    || Boolean(filters.hasAttachments)
    || (filters.sortBy && filters.sortBy !== 'date')
    || (filters.sortOrder && filters.sortOrder !== 'desc')
  );

  const handleFilterChange = (key, value) => {
    onFilterChange?.({ [key]: value });
  };

  const handleReset = () => {
    onReset?.();
  };

  const clearFilter = (key) => {
    if (key === 'hasAttachments') {
      handleFilterChange(key, false);
      return;
    }
    if (key === 'sortOrder') {
      handleFilterChange(key, 'desc');
      return;
    }
    if (key === 'sortBy') {
      handleFilterChange(key, 'date');
      return;
    }
    handleFilterChange(key, 'all');
  };

  const activeFilters = [
    filters.dateRange && filters.dateRange !== 'all'
      ? { key: 'dateRange', label: `Date: ${dateRangeLabel}` }
      : null,
    filters.hasAttachments
      ? { key: 'hasAttachments', label: 'With attachments' }
      : null,
    filters.sortBy && filters.sortBy !== 'date'
      ? { key: 'sortBy', label: `Sort: ${sortByLabel}` }
      : null,
    filters.sortOrder && filters.sortOrder !== 'desc'
      ? { key: 'sortOrder', label: sortOrderLabel }
      : null,
  ].filter(Boolean);

  return (
    <div className="search-filters">
      <div className="search-filters__summary">
        <div className="search-filters__summary-copy">
          <span className="search-filters__summary-eyebrow">Archive filters</span>
          <p className="search-filters__summary-body">
            Refine this view by time, attachments, and ordering.
          </p>
        </div>
        {activeFilters.length > 0 && (
          <div className="search-filters__active-pills" aria-label="Active filters">
            {activeFilters.map((item) => (
              <button
                key={item.key}
                type="button"
                className="search-filters__pill"
                onClick={() => clearFilter(item.key)}
              >
                <span>{item.label}</span>
                <X size={12} />
              </button>
            ))}
          </div>
        )}
      </div>

      <div className="search-filters__bar">
        <button
          type="button"
          className={`search-filters__toggle ${isOpen ? 'search-filters__toggle--open' : ''}`}
          onClick={() => setIsOpen(!isOpen)}
          aria-expanded={isOpen}
        >
          <div className="search-filters__toggle-copy">
            <span className="search-filters__toggle-label">
              <Filter size={16} />
              Filter archive
            </span>
            <span className="search-filters__toggle-value">
              {hasActiveFilters ? `${activeFilters.length} active · ${sortByLabel} · ${sortOrderLabel}` : 'Open controls'}
            </span>
          </div>
          {hasActiveFilters && <span className="search-filters__badge">{activeFilters.length}</span>}
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
            <label className="search-filters__label">
              <CalendarRange size={14} />
              Date Range
            </label>
            <select
              className="search-filters__select"
              value={filters.dateRange || 'all'}
              onChange={(e) => handleFilterChange('dateRange', e.target.value)}
            >
              <option value="all">Any time</option>
              <option value="today">Past 24 hours</option>
              <option value="week">Past week</option>
              <option value="month">Past month</option>
              <option value="year">Past year</option>
            </select>
          </div>

          <div className="search-filters__group">
            <label className="search-filters__label">
              <Paperclip size={14} />
              Attachments
            </label>
            <select
              className="search-filters__select"
              value={filters.hasAttachments ? 'with-attachments' : 'all'}
              onChange={(e) => handleFilterChange('hasAttachments', e.target.value === 'with-attachments')}
            >
              <option value="all">All sessions</option>
              <option value="with-attachments">With attachments</option>
            </select>
          </div>

          <div className="search-filters__group">
            <label className="search-filters__label">
              <ArrowUpDown size={14} />
              Sort By
            </label>
            <select
              className="search-filters__select"
              value={filters.sortBy || 'date'}
              onChange={(e) => handleFilterChange('sortBy', e.target.value)}
            >
              <option value="date">Date</option>
              <option value="title">Title</option>
              <option value="sources">Sources</option>
            </select>
          </div>

          <div className="search-filters__group">
            <label className="search-filters__label">
              <ArrowUpDown size={14} />
              Order
            </label>
            <select
              className="search-filters__select"
              value={filters.sortOrder || 'desc'}
              onChange={(e) => handleFilterChange('sortOrder', e.target.value)}
            >
              <option value="desc">Newest first</option>
              <option value="asc">Oldest first</option>
            </select>
          </div>
        </div>
      )}
    </div>
  );
}

export default SearchFilters;
