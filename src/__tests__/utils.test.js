/**
 * Utility Function Tests
 */

import { describe, it, expect } from 'vitest';

describe('Utility Functions', () => {
  describe('formatBytes', () => {
    const formatBytes = (bytes) => {
      if (!bytes || bytes === 0) return '0 B';
      const k = 1024;
      const sizes = ['B', 'KB', 'MB', 'GB'];
      const i = Math.floor(Math.log(bytes) / Math.log(k));
      return `${parseFloat((bytes / Math.pow(k, i)).toFixed(2))} ${sizes[i]}`;
    };

    it('formats bytes correctly', () => {
      expect(formatBytes(0)).toBe('0 B');
      expect(formatBytes(512)).toBe('512 B');
      expect(formatBytes(1024)).toBe('1 KB');
      expect(formatBytes(1536)).toBe('1.5 KB');
      expect(formatBytes(1048576)).toBe('1 MB');
      expect(formatBytes(1073741824)).toBe('1 GB');
    });

    it('handles null and undefined', () => {
      expect(formatBytes(null)).toBe('0 B');
      expect(formatBytes(undefined)).toBe('0 B');
    });
  });

  describe('getDomain', () => {
    const getDomain = (url) => {
      try {
        return new URL(url).hostname.replace(/^www\./, '');
      } catch {
        return url;
      }
    };

    it('extracts domain from URL', () => {
      expect(getDomain('https://example.com')).toBe('example.com');
      expect(getDomain('https://www.example.com')).toBe('example.com');
      expect(getDomain('https://sub.example.com/path')).toBe('sub.example.com');
    });

    it('returns original string for invalid URLs', () => {
      expect(getDomain('not-a-url')).toBe('not-a-url');
      expect(getDomain('')).toBe('');
    });
  });

  describe('getFavicon', () => {
    const getFavicon = (url) => {
      try {
        return `https://www.google.com/s2/favicons?domain=${new URL(url).origin}&sz=32`;
      } catch {
        return null;
      }
    };

    it('generates favicon URL', () => {
      expect(getFavicon('https://example.com')).toBe(
        'https://www.google.com/s2/favicons?domain=https://example.com&sz=32'
      );
    });

    it('returns null for invalid URLs', () => {
      expect(getFavicon('not-a-url')).toBeNull();
    });
  });

  describe('createId', () => {
    const createId = () => `${Date.now()}-${Math.random().toString(16).slice(2)}`;

    it('generates unique IDs', () => {
      const ids = new Set();
      for (let i = 0; i < 100; i++) {
        ids.add(createId());
      }
      expect(ids.size).toBe(100);
    });

    it('includes timestamp', () => {
      const id = createId();
      const timestamp = id.split('-')[0];
      expect(Number(timestamp)).toBeGreaterThan(0);
    });
  });

  describe('truncateText', () => {
    const truncateText = (text, maxLength = 100) => {
      if (text.length <= maxLength) return text;
      return text.slice(0, maxLength) + '...';
    };

    it('truncates long text', () => {
      const longText = 'a'.repeat(200);
      expect(truncateText(longText, 100)).toHaveLength(103); // 100 + '...'
      expect(truncateText(longText, 100)).toMatch(/\.\.\.$/);
    });

    it('does not truncate short text', () => {
      const shortText = 'Hello';
      expect(truncateText(shortText, 100)).toBe('Hello');
    });

    it('handles empty string', () => {
      expect(truncateText('', 100)).toBe('');
    });
  });
});
