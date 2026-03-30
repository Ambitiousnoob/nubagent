/**
 * Test Setup
 * Configures testing environment
 */

import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";

// Cleanup after each test
afterEach(() => {
  cleanup();
  window.localStorage.clear();
  global.fetch.mockReset();
});

// Mock window.matchMedia
Object.defineProperty(window, "matchMedia", {
  writable: true,
  value: vi.fn().mockImplementation((query) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: vi.fn(),
    removeListener: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  })),
});

// Mock localStorage
const localStorageState = new Map();
const localStorageMock = {
  getItem: vi.fn((key) =>
    localStorageState.has(key) ? localStorageState.get(key) : null,
  ),
  setItem: vi.fn((key, value) => {
    localStorageState.set(String(key), String(value));
  }),
  removeItem: vi.fn((key) => {
    localStorageState.delete(String(key));
  }),
  clear: vi.fn(() => {
    localStorageState.clear();
  }),
  key: vi.fn((index) => Array.from(localStorageState.keys())[index] ?? null),
};
Object.defineProperty(localStorageMock, "length", {
  get() {
    return localStorageState.size;
  },
});
Object.defineProperty(window, "localStorage", {
  value: localStorageMock,
});

if (!window.HTMLElement.prototype.scrollIntoView) {
  window.HTMLElement.prototype.scrollIntoView = vi.fn();
}

// Mock fetch
global.fetch = vi.fn();
