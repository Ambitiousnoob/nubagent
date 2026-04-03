import test from "node:test";
import assert from "node:assert/strict";

import { normalizePostgresConnectionString } from "../lib/history.js";

test("normalizePostgresConnectionString upgrades alias sslmodes to verify-full", () => {
  const result = normalizePostgresConnectionString(
    "postgresql://user:pass@db.example.com:5432/app?sslmode=require",
  );

  assert.equal(
    result,
    "postgresql://user:pass@db.example.com:5432/app?sslmode=verify-full",
  );
});

test("normalizePostgresConnectionString preserves libpq compatibility opt-in", () => {
  const result = normalizePostgresConnectionString(
    "postgresql://user:pass@db.example.com:5432/app?uselibpqcompat=true&sslmode=require",
  );

  assert.equal(
    result,
    "postgresql://user:pass@db.example.com:5432/app?uselibpqcompat=true&sslmode=require",
  );
});

test("normalizePostgresConnectionString leaves non-alias sslmodes unchanged", () => {
  const result = normalizePostgresConnectionString(
    "postgresql://user:pass@db.example.com:5432/app?sslmode=verify-full",
  );

  assert.equal(
    result,
    "postgresql://user:pass@db.example.com:5432/app?sslmode=verify-full",
  );
});
