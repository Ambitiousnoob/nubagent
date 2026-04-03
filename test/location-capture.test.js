import test from "node:test";
import assert from "node:assert/strict";
import { Readable } from "node:stream";

import { createLocationCaptureHandler } from "../api/location-capture.js";

function createRequest({ method = "GET", url, body } = {}) {
  const payload =
    body === undefined ? [] : [Buffer.from(JSON.stringify(body), "utf8")];
  const req = Readable.from(payload);
  req.method = method;
  req.url = url;
  req.headers = {
    host: "example.test",
  };
  return req;
}

function createResponse() {
  return {
    statusCode: 0,
    headers: {},
    body: "",
    setHeader(name, value) {
      this.headers[name] = value;
    },
    end(payload = "") {
      this.body += payload;
    },
  };
}

test("location capture GET renders the browser geolocation page for ready tokens", async () => {
  const handler = createLocationCaptureHandler({
    configLoader: () => ({}),
    conversationStoreFactory: async () => ({
      async getLocationCaptureSession(token) {
        assert.equal(token, "capture-token-1");
        return {
          status: "ready",
        };
      },
    }),
  });
  const res = createResponse();

  await handler(
    createRequest({
      url: "/api/location-capture?token=capture-token-1",
    }),
    res,
  );

  assert.equal(res.statusCode, 200);
  assert.equal(res.headers["Content-Type"], "text/html; charset=utf-8");
  assert.match(res.body, /Share Location/);
  assert.match(res.body, /Share my location/);
});

test("location capture POST saves browser coordinates", async () => {
  const saves = [];
  const handler = createLocationCaptureHandler({
    configLoader: () => ({}),
    conversationStoreFactory: async () => ({
      async saveLocationFromCapture(payload) {
        saves.push(payload);
        return {
          ok: true,
          latitude: payload.latitude,
          longitude: payload.longitude,
          updatedAt: "2026-04-03T00:00:00.000Z",
        };
      },
    }),
  });
  const res = createResponse();

  await handler(
    createRequest({
      method: "POST",
      url: "/api/location-capture",
      body: {
        token: "capture-token-2",
        latitude: 6.5244,
        longitude: 3.3792,
      },
    }),
    res,
  );

  assert.equal(res.statusCode, 200);
  assert.deepEqual(saves, [
    {
      token: "capture-token-2",
      latitude: 6.5244,
      longitude: 3.3792,
    },
  ]);
  assert.deepEqual(JSON.parse(res.body), {
    ok: true,
    latitude: 6.5244,
    longitude: 3.3792,
    updatedAt: "2026-04-03T00:00:00.000Z",
  });
});

test("location capture POST rejects expired tokens", async () => {
  const handler = createLocationCaptureHandler({
    configLoader: () => ({}),
    conversationStoreFactory: async () => ({
      async saveLocationFromCapture() {
        return {
          ok: false,
          status: "expired",
        };
      },
    }),
  });
  const res = createResponse();

  await handler(
    createRequest({
      method: "POST",
      url: "/api/location-capture",
      body: {
        token: "capture-token-3",
        latitude: 6.5244,
        longitude: 3.3792,
      },
    }),
    res,
  );

  assert.equal(res.statusCode, 410);
  assert.deepEqual(JSON.parse(res.body), {
    ok: false,
    error: "This location link has expired.",
  });
});
