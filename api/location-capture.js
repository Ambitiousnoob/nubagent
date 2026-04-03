import { getRuntimeConfig } from "../lib/config.js";
import { getConversationStore } from "../lib/history.js";
import { summarizeError } from "../lib/reliability.js";

function buildRequestUrl(req) {
  return new URL(req.url, `https://${req.headers.host || "localhost"}`);
}

function sendJson(res, statusCode, payload) {
  res.statusCode = statusCode;
  res.setHeader("Content-Type", "application/json; charset=utf-8");
  res.end(JSON.stringify(payload));
}

function sendHtml(res, statusCode, payload) {
  res.statusCode = statusCode;
  res.setHeader("Content-Type", "text/html; charset=utf-8");
  res.end(payload);
}

function normalizeToken(value) {
  return typeof value === "string" ? value.trim() : "";
}

function normalizeCoordinate(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

async function readJsonBody(req) {
  const chunks = [];

  for await (const chunk of req) {
    chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
  }

  if (chunks.length === 0) {
    return {};
  }

  return JSON.parse(Buffer.concat(chunks).toString("utf8"));
}

function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function renderCapturePage({ token, status, introOverride = "", followUpOverride = "" }) {
  const isReady = status === "ready";
  const pageTitle = isReady ? "Share Location" : "Link Unavailable";
  const intro =
    introOverride ||
    (status === "used"
      ? "This location link has already been used."
      : status === "expired"
        ? "This location link has expired."
        : status === "invalid"
          ? "This location link is invalid."
          : "Allow location access and NubAgent will save your current location automatically.");
  const followUp =
    followUpOverride ||
    (isReady
      ? "If the prompt does not appear, tap the button below."
      : "Go back to Messenger and send !location to get a fresh link.");

  return `<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>${escapeHtml(pageTitle)} | NubAgent</title>
    <style>
      :root {
        color-scheme: light;
        --bg: #f4efe4;
        --panel: rgba(255, 252, 245, 0.88);
        --ink: #1d2329;
        --muted: #5d6a75;
        --accent: #176f57;
        --accent-strong: #0f5744;
        --error: #9a3412;
        --success: #166534;
        --border: rgba(29, 35, 41, 0.12);
      }

      * {
        box-sizing: border-box;
      }

      body {
        margin: 0;
        min-height: 100vh;
        display: grid;
        place-items: center;
        padding: 24px;
        background:
          radial-gradient(circle at top left, rgba(23, 111, 87, 0.18), transparent 38%),
          radial-gradient(circle at bottom right, rgba(217, 119, 6, 0.12), transparent 30%),
          linear-gradient(180deg, #fbf6ea 0%, #efe4d0 100%);
        color: var(--ink);
        font-family:
          "Newsreader",
          Georgia,
          serif;
      }

      main {
        width: min(100%, 480px);
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 24px;
        padding: 28px 24px;
        box-shadow: 0 24px 70px rgba(29, 35, 41, 0.12);
        backdrop-filter: blur(12px);
      }

      h1 {
        margin: 0 0 12px;
        font-size: clamp(2.2rem, 7vw, 3.4rem);
        font-weight: 500;
        line-height: 0.94;
      }

      p {
        margin: 0;
        line-height: 1.55;
        color: var(--muted);
        font-size: 1rem;
      }

      .stack {
        display: grid;
        gap: 16px;
      }

      .status-card {
        padding: 16px;
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.7);
        border: 1px solid rgba(29, 35, 41, 0.08);
      }

      .status {
        margin: 0;
        color: var(--ink);
        font-size: 1rem;
      }

      .status[data-kind="error"] {
        color: var(--error);
      }

      .status[data-kind="success"] {
        color: var(--success);
      }

      .details {
        margin-top: 8px;
        color: var(--muted);
        font-size: 0.95rem;
        white-space: pre-line;
      }

      button {
        appearance: none;
        border: 0;
        border-radius: 999px;
        padding: 14px 18px;
        font: inherit;
        font-weight: 600;
        color: #f7fbf9;
        background: linear-gradient(135deg, var(--accent) 0%, var(--accent-strong) 100%);
        cursor: pointer;
        transition:
          transform 140ms ease,
          opacity 140ms ease;
      }

      button:hover {
        transform: translateY(-1px);
      }

      button:disabled {
        opacity: 0.72;
        cursor: wait;
        transform: none;
      }
    </style>
  </head>
  <body>
    <main class="stack">
      <div class="stack">
        <p>NubAgent</p>
        <h1>${escapeHtml(pageTitle)}</h1>
        <p>${escapeHtml(intro)}</p>
        <p>${escapeHtml(followUp)}</p>
      </div>

      <section class="status-card">
        <p id="status" class="status">Waiting for location access.</p>
        <p id="details" class="details"></p>
      </section>

      <button id="share-button"${isReady ? "" : " hidden"}>Share my location</button>
    </main>

    <script>
      const token = ${JSON.stringify(token)};
      const canCapture = ${JSON.stringify(isReady)};
      const statusElement = document.getElementById("status");
      const detailsElement = document.getElementById("details");
      const shareButton = document.getElementById("share-button");
      let inFlight = false;

      function setStatus(text, kind) {
        statusElement.textContent = text;
        if (kind) {
          statusElement.dataset.kind = kind;
        } else {
          delete statusElement.dataset.kind;
        }
      }

      function setDetails(text) {
        detailsElement.textContent = text || "";
      }

      function describeGeoError(error) {
        if (!error) {
          return "Location access failed. Try again.";
        }

        if (error.code === 1) {
          return "Location permission was denied. Allow it in your browser and try again.";
        }

        if (error.code === 2) {
          return "Your browser could not determine your location right now.";
        }

        if (error.code === 3) {
          return "Location access timed out. Try again in a spot with better signal.";
        }

        return error.message || "Location access failed. Try again.";
      }

      async function saveLocation(latitude, longitude) {
        const response = await fetch(window.location.pathname, {
          method: "POST",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify({
            token,
            latitude,
            longitude
          })
        });
        const payload = await response.json().catch(() => null);

        if (!response.ok || !payload?.ok) {
          throw new Error(payload?.error || "Could not save your location.");
        }

        return payload;
      }

      function captureLocation() {
        if (!canCapture || inFlight) {
          return;
        }

        if (!navigator.geolocation) {
          setStatus("This browser does not support geolocation.", "error");
          setDetails("Open the link in a browser with location access and try again.");
          return;
        }

        inFlight = true;
        shareButton.disabled = true;
        setStatus("Requesting your location...", "");
        setDetails("");

        navigator.geolocation.getCurrentPosition(
          async (position) => {
            try {
              setStatus("Saving your location...", "");
              const payload = await saveLocation(
                position.coords.latitude,
                position.coords.longitude
              );
              setStatus("Location saved. You can return to Messenger now.", "success");
              setDetails(
                "Latitude: " +
                  payload.latitude +
                  "\\nLongitude: " +
                  payload.longitude
              );
              shareButton.textContent = "Saved";
            } catch (error) {
              inFlight = false;
              shareButton.disabled = false;
              shareButton.textContent = "Try again";
              setStatus(error.message || "Could not save your location.", "error");
              setDetails("If the link expired, go back to Messenger and send !location again.");
            }
          },
          (error) => {
            inFlight = false;
            shareButton.disabled = false;
            shareButton.textContent = "Try again";
            setStatus(describeGeoError(error), "error");
            setDetails("If Messenger blocks location access, open the link in your system browser and retry.");
          },
          {
            enableHighAccuracy: true,
            timeout: 15000,
            maximumAge: 0
          }
        );
      }

      if (!canCapture) {
        setStatus(${JSON.stringify(intro)}, "error");
        setDetails(${JSON.stringify(followUp)});
      } else {
        captureLocation();
      }

      shareButton?.addEventListener("click", captureLocation);
    </script>
  </body>
</html>`;
}

export function createLocationCaptureHandler({
  configLoader = getRuntimeConfig,
  conversationStoreFactory = getConversationStore,
} = {}) {
  return async function handler(req, res) {
    if (req.method !== "GET" && req.method !== "POST") {
      sendJson(res, 405, { error: "Method not allowed." });
      return;
    }

    const url = buildRequestUrl(req);
    const queryToken = normalizeToken(url.searchParams.get("token"));

    try {
      const config = configLoader();
      const store = await conversationStoreFactory(config);

      if (req.method === "GET") {
        const session = queryToken
          ? await store.getLocationCaptureSession(queryToken)
          : null;
        const status = session?.status || (queryToken ? "invalid" : "missing");

        sendHtml(
          res,
          status === "ready" ? 200 : status === "missing" ? 400 : 410,
          renderCapturePage({
            token: queryToken,
            status: status === "missing" ? "invalid" : status,
          }),
        );
        return;
      }

      const body = await readJsonBody(req);
      const token = normalizeToken(body?.token || queryToken);
      const latitude = normalizeCoordinate(body?.latitude);
      const longitude = normalizeCoordinate(body?.longitude);

      if (!token || latitude === null || longitude === null) {
        sendJson(res, 400, {
          ok: false,
          error: "token, latitude, and longitude are required.",
        });
        return;
      }

      const result = await store.saveLocationFromCapture({
        token,
        latitude,
        longitude,
      });

      if (!result.ok) {
        sendJson(res, result.status === "invalid" ? 404 : 410, {
          ok: false,
          error:
            result.status === "used"
              ? "This location link has already been used."
              : result.status === "expired"
                ? "This location link has expired."
                : "This location link is invalid.",
        });
        return;
      }

      sendJson(res, 200, {
        ok: true,
        latitude: result.latitude,
        longitude: result.longitude,
        updatedAt: result.updatedAt,
      });
    } catch (error) {
      const message = summarizeError(error);

      if (req.method === "GET") {
        sendHtml(
          res,
          500,
          renderCapturePage({
            token: queryToken,
            status: "invalid",
            introOverride: `Could not load the location page: ${message}`,
          }),
        );
        return;
      }

      sendJson(res, 500, {
        ok: false,
        error: message,
      });
    }
  };
}

export default createLocationCaptureHandler();
