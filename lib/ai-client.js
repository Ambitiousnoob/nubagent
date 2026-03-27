const { GoogleGenAI } = require("@google/genai");
const dotenv = require("dotenv");
const path = require("node:path");

// Load env from process first, then fallback to local .env in repo root
dotenv.config();
dotenv.config({ path: path.join(__dirname, "..", ".env") });

// ==========================================
// 1. PROACTIVE: The Smart Rate Limiter (mutex + queue)
// ==========================================
class SmartRateLimiter {
    constructor(maxRequests, timeWindowMs, minSpacingMs = 0) {
        this.maxRequests = maxRequests;
        this.timeWindowMs = timeWindowMs;
        this.minSpacingMs = minSpacingMs;
        this.requestTimestamps = [];
        this.lastRequestTime = 0;
        this.queueLock = Promise.resolve(); // serialize concurrent callers
    }

    async waitForSlot() {
        const releaseLock = await new Promise((resolve) => {
            const previous = this.queueLock;
            this.queueLock = previous.then(() => resolve);
        });

        try {
            await this._evaluateSlot();
        } finally {
            releaseLock();
        }
    }

    async _evaluateSlot() {
        let now = Date.now();

        if (this.minSpacingMs > 0) {
            const timeSinceLast = now - this.lastRequestTime;
            if (timeSinceLast < this.minSpacingMs) {
                const burstWait = this.minSpacingMs - timeSinceLast;
                await new Promise((resolve) => setTimeout(resolve, burstWait));
                now = Date.now();
            }
        }

        this.requestTimestamps = this.requestTimestamps.filter(
            (timestamp) => (now - timestamp) < this.timeWindowMs
        );

        if (this.requestTimestamps.length >= this.maxRequests) {
            const oldestRequest = this.requestTimestamps[0];
            const windowWait = this.timeWindowMs - (now - oldestRequest);
            console.log(`[Queue] API capacity reached. Holding request for ${(windowWait / 1000).toFixed(1)}s...`);
            await new Promise((resolve) => setTimeout(resolve, windowWait));
            return this._evaluateSlot();
        }

        const executionTime = Date.now();
        this.requestTimestamps.push(executionTime);
        this.lastRequestTime = executionTime;
        return true;
    }
}

// ==========================================
// 2. REACTIVE: The Resilient API Client
// ==========================================
class ResilientAIClient {
    constructor() {
        const keyString = process.env.GEMINI_API_KEYS || process.env.GEMINI_API_KEY;
        this.apiKeys = (keyString || "")
            .split(",")
            .map((key) => key.trim())
            .filter(Boolean);
        if (!this.apiKeys.length) {
            console.error("[CRITICAL] GEMINI_API_KEYS or GEMINI_API_KEY missing from environment, falling back to bundled keys.");
            this.apiKeys = [
                "AIzaSyCMmk7zZ3a6h5CYdt7zdRc2SH4IX4Dg7x8",
                "AIzaSyBWnzWbk72GMPstEB9k5hcMQRoka6s69wk",
                "AIzaSyDbHfg4kYKRxmTVo9YLgr5uBckYdYqNVCY",
            ];
        }

        this.currentKeyIndex = 0;
        this.ai = this.apiKeys.length ? new GoogleGenAI({ apiKey: this.apiKeys[this.currentKeyIndex] }) : null;
        if (this.ai) console.log(`[System] AI Client ready. Pool: ${this.apiKeys.length} API keys.`);

        // 14 requests per minute with 2s spacing between calls
        this.rateLimiter = new SmartRateLimiter(14, 60000, 2000);
    }

    rotateKey() {
        if (!this.apiKeys.length) return;
        this.currentKeyIndex = (this.currentKeyIndex + 1) % this.apiKeys.length;
        const newKey = this.apiKeys[this.currentKeyIndex];
        this.ai = new GoogleGenAI({ apiKey: newKey });
        console.log(`[System] Rotated to API Key #${this.currentKeyIndex + 1}`);
    }

    async generateContent(params, maxRetries = 3) {
        if (!this.ai) {
            throw new Error("GEMINI_API_KEYS not configured on the server");
        }

        let attempt = 0;
        let baseDelay = 2000;

        while (attempt <= maxRetries) {
            try {
                await this.rateLimiter.waitForSlot();
                return await this.ai.models.generateContent(params);
            } catch (error) {
                const isRateLimit = error?.status === 429 ||
                    (error?.message && error.message.includes("429")) ||
                    (error?.message && error.message.includes("rate_limited"));

                if (isRateLimit) {
                    attempt += 1;
                    if (attempt > maxRetries) {
                        console.error("[Error] All keys rate limited. Retries exhausted.");
                        throw new Error("System is currently overloaded. Please try again later.");
                    }

                    console.log(`[Warning] Rate limit hit on Key #${this.currentKeyIndex + 1}.`);
                    this.rotateKey();
                    const waitTime = baseDelay * Math.pow(2, attempt - 1);
                    console.log(`[System] Backing off for ${waitTime / 1000}s before retry...`);
                    await new Promise((resolve) => setTimeout(resolve, waitTime));
                } else {
                    throw error;
                }
            }
        }
    }
}

const aiClient = new ResilientAIClient();
let singleton = aiClient;
const getAiClient = () => {
    if (!singleton) singleton = new ResilientAIClient();
    return singleton;
};

const resetAiClient = () => {
    singleton = null;
};

module.exports = { aiClient: getAiClient(), getAiClient, resetAiClient, ResilientAIClient, SmartRateLimiter };
