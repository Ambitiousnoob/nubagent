const SUPPORTED_IMAGE_MIME_TYPES = new Set([
  "image/png",
  "image/jpeg",
  "image/webp",
  "image/heic",
  "image/heif",
]);

// Keep raw image bytes below Gemini's practical inline-request ceiling after base64 expansion.
const MAX_INLINE_IMAGE_BYTES = 14 * 1024 * 1024;
const IMAGE_FETCH_TIMEOUT_MS = 12000;

export const IMAGE_READY_REPLY =
  "I checked the image. Now send your question or instruction about it.";

function normalizeMimeType(value) {
  return typeof value === "string"
    ? value.split(";")[0].trim().toLowerCase()
    : "";
}

export function extractImageAttachments(event) {
  return (event?.message?.attachments ?? []).filter(
    (attachment) =>
      attachment?.type === "image" &&
      typeof attachment?.payload?.url === "string" &&
      attachment.payload.url.trim(),
  );
}

export function buildVisionPrompt(basePrompt, imageCount) {
  const normalizedPrompt =
    typeof basePrompt === "string" ? basePrompt.trim() : "";

  if (normalizedPrompt) {
    return normalizedPrompt;
  }

  if (imageCount <= 0) {
    return "";
  }

  return imageCount > 1
    ? "Analyze the attached images and briefly describe the key details."
    : "Analyze the attached image and briefly describe the key details.";
}

export function buildImageContextPrompt(imageCount) {
  return imageCount > 1
    ? "Analyze the attached images and write a concise factual visual summary for follow-up questions. Focus only on what is visible. Do not answer an implied question and do not add extra commentary."
    : "Analyze the attached image and write a concise factual visual summary for follow-up questions. Focus only on what is visible. Do not answer an implied question and do not add extra commentary.";
}

export function buildStoredInboundText(prompt, imageCount) {
  const normalizedPrompt = typeof prompt === "string" ? prompt.trim() : "";

  if (imageCount <= 0) {
    return normalizedPrompt;
  }

  const imageNote =
    imageCount === 1
      ? "[User attached 1 image]"
      : `[User attached ${imageCount} images]`;

  return normalizedPrompt ? `${normalizedPrompt}\n\n${imageNote}` : imageNote;
}

export function buildPromptWithImageContext(prompt, imageContext) {
  const normalizedPrompt = typeof prompt === "string" ? prompt.trim() : "";
  const normalizedContext =
    typeof imageContext === "string" ? imageContext.trim() : "";

  if (!normalizedContext) {
    return normalizedPrompt;
  }

  return `Use the following image context only if the user's message appears to refer to the previously sent image.

Image context:
${normalizedContext}

User message:
${normalizedPrompt}`;
}

function isSupportedImageMimeType(mimeType) {
  return SUPPORTED_IMAGE_MIME_TYPES.has(normalizeMimeType(mimeType));
}

function getFetchOptions() {
  if (typeof AbortSignal?.timeout === "function") {
    return {
      signal: AbortSignal.timeout(IMAGE_FETCH_TIMEOUT_MS),
    };
  }

  return undefined;
}

export async function loadInlineImageParts(attachments) {
  const selectedAttachments = Array.isArray(attachments) ? attachments : [];
  const parts = [];
  let totalBytes = 0;

  for (const attachment of selectedAttachments) {
    let response;

    try {
      response = await fetch(attachment.payload.url, getFetchOptions());
    } catch {
      continue;
    }

    if (!response.ok) {
      continue;
    }

    const mimeType = normalizeMimeType(
      response.headers.get("content-type") || "image/jpeg",
    );

    if (!isSupportedImageMimeType(mimeType)) {
      continue;
    }

    let bytes;

    try {
      bytes = Buffer.from(await response.arrayBuffer());
    } catch {
      continue;
    }

    if (bytes.length === 0) {
      continue;
    }

    if (totalBytes + bytes.length > MAX_INLINE_IMAGE_BYTES) {
      continue;
    }

    totalBytes += bytes.length;
    parts.push({
      inline_data: {
        mime_type: mimeType,
        data: bytes.toString("base64"),
      },
    });
  }

  return parts;
}
