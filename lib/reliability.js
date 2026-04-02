function sleep(ms) {
  return new Promise((resolve) => {
    const timer = setTimeout(resolve, ms);

    if (typeof timer?.unref === "function") {
      timer.unref();
    }
  });
}

export function summarizeError(error) {
  if (error instanceof Error) {
    return error.message;
  }

  return String(error);
}

export async function retryAsync(
  task,
  {
    retries = 0,
    baseDelayMs = 250,
    shouldRetry = () => false,
    onRetry,
  } = {},
) {
  let attempt = 0;

  while (true) {
    try {
      return {
        value: await task(attempt),
        retryCount: attempt,
      };
    } catch (error) {
      if (attempt >= retries || !shouldRetry(error)) {
        error.retryCount = attempt;
        throw error;
      }

      attempt += 1;
      onRetry?.(error, attempt);
      await sleep(baseDelayMs * attempt);
    }
  }
}
