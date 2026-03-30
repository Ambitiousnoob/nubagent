import { describe, expect, it, vi } from "vitest";
import { loadCommonJsModule } from "../../src/__tests__/loadCommonJsModule.js";

const buildMemoryModule = () => {
  const query = vi.fn(async (sql) => {
    if (/UPDATE research_controls/i.test(String(sql || ""))) {
      return [{ affectedRows: 2 }];
    }
    return [{}];
  });

  const module = loadCommonJsModule(
    "/root/.bot/.downloads/nubagent/lib/research-memory.js",
    {
      "./db": {
        getPool: () => ({ query }),
      },
    },
  );

  return {
    ...module,
    query,
  };
};

describe("research memory controls", () => {
  it("deduplicates control ids before applying a status update", async () => {
    const memory = buildMemoryModule();

    const updated = await memory.markResearchControlsApplied([7, 7, 9, 0, -1]);

    expect(updated).toBe(2);
    expect(memory.query).toHaveBeenLastCalledWith(
      expect.stringContaining("UPDATE research_controls"),
      ["applied", 7, 9],
    );
  });
});
