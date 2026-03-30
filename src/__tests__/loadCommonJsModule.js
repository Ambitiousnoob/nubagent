import { existsSync, readFileSync } from "node:fs";
import { createRequire } from "node:module";
import path from "node:path";
import vm from "node:vm";

const nativeRequire = createRequire(import.meta.url);

const resolveCandidates = (baseDir, specifier) => {
  const resolved = path.resolve(baseDir, specifier);
  return [
    resolved,
    `${resolved}.js`,
    `${resolved}.json`,
    path.join(resolved, "index.js"),
  ];
};

export const loadCommonJsModule = (modulePath, mocks = {}, cache = new Map()) => {
  const resolvedPath = path.resolve(modulePath);
  if (cache.has(resolvedPath)) {
    return cache.get(resolvedPath).exports;
  }

  const source = readFileSync(resolvedPath, "utf8");
  const module = { exports: {} };
  cache.set(resolvedPath, module);

  const requireFromModule = (specifier) => {
    if (Object.prototype.hasOwnProperty.call(mocks, specifier)) {
      return mocks[specifier];
    }

    if (specifier.startsWith(".") || specifier.startsWith("/")) {
      const candidates = resolveCandidates(path.dirname(resolvedPath), specifier);
      for (const candidate of candidates) {
        if (Object.prototype.hasOwnProperty.call(mocks, candidate)) {
          return mocks[candidate];
        }
        if (existsSync(candidate)) {
          return loadCommonJsModule(candidate, mocks, cache);
        }
      }
    }

    return nativeRequire(specifier);
  };

  const wrapped = `(function (exports, require, module, __filename, __dirname) {\n${source}\n})`;
  const compiled = vm.runInThisContext(wrapped, { filename: resolvedPath });
  compiled(module.exports, requireFromModule, module, resolvedPath, path.dirname(resolvedPath));
  return module.exports;
};
