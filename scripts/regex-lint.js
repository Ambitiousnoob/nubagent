#!/usr/bin/env node
const fs = require("fs");
const path = require("path");
const acorn = require("acorn");
const jsx = require("acorn-jsx");
const walk = require("acorn-walk");
const Parser = acorn.Parser.extend(jsx());
const base = { ...walk.base };

// Minimal JSX walkers to allow traversal without errors
base.JSXText = () => {};
base.JSXIdentifier = () => {};
base.JSXNamespacedName = () => {};
base.JSXMemberExpression = () => {};
base.JSXEmptyExpression = () => {};
base.JSXExpressionContainer = () => {};
base.JSXSpreadChild = () => {};
base.JSXAttribute = () => {};
base.JSXSpreadAttribute = () => {};
base.JSXOpeningElement = () => {};
base.JSXClosingElement = () => {};
base.JSXFragment = () => {};
base.JSXElement = () => {};

const roots = ["api", "src", "lib"];
const exts = new Set([".js", ".jsx", ".mjs", ".cjs"]);

const files = [];
function collect(dir) {
    if (!fs.existsSync(dir)) return;
    const entries = fs.readdirSync(dir, { withFileTypes: true });
    for (const entry of entries) {
        const full = path.join(dir, entry.name);
        if (entry.isDirectory()) {
            collect(full);
        } else if (entry.isFile() && exts.has(path.extname(entry.name))) {
            files.push(full);
        }
    }
}
roots.forEach(collect);

const failures = [];

for (const file of files) {
    let src;
    try {
        src = fs.readFileSync(file, "utf8");
    } catch (e) {
        failures.push({ file, message: `read error: ${e.message}` });
        continue;
    }

    let ast;
    try {
    ast = Parser.parse(src, {
        ecmaVersion: "latest",
        sourceType: "module",
        allowHashBang: true,
        locations: true,
        });
    } catch (e) {
        failures.push({ file, message: `parse error: ${e.message}` });
        continue;
    }

    const testRegex = (pattern, flags, loc) => {
        try {
            // eslint-disable-next-line no-new
            new RegExp(pattern, flags);
        } catch (e) {
            failures.push({ file, message: `invalid regex /${pattern}/${flags || ""} at ${loc}` });
        }
    };

    walk.simple(ast, {
        Literal(node) {
            if (node.regex) {
                testRegex(node.regex.pattern, node.regex.flags, formatLoc(node.loc));
            }
        },
        NewExpression(node) {
            if (node.callee.type === "Identifier" && node.callee.name === "RegExp") {
                const arg = node.arguments[0];
                const flagsArg = node.arguments[1];
                if (arg && arg.type === "Literal" && typeof arg.value === "string") {
                    const flags = flagsArg && flagsArg.type === "Literal" ? String(flagsArg.value || "") : "";
                    testRegex(arg.value, flags, formatLoc(node.loc));
                }
            }
        },
        CallExpression(node) {
            if (node.callee.type === "Identifier" && node.callee.name === "RegExp") {
                const arg = node.arguments[0];
                const flagsArg = node.arguments[1];
                if (arg && arg.type === "Literal" && typeof arg.value === "string") {
                    const flags = flagsArg && flagsArg.type === "Literal" ? String(flagsArg.value || "") : "";
                    testRegex(arg.value, flags, formatLoc(node.loc));
                }
            }
        },
    }, base);
}

if (failures.length) {
    console.error(`\nInvalid regexes found (${failures.length}):`);
    failures.forEach((f) => console.error(`- ${f.file}: ${f.message}`));
    process.exitCode = 1;
} else {
    console.log("All regex literals validated.");
}

function formatLoc(loc) {
    if (!loc) return "unknown";
    return `${loc.start.line}:${loc.start.column + 1}`;
}
