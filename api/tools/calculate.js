/**
 * Calculate Tool
 * Evaluates basic arithmetic expressions
 */

/**
 * Tool definition for function calling
 */
const definition = {
    type: "function",
    function: {
        name: "calculate",
        strict: true,
        description: "Evaluate a basic arithmetic expression. Supports + - * / and parentheses. Use for math.",
        parameters: {
            type: "object",
            properties: {
                expression: {
                    type: "string",
                    description: "Arithmetic expression, e.g. 15*7+(2/3)",
                },
            },
            required: ["expression"],
            additionalProperties: false,
        },
    },
};

/**
 * Calculate handler function
 * @param {object} args - Calculation arguments
 * @param {string} args.expression - Arithmetic expression to evaluate
 * @returns {Promise<string>} Calculation result or error message
 */
const handler = async (args) => {
    const expr = String(args.expression || "");
    const safeExpr = expr.replace(/[^0-9+*/().\s-]/g, "");

    // Validate balanced parentheses
    const openParens = (safeExpr.match(/\(/g) || []).length;
    const closeParens = (safeExpr.match(/\)/g) || []).length;
    if (openParens !== closeParens) {
        return "Error: unbalanced parentheses in expression";
    }

    // Prevent consecutive operators (except for negative numbers)
    if (/[*+/]{2,}/.test(safeExpr) || /[)(][0-9]/.test(safeExpr)) {
        return "Error: invalid operator sequence in expression";
    }

    try {
        // Use Function constructor with strict validation instead of eval
        // This is safer because we've already sanitized the input
        const result = Function(`"use strict"; return (${safeExpr})`)();
        if (!Number.isFinite(result)) return "Error: calculation produced non-finite result";
        return String(result);
    } catch (e) {
        return `Error: invalid expression (${e.message})`;
    }
};

module.exports = {
    definition,
    handler,
};
