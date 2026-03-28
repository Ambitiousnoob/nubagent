module.exports = {
    definition: {
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
    },
    handler: async (args) => {
        const expr = String(args.expression || "");
        const safeExpr = expr.replace(/[^0-9+*/().\s-]/g, "");
        try {
            // eslint-disable-next-line no-eval
            const result = eval(safeExpr);
            if (!Number.isFinite(result)) return "Error: calculation produced non-finite result";
            return String(result);
        } catch (e) {
            return `Error: invalid expression (${e.message})`;
        }
    },
};
