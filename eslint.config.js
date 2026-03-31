import globals from "globals";

const sharedLanguageOptions = {
  ecmaVersion: "latest",
  sourceType: "module",
  parserOptions: {
    ecmaFeatures: {
      jsx: true,
    },
  },
};

const sharedRules = {
  "no-undef": "error",
  "no-unreachable": "error",
};

export default [
  {
    ignores: ["dist/**", "node_modules/**"],
  },
  {
    files: ["src/**/*.{js,jsx}"],
    languageOptions: {
      ...sharedLanguageOptions,
      globals: {
        ...globals.browser,
        ...globals.es2024,
      },
    },
    rules: sharedRules,
  },
  {
    files: ["api/**/*.js", "lib/**/*.js"],
    languageOptions: {
      ...sharedLanguageOptions,
      globals: {
        ...globals.node,
        ...globals.es2024,
      },
    },
    rules: sharedRules,
  },
];
