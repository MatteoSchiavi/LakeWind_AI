import type { Config } from "tailwindcss";

const config: Config = {
  // Phase 4 (W4): class-based dark mode so the in-app toggle (with
  // system-preference default) controls the theme; media-only dark variants
  // cannot be toggled manually.
  darkMode: "class",
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        // Design tokens (W5) — resolved from CSS variables in globals.css,
        // light + dark sets. The speed palette itself lives in src/lib/palette.ts
        // (shared with the Python map renderer and the bot).
        background: "var(--background)",
        foreground: "var(--foreground)",
        card: {
          DEFAULT: "var(--card)",
          foreground: "var(--card-foreground)",
        },
        muted: {
          DEFAULT: "var(--muted)",
          foreground: "var(--muted-foreground)",
        },
        border: "var(--border)",
        primary: {
          DEFAULT: "var(--primary)",
          foreground: "var(--primary-foreground)",
        },
        ring: "var(--ring)",
      },
      borderRadius: {
        xl: "0.75rem",
      },
    },
  },
  plugins: [],
};
export default config;
