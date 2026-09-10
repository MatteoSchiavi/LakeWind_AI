# LakeWind AI — multi-stage image (Phase 5 S1).
#
# Phase 4 retired Streamlit; the old image still exposed :8501 and could not
# serve the real UI. This build:
#   stage 1: compiles the Next.js web-ui (output: "standalone")
#   stage 2: python runtime + the node binary needed to run the standalone
#            server — bot + pipeline + API + web UI all in one container,
#            consistent with the single-writer DuckDB architecture.

# --- Stage 1: web UI ---------------------------------------------------------
FROM node:22-slim AS web-builder
WORKDIR /build/web-ui

COPY web-ui/package.json web-ui/package-lock.json ./
RUN npm ci

COPY web-ui/ ./
RUN npm run build

# --- Stage 2: runtime --------------------------------------------------------
FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# Node runtime for the standalone Next server (same debian base, so the
# binary is glibc-compatible). No npm needed at runtime.
COPY --from=node:22-slim /usr/local/bin/node /usr/local/bin/node

COPY pyproject.toml ./
COPY lakewind/ lakewind/

RUN pip install --no-cache-dir -e .

# Web UI: standalone server bundle + static assets (node_modules NOT needed).
COPY --from=web-builder /build/web-ui/.next/standalone /app/web-ui
COPY --from=web-builder /build/web-ui/.next/static /app/web-ui/.next/static
COPY --from=web-builder /build/web-ui/public /app/web-ui/public

RUN mkdir -p data models

COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

EXPOSE 3000 8000

ENTRYPOINT ["/docker-entrypoint.sh"]
