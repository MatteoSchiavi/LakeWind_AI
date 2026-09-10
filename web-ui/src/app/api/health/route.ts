/**
 * LakeWind API — health proxy (Phase 2 architecture).
 *
 * Previously opened a read-write DuckDB handle per request. Now proxies to
 * the internal FastAPI /api/health, which additionally reports pipeline-loop
 * status, cache statistics and per-source freshness SLAs.
 */
import { NextResponse } from 'next/server';

export const dynamic = 'force-dynamic';

const API_URL = process.env.LAKEWIND_API_URL || 'http://127.0.0.1:8000';

interface SourceHealth {
  source: string;
  ok: boolean;
  latency_ms: number;
  checked_at: string;
  error_msg: string | null;
}

export async function GET() {
  try {
    const res = await fetch(`${API_URL}/api/health`, { cache: 'no-store' });
    if (!res.ok) {
      const detail = await res.text();
      console.error('LakeWind upstream error:', res.status, detail.slice(0, 200));
      return NextResponse.json(
        { status: 'error', error: `LakeWind API returned ${res.status}`, sources: [] },
        { status: 502 }
      );
    }

    const payload = (await res.json()) as {
      status: string;
      freshness?: Array<{ source: string; age_minutes: number; is_fresh: boolean }>;
      source_health?: Array<Record<string, unknown>>;
      pipeline?: { active?: boolean; last_predict_summary?: { n_forecasts?: number } | null };
      store?: Record<string, unknown>;
    };

    // Preserve the original web-ui contract, enriched with pipeline data.
    const sources: SourceHealth[] = (payload.source_health ?? []).map((h) => ({
      source: String(h.source ?? ''),
      ok: Boolean(h.ok),
      latency_ms: Number(h.latency_ms ?? 0),
      checked_at: String(h.checked_at ?? ''),
      error_msg: h.error_msg ? String(h.error_msg) : null,
    }));

    return NextResponse.json({
      status: payload.pipeline?.active ? 'ok' : 'degraded',
      sources,
      freshness: payload.freshness ?? [],
      last_pipeline_cycle: payload.pipeline ?? null,
      cache: payload.store ?? {},
    });
  } catch (error) {
    console.error('Health API error:', error);
    return NextResponse.json(
      {
        status: 'error',
        error: error instanceof Error ? error.message : 'Unknown error',
        sources: [],
      },
      { status: 502 }
    );
  }
}
