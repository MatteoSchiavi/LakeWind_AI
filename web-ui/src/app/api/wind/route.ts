/**
 * LakeWind API — wind forecast proxy (Phase 2 architecture).
 *
 * HISTORY: this route opened a fresh read-write `duckdb.Database` per request
 * in the Node process, racing the Python writer for DuckDB's single-writer
 * file lock (intermittent "Could not set lock on file" errors under load and
 * duplicated page-cache). Phase 2 introduced an internal FastAPI service
 * inside the Python process; this route now proxies to it and benefits from
 * the same in-memory forecast cache as the Telegram bot.
 *
 * Config: LAKEWIND_API_URL (default http://127.0.0.1:8000).
 */
import { NextRequest, NextResponse } from 'next/server';

export const dynamic = 'force-dynamic';

const API_URL = process.env.LAKEWIND_API_URL || 'http://127.0.0.1:8000';

interface Prediction {
  point_id: string;
  generated_at: string;
  valid_time: string;
  model_version: string;
  wind_speed_kn: number | null;
  wind_dir_deg: number | null;
  wind_gust_kn: number | null;
  confidence_pct: number | null;
  expected_error_kn: number | null;
  wind_speed_q10_kn?: number | null;
  wind_speed_q90_kn?: number | null;
  regime?: string | null;
}

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const horizon = searchParams.get('horizon') || '0';
  const pointId = searchParams.get('point');

  const targetTime = new Date(Date.now() + parseInt(horizon, 10) * 3600 * 1000);

  try {
    const upstream = new URL(`${API_URL}/api/wind`);
    if (pointId) upstream.searchParams.set('point', pointId);
    upstream.searchParams.set('horizon', String(parseInt(horizon, 10) || 0));

    const res = await fetch(upstream.toString(), { cache: 'no-store' });
    if (!res.ok) {
      const detail = await res.text();
      console.error('LakeWind upstream error:', res.status, detail.slice(0, 200));
      return NextResponse.json(
        {
          status: 'error',
          error: `LakeWind API returned ${res.status}`,
          target_time: targetTime.toISOString(),
          predictions: [],
        },
        { status: 502 }
      );
    }

    const payload = (await res.json()) as Record<string, unknown>;
    const predictions: Prediction[] = Object.entries(payload).map(([pid, row]) => {
      const r = row as Record<string, unknown>;
      return {
        point_id: pid,
        generated_at: String(r.generated_at ?? ''),
        valid_time: String(r.valid_time ?? ''),
        model_version: String(r.model_version ?? ''),
        wind_speed_kn: (r.wind_speed_kn as number | null) ?? null,
        wind_dir_deg: (r.wind_dir_deg as number | null) ?? null,
        wind_gust_kn: (r.wind_gust_kn as number | null) ?? null,
        confidence_pct: (r.confidence_pct as number | null) ?? null,
        expected_error_kn: (r.expected_error_kn as number | null) ?? null,
        // Phase 4 (W1): calibrated 80% band + regime pass through untouched
        wind_speed_q10_kn: (r.wind_speed_q10_kn as number | null) ?? null,
        wind_speed_q90_kn: (r.wind_speed_q90_kn as number | null) ?? null,
        regime: r.regime ? String(r.regime) : null,
      };
    });

    return NextResponse.json({
      status: 'ok',
      target_time: targetTime.toISOString(),
      horizon_hours: parseInt(horizon, 10) || 0,
      predictions,
    });
  } catch (error) {
    console.error('LakeWind API error:', error);
    return NextResponse.json(
      {
        status: 'error',
        error: error instanceof Error ? error.message : 'Unknown error',
        predictions: [],
      },
      { status: 502 }
    );
  }
}
