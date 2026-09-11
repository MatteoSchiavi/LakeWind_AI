/**
 * LakeWind API — trend series proxy (Phase 2 architecture).
 *
 * Previously opened a read-write DuckDB handle per request (lock contention
 * with the Python writer). Now proxies to the internal FastAPI service,
 * which serves from the in-memory forecast store.
 */
import { NextRequest, NextResponse } from 'next/server';

export const dynamic = 'force-dynamic';

const API_URL = process.env.LAKEWIND_API_URL || 'http://127.0.0.1:8000';

interface TrendRow {
  point_id: string;
  valid_time: string;
  wind_speed_kn: number | null;
  wind_dir_deg: number | null;
  wind_gust_kn: number | null;
  confidence_pct: number | null;
  expected_error_kn: number | null;
  wind_speed_q10_kn?: number | null;
  wind_speed_q90_kn?: number | null;
  regime?: string | null;
  time: number;
}

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const pointId = searchParams.get('point') || 'dongo';
  const hours = parseInt(searchParams.get('hours') || '24', 10);

  try {
    const upstream = new URL(`${API_URL}/api/trend`);
    upstream.searchParams.set('point', pointId);
    upstream.searchParams.set('hours', String(Math.min(Math.max(hours, 1), 48)));

    const res = await fetch(upstream.toString(), { cache: 'no-store' });
    if (!res.ok) {
      const detail = await res.text();
      console.error('LakeWind upstream error:', res.status, detail.slice(0, 200));
      return NextResponse.json(
        { status: 'error', error: `LakeWind API returned ${res.status}`, data: [] },
        { status: 502 }
      );
    }

    // Upstream shape: { "<point_id>": [prediction rows sorted by valid_time] }
    const payload = (await res.json()) as Record<string, Array<Record<string, unknown>>>;
    const rows = payload[pointId] ?? [];

    const data: TrendRow[] = rows.map((r) => ({
      point_id: pointId,
      valid_time: String(r.valid_time ?? ''),
      wind_speed_kn: (r.wind_speed_kn as number | null) ?? null,
      wind_dir_deg: (r.wind_dir_deg as number | null) ?? null,
      wind_gust_kn: (r.wind_gust_kn as number | null) ?? null,
      confidence_pct: (r.confidence_pct as number | null) ?? null,
      expected_error_kn: (r.expected_error_kn as number | null) ?? null,
      // Phase 4 (W1): calibrated 80% band + regime pass through untouched
      wind_speed_q10_kn: (r.wind_speed_q10_kn as number | null) ?? null,
      wind_speed_q90_kn: (r.wind_speed_q90_kn as number | null) ?? null,
      regime: r.regime ? String(r.regime) : null,
      time: new Date(String(r.valid_time ?? '')).getTime(),
    }));

    return NextResponse.json({ status: 'ok', point_id: pointId, hours, data });
  } catch (error) {
    console.error('Trend API error:', error);
    return NextResponse.json(
      { status: 'error', error: error instanceof Error ? error.message : 'Unknown error', data: [] },
      { status: 502 }
    );
  }
}
