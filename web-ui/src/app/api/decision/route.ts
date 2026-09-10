/**
 * LakeWind API — sailing-decision proxy (Phase 4 / W2).
 *
 * Serves the "Go sailing?" hero card from the shared decision module
 * (`lakewind/prediction/decision.py`) — the SAME math the bot's /sailing
 * uses, so the web and the bot always agree on GO / MARGINAL / NO-GO.
 */
import { NextRequest, NextResponse } from 'next/server';

export const dynamic = 'force-dynamic';

const API_URL = process.env.LAKEWIND_API_URL || 'http://127.0.0.1:8000';

export interface DecisionHourDTO {
  hour: number | null;
  valid_time: string | null;
  speed_kn: number;
  q10_kn: number | null;
  q90_kn: number | null;
  dir_deg: number | null;
  p_go: number;
  p_strong: number;
  regime: string | null;
}

export interface DecisionDTO {
  point_id: string;
  verdict: 'go' | 'marginal' | 'no_go';
  best_hour: number | null;
  best_speed_kn: number | null;
  n_go_hours: number;
  peak_p_go: number;
  regime: string | null;
  hours: DecisionHourDTO[];
}

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const point = searchParams.get('point');
  const hours = Math.min(Math.max(parseInt(searchParams.get('hours') || '14', 10) || 14, 4), 24);

  try {
    const upstream = new URL(`${API_URL}/api/decision`);
    if (point) upstream.searchParams.set('point', point);
    upstream.searchParams.set('hours', String(hours));

    const res = await fetch(upstream.toString(), { cache: 'no-store' });
    if (!res.ok) {
      const detail = await res.text();
      console.error('LakeWind upstream error:', res.status, detail.slice(0, 200));
      return NextResponse.json(
        { status: 'error', error: `LakeWind API returned ${res.status}`, decisions: {} },
        { status: 502 },
      );
    }

    const payload = (await res.json()) as {
      generated_at?: string;
      timezone?: string;
      thresholds?: { go_kn?: number; strong_kn?: number };
      decisions?: Record<string, DecisionDTO>;
    };

    return NextResponse.json({
      status: 'ok',
      generated_at: payload.generated_at ?? null,
      timezone: payload.timezone ?? 'Europe/Rome',
      thresholds: payload.thresholds ?? { go_kn: 8, strong_kn: 12 },
      decisions: payload.decisions ?? {},
    });
  } catch (error) {
    console.error('Decision API error:', error);
    return NextResponse.json(
      { status: 'error', error: error instanceof Error ? error.message : 'Unknown error', decisions: {} },
      { status: 502 },
    );
  }
}
