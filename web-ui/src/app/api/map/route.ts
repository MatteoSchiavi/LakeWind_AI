/**
 * LakeWind API — precomputed heatmap PNG proxy (Phase 4 / W3).
 *
 * The interactive Leaflet map answers "where exactly?"; the precomputed
 * v3 heatmap answers "what does the whole corridor look like?" with the
 * richer overlays (interpolation, station models, regime badge). Both
 * views are served: tab switcher on the dashboard.
 *
 * Passes through the FastAPI provenance headers (X-Map-Source,
 * X-Map-Valid-Time) so the UI can stamp the map's freshness.
 */
import { NextRequest, NextResponse } from 'next/server';

export const dynamic = 'force-dynamic';

const API_URL = process.env.LAKEWIND_API_URL || 'http://127.0.0.1:8000';

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const offset = Math.min(Math.max(parseInt(searchParams.get('offset') || '0', 10) || 0, 0), 24);

  try {
    const upstream = `${API_URL}/api/map.png?offset=${offset}`;
    const res = await fetch(upstream, { cache: 'no-store' });
    if (!res.ok) {
      return NextResponse.json(
        { status: 'error', error: `LakeWind API returned ${res.status}` },
        { status: 502 },
      );
    }

    const buf = await res.arrayBuffer();
    return new NextResponse(buf, {
      status: 200,
      headers: {
        'Content-Type': 'image/png',
        'Cache-Control': 'no-store',
        'X-Map-Source': res.headers.get('X-Map-Source') ?? 'unknown',
        'X-Map-Valid-Time': res.headers.get('X-Map-Valid-Time') ?? '',
      },
    });
  } catch (error) {
    console.error('Map proxy error:', error);
    return NextResponse.json(
      { status: 'error', error: error instanceof Error ? error.message : 'Unknown error' },
      { status: 502 },
    );
  }
}
