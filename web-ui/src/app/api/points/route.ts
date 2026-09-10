import { NextResponse } from 'next/server';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { parse as parseYaml } from 'yaml';

interface PointInfo {
  id: string;
  lat: number;
  lon: number;
  is_operational: boolean;
  sector: string;
}

/**
 * V6.6 FIX (Phase 1 audit): this route hardcoded the V3 list of 15 points —
 * including 8 points that were DELETED in V4 (shore/offshore pairs closer
 * than NWP resolution) and coordinates that were corrected in V5. The map was
 * rendering points that no longer exist at wrong positions.
 *
 * Now: points are read live from settings.yaml (single source of truth).
 * Candidates for the file: LAKEWIND_SETTINGS_PATH env var, /app/settings.yaml
 * (Docker), ../settings.yaml (repo checkout). If the file cannot be read, we
 * fall back to the current V6 list below.
 */
const FALLBACK_POINTS: PointInfo[] = [
  // Operational (7) — coordinates validated on water (V5)
  { id: 'dongo_shore', lat: 46.1230, lon: 9.2850, is_operational: true, sector: 'North' },
  { id: 'gravedona_shore', lat: 46.1460, lon: 9.3050, is_operational: true, sector: 'North' },
  { id: 'domaso_offshore', lat: 46.1500, lon: 9.3230, is_operational: true, sector: 'North' },
  { id: 'mid_channel', lat: 46.1000, lon: 9.3040, is_operational: true, sector: 'Mid-lake' },
  { id: 'piona_entrance', lat: 46.1140, lon: 9.3100, is_operational: true, sector: 'Mid-lake' },
  { id: 'dervio_shore', lat: 46.0763, lon: 9.2980, is_operational: true, sector: 'Mid-lake' },
  { id: 'bellano_offshore', lat: 46.0550, lon: 9.3000, is_operational: true, sector: 'South' },
  // Auxiliary (4) — macro-area pressure-gradient inputs, not forecast points
  { id: 'zurich', lat: 47.376, lon: 8.541, is_operational: false, sector: 'Auxiliary' },
  { id: 'milano_linate', lat: 45.445, lon: 9.278, is_operational: false, sector: 'Auxiliary' },
  { id: 'sondrio', lat: 46.170, lon: 9.870, is_operational: false, sector: 'Auxiliary' },
  { id: 'lugano', lat: 46.005, lon: 8.952, is_operational: false, sector: 'Auxiliary' },
];

function sectorFor(id: string): string {
  if (['dongo_shore', 'gravedona_shore', 'domaso_offshore'].includes(id)) return 'North';
  if (['mid_channel', 'piona_entrance', 'dervio_shore'].includes(id)) return 'Mid-lake';
  if (id === 'bellano_offshore') return 'South';
  return 'Auxiliary';
}

async function loadPointsFromSettings(): Promise<PointInfo[] | null> {
  const candidates = [
    process.env.LAKEWIND_SETTINGS_PATH,
    '/app/settings.yaml',
    path.resolve(process.cwd(), '..', 'settings.yaml'),
    path.resolve(process.cwd(), 'settings.yaml'),
  ].filter((p): p is string => Boolean(p));

  for (const candidate of candidates) {
    try {
      const text = await readFile(candidate, 'utf8');
      const doc = parseYaml(text) as {
        virtual_points?: Array<{ id: string; lat: number; lon: number }>;
        operational_point_ids?: string[];
      };
      if (!doc.virtual_points?.length) continue;
      const opIds = new Set(doc.operational_point_ids ?? []);
      return doc.virtual_points.map((vp) => ({
        id: vp.id,
        lat: vp.lat,
        lon: vp.lon,
        is_operational: opIds.size === 0 ? true : opIds.has(vp.id),
        sector: sectorFor(vp.id),
      }));
    } catch {
      // try next candidate
    }
  }
  return null;
}

let cache: { points: PointInfo[]; loaded_at: number } | null = null;
const CACHE_TTL_MS = 60_000;

export async function GET() {
  // Cache for 60s — settings.yaml changes are rare
  if (!cache || Date.now() - cache.loaded_at > CACHE_TTL_MS) {
    const points = (await loadPointsFromSettings()) ?? FALLBACK_POINTS;
    cache = { points, loaded_at: Date.now() };
  }
  return NextResponse.json({
    status: 'ok',
    source: cache.points === FALLBACK_POINTS ? 'fallback' : 'settings.yaml',
    points: cache.points,
    operational_count: cache.points.filter((p) => p.is_operational).length,
  });
}
