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
  label: string;
  lake?: string;
}

/**
 * Points are read live from settings.yaml (single source of truth) — including
 * the Phase 5.5 per-spot metadata (label, sector) so the UI never hardcodes
 * coordinates or sector maps. The old hardcoded sectorFor() mapping silently
 * dumped any newly added spot into "Auxiliary", hiding it from the sector
 * grid — new spots now flow through automatically.
 *
 * Candidates for the file: LAKEWIND_SETTINGS_PATH env var, /app/settings.yaml
 * (Docker), ../settings.yaml (repo checkout). If the file cannot be read, we
 * fall back to the verified Phase 5.5 list below (same data, snapshotted).
 */
const FALLBACK_POINTS: PointInfo[] = [
  // Operational (15) — verified anchors, water-projected sampling points
  // (source: settings.yaml Phase 5.5; each spot cross-checked OSM + Wikipedia)
  { id: 'colico', lat: 46.14191, lon: 9.36743, is_operational: true, sector: 'alto_lario', label: 'Colico' },
  { id: 'sorico', lat: 46.16686, lon: 9.37255, is_operational: true, sector: 'alto_lario', label: 'Sorico' },
  { id: 'gera_lario', lat: 46.16389, lon: 9.37257, is_operational: true, sector: 'alto_lario', label: 'Gera Lario' },
  { id: 'domaso', lat: 46.14699, lon: 9.32155, is_operational: true, sector: 'alto_lario', label: 'Domaso' },
  { id: 'gravedona', lat: 46.14492, lon: 9.31233, is_operational: true, sector: 'alto_lario', label: 'Gravedona ed Uniti' },
  { id: 'dongo', lat: 46.12030, lon: 9.28634, is_operational: true, sector: 'alto_lario', label: 'Dongo' },
  { id: 'piona', lat: 46.12774, lon: 9.33569, is_operational: true, sector: 'alto_lario', label: 'Piona (Olgiasca)' },
  { id: 'cremia', lat: 46.08681, lon: 9.28480, is_operational: true, sector: 'alto_lario', label: 'Cremia' },
  { id: 'dervio', lat: 46.07980, lon: 9.29490, is_operational: true, sector: 'alto_lario', label: 'Dervio' },
  { id: 'varenna', lat: 46.00439, lon: 9.28407, is_operational: true, sector: 'lario_centrale', label: 'Varenna' },
  { id: 'menaggio', lat: 46.01614, lon: 9.24443, is_operational: true, sector: 'lario_centrale', label: 'Menaggio' },
  { id: 'bellagio', lat: 45.98757, lon: 9.25324, is_operational: true, sector: 'lario_centrale', label: 'Bellagio' },
  { id: 'mandello', lat: 45.91524, lon: 9.30640, is_operational: true, sector: 'branca_lecco', label: 'Mandello del Lario' },
  { id: 'lecco', lat: 45.85402, lon: 9.38194, is_operational: true, sector: 'branca_lecco', label: 'Lecco / Valmadrera' },
  { id: 'como_city', lat: 45.81676, lon: 9.07533, is_operational: true, sector: 'branca_como', label: 'Como', lake: 'lake_como' },
  // Phase 6: Garda + Maggiore north basins (verified on-water, settings.yaml)
  { id: 'riva_del_garda', lat: 45.8765, lon: 10.8495, is_operational: true, sector: 'garda_nord', label: 'Riva del Garda', lake: 'lake_garda' },
  { id: 'torbole', lat: 45.8625, lon: 10.858, is_operational: true, sector: 'garda_nord', label: 'Torbole', lake: 'lake_garda' },
  { id: 'malcesine', lat: 45.7672, lon: 10.8025, is_operational: true, sector: 'garda_sud', label: 'Malcesine', lake: 'lake_garda' },
  { id: 'brenzone', lat: 45.7266, lon: 10.7725, is_operational: true, sector: 'garda_sud', label: 'Brenzone', lake: 'lake_garda' },
  { id: 'luino', lat: 46.0015, lon: 8.729, is_operational: true, sector: 'maggiore_nord', label: 'Luino', lake: 'lake_maggiore' },
  { id: 'cannero', lat: 46.018, lon: 8.7, is_operational: true, sector: 'maggiore_nord', label: 'Cannero Riviera', lake: 'lake_maggiore' },
  { id: 'cannobio', lat: 46.063, lon: 8.71, is_operational: true, sector: 'maggiore_nord', label: 'Cannobio', lake: 'lake_maggiore' },
  // Auxiliary (4) — macro-area pressure-gradient inputs, not forecast points
  { id: 'zurich', lat: 47.376, lon: 8.541, is_operational: false, sector: 'auxiliary', label: 'Zurich' },
  { id: 'milano_linate', lat: 45.445, lon: 9.278, is_operational: false, sector: 'auxiliary', label: 'Milano Linate' },
  { id: 'sondrio', lat: 46.170, lon: 9.870, is_operational: false, sector: 'auxiliary', label: 'Sondrio' },
  { id: 'lugano', lat: 46.005, lon: 8.952, is_operational: false, sector: 'auxiliary', label: 'Lugano' },
];

const SECTOR_FALLBACKS: Record<string, string> = {
  alto_lario: 'Alto Lario',
  lario_centrale: 'Lario Centrale',
  branca_lecco: 'Branca di Lecco',
  branca_como: 'Branca di Como',
  garda_nord: 'Garda Nord',
  garda_sud: 'Garda Sud',
  maggiore_nord: 'Maggiore Nord',
  auxiliary: 'Auxiliary',
};

function sectorLabel(sector: string | undefined): string {
  return SECTOR_FALLBACKS[sector ?? 'auxiliary'] ?? sector ?? 'Auxiliary';
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
        virtual_points?: Array<{
          id: string; lat: number; lon: number;
          label?: string; sector?: string; lake?: string;
        }>;
        operational_point_ids?: string[];
      };
      if (!doc.virtual_points?.length) continue;
      const opIds = new Set(doc.operational_point_ids ?? []);
      return doc.virtual_points.map((vp) => ({
        id: vp.id,
        lat: vp.lat,
        lon: vp.lon,
        is_operational: opIds.size === 0 ? true : opIds.has(vp.id),
        sector: sectorLabel(vp.sector),
        label: vp.label ?? vp.id,
        lake: vp.lake,
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
