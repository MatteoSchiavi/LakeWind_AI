'use client';

/**
 * Interactive wind map (Phase 4 / W3, overhauled in Phase 5.5).
 *
 * Phase 5.5 upgrades:
 *  - REAL wind-field layer: anisotropic IDW interpolation over the 15 spots,
 *    rendered client-side into a canvas and masked by the verified lake
 *    shoreline (/lake-como.geojson — OSM relation 541757). The field mirrors
 *    the Python heatmap math (same palette, same 10° valley-axis anisotropy)
 *    so both views tell the same story.
 *  - All 15 verified spots with permanent name labels (coordinates come from
 *    settings.yaml via /api/points — nothing hardcoded here).
 *  - Legend built from the shared palette constants (no duplicated hexes).
 *  - Click a marker → selects the point across the whole dashboard.
 */
import { Fragment, useEffect, useMemo, useRef, useState } from 'react';
import {
  MapContainer, TileLayer, CircleMarker, Popup, Polyline, Tooltip, useMap,
  ImageOverlay, Polygon,
} from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import { SPEED_COLORS, SPEED_BREAKS } from '@/lib/palette';

// --- Types ---
interface WindPoint {
  id: string;
  lat: number;
  lon: number;
  label?: string;
  speed: number;
  q10?: number | null;
  q90?: number | null;
  direction: number;
  gust: number;
  confidence: number;
  sector: string;
}

interface LakeGeo {
  type: string;
  features: Array<{ geometry: { type: string; coordinates: number[][][] | number[][][][] } }>;
}

const CARDINALS = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'];

function degToCardinal(deg: number): string {
  return CARDINALS[Math.round(deg / 22.5) % 16];
}

function arrowDelta(direction: number, speed: number): [number, number] {
  // Arrow points in the direction wind is GOING TO (opposite of FROM)
  const goToDir = (direction + 180) % 360;
  const rad = (goToDir * Math.PI) / 180;
  const len = Math.min(speed * 0.003 + 0.003, 0.015); // scale by speed
  return [Math.sin(rad) * len, Math.cos(rad) * len];
}

function hexToRgb(hex: string): [number, number, number] {
  return [
    parseInt(hex.slice(1, 3), 16),
    parseInt(hex.slice(3, 5), 16),
    parseInt(hex.slice(5, 7), 16),
  ];
}

// Piecewise gradient anchored to the shared palette — mirrors the Python
// heatmap colormap exactly (band midpoints 2.5/6.5/10/14/22 on 0-30 kn).
const GRAD_ANCHORS: Array<[number, [number, number, number]]> = (() => {
  const mids = [2.5, 6.5, 10.0, 14.0, 22.0];
  const cols = SPEED_COLORS.map(hexToRgb);
  return [
    [0.0, cols[0]],
    ...mids.map((m, i) => [m, cols[i]] as [number, [number, number, number]]),
    [30.0, cols[cols.length - 1]],
  ];
})();

function gradientColor(speedKn: number): [number, number, number] {
  const v = Math.max(0, Math.min(30, speedKn));
  for (let i = 1; i < GRAD_ANCHORS.length; i++) {
    const [x1, c1] = GRAD_ANCHORS[i];
    const [x0, c0] = GRAD_ANCHORS[i - 1];
    if (v <= x1) {
      const t = x1 === x0 ? 0 : (v - x0) / (x1 - x0);
      return [
        c0[0] + (c1[0] - c0[0]) * t,
        c0[1] + (c1[1] - c0[1]) * t,
        c0[2] + (c1[2] - c0[2]) * t,
      ];
    }
  }
  return GRAD_ANCHORS[GRAD_ANCHORS.length - 1][1];
}

// Lake geometry cache (fetched once per page load)
let lakeGeoPromise: Promise<number[][] | null> | null = null;

function outerRingOf(geo: LakeGeo): number[][] {
  const f = geo.features[0];
  const c = f.geometry.coordinates as number[][][] | number[][][][];
  if (f.geometry.type === 'MultiPolygon') {
    const mp = c as number[][][][];
    let best: number[][] = [];
    for (const poly of mp) {
      if (poly[0].length > best.length) best = poly[0];
    }
    return best;
  }
  return (c as number[][][])[0];
}

function fetchLakeRing(): Promise<number[][] | null> {
  if (!lakeGeoPromise) {
    lakeGeoPromise = fetch('/lake-como.geojson')
      .then((r) => (r.ok ? r.json() : null))
      .then((geo: LakeGeo | null) => (geo ? outerRingOf(geo) : null))
      .catch(() => null);
  }
  return lakeGeoPromise;
}

// Same geometry transform as the Python heatmap: km-space, rotated onto the
// 10° valley axis, cross-axis compressed by 3 -> the field elongates along
// the lake corridor instead of smearing across the ridges.
const VALLEY_AXIS_DEG = 10.0;
const ANISOTROPY = 3.0;
const CANVAS_W = 320;
const CANVAS_H = 560;
const FIELD_BOUNDS: [[number, number], [number, number]] = [[45.70, 8.98], [46.26, 9.46]];

function buildWindFieldCanvas(points: WindPoint[], ring: number[][]): string | null {
  if (points.length < 3) return null;
  const [latMin, lonMin] = FIELD_BOUNDS[0];
  const [latMax, lonMax] = FIELD_BOUNDS[1];
  const lat0 = (latMin + latMax) / 2;
  const kmLat = 110.574;
  const kmLon = 111.32 * Math.cos((lat0 * Math.PI) / 180);
  const th = (VALLEY_AXIS_DEG * Math.PI) / 180;
  const cos = Math.cos(th);
  const sin = Math.sin(th);

  const proj = (lat: number, lon: number): [number, number] => {
    const x = (lon - (lonMin + lonMax) / 2) * kmLon;
    const y = (lat - lat0) * kmLat;
    // along-axis = first, cross-axis = second (compressed)
    return [x * cos + y * sin, (-x * sin + y * cos) / ANISOTROPY];
  };

  const ptsKm = points.map((p) => ({ v: p.speed, a: proj(p.lat, p.lon)[0], c: proj(p.lat, p.lon)[1] }));
  const meanV = points.reduce((s, p) => s + p.speed, 0) / points.length;

  const canvas = document.createElement('canvas');
  canvas.width = CANVAS_W;
  canvas.height = CANVAS_H;
  const ctx = canvas.getContext('2d');
  if (!ctx) return null;

  // Mask: fill the lake polygon path once, then sample the alpha channel.
  const lonToX = (lon: number) => ((lon - lonMin) / (lonMax - lonMin)) * CANVAS_W;
  const latToY = (lat: number) => CANVAS_H - ((lat - latMin) / (latMax - latMin)) * CANVAS_H;
  ctx.beginPath();
  ring.forEach(([lon, lat], i) => {
    const x = lonToX(lon);
    const y = latToY(lat);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.closePath();
  ctx.fillStyle = '#fff';
  ctx.fill();
  const mask = ctx.getImageData(0, 0, CANVAS_W, CANVAS_H);

  // IDW field over masked pixels
  const out = ctx.createImageData(CANVAS_W, CANVAS_H);
  for (let py = 0; py < CANVAS_H; py++) {
    for (let px = 0; px < CANVAS_W; px++) {
      const idx = (py * CANVAS_W + px) * 4;
      if (mask.data[idx + 3] < 128) continue; // outside the lake
      const lon = lonMin + (px / CANVAS_W) * (lonMax - lonMin);
      const lat = latMin + ((CANVAS_H - py) / CANVAS_H) * (latMax - latMin);
      const [pa, pc] = proj(lat, lon);
      let wSum = 0;
      let vSum = 0;
      for (const p of ptsKm) {
        const d2 = (pa - p.a) * (pa - p.a) + (pc - p.c) * (pc - p.c) + 0.02;
        const w = 1 / (d2 * d2); // ~1/d^4 -> crisper near spots
        wSum += w;
        vSum += w * p.v;
      }
      const v = wSum > 0 ? vSum / wSum : meanV;
      const [r, g, b] = gradientColor(v);
      out.data[idx] = r;
      out.data[idx + 1] = g;
      out.data[idx + 2] = b;
      out.data[idx + 3] = 168; // ~66% opacity baked in
    }
  }
  ctx.putImageData(out, 0, 0);
  return canvas.toDataURL('image/png');
}

function FitBounds({ points }: { points: WindPoint[] }) {
  const map = useMap();
  useEffect(() => {
    if (points.length === 0) return;
    const lats = points.map((p) => p.lat);
    const lons = points.map((p) => p.lon);
    const bounds: [[number, number], [number, number]] = [
      [Math.min(...lats) - 0.01, Math.min(...lons) - 0.01],
      [Math.max(...lats) + 0.01, Math.max(...lons) + 0.01],
    ];
    map.fitBounds(bounds);
  }, [points, map]);
  return null;
}

export default function WindMap({ points, onSelect }: {
  points: WindPoint[];
  onSelect?: (pointId: string) => void;
}) {
  // Lake Como center (whole-basin view)
  const center: [number, number] = [46.06, 9.26];
  const [ring, setRing] = useState<number[][] | null>(null);
  const ringRef = useRef<number[][] | null>(null);

  useEffect(() => {
    let alive = true;
    fetchLakeRing().then((r) => {
      if (!alive) return;
      ringRef.current = r;
      setRing(r);
    });
    return () => { alive = false; };
  }, []);

  // Rebuild the field whenever the data or the shoreline arrive
  const fieldUrl = useMemo(
    () => (ring ? buildWindFieldCanvas(points, ring) : null),
    [points, ring],
  );

  return (
    <div className="relative h-[420px] w-full overflow-hidden rounded-xl border">
      <MapContainer
        center={center}
        zoom={11}
        style={{ height: '100%', width: '100%' }}
        scrollWheelZoom={false}
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        <FitBounds points={points} />

        {/* Interpolated wind field (masked to the verified lake polygon) */}
        {fieldUrl && (
          <ImageOverlay
            url={fieldUrl}
            bounds={FIELD_BOUNDS}
            opacity={0.75}
            interactive={false}
            zIndex={350}
          />
        )}

        {/* Verified shoreline outline */}
        {ring && (
          <Polygon
            positions={ring.map(([lon, lat]) => [lat, lon] as [number, number])}
            pathOptions={{ color: '#0a3d5c', weight: 1.5, opacity: 0.8, fill: false }}
            interactive={false}
          />
        )}

        {points.map((pt) => {
          const color = SPEED_COLORS[
            pt.speed < SPEED_BREAKS[0] ? 0
              : pt.speed < SPEED_BREAKS[1] ? 1
                : pt.speed < SPEED_BREAKS[2] ? 2
                  : pt.speed < SPEED_BREAKS[3] ? 3 : 4
          ];
          const [dx, dy] = arrowDelta(pt.direction, pt.speed);
          const arrowEnd: [number, number] = [pt.lat + dy, pt.lon + dx];
          const radius = Math.max(6, Math.min(pt.speed * 1.2, 20));

          return (
            <Fragment key={pt.id}>
              {/* Wind arrow */}
              <Polyline
                positions={[[pt.lat, pt.lon], arrowEnd]}
                pathOptions={{ color, weight: 2.5, opacity: 0.9 }}
              />
              {/* Point circle */}
              <CircleMarker
                center={[pt.lat, pt.lon]}
                radius={radius}
                eventHandlers={{ click: () => onSelect?.(pt.id) }}
                pathOptions={{
                  color: '#0b1220',
                  fillColor: color,
                  fillOpacity: 0.85,
                  weight: 1.5,
                }}
              >
                <Tooltip permanent direction="right" offset={[6, 0]} opacity={0.9}
                  className="lakewind-spot-label">
                  <span style={{ fontWeight: 700, fontSize: 11 }}>{pt.label ?? pt.id}</span>
                </Tooltip>
                <Popup>
                  <div className="text-sm">
                    <div className="font-bold">{pt.label ?? pt.id.replace(/_/g, ' ')}</div>
                    <div className="text-muted-foreground text-xs">
                      {pt.lat.toFixed(4)}, {pt.lon.toFixed(4)} · verified spot
                    </div>
                    <div className="mt-1">
                      <span className="font-bold" style={{ color }}>{pt.speed.toFixed(1)} kn</span>
                      {' '}
                      {degToCardinal(pt.direction)} ({pt.direction.toFixed(0)}°)
                    </div>
                    {pt.q10 !== null && pt.q10 !== undefined && pt.q90 !== null && pt.q90 !== undefined && (
                      <div className="text-muted-foreground">
                        Range: {pt.q10.toFixed(1)}–{pt.q90.toFixed(1)} kn (80%)
                      </div>
                    )}
                    <div className="text-muted-foreground">
                      Gust: {pt.gust.toFixed(1)} kn · Conf: {pt.confidence.toFixed(0)}%
                    </div>
                    <div className="text-muted-foreground">Sector: {pt.sector}</div>
                  </div>
                </Popup>
              </CircleMarker>
            </Fragment>
          );
        })}
      </MapContainer>

      {/* Legend overlay — built from the shared palette constants */}
      <div className="absolute bottom-3 right-3 z-[1000] rounded-lg border bg-white/90 p-2 text-xs shadow-lg dark:bg-slate-900/90">
        <div className="mb-1 font-semibold">Wind Speed (kn)</div>
        {[
          { c: SPEED_COLORS[0], label: `<${SPEED_BREAKS[0]}` },
          { c: SPEED_COLORS[1], label: `${SPEED_BREAKS[0]}–${SPEED_BREAKS[1]}` },
          { c: SPEED_COLORS[2], label: `${SPEED_BREAKS[1]}–${SPEED_BREAKS[2]} ⛵` },
          { c: SPEED_COLORS[3], label: `${SPEED_BREAKS[2]}–${SPEED_BREAKS[3]}` },
          { c: SPEED_COLORS[4], label: `${SPEED_BREAKS[3]}+` },
        ].map(({ c, label }) => (
          <div key={label} className="flex items-center gap-1">
            <span className="inline-block h-3 w-3 rounded-full" style={{ background: c }} />
            <span>{label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
