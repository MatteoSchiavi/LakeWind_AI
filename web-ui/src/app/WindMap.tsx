'use client';

/**
 * Interactive wind map (Phase 4 / W3).
 *
 * Fixes applied over the orphaned pre-Phase-4 version:
 *  - the raw `<div key>` as a direct MapContainer child was a react-leaflet
 *    anti-pattern that broke the layer context — children are now wrapped
 *    in `<Fragment key>` (no DOM node inside the layer tree);
 *  - CARDINALS was defined at the bottom of the module (used before
 *    definition) — moved to the top;
 *  - speed colors come from the SHARED palette (`lib/palette.ts`), so a
 *    green dot here means exactly "sailable band" everywhere else;
 *  - clicking a marker selects the point across the whole dashboard
 *    (`onSelect` prop — W3.1 "click marker → selects point everywhere").
 */
import { Fragment, useEffect } from 'react';
import { MapContainer, TileLayer, CircleMarker, Popup, Polyline, useMap } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import { speedColor } from '@/lib/palette';

// --- Types ---
interface WindPoint {
  id: string;
  lat: number;
  lon: number;
  speed: number;
  q10?: number | null;
  q90?: number | null;
  direction: number;
  gust: number;
  confidence: number;
  sector: string;
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
  // Lake Como center
  const center: [number, number] = [46.09, 9.30];

  return (
    <div className="relative h-[420px] w-full overflow-hidden rounded-xl border">
      <MapContainer
        center={center}
        zoom={12}
        style={{ height: '100%', width: '100%' }}
        scrollWheelZoom={false}
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        <FitBounds points={points} />

        {points.map((pt) => {
          const color = speedColor(pt.speed);
          const [dx, dy] = arrowDelta(pt.direction, pt.speed);
          const arrowEnd: [number, number] = [pt.lat + dy, pt.lon + dx];
          const radius = Math.max(6, Math.min(pt.speed * 1.2, 20));

          return (
            <Fragment key={pt.id}>
              {/* Wind arrow */}
              <Polyline
                positions={[[pt.lat, pt.lon], arrowEnd]}
                pathOptions={{ color, weight: 2.5, opacity: 0.8 }}
              />
              {/* Point circle */}
              <CircleMarker
                center={[pt.lat, pt.lon]}
                radius={radius}
                eventHandlers={{ click: () => onSelect?.(pt.id) }}
                pathOptions={{
                  color,
                  fillColor: color,
                  fillOpacity: 0.6,
                  weight: 2,
                }}
              >
                <Popup>
                  <div className="text-sm">
                    <div className="font-bold">{pt.id.replace(/_/g, ' ')}</div>
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

      {/* Legend overlay — shared palette bands */}
      <div className="absolute bottom-3 right-3 z-[1000] rounded-lg border bg-white/90 p-2 text-xs shadow-lg dark:bg-slate-900/90">
        <div className="mb-1 font-semibold">Wind Speed (kn)</div>
        {[
          { c: '#3b82f6', label: '<5' },
          { c: '#06b6d4', label: '5–8' },
          { c: '#22c55e', label: '8–12 ⛵' },
          { c: '#f59e0b', label: '12–16' },
          { c: '#dc2626', label: '16+' },
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
