'use client';

import { useEffect } from 'react';
import { MapContainer, TileLayer, CircleMarker, Popup, Polyline, useMap } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';

// --- Types ---
interface WindPoint {
  id: string;
  lat: number;
  lon: number;
  speed: number;
  direction: number;
  gust: number;
  confidence: number;
  sector: string;
}

// Wind arrow component using SVG polyline
function WindArrow({ direction, speed }: { direction: number; speed: number }) {
  // Arrow points in the direction wind is GOING TO (opposite of FROM)
  const goToDir = (direction + 180) % 360;
  const rad = (goToDir * Math.PI) / 180;
  const len = Math.min(speed * 0.003 + 0.003, 0.015); // scale by speed
  const dx = Math.sin(rad) * len;
  const dy = Math.cos(rad) * len;
  return [dx, dy];
}

function speedColor(speed: number): string {
  if (speed < 5) return '#2b83ba';
  if (speed < 10) return '#abdda4';
  if (speed < 16) return '#fdae61';
  if (speed < 22) return '#f46d43';
  return '#d73027';
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

export default function WindMap({ points }: { points: WindPoint[] }) {
  // Lake Como center
  const center: [number, number] = [46.09, 9.30];

  return (
    <div className="relative h-[500px] w-full overflow-hidden rounded-xl border">
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
          const [arrowDx, arrowDy] = WindArrow({ direction: pt.direction, speed: pt.speed });
          const arrowEnd: [number, number] = [pt.lat + arrowDy, pt.lon + arrowDx];
          const radius = Math.max(6, Math.min(pt.speed * 1.2, 20));

          return (
            <div key={pt.id}>
              {/* Wind arrow */}
              <Polyline
                positions={[[pt.lat, pt.lon], arrowEnd]}
                pathOptions={{ color: color, weight: 2.5, opacity: 0.8 }}
              />
              {/* Point circle */}
              <CircleMarker
                center={[pt.lat, pt.lon]}
                radius={radius}
                pathOptions={{
                  color: color,
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
                      {CARDINALS[Math.round(pt.direction / 22.5) % 16]} ({pt.direction.toFixed(0)}°)
                    </div>
                    <div className="text-muted-foreground">
                      Gust: {pt.gust.toFixed(1)} kn · Conf: {pt.confidence.toFixed(0)}%
                    </div>
                    <div className="text-muted-foreground">Sector: {pt.sector}</div>
                  </div>
                </Popup>
              </CircleMarker>
            </div>
          );
        })}
      </MapContainer>

      {/* Legend overlay */}
      <div className="absolute bottom-3 right-3 z-[1000] rounded-lg border bg-white/90 p-2 text-xs shadow-lg">
        <div className="mb-1 font-semibold">Wind Speed (kn)</div>
        <div className="flex items-center gap-1">
          <span className="inline-block h-3 w-3 rounded-full" style={{ background: '#2b83ba' }} />
          <span>&lt;5</span>
        </div>
        <div className="flex items-center gap-1">
          <span className="inline-block h-3 w-3 rounded-full" style={{ background: '#abdda4' }} />
          <span>5-10</span>
        </div>
        <div className="flex items-center gap-1">
          <span className="inline-block h-3 w-3 rounded-full" style={{ background: '#fdae61' }} />
          <span>10-16</span>
        </div>
        <div className="flex items-center gap-1">
          <span className="inline-block h-3 w-3 rounded-full" style={{ background: '#f46d43' }} />
          <span>16-22</span>
        </div>
        <div className="flex items-center gap-1">
          <span className="inline-block h-3 w-3 rounded-full" style={{ background: '#d73027' }} />
          <span>&gt;22</span>
        </div>
      </div>
    </div>
  );
}

const CARDINALS = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'];
