'use client';

import { useState, useEffect, useCallback } from 'react';
import dynamic from 'next/dynamic';
import {
  Wind,
  Navigation,
  TrendingUp,
  Activity,
  MapPin,
  Clock,
  Zap,
  Gauge,
  RefreshCw,
  ChevronRight,
  AlertCircle,
  CheckCircle2,
  XCircle,
  Compass,
  Waves,
  Calendar,
  ArrowUp,
  ArrowDown,
  Minus,
} from 'lucide-react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Area,
  AreaChart,
  ReferenceLine,
  Legend,
} from 'recharts';

// Dynamically import WindMap (leaflet needs window)
const WindMap = dynamic(() => import('./WindMap'), { ssr: false });

// --- Types ---

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
}

interface PointInfo {
  id: string;
  lat: number;
  lon: number;
  is_operational: boolean;
  sector: string;
}

interface HealthInfo {
  source: string;
  ok: boolean;
  latency_ms: number;
  checked_at: string;
  error_msg: string | null;
}

interface TrendPoint {
  time: number;
  valid_time: string;
  wind_speed_kn: number | null;
  wind_dir_deg: number | null;
  wind_gust_kn: number | null;
  confidence_pct: number | null;
  expected_error_kn: number | null;
}

// --- Constants ---

const HORIZONS = [
  { hours: 0, label: 'Now' },
  { hours: 1, label: '+1h' },
  { hours: 3, label: '+3h' },
  { hours: 6, label: '+6h' },
  { hours: 12, label: '+12h' },
  { hours: 24, label: '+24h' },
];

const CARDINALS = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'];

function degToCardinal(deg: number): string {
  return CARDINALS[Math.round(deg / 22.5) % 16];
}

function speedColor(speed: number): string {
  if (speed < 5) return 'text-blue-500';
  if (speed < 10) return 'text-green-500';
  if (speed < 16) return 'text-yellow-500';
  if (speed < 22) return 'text-orange-500';
  return 'text-red-500';
}

function confidenceColor(conf: number): string {
  if (conf >= 75) return 'text-green-500';
  if (conf >= 50) return 'text-yellow-500';
  return 'text-red-500';
}

// --- Components ---

function WindCompass({ direction, size = 80 }: { direction: number; size?: number }) {
  const rad = (direction * Math.PI) / 180;
  // Arrow points in direction wind is GOING TO (opposite of FROM)
  const goToDir = (direction + 180) % 360;
  const goRad = (goToDir * Math.PI) / 180;
  const cx = size / 2;
  const cy = size / 2;
  const arrowLen = size * 0.35;
  const dx = Math.sin(goRad) * arrowLen;
  const dy = -Math.cos(goRad) * arrowLen;

  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
      {/* Circle */}
      <circle cx={cx} cy={cy} r={size * 0.42} fill="none" stroke="currentColor" strokeWidth="1.5" opacity="0.3" />
      {/* N/E/S/W markers */}
      <text x={cx} y={size * 0.12} textAnchor="middle" fontSize="10" fontWeight="bold" fill="currentColor">N</text>
      <text x={size * 0.88} y={cy + 4} textAnchor="middle" fontSize="10" fill="currentColor" opacity="0.5">E</text>
      <text x={cx} y={size * 0.94} textAnchor="middle" fontSize="10" fill="currentColor" opacity="0.5">S</text>
      <text x={size * 0.12} y={cy + 4} textAnchor="middle" fontSize="10" fill="currentColor" opacity="0.5">W</text>
      {/* Wind direction arrow */}
      <line x1={cx} y1={cy} x2={cx + dx} y2={cy + dy} stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" />
      <polygon
        points={`${cx + dx},${cy + dy} ${cx + dx - 4},${cy + dy - 4} ${cx + dx + 4},${cy + dy - 4}`}
        fill="currentColor"
      />
      {/* Center dot */}
      <circle cx={cx} cy={cy} r="3" fill="currentColor" />
    </svg>
  );
}

function StatCard({
  icon: Icon,
  label,
  value,
  unit,
  color = 'text-foreground',
  sublabel,
}: {
  icon: React.ElementType;
  label: string;
  value: string | number;
  unit?: string;
  color?: string;
  sublabel?: string;
}) {
  return (
    <div className="rounded-xl border bg-card p-4 shadow-sm">
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-muted-foreground uppercase tracking-wide">{label}</span>
        <Icon className="h-4 w-4 text-muted-foreground" />
      </div>
      <div className="mt-2 flex items-baseline gap-1">
        <span className={`text-2xl font-bold ${color}`}>{value}</span>
        {unit && <span className="text-sm text-muted-foreground">{unit}</span>}
      </div>
      {sublabel && <div className="mt-1 text-xs text-muted-foreground">{sublabel}</div>}
    </div>
  );
}

function PointCard({ pred, onClick, isSelected }: { pred: Prediction; onClick: () => void; isSelected: boolean }) {
  const speed = pred.wind_speed_kn ?? 0;
  const dir = pred.wind_dir_deg ?? 0;
  const gust = pred.wind_gust_kn ?? 0;
  const conf = pred.confidence_pct ?? 0;
  const err = pred.expected_error_kn ?? 0;

  return (
    <button
      onClick={onClick}
      className={`w-full rounded-xl border p-4 text-left transition-all hover:shadow-md ${
        isSelected ? 'border-primary ring-2 ring-primary/20' : 'border-border bg-card'
      }`}
    >
      <div className="flex items-start justify-between">
        <div>
          <div className="text-sm font-semibold">{pred.point_id.replace(/_/g, ' ')}</div>
          <div className="mt-1 flex items-baseline gap-1">
            <span className={`text-2xl font-bold ${speedColor(speed)}`}>{speed.toFixed(1)}</span>
            <span className="text-xs text-muted-foreground">kn</span>
          </div>
          <div className="mt-0.5 text-xs text-muted-foreground">
            {degToCardinal(dir)} ({dir.toFixed(0)}°) · Gust {gust.toFixed(1)}
          </div>
        </div>
        <div className="flex flex-col items-center">
          <WindCompass direction={dir} size={56} />
        </div>
      </div>
      <div className="mt-3 flex items-center justify-between text-xs">
        <span className={confidenceColor(conf)}>Conf {conf.toFixed(0)}%</span>
        <span className="text-muted-foreground">±{err.toFixed(1)} kn</span>
      </div>
      {/* Confidence bar */}
      <div className="mt-1.5 h-1.5 w-full overflow-hidden rounded-full bg-muted">
        <div
          className={`h-full ${conf >= 75 ? 'bg-green-500' : conf >= 50 ? 'bg-yellow-500' : 'bg-red-500'}`}
          style={{ width: `${conf}%` }}
        />
      </div>
    </button>
  );
}

function TrendChart({ data }: { data: TrendPoint[] }) {
  if (!data.length) {
    return (
      <div className="flex h-64 items-center justify-center text-muted-foreground">
        <Activity className="mr-2 h-5 w-5" />
        Loading trend data...
      </div>
    );
  }

  const chartData = data.map((d) => ({
    time: new Date(d.time).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }),
    speed: d.wind_speed_kn,
    gust: d.wind_gust_kn,
    confidence: d.confidence_pct,
  }));

  return (
    <ResponsiveContainer width="100%" height={280}>
      <AreaChart data={chartData} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
        <defs>
          <linearGradient id="speedGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#2b83ba" stopOpacity={0.8} />
            <stop offset="95%" stopColor="#2b83ba" stopOpacity={0.1} />
          </linearGradient>
          <linearGradient id="gustGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#d73027" stopOpacity={0.6} />
            <stop offset="95%" stopColor="#d73027" stopOpacity={0.05} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" vertical={false} />
        <XAxis dataKey="time" tick={{ fontSize: 11 }} interval="preserveStartEnd" />
        <YAxis tick={{ fontSize: 11 }} label={{ value: 'kn', angle: -90, position: 'insideLeft', style: { fontSize: 11 } }} />
        <Tooltip
          contentStyle={{ borderRadius: '8px', border: '1px solid #e0e0e0', fontSize: '12px' }}
          formatter={(value: number) => [`${value?.toFixed(1)} kn`, '']}
        />
        <ReferenceLine y={8} stroke="#22c55e" strokeDasharray="3 3" label={{ value: 'Sailing', fontSize: 10, fill: '#22c55e' }} />
        <Area type="monotone" dataKey="gust" stroke="#d73027" strokeWidth={1.5} fill="url(#gustGradient)" name="Gust" />
        <Area type="monotone" dataKey="speed" stroke="#2b83ba" strokeWidth={2} fill="url(#speedGradient)" name="Speed" />
        <Legend wrapperStyle={{ fontSize: '11px' }} />
      </AreaChart>
    </ResponsiveContainer>
  );
}

function DirectionChart({ data }: { data: TrendPoint[] }) {
  if (!data.length) return null;
  const chartData = data.map((d) => ({
    time: new Date(d.time).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }),
    dir: d.wind_dir_deg,
  }));

  return (
    <ResponsiveContainer width="100%" height={160}>
      <LineChart data={chartData} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" vertical={false} />
        <XAxis dataKey="time" tick={{ fontSize: 11 }} interval="preserveStartEnd" />
        <YAxis domain={[0, 360]} ticks={[0, 90, 180, 270, 360]} tickFormatter={(v) => degToCardinal(v)} tick={{ fontSize: 10 }} />
        <Tooltip formatter={(value: number) => [`${degToCardinal(value)} (${value?.toFixed(0)}°)`, 'Direction']} contentStyle={{ borderRadius: '8px', fontSize: '12px' }} />
        <Line type="monotone" dataKey="dir" stroke="#abdda4" strokeWidth={2} dot={false} />
      </LineChart>
    </ResponsiveContainer>
  );
}

function HealthBadge({ source, ok, latency }: { source: string; ok: boolean; latency: number }) {
  return (
    <div className="flex items-center gap-2 rounded-lg border px-3 py-1.5 text-xs">
      {ok ? <CheckCircle2 className="h-3.5 w-3.5 text-green-500" /> : <XCircle className="h-3.5 w-3.5 text-red-500" />}
      <span className="font-medium">{source}</span>
      <span className="text-muted-foreground">{latency.toFixed(0)}ms</span>
    </div>
  );
}

// --- Main Page ---

export default function LakeWindDashboard() {
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [points, setPoints] = useState<PointInfo[]>([]);
  const [health, setHealth] = useState<HealthInfo[]>([]);
  const [trendData, setTrendData] = useState<TrendPoint[]>([]);
  const [selectedHorizon, setSelectedHorizon] = useState(0);
  const [selectedPoint, setSelectedPoint] = useState<string>('mid_channel');
  const [loading, setLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());

  const fetchWind = useCallback(async (horizon: number) => {
    try {
      const res = await fetch(`/api/wind?horizon=${horizon}`);
      const data = await res.json();
      if (data.status === 'ok') {
        setPredictions(data.predictions);
      }
    } catch (err) {
      console.error('Failed to fetch wind:', err);
    }
  }, []);

  const fetchHealth = useCallback(async () => {
    try {
      const res = await fetch('/api/health');
      const data = await res.json();
      if (data.status === 'ok') {
        setHealth(data.sources || []);
      }
    } catch (err) {
      console.error('Failed to fetch health:', err);
    }
  }, []);

  const fetchTrend = useCallback(async (point: string) => {
    try {
      const res = await fetch(`/api/trend?point=${point}&hours=24`);
      const data = await res.json();
      if (data.status === 'ok') {
        setTrendData(data.data || []);
      }
    } catch (err) {
      console.error('Failed to fetch trend:', err);
    }
  }, []);

  const fetchPoints = useCallback(async () => {
    try {
      const res = await fetch('/api/points');
      const data = await res.json();
      if (data.status === 'ok') {
        setPoints(data.points || []);
      }
    } catch (err) {
      console.error('Failed to fetch points:', err);
    }
  }, []);

  // Initial load
  useEffect(() => {
    let cancelled = false;
    (async () => {
      setLoading(true);
      await Promise.all([fetchPoints(), fetchWind(selectedHorizon), fetchHealth(), fetchTrend(selectedPoint)]);
      if (!cancelled) {
        setLoading(false);
        setLastUpdate(new Date());
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // Refetch when horizon changes
  useEffect(() => {
    let cancelled = false;
    (async () => {
      await fetchWind(selectedHorizon);
      if (!cancelled) setLastUpdate(new Date());
    })();
    return () => {
      cancelled = true;
    };
  }, [selectedHorizon, fetchWind]);

  // Refetch trend when point changes
  useEffect(() => {
    let cancelled = false;
    (async () => {
      await fetchTrend(selectedPoint);
      if (!cancelled) {
        // no setState needed — fetchTrend updates trendData internally
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [selectedPoint, fetchTrend]);

  // Auto-refresh every 5 min
  useEffect(() => {
    const interval = setInterval(() => {
      fetchWind(selectedHorizon);
      fetchHealth();
      setLastUpdate(new Date());
    }, 300000);
    return () => clearInterval(interval);
  }, [selectedHorizon, fetchWind, fetchHealth]);

  // Group predictions by sector
  const pointsBySector = points.reduce((acc, p) => {
    if (!p.is_operational) return acc;
    if (!acc[p.sector]) acc[p.sector] = [];
    acc[p.sector].push(p);
    return acc;
  }, {} as Record<string, PointInfo[]>);

  const selectedPred = predictions.find((p) => p.point_id === selectedPoint) || predictions[0];
  const bestPoint = predictions
    .filter((p) => p.wind_speed_kn !== null && (p.confidence_pct ?? 0) >= 50)
    .sort((a, b) => (b.wind_speed_kn ?? 0) - (a.wind_speed_kn ?? 0))[0];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50 dark:from-slate-950 dark:to-slate-900">
      {/* Header */}
      <header className="sticky top-0 z-50 border-b bg-white/80 backdrop-blur-md dark:bg-slate-950/80">
        <div className="mx-auto max-w-7xl px-4 py-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-blue-500 to-cyan-500 text-white shadow-lg">
                <Waves className="h-6 w-6" />
              </div>
              <div>
                <h1 className="text-xl font-bold tracking-tight">LakeWind AI</h1>
                <p className="text-xs text-muted-foreground">
                  Hyperlocal wind forecasting · Dongo-Dervio · Lake Como
                </p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <div className="hidden items-center gap-1.5 text-xs text-muted-foreground sm:flex">
                <Clock className="h-3.5 w-3.5" />
                Updated {lastUpdate.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })}
              </div>
              <button
                onClick={() => {
                  fetchWind(selectedHorizon);
                  fetchHealth();
                  setLastUpdate(new Date());
                }}
                className="rounded-lg border p-2 hover:bg-muted"
                aria-label="Refresh"
              >
                <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
              </button>
            </div>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-4 py-6">
        {/* Hero stats */}
        {selectedPred && (
          <section className="mb-6 grid grid-cols-2 gap-4 md:grid-cols-4">
            <StatCard
              icon={Wind}
              label="Wind Speed"
              value={selectedPred.wind_speed_kn?.toFixed(1) ?? '—'}
              unit="kn"
              color={speedColor(selectedPred.wind_speed_kn ?? 0)}
              sublabel={`Gust ${(selectedPred.wind_gust_kn ?? 0).toFixed(1)} kn`}
            />
            <StatCard
              icon={Compass}
              label="Direction"
              value={degToCardinal(selectedPred.wind_dir_deg ?? 0)}
              color="text-foreground"
              sublabel={`${(selectedPred.wind_dir_deg ?? 0).toFixed(0)}°`}
            />
            <StatCard
              icon={Gauge}
              label="Confidence"
              value={`${(selectedPred.confidence_pct ?? 0).toFixed(0)}`}
              unit="%"
              color={confidenceColor(selectedPred.confidence_pct ?? 0)}
              sublabel={`±${(selectedPred.expected_error_kn ?? 0).toFixed(1)} kn`}
            />
            <StatCard
              icon={Zap}
              label="Best Spot Now"
              value={bestPoint?.point_id.replace(/_/g, ' ') ?? '—'}
              color="text-green-600"
              sublabel={bestPoint ? `${bestPoint.wind_speed_kn?.toFixed(1)} kn` : ''}
            />
          </section>
        )}

        {/* Horizon selector */}
        <section className="mb-6">
          <div className="flex items-center gap-2 overflow-x-auto pb-2">
            {HORIZONS.map((h) => (
              <button
                key={h.hours}
                onClick={() => setSelectedHorizon(h.hours)}
                className={`flex-shrink-0 rounded-lg px-4 py-2 text-sm font-medium transition-all ${
                  selectedHorizon === h.hours
                    ? 'bg-primary text-primary-foreground shadow-md'
                    : 'bg-card border hover:bg-muted'
                }`}
              >
                {h.label}
              </button>
            ))}
          </div>
        </section>

        {/* Interactive Map */}
        <section className="mb-6">
          <div className="mb-3 flex items-center justify-between">
            <h2 className="text-lg font-semibold flex items-center gap-2">
              <MapPin className="h-5 w-5 text-primary" />
              Wind Map
            </h2>
            <span className="text-sm text-muted-foreground">
              {predictions.filter(p => p.wind_speed_kn !== null).length} active points
            </span>
          </div>
          <WindMap
            points={predictions
              .filter((p) => p.wind_speed_kn !== null)
              .map((p) => {
                const ptInfo = points.find((pt) => pt.id === p.point_id);
                return {
                  id: p.point_id,
                  lat: ptInfo?.lat || 46.1,
                  lon: ptInfo?.lon || 9.3,
                  speed: p.wind_speed_kn || 0,
                  direction: p.wind_dir_deg || 0,
                  gust: p.wind_gust_kn || 0,
                  confidence: p.confidence_pct || 0,
                  sector: ptInfo?.sector || 'Unknown',
                };
              })}
          />
        </section>

        {/* Main grid */}
        <div className="grid gap-6 lg:grid-cols-3">
          {/* Left: Points list */}
          <section className="lg:col-span-2">
            <div className="mb-3 flex items-center justify-between">
              <h2 className="text-lg font-semibold">
                All Points
                <span className="ml-2 text-sm font-normal text-muted-foreground">
                  {predictions.length} points · {selectedHorizon}h ahead
                </span>
              </h2>
            </div>

            {Object.entries(pointsBySector).map(([sector, sectorPoints]) => (
              <div key={sector} className="mb-4">
                <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                  {sector}
                </h3>
                <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
                  {sectorPoints.map((pt) => {
                    const pred = predictions.find((p) => p.point_id === pt.id);
                    if (!pred) {
                      return (
                        <div key={pt.id} className="rounded-xl border border-dashed p-4 text-center text-sm text-muted-foreground">
                          {pt.id.replace(/_/g, ' ')}
                          <div className="mt-1 text-xs">No data</div>
                        </div>
                      );
                    }
                    return (
                      <PointCard
                        key={pt.id}
                        pred={pred}
                        onClick={() => setSelectedPoint(pt.id)}
                        isSelected={selectedPoint === pt.id}
                      />
                    );
                  })}
                </div>
              </div>
            ))}
          </section>

          {/* Right: Trend chart + details */}
          <section className="space-y-6">
            {/* Selected point detail */}
            {selectedPred && (
              <div className="rounded-xl border bg-card p-5 shadow-sm">
                <div className="mb-3 flex items-center justify-between">
                  <h3 className="font-semibold">
                    {selectedPoint.replace(/_/g, ' ')}
                  </h3>
                  <span className="text-xs text-muted-foreground">24h forecast</span>
                </div>

                {/* Wind compass large */}
                <div className="mb-4 flex justify-center">
                  <div className="flex flex-col items-center">
                    <WindCompass direction={selectedPred.wind_dir_deg ?? 0} size={100} />
                    <div className="mt-2 text-center">
                      <div className={`text-3xl font-bold ${speedColor(selectedPred.wind_speed_kn ?? 0)}`}>
                        {selectedPred.wind_speed_kn?.toFixed(1)}
                      </div>
                      <div className="text-xs text-muted-foreground">kn · {degToCardinal(selectedPred.wind_dir_deg ?? 0)}</div>
                    </div>
                  </div>
                </div>

                {/* Trend chart */}
                <div className="mt-4">
                  <div className="mb-2 flex items-center gap-2 text-xs font-medium text-muted-foreground">
                    <TrendingUp className="h-3.5 w-3.5" />
                    WIND TREND (24H)
                  </div>
                  <TrendChart data={trendData} />
                </div>

                {/* Direction chart */}
                <div className="mt-4">
                  <div className="mb-2 flex items-center gap-2 text-xs font-medium text-muted-foreground">
                    <Navigation className="h-3.5 w-3.5" />
                    DIRECTION
                  </div>
                  <DirectionChart data={trendData} />
                </div>
              </div>
            )}

            {/* Data source health */}
            <div className="rounded-xl border bg-card p-5 shadow-sm">
              <div className="mb-3 flex items-center gap-2">
                <Activity className="h-4 w-4 text-muted-foreground" />
                <h3 className="text-sm font-semibold">Data Sources</h3>
              </div>
              <div className="flex flex-wrap gap-2">
                {health.length === 0 ? (
                  <span className="text-xs text-muted-foreground">Loading...</span>
                ) : (
                  health.map((h) => (
                    <HealthBadge
                      key={h.source}
                      source={h.source}
                      ok={h.ok}
                      latency={h.latency_ms}
                    />
                  ))
                )}
              </div>
            </div>
          </section>
        </div>

        {/* Footer */}
        <footer className="mt-12 border-t pt-6 text-center text-xs text-muted-foreground">
          <p>
            LakeWind AI · MOS bias-corrected forecasts · 15 virtual points · Open-Meteo + ERA5 + ARPA + Domaso + CML
          </p>
          <p className="mt-1">
            Model: {selectedPred?.model_version ?? '—'} · Generated: {selectedPred ? new Date(selectedPred.generated_at).toLocaleString('en-US') : '—'}
          </p>
        </footer>
      </main>
    </div>
  );
}
