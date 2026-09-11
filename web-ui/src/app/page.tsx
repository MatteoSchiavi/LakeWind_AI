'use client';

/**
 * LakeWind dashboard — Phase 4 (W1-W4) full redesign.
 *
 * What changed vs the pre-Phase-4 page (audit findings F1-F7, F10):
 *  W1  uncertainty: shaded 80% band in the trend chart, `12.4 (10.1–14.8)`
 *      ranges on point cards and the hero stat — the conformal calibration
 *      work is finally visible;
 *  W2  decision surface: "Go sailing?" hero card fed by the shared decision
 *      module via /api/decision (same math as the bot's /sailing);
 *  W3  spatial view: interactive Leaflet map (click marker = select point)
 *      + precomputed model heatmap tab with freshness stamp;
 *  W4  UX states: skeleton loaders, error banner + retry, stale-data
 *      banner (pipeline freshness), ?point=&h= URL state, dark-mode toggle,
 *      en/it language toggle, PWA manifest + iOS meta (layout.tsx).
 */
import { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import dynamic from 'next/dynamic';
import {
  Wind, Navigation, TrendingUp, Activity, Clock,
  Zap, Gauge, RefreshCw, Compass, Waves, AlertCircle,
  CheckCircle2, XCircle, Sailboat, Moon, Sun, Languages,
  Map as MapIcon, ChevronDown,
} from 'lucide-react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Area, AreaChart, ReferenceLine, Legend,
} from 'recharts';
import { speedColor, confColor, speedLabel } from '@/lib/palette';
import { t, type Lang } from '@/lib/i18n';

// react-leaflet touches window at import time — client-only.
const WindMap = dynamic(() => import('./WindMap'), {
  ssr: false,
  loading: () => (
    <div className="flex h-[420px] items-center justify-center rounded-xl border bg-card text-sm text-muted-foreground">
      <Activity className="mr-2 h-5 w-5 animate-pulse" /> Map…
    </div>
  ),
});

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
  wind_speed_q10_kn?: number | null;
  wind_speed_q90_kn?: number | null;
  regime?: string | null;
}
interface PointInfo { id: string; lat: number; lon: number; sector: string; label?: string; is_operational: boolean; }
interface HealthInfo { source: string; ok: boolean; latency_ms: number; }
interface TrendPoint {
  time: number; valid_time: string;
  wind_speed_kn: number | null; wind_dir_deg: number | null;
  wind_gust_kn: number | null; confidence_pct: number | null;
  wind_speed_q10_kn?: number | null; wind_speed_q90_kn?: number | null;
  regime?: string | null;
}
interface DecisionHour {
  hour: number | null; valid_time: string | null;
  speed_kn: number; q10_kn: number | null; q90_kn: number | null;
  dir_deg: number | null; p_go: number; p_strong: number; regime: string | null;
}
interface Decision {
  point_id: string;
  verdict: 'go' | 'marginal' | 'no_go';
  best_hour: number | null;
  best_speed_kn: number | null;
  n_go_hours: number;
  peak_p_go: number;
  regime: string | null;
  hours: DecisionHour[];
}

const HORIZONS = [
  { hours: 0, label: 'Now' }, { hours: 1, label: '+1h' },
  { hours: 3, label: '+3h' }, { hours: 6, label: '+6h' },
  { hours: 12, label: '+12h' }, { hours: 24, label: '+24h' },
];
const CARDINALS = ['N','NNE','NE','ENE','E','ESE','SE','SSE','S','SSW','SW','WSW','W','WNW','NW','NNW'];

function degToCardinal(deg: number): string {
  return CARDINALS[Math.round(deg / 22.5) % 16];
}

const REGIME_ICON: Record<string, string> = {
  breva: '🌤', tivano: '❄️', foehn: '🌡', storm: '⛈', calm: '😌',
};

// --- Wind Compass Component ---
function WindCompass({ direction, size = 80 }: { direction: number; size?: number }) {
  const goRad = ((direction + 180) % 360) * Math.PI / 180;
  const cx = size / 2, cy = size / 2;
  const arrowLen = size * 0.35;
  const dx = Math.sin(goRad) * arrowLen;
  const dy = -Math.cos(goRad) * arrowLen;
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className="text-foreground">
      <circle cx={cx} cy={cy} r={size * 0.42} fill="none" stroke="currentColor" strokeWidth="1.5" opacity="0.2" />
      <text x={cx} y={size * 0.12} textAnchor="middle" fontSize="10" fontWeight="bold" fill="currentColor">N</text>
      <text x={size * 0.88} y={cy + 4} textAnchor="middle" fontSize="9" fill="currentColor" opacity="0.4">E</text>
      <text x={cx} y={size * 0.94} textAnchor="middle" fontSize="9" fill="currentColor" opacity="0.4">S</text>
      <text x={size * 0.12} y={cy + 4} textAnchor="middle" fontSize="9" fill="currentColor" opacity="0.4">W</text>
      <line x1={cx} y1={cy} x2={cx + dx} y2={cy + dy} stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" />
      <polygon points={`${cx+dx},${cy+dy} ${cx+dx-4},${cy+dy-4} ${cx+dx+4},${cy+dy-4}`} fill="currentColor" />
      <circle cx={cx} cy={cy} r="3" fill="currentColor" />
    </svg>
  );
}

// --- Stat Card ---
function StatCard({ icon: Icon, label, value, unit, color, sublabel }: {
  icon: React.ElementType; label: string; value: string | number;
  unit?: string; color?: string; sublabel?: string;
}) {
  return (
    <div className="rounded-xl border bg-card p-4 shadow-sm transition-shadow hover:shadow-md">
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-muted-foreground uppercase tracking-wide">{label}</span>
        <Icon className="h-4 w-4 text-muted-foreground" />
      </div>
      <div className="mt-2 flex items-baseline gap-1">
        <span className="text-2xl font-bold" style={color ? { color } : undefined}>{value}</span>
        {unit && <span className="text-sm text-muted-foreground">{unit}</span>}
      </div>
      {sublabel && <div className="mt-1 text-xs text-muted-foreground">{sublabel}</div>}
    </div>
  );
}

// --- Skeleton Card ---
function SkeletonCard() {
  return (
    <div className="animate-pulse rounded-xl border bg-card p-4">
      <div className="h-3 w-24 rounded bg-muted" />
      <div className="mt-3 h-7 w-16 rounded bg-muted" />
      <div className="mt-2 h-3 w-28 rounded bg-muted" />
      <div className="mt-3 h-1.5 w-full rounded-full bg-muted" />
    </div>
  );
}

// --- Point Card ---
function PointCard({ pred, onClick, isSelected, lang }: {
  pred: Prediction; onClick: () => void; isSelected: boolean; lang: Lang;
}) {
  const speed = pred.wind_speed_kn ?? 0;
  const dir = pred.wind_dir_deg ?? 0;
  const gust = pred.wind_gust_kn ?? 0;
  const conf = pred.confidence_pct ?? 0;
  const err = pred.expected_error_kn ?? 0;
  const q10 = pred.wind_speed_q10_kn;
  const q90 = pred.wind_speed_q90_kn;
  const hasBand = q10 !== null && q10 !== undefined && q90 !== null && q90 !== undefined;
  return (
    <button onClick={onClick}
      className={`w-full rounded-xl border p-4 text-left transition-all hover:shadow-md ${
        isSelected ? 'border-primary ring-2 ring-primary/20' : 'border-border bg-card'
      }`}>
      <div className="flex items-start justify-between">
        <div>
          <div className="text-sm font-semibold">{pred.point_id.replace(/_/g, ' ')}</div>
          <div className="mt-1 flex items-baseline gap-1">
            <span className="text-2xl font-bold" style={{ color: speedColor(speed) }}>{speed.toFixed(1)}</span>
            {hasBand && (
              <span className="text-xs text-muted-foreground">
                ({q10!.toFixed(1)}–{q90!.toFixed(1)})
              </span>
            )}
            <span className="text-xs text-muted-foreground">kn</span>
          </div>
          <div className="mt-0.5 text-xs text-muted-foreground">
            {speedLabel(speed, lang)} · {degToCardinal(dir)} ({dir.toFixed(0)}°) · {t(lang, 'gust')} {gust.toFixed(1)}
          </div>
        </div>
        <WindCompass direction={dir} size={56} />
      </div>
      <div className="mt-3 flex items-center justify-between text-xs">
        <span style={{ color: confColor(conf) }}>{t(lang, 'confidence')} {conf.toFixed(0)}%</span>
        <span className="text-muted-foreground">±{err.toFixed(1)} kn</span>
      </div>
      <div className="mt-1.5 h-1.5 w-full overflow-hidden rounded-full bg-muted">
        <div className="h-full" style={{
          width: `${conf}%`,
          backgroundColor: confColor(conf),
        }} />
      </div>
    </button>
  );
}

// --- Trend Chart (with calibrated 80% band) ---
function TrendChart({ data, lang }: { data: TrendPoint[]; lang: Lang }) {
  const chartData = useMemo(() => data.map(d => ({
    time: new Date(d.time).toLocaleTimeString(lang === 'it' ? 'it-IT' : 'en-US', { hour: '2-digit', minute: '2-digit' }),
    speed: d.wind_speed_kn, gust: d.wind_gust_kn,
    q10: d.wind_speed_q10_kn ?? null, q90: d.wind_speed_q90_kn ?? null,
  })), [data, lang]);

  if (!chartData.length) return (
    <div className="flex h-64 items-center justify-center text-muted-foreground">
      <Activity className="mr-2 h-5 w-5" /> {t(lang, 'loading')}
    </div>
  );

  const hasBand = chartData.some(d => d.q10 !== null && d.q90 !== null);

  return (
    <div>
      <ResponsiveContainer width="100%" height={260}>
        <AreaChart data={chartData} margin={{ top: 5, right: 20, bottom: 5, left: -10 }}>
          <defs>
            <linearGradient id="speedG" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.8} />
              <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.1} />
            </linearGradient>
            <linearGradient id="gustG" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#dc2626" stopOpacity={0.5} />
              <stop offset="95%" stopColor="#dc2626" stopOpacity={0.05} />
            </linearGradient>
            <linearGradient id="bandG" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#3b82f6" stopOpacity={0.18} />
              <stop offset="100%" stopColor="#3b82f6" stopOpacity={0.18} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#94a3b8" strokeOpacity={0.35} vertical={false} />
          <XAxis dataKey="time" tick={{ fontSize: 11 }} interval="preserveStartEnd" />
          <YAxis tick={{ fontSize: 11 }} />
          <Tooltip contentStyle={{ borderRadius: '8px', border: '1px solid #94a3b8', fontSize: '12px' }}
            formatter={(v: number, name: string) => {
              const labels: Record<string, string> = {
                speed: lang === 'it' ? 'Vento' : 'Speed',
                gust: t(lang, 'gust'),
                band: t(lang, 'range'),
              };
              return [`${v?.toFixed(1)} kn`, labels[name] ?? name];
            }} />
          <ReferenceLine y={8} stroke="#22c55e" strokeDasharray="3 3"
            label={{ value: '⛵ 8', fontSize: 10, fill: '#22c55e', position: 'insideTopLeft' }} />
          {hasBand && (
            // Range area: array dataKey = recharts range band (q10–q90).
            // Runtime is supported since recharts 2.x; the type defs lag
            // behind, hence the cast.
            <Area
              type="monotone"
              dataKey={['q10', 'q90'] as unknown as string}
              stroke="none"
              fill="url(#bandG)"
              name="band"
            />
          )}
          <Area type="monotone" dataKey="gust" stroke="#dc2626" strokeWidth={1} fill="url(#gustG)" name="gust" />
          <Area type="monotone" dataKey="speed" stroke="#3b82f6" strokeWidth={2} fill="url(#speedG)" name="speed" />
          <Legend wrapperStyle={{ fontSize: '11px' }} />
        </AreaChart>
      </ResponsiveContainer>
      {hasBand && (
        <div className="mt-1 text-center text-[10px] text-muted-foreground">{t(lang, 'bandNote')}</div>
      )}
    </div>
  );
}

// --- Direction Chart ---
function DirectionChart({ data, lang }: { data: TrendPoint[]; lang: Lang }) {
  const chartData = data.map(d => ({
    time: new Date(d.time).toLocaleTimeString(lang === 'it' ? 'it-IT' : 'en-US', { hour: '2-digit', minute: '2-digit' }),
    dir: d.wind_dir_deg,
  }));
  if (!chartData.length) return null;
  return (
    <ResponsiveContainer width="100%" height={140}>
      <LineChart data={chartData} margin={{ top: 5, right: 20, bottom: 5, left: -10 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#94a3b8" strokeOpacity={0.35} vertical={false} />
        <XAxis dataKey="time" tick={{ fontSize: 11 }} interval="preserveStartEnd" />
        <YAxis domain={[0, 360]} ticks={[0, 90, 180, 270, 360]}
          tickFormatter={(v) => degToCardinal(v)} tick={{ fontSize: 10 }} />
        <Tooltip formatter={(v: number) => [`${degToCardinal(v)} (${v?.toFixed(0)}°)`, 'Dir']}
          contentStyle={{ borderRadius: '8px', fontSize: '12px' }} />
        <Line type="monotone" dataKey="dir" stroke="#22c55e" strokeWidth={2} dot={false} />
      </LineChart>
    </ResponsiveContainer>
  );
}

// --- Sailing Decision Card (W2) ---
function DecisionCard({ decision, lang }: { decision: Decision | null; lang: Lang }) {
  if (!decision) return null;
  const verdictText = decision.verdict === 'go' ? t(lang, 'verdictGo')
    : decision.verdict === 'marginal' ? t(lang, 'verdictMarginal') : t(lang, 'verdictNoGo');
  const verdictColor = decision.verdict === 'go' ? '#22c55e'
    : decision.verdict === 'marginal' ? '#f59e0b' : '#dc2626';
  const regimeIcon = decision.regime ? (REGIME_ICON[decision.regime] ?? '🌊') : '🌊';
  const regimeName = decision.regime ? decision.regime.charAt(0).toUpperCase() + decision.regime.slice(1) : null;

  return (
    <div className="rounded-xl border bg-card p-5 shadow-sm">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Sailboat className="h-5 w-5 text-muted-foreground" />
          <h3 className="font-semibold">{t(lang, 'sailing')}</h3>
        </div>
        {regimeName && (
          <span className="rounded-full border px-2.5 py-0.5 text-xs font-medium text-muted-foreground">
            {regimeIcon} {regimeName}
          </span>
        )}
      </div>
      <div className="mt-3 flex items-end justify-between">
        <div>
          <div className="text-2xl font-bold" style={{ color: verdictColor }}>{verdictText}</div>
          <div className="mt-1 text-xs text-muted-foreground">
            {decision.best_hour !== null && (
              <>
                {t(lang, 'bestWindow')}: <span className="font-semibold text-foreground">{decision.best_hour}:00</span>
                {' · '}{decision.best_speed_kn?.toFixed(1)} kn ·{' '}
              </>
            )}
            {t(lang, 'chanceOfWind')} {Math.round(decision.peak_p_go * 100)}%
          </div>
        </div>
      </div>
      {/* Per-hour probability bars */}
      <div className="mt-4 space-y-1.5">
        {decision.hours
          .filter(h => h.hour !== null && h.p_go > 0)
          .slice(0, 10)
          .map(h => (
            <div key={h.hour} className="flex items-center gap-2 text-xs">
              <span className="w-10 shrink-0 text-muted-foreground">{h.hour}:00</span>
              <div className="h-2.5 flex-1 overflow-hidden rounded-full bg-muted">
                <div className="h-full rounded-full" style={{
                  width: `${Math.round(h.p_go * 100)}%`,
                  backgroundColor: h.p_go >= 0.5 ? '#22c55e' : h.p_go >= 0.25 ? '#f59e0b' : '#94a3b8',
                }} />
              </div>
              <span className="w-20 shrink-0 text-right text-muted-foreground">
                {Math.round(h.p_go * 100)}% · {h.speed_kn.toFixed(1)} kn
              </span>
            </div>
          ))}
      </div>
    </div>
  );
}

// --- Health Badge ---
function HealthBadge({ source, ok, latency }: { source: string; ok: boolean; latency: number }) {
  return (
    <div className="flex items-center gap-2 rounded-lg border px-3 py-1.5 text-xs">
      {ok ? <CheckCircle2 className="h-3.5 w-3.5 text-green-500" /> : <XCircle className="h-3.5 w-3.5 text-red-500" />}
      <span className="font-medium">{source}</span>
      <span className="text-muted-foreground">{latency.toFixed(0)}ms</span>
    </div>
  );
}

// --- Main Dashboard ---
export default function LakeWindDashboard() {
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [points, setPoints] = useState<PointInfo[]>([]);
  const [health, setHealth] = useState<HealthInfo[]>([]);
  const [trendData, setTrendData] = useState<TrendPoint[]>([]);
  const [decisions, setDecisions] = useState<Record<string, Decision>>({});
  const [selectedHorizon, setSelectedHorizon] = useState(0);
  // Phase 5.5: default spot is dongo — the product's home shore
  const [selectedPoint, setSelectedPoint] = useState('dongo');
  const [loading, setLoading] = useState(true);
  const [fetchError, setFetchError] = useState<string | null>(null);
  const [lastGeneration, setLastGeneration] = useState<string | null>(null);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);
  const [lang, setLang] = useState<Lang>('en');
  const [dark, setDark] = useState(false);
  const [mapTab, setMapTab] = useState<'interactive' | 'heatmap'>('interactive');
  const [mapUrl, setMapUrl] = useState<string | null>(null);
  const [mapSource, setMapSource] = useState<string | null>(null);
  const [mapLoading, setMapLoading] = useState(false);
  const bootedRef = useRef(false);

  // --- URL state + persisted prefs on mount (W4: shareable ?point=&h=) ---
  useEffect(() => {
    if (bootedRef.current) return;
    bootedRef.current = true;
    const params = new URLSearchParams(window.location.search);
    const p = params.get('point');
    const h = parseInt(params.get('h') || '0', 10);
    const l = params.get('lang');
    if (p) setSelectedPoint(p);
    if (!Number.isNaN(h) && HORIZONS.some(x => x.hours === h)) setSelectedHorizon(h);
    const storedLang = (l === 'it' || l === 'en' ? l : localStorage.getItem('lw_lang')) as Lang | null;
    if (storedLang === 'it' || storedLang === 'en') setLang(storedLang);
    const storedTheme = localStorage.getItem('lw_theme');
    const prefersDark = storedTheme === 'dark' || (
      storedTheme === null && window.matchMedia('(prefers-color-scheme: dark)').matches
    );
    setDark(prefersDark);
    document.documentElement.classList.toggle('dark', prefersDark);
  }, []);

  const updateUrl = useCallback((point: string, horizon: number) => {
    const params = new URLSearchParams();
    params.set('point', point);
    if (horizon !== 0) params.set('h', String(horizon));
    if (lang !== 'en') params.set('lang', lang);
    window.history.replaceState(null, '', `?${params.toString()}`);
  }, [lang]);

  const selectPoint = useCallback((pid: string) => {
    setSelectedPoint(pid);
    updateUrl(pid, selectedHorizon);
  }, [selectedHorizon, updateUrl]);

  const selectHorizon = useCallback((h: number) => {
    setSelectedHorizon(h);
    updateUrl(selectedPoint, h);
  }, [selectedPoint, updateUrl]);

  const toggleLang = useCallback(() => {
    setLang(prev => {
      const next: Lang = prev === 'en' ? 'it' : 'en';
      localStorage.setItem('lw_lang', next);
      updateUrl(selectedPoint, selectedHorizon);
      return next;
    });
  }, [selectedHorizon, selectedPoint, updateUrl]);

  const toggleDark = useCallback(() => {
    setDark(prev => {
      const next = !prev;
      localStorage.setItem('lw_theme', next ? 'dark' : 'light');
      document.documentElement.classList.toggle('dark', next);
      return next;
    });
  }, []);

  // --- Data fetching (W4: visible errors, never console-only) ---
  const fetchAll = useCallback(async (horizon: number, point: string) => {
    setFetchError(null);
    const results = await Promise.allSettled([
      fetch(`/api/wind?horizon=${horizon}`).then(r => r.json()),
      fetch('/api/health').then(r => r.json()),
      fetch(`/api/trend?point=${point}&hours=24`).then(r => r.json()),
      fetch(`/api/decision?point=${point}&hours=14`).then(r => r.json()),
      fetch('/api/points').then(r => r.json()),
    ]);
    let anyError = false;

    const [windR, healthR, trendR, decisionR, pointsR] = results;
    if (windR.status === 'fulfilled' && windR.value.status === 'ok') {
      setPredictions(windR.value.predictions ?? []);
    } else anyError = true;

    if (healthR.status === 'fulfilled') {
      const hv = healthR.value;
      setHealth(hv.sources ?? []);
      const gen = hv.cache?.last_generation ?? null;
      setLastGeneration(typeof gen === 'string' ? gen : null);
    }

    if (trendR.status === 'fulfilled' && trendR.value.status === 'ok') {
      setTrendData(trendR.value.data ?? []);
    }

    if (decisionR.status === 'fulfilled' && decisionR.value.status === 'ok') {
      setDecisions(decisionR.value.decisions ?? {});
    }

    if (pointsR.status === 'fulfilled' && pointsR.value.status === 'ok') {
      setPoints(pointsR.value.points ?? []);
    }

    if (anyError) setFetchError('api');
    setLastUpdate(new Date());
    setLoading(false);
  }, []);

  // Initial load
  useEffect(() => {
    setLoading(true);
    fetchAll(selectedHorizon, selectedPoint);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Refetch on horizon / point change
  useEffect(() => {
    if (!bootedRef.current) return;
    fetchAll(selectedHorizon, selectedPoint);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedHorizon, selectedPoint]);

  // Auto-refresh every 5 min
  useEffect(() => {
    const interval = setInterval(() => {
      fetchAll(selectedHorizon, selectedPoint);
    }, 300000);
    return () => clearInterval(interval);
  }, [selectedHorizon, selectedPoint, fetchAll]);

  // --- Precomputed heatmap fetch (blob, with provenance headers) ---
  useEffect(() => {
    if (mapTab !== 'heatmap') return;
    let objectUrl: string | null = null;
    setMapLoading(true);
    fetch(`/api/map?offset=${selectedHorizon}`)
      .then(res => {
        if (!res.ok) throw new Error(String(res.status));
        setMapSource(res.headers.get('X-Map-Source'));
        return res.blob();
      })
      .then(blob => {
        objectUrl = URL.createObjectURL(blob);
        setMapUrl(objectUrl);
      })
      .catch(() => setMapUrl(null))
      .finally(() => setMapLoading(false));
    return () => { if (objectUrl) URL.revokeObjectURL(objectUrl); };
  }, [mapTab, selectedHorizon]);

  const pointsBySector = points
    .filter(p => p.is_operational)
    .reduce((acc, p) => {
      if (!acc[p.sector]) acc[p.sector] = [];
      acc[p.sector].push(p);
      return acc;
    }, {} as Record<string, PointInfo[]>);

  const selectedPred = predictions.find(p => p.point_id === selectedPoint) || predictions[0];
  const bestPoint = predictions
    .filter(p => p.wind_speed_kn !== null && (p.confidence_pct ?? 0) >= 50)
    .sort((a, b) => (b.wind_speed_kn ?? 0) - (a.wind_speed_kn ?? 0))[0];
  const selectedDecision = decisions[selectedPoint] ?? null;

  // Best-of-all-points decision for the hero badge row
  const bestDecision = useMemo(() => {
    const list = Object.values(decisions);
    if (!list.length) return null;
    return list.reduce((a, b) => (b.peak_p_go > a.peak_p_go ? b : a));
  }, [decisions]);

  // Stale detection (W4/F6): pipeline generation older than 3h => banner
  const staleMinutes = useMemo(() => {
    if (!lastGeneration) return null;
    const gen = new Date(lastGeneration).getTime();
    if (Number.isNaN(gen)) return null;
    return Math.round((Date.now() - gen) / 60000);
  }, [lastGeneration]);
  const isStale = staleMinutes !== null && staleMinutes > 180;

  const mapPoints = useMemo(() => predictions
    .map(pr => {
      const info = points.find(pt => pt.id === pr.point_id);
      if (!info) return null;
      return {
        id: pr.point_id, lat: info.lat, lon: info.lon,
        label: info.label,
        speed: pr.wind_speed_kn ?? 0,
        q10: pr.wind_speed_q10_kn ?? null, q90: pr.wind_speed_q90_kn ?? null,
        direction: pr.wind_dir_deg ?? 0, gust: pr.wind_gust_kn ?? 0,
        confidence: pr.confidence_pct ?? 0, sector: info.sector,
      };
    })
    .filter((x): x is NonNullable<typeof x> => x !== null), [predictions, points]);

  const q10 = selectedPred?.wind_speed_q10_kn;
  const q90 = selectedPred?.wind_speed_q90_kn;
  const heroBand = q10 !== null && q10 !== undefined && q90 !== null && q90 !== undefined
    ? `${q10.toFixed(1)}–${q90.toFixed(1)} kn (80%)` : null;

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
                <p className="text-xs text-muted-foreground">{t(lang, 'subtitle')}</p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <div className="hidden items-center gap-1.5 text-xs text-muted-foreground sm:flex">
                <Clock className="h-3.5 w-3.5" />
                {lastUpdate?.toLocaleTimeString(lang === 'it' ? 'it-IT' : 'en-US', { hour: '2-digit', minute: '2-digit' }) ?? '—'}
              </div>
              <button onClick={toggleLang}
                className="flex items-center gap-1 rounded-lg border px-2.5 py-2 text-xs font-semibold hover:bg-muted"
                aria-label="Language">
                <Languages className="h-4 w-4" />
                {lang === 'en' ? 'IT' : 'EN'}
              </button>
              <button onClick={toggleDark} className="rounded-lg border p-2 hover:bg-muted"
                aria-label="Toggle dark mode">
                {dark ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
              </button>
              <button onClick={() => fetchAll(selectedHorizon, selectedPoint)}
                className="rounded-lg border p-2 hover:bg-muted" aria-label="Refresh">
                <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
              </button>
            </div>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-4 py-6">
        {/* Error banner (W4/F6) */}
        {fetchError && (
          <div className="mb-4 flex items-center justify-between rounded-xl border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-800 dark:border-red-800 dark:bg-red-950/50 dark:text-red-300">
            <div className="flex items-center gap-2">
              <AlertCircle className="h-4 w-4" />
              {t(lang, 'errBanner')}
            </div>
            <button onClick={() => fetchAll(selectedHorizon, selectedPoint)}
              className="rounded-lg border border-red-300 px-3 py-1 text-xs font-semibold hover:bg-red-100 dark:border-red-700 dark:hover:bg-red-900/40">
              {t(lang, 'retry')}
            </button>
          </div>
        )}

        {/* Stale banner (W4/F6) */}
        {!fetchError && isStale && staleMinutes !== null && (
          <div className="mb-4 flex items-center gap-2 rounded-xl border border-amber-300 bg-amber-50 px-4 py-3 text-sm text-amber-800 dark:border-amber-800 dark:bg-amber-950/50 dark:text-amber-300">
            <AlertCircle className="h-4 w-4" />
            {t(lang, 'staleBanner', {
              t: lastGeneration ? new Date(lastGeneration).toLocaleTimeString(lang === 'it' ? 'it-IT' : 'en-US', { hour: '2-digit', minute: '2-digit' }) : '—',
            })}
          </div>
        )}

        {/* Hero stats + decision */}
        <section className="mb-6 grid gap-4 md:grid-cols-5">
          <div className="md:col-span-4">
            {loading && !selectedPred ? (
              <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
                <SkeletonCard /><SkeletonCard /><SkeletonCard /><SkeletonCard />
              </div>
            ) : selectedPred ? (
              <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
                <StatCard icon={Wind} label={t(lang, 'windSpeed')} value={selectedPred.wind_speed_kn?.toFixed(1) ?? '—'}
                  unit="kn" color={speedColor(selectedPred.wind_speed_kn ?? 0)}
                  sublabel={heroBand ?? `±${(selectedPred.expected_error_kn ?? 0).toFixed(1)} kn`} />
                <StatCard icon={Compass} label={t(lang, 'direction')}
                  value={degToCardinal(selectedPred.wind_dir_deg ?? 0)}
                  sublabel={`${(selectedPred.wind_dir_deg ?? 0).toFixed(0)}°${selectedPred.regime ? ` · ${REGIME_ICON[selectedPred.regime] ?? ''} ${selectedPred.regime}` : ''}`} />
                <StatCard icon={Gauge} label={t(lang, 'confidence')}
                  value={`${(selectedPred.confidence_pct ?? 0).toFixed(0)}`} unit="%"
                  color={confColor(selectedPred.confidence_pct ?? 0)}
                  sublabel={speedLabel(selectedPred.wind_speed_kn ?? 0, lang)} />
                <StatCard icon={Zap} label={t(lang, 'bestSpot')}
                  value={bestPoint?.point_id.replace(/_/g, ' ') ?? '—'}
                  color="#16a34a" sublabel={bestPoint ? `${bestPoint.wind_speed_kn?.toFixed(1)} kn` : ''} />
              </div>
            ) : null}
          </div>
          {/* Decision verdict badge (compact) — full card below the grid */}
          {bestDecision && (
            <div className="flex flex-col justify-center rounded-xl border bg-card p-4 shadow-sm">
              <div className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                {t(lang, 'sailing')}
              </div>
              <div className="mt-1 text-xl font-bold" style={{
                color: bestDecision.verdict === 'go' ? '#22c55e'
                  : bestDecision.verdict === 'marginal' ? '#f59e0b' : '#dc2626',
              }}>
                {bestDecision.verdict === 'go' ? t(lang, 'verdictGo')
                  : bestDecision.verdict === 'marginal' ? t(lang, 'verdictMarginal')
                  : t(lang, 'verdictNoGo')}
              </div>
              <div className="mt-1 text-xs text-muted-foreground">
                {bestDecision.point_id.replace(/_/g, ' ')}
                {bestDecision.best_hour !== null ? ` · ${bestDecision.best_hour}:00` : ''}
                {' · '}{Math.round(bestDecision.peak_p_go * 100)}%
              </div>
            </div>
          )}
        </section>

        {/* Horizon selector */}
        <section className="mb-6">
          <div className="flex items-center gap-2 overflow-x-auto pb-2">
            {HORIZONS.map(h => (
              <button key={h.hours} onClick={() => selectHorizon(h.hours)}
                className={`flex-shrink-0 rounded-lg px-4 py-2 text-sm font-medium transition-all ${
                  selectedHorizon === h.hours
                    ? 'bg-primary text-primary-foreground shadow-md'
                    : 'bg-card border hover:bg-muted'
                }`}>
                {h.hours === 0 ? t(lang, 'now') : h.label}
              </button>
            ))}
          </div>
        </section>

        {/* Map section (W3): interactive + precomputed heatmap tabs */}
        <section className="mb-6">
          <div className="mb-3 flex items-center justify-between">
            <h2 className="text-lg font-semibold">{lang === 'it' ? 'Mappa del lago' : 'Lake map'}</h2>
            <div className="flex rounded-lg border p-1">
              <button onClick={() => setMapTab('interactive')}
                className={`flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-medium ${
                  mapTab === 'interactive' ? 'bg-primary text-primary-foreground' : 'hover:bg-muted'
                }`}>
                <MapIcon className="h-3.5 w-3.5" /> {t(lang, 'mapInteractive')}
              </button>
              <button onClick={() => setMapTab('heatmap')}
                className={`flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-medium ${
                  mapTab === 'heatmap' ? 'bg-primary text-primary-foreground' : 'hover:bg-muted'
                }`}>
                <Activity className="h-3.5 w-3.5" /> {t(lang, 'mapHeatmap')}
              </button>
            </div>
          </div>
          {mapTab === 'interactive' ? (
            <WindMap points={mapPoints} onSelect={selectPoint} />
          ) : (
            <div>
              {mapLoading ? (
                <div className="flex h-[420px] items-center justify-center rounded-xl border bg-card text-sm text-muted-foreground">
                  <RefreshCw className="mr-2 h-5 w-5 animate-spin" /> {t(lang, 'loading')}
                </div>
              ) : mapUrl ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img src={mapUrl} alt="LakeWind model heatmap"
                  className="w-full rounded-xl border" />
              ) : (
                <div className="flex h-[420px] items-center justify-center rounded-xl border bg-card text-sm text-muted-foreground">
                  <AlertCircle className="mr-2 h-5 w-5" /> {t(lang, 'mapError')}
                </div>
              )}
              <div className="mt-2 flex items-center justify-between text-xs text-muted-foreground">
                <span>{t(lang, 'mapCaption')}</span>
                {mapSource && (
                  <span className="flex items-center gap-1">
                    <ChevronDown className="h-3 w-3" />
                    {mapSource === 'artifact' ? 'precomputed ✓' : 'on-demand render'}
                  </span>
                )}
              </div>
            </div>
          )}
        </section>

        {/* Main grid */}
        <div className="grid gap-6 lg:grid-cols-3">
          {/* Left: Points list */}
          <section className="lg:col-span-2">
            <div className="mb-3 flex items-center justify-between">
              <h2 className="text-lg font-semibold">
                {t(lang, 'allPoints')}
                <span className="ml-2 text-sm font-normal text-muted-foreground">
                  {t(lang, 'pointsAhead', { h: selectedHorizon }).replace(String(predictions.length) + ' ', predictions.length + ' ')}
                </span>
              </h2>
            </div>
            {loading && !predictions.length ? (
              <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
                <SkeletonCard /><SkeletonCard /><SkeletonCard />
              </div>
            ) : (
              Object.entries(pointsBySector).map(([sector, sectorPoints]) => (
                <div key={sector} className="mb-4">
                  <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">{sector}</h3>
                  <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
                    {sectorPoints.map(pt => {
                      const pred = predictions.find(p => p.point_id === pt.id);
                      if (!pred) return (
                        <div key={pt.id} className="rounded-xl border border-dashed p-4 text-center text-sm text-muted-foreground">
                          {pt.id.replace(/_/g, ' ')}<div className="mt-1 text-xs">{t(lang, 'noData')}</div>
                        </div>
                      );
                      return <PointCard key={pt.id} pred={pred} lang={lang} onClick={() => selectPoint(pt.id)}
                        isSelected={selectedPoint === pt.id} />;
                    })}
                  </div>
                </div>
              ))
            )}
          </section>

          {/* Right: decision + detail + trend */}
          <section className="space-y-6">
            {selectedDecision && <DecisionCard decision={selectedDecision} lang={lang} />}

            {selectedPred && (
              <div className="rounded-xl border bg-card p-5 shadow-sm">
                <div className="mb-3 flex items-center justify-between">
                  <h3 className="font-semibold">{selectedPoint.replace(/_/g, ' ')}</h3>
                  <span className="text-xs text-muted-foreground">{t(lang, 'forecast24h')}</span>
                </div>
                <div className="mb-4 flex justify-center">
                  <div className="flex flex-col items-center">
                    <WindCompass direction={selectedPred.wind_dir_deg ?? 0} size={100} />
                    <div className="mt-2 text-center">
                      <div className="text-3xl font-bold" style={{ color: speedColor(selectedPred.wind_speed_kn ?? 0) }}>
                        {selectedPred.wind_speed_kn?.toFixed(1)}
                      </div>
                      <div className="text-xs text-muted-foreground">
                        kn · {degToCardinal(selectedPred.wind_dir_deg ?? 0)}
                      </div>
                    </div>
                  </div>
                </div>
                <div className="mt-4">
                  <div className="mb-2 flex items-center gap-2 text-xs font-medium text-muted-foreground">
                    <TrendingUp className="h-3.5 w-3.5" /> {t(lang, 'windTrend')}
                  </div>
                  <TrendChart data={trendData} lang={lang} />
                </div>
                <div className="mt-4">
                  <div className="mb-2 flex items-center gap-2 text-xs font-medium text-muted-foreground">
                    <Navigation className="h-3.5 w-3.5" /> {t(lang, 'directionLabel')}
                  </div>
                  <DirectionChart data={trendData} lang={lang} />
                </div>
              </div>
            )}

            {/* Data source health */}
            <div className="rounded-xl border bg-card p-5 shadow-sm">
              <div className="mb-3 flex items-center gap-2">
                <Activity className="h-4 w-4 text-muted-foreground" />
                <h3 className="text-sm font-semibold">{t(lang, 'dataSources')}</h3>
              </div>
              <div className="flex flex-wrap gap-2">
                {health.length === 0 ? (
                  <span className="text-xs text-muted-foreground">{t(lang, 'loading')}</span>
                ) : health.map(h => (
                  <HealthBadge key={h.source} source={h.source} ok={h.ok} latency={h.latency_ms} />
                ))}
              </div>
            </div>
          </section>
        </div>

        {/* Footer */}
        <footer className="mt-12 border-t pt-6 text-center text-xs text-muted-foreground">
          <p>{t(lang, 'footerLine', { n: points.filter(p => p.is_operational).length })}</p>
          <p className="mt-1">
            {t(lang, 'model')}: {selectedPred?.model_version ?? '—'} ·
            {t(lang, 'generated')}: {selectedPred ? new Date(selectedPred.generated_at).toLocaleString(lang === 'it' ? 'it-IT' : 'en-US') : '—'}
          </p>
        </footer>
      </main>
    </div>
  );
}
