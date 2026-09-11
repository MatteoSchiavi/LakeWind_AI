/**
 * LakeWind web i18n (Phase 4 / W4) — lightweight en/it dictionary.
 *
 * Deliberately NOT next-intl: a single dashboard page needs ~40 strings,
 * and a runtime dependency + routing config would outweigh the benefit.
 * The bot's per-user language lives in DuckDB; the web toggle lives in
 * localStorage — both feed the same en/it pair of vocabularies.
 */
export type Lang = 'en' | 'it';

export const STRINGS = {
  en: {
    subtitle: 'Hyperlocal wind forecasting · Lake Como',
    windSpeed: 'Wind Speed',
    direction: 'Direction',
    confidence: 'Confidence',
    bestSpot: 'Best Spot',
    allPoints: 'All Points',
    pointsAhead: 'points · {h}h ahead',
    noData: 'No data',
    forecast24h: '24h forecast',
    windTrend: 'WIND TREND (24H)',
    directionLabel: 'DIRECTION',
    dataSources: 'Data Sources',
    loading: 'Loading…',
    now: 'Now',
    sailing: 'Go sailing?',
    verdictGo: 'GO SAILING',
    verdictMarginal: 'MARGINAL',
    verdictNoGo: 'STAY HOME',
    bestWindow: 'Best window',
    chanceOfWind: 'chance of ≥8 kn',
    chanceStrong: '≥12 kn',
    byHour: 'By hour',
    regime: 'Regime',
    mapInteractive: 'Interactive map',
    mapHeatmap: 'Heatmap (model)',
    mapCaption: 'Precomputed model heatmap — refreshed every pipeline cycle',
    mapFreshness: 'Rendered {t}',
    errBanner: 'Couldn’t reach the forecast API.',
    retry: 'Retry',
    staleBanner: 'Forecast data may be stale — last pipeline run {t}.',
    range: 'Range',
    gust: 'Gust',
    updated: 'Updated',
    footerLine: 'LakeWind AI · MOS bias-corrected · {n} virtual points',
    model: 'Model',
    generated: 'Generated',
    bandNote: 'Shaded band: calibrated 80% interval',
    loadingSkeleton: 'Loading…',
    mapError: 'Map image unavailable.',
  },
  it: {
    subtitle: 'Previsioni del vento iperlocali · Lago di Como',
    windSpeed: 'Velocità vento',
    direction: 'Direzione',
    confidence: 'Affidabilità',
    bestSpot: 'Spot migliore',
    allPoints: 'Tutti i punti',
    pointsAhead: 'punti · tra {h}h',
    noData: 'Nessun dato',
    forecast24h: 'previsione 24h',
    windTrend: 'ANDAMENTO VENTO (24H)',
    directionLabel: 'DIREZIONE',
    dataSources: 'Sorgenti dati',
    loading: 'Caricamento…',
    now: 'Adesso',
    sailing: 'Si va in barca?',
    verdictGo: 'SI ESCE',
    verdictMarginal: 'AL LIMITE',
    verdictNoGo: 'MEGLIO CASA',
    bestWindow: 'Finestra migliore',
    chanceOfWind: 'probabilità ≥8 kn',
    chanceStrong: '≥12 kn',
    byHour: 'Per ora',
    regime: 'Regime',
    mapInteractive: 'Mappa interattiva',
    mapHeatmap: 'Mappa (modello)',
    mapCaption: 'Mappa precalcolata dal modello — aggiornata a ogni ciclo',
    mapFreshness: 'Generata alle {t}',
    errBanner: 'Impossibile raggiungere le previsioni.',
    retry: 'Riprova',
    staleBanner: 'I dati potrebbero non essere aggiornati — ultimo ciclo alle {t}.',
    range: 'Range',
    gust: 'Raffica',
    updated: 'Aggiornato',
    footerLine: 'LakeWind AI · correzione MOS · {n} punti virtuali',
    model: 'Modello',
    generated: 'Generato',
    bandNote: 'Area ombreggiata: intervallo calibrato all’80%',
    loadingSkeleton: 'Caricamento…',
    mapError: 'Immagine mappa non disponibile.',
  },
} as const;

export type StringKey = keyof typeof STRINGS.en;

export function t(lang: Lang, key: StringKey, vars?: Record<string, string | number>): string {
  let s: string = STRINGS[lang][key] ?? STRINGS.en[key];
  if (vars) {
    for (const [k, v] of Object.entries(vars)) s = s.replace(`{${k}}`, String(v));
  }
  return s;
}
