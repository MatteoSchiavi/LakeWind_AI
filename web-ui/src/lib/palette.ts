/**
 * LakeWind design tokens — wind-speed visual encoding (Phase 4 / W5).
 *
 * Single source of truth for the web surface. The constants below MIRROR
 * `lakewind/utils/palette.py` (Python) and are kept in lockstep by
 * `tests/test_phase4_uiux.py::test_palette_web_python_mirror`, which fails
 * the build if the two files drift apart. There, "green" means exactly
 * P(>= 8 kn) territory — the same 8/12 kn anchors the decision module and
 * the R9 decision-precision metrics use.
 *
 *   <5 kn  calm     (blue)   — not sailable
 *   5-8    light    (teal)   — marginal
 *   8-12   sailable (green)  — the Breva sweet spot
 *   12-16  strong   (amber)  — powered sailing, reef territory
 *   16+    extreme  (red)    — small-craft caution
 */

export const SPEED_BREAKS = [5.0, 8.0, 12.0, 16.0] as const;

export const SPEED_COLORS = [
  '#3b82f6', // blue  — calm
  '#06b6d4', // teal  — light
  '#22c55e', // green — sailable
  '#f59e0b', // amber — strong
  '#dc2626', // red   — extreme
] as const;

export const BAND_LABELS = {
  en: ['calm', 'light', 'sailable', 'strong', 'extreme'],
  it: ['calma', 'leggero', 'navigabile', 'forte', 'estremo'],
} as const;

export function bandIndex(speedKn: number): number {
  if (speedKn === null || speedKn === undefined || Number.isNaN(speedKn)) return 0;
  for (let i = 0; i < SPEED_BREAKS.length; i++) {
    if (speedKn < SPEED_BREAKS[i]) return i;
  }
  return SPEED_BREAKS.length; // open-ended top band (>= 16 kn)
}

export function speedColor(speedKn: number): string {
  return SPEED_COLORS[bandIndex(speedKn)];
}

export function speedLabel(speedKn: number, lang: 'en' | 'it' = 'en'): string {
  return BAND_LABELS[lang][bandIndex(speedKn)];
}

/** Confidence color — shared by point cards and hero stat. */
export function confColor(conf: number): string {
  if (conf >= 75) return '#22c55e';
  if (conf >= 50) return '#f59e0b';
  return '#dc2626';
}
