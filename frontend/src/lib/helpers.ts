import type { Lang } from './i18n';
import { t } from './i18n';

export interface ProbabilityTier {
  /** translated label, e.g. "Sehr wahrscheinlich" */
  label: string;
  /** stable key for the tier */
  key: 'confirmed' | 'veryLikely' | 'possible' | 'unlikely' | 'ruledOut';
  /** representative accent colour (hex) for badges / text */
  accent: string;
}

export function getProbabilityTier(probability: number, lang: Lang): ProbabilityTier {
  const s = t(lang);
  if (probability >= 1.0) return { label: s.confirmed, key: 'confirmed', accent: '#ffd166' };
  if (probability >= 0.7) return { label: s.veryLikely, key: 'veryLikely', accent: '#ff5599' };
  if (probability >= 0.3) return { label: s.possible, key: 'possible', accent: '#c86bff' };
  if (probability > 0) return { label: s.unlikely, key: 'unlikely', accent: '#6f7bd6' };
  return { label: s.ruledOutLabel, key: 'ruledOut', accent: '#6f6878' };
}

/* ---- continuous "heat" ramp: cold (low p) → hot (high p) ---- */
type RGB = [number, number, number];

const HEAT_STOPS: { at: number; rgb: RGB }[] = [
  { at: 0.0, rgb: [42, 36, 53] },    // --heat-0  ruled-out / near-zero
  { at: 0.18, rgb: [58, 95, 176] },  // --heat-1  faint blue
  { at: 0.45, rgb: [123, 63, 214] }, // --heat-2  violet
  { at: 0.72, rgb: [176, 38, 255] }, // --heat-3  electric violet
  { at: 1.0, rgb: [255, 61, 139] },  // --heat-4  hot magenta
];

function lerp(a: number, b: number, t: number): number {
  return Math.round(a + (b - a) * t);
}

/** Interpolated heat colour for a probability in [0,1]. */
export function heatRgb(probability: number): RGB {
  const p = Math.max(0, Math.min(1, probability));
  for (let i = 1; i < HEAT_STOPS.length; i++) {
    const lo = HEAT_STOPS[i - 1];
    const hi = HEAT_STOPS[i];
    if (p <= hi.at) {
      const t = (p - lo.at) / (hi.at - lo.at || 1);
      return [lerp(lo.rgb[0], hi.rgb[0], t), lerp(lo.rgb[1], hi.rgb[1], t), lerp(lo.rgb[2], hi.rgb[2], t)];
    }
  }
  return HEAT_STOPS[HEAT_STOPS.length - 1].rgb;
}

export function heatColor(probability: number, alpha = 1): string {
  const [r, g, b] = heatRgb(probability);
  return alpha >= 1 ? `rgb(${r}, ${g}, ${b})` : `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

/** Light vs. dark text on a given heat background. */
export function heatTextColor(probability: number): string {
  return probability >= 0.32 ? '#fff' : 'rgba(243,239,233,0.62)';
}

export function formatProbability(probability: number): string {
  if (probability >= 1.0) return '100%';
  if (probability <= 0) return '0%';
  return `${Math.round(probability * 100)}%`;
}

export function formatProbabilityLabel(probability: number, lang: Lang): string {
  const tier = getProbabilityTier(probability, lang);
  if (probability >= 1.0) return `${tier.label} ✓`;
  if (probability <= 0) return tier.label;
  return `${tier.label} — ${formatProbability(probability)}`;
}

export function seasonSlug(id: string): string {
  return id;
}

export function seasonUrl(id: string): string {
  return `/staffel/${id}/`;
}
