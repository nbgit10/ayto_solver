import type { Lang } from './i18n';
import { t } from './i18n';

export interface ProbabilityTier {
  label: string;
  color: string;
  bgClass: string;
  textClass: string;
  barClass: string;
}

export function getProbabilityTier(probability: number, lang: Lang): ProbabilityTier {
  const s = t(lang);

  if (probability >= 1.0) {
    return {
      label: s.confirmed,
      color: 'emerald',
      bgClass: 'bg-emerald-100 dark:bg-emerald-900/30',
      textClass: 'text-emerald-700 dark:text-emerald-300',
      barClass: 'bg-emerald-500',
    };
  }
  if (probability >= 0.7) {
    return {
      label: s.veryLikely,
      color: 'green',
      bgClass: 'bg-green-50 dark:bg-green-900/20',
      textClass: 'text-green-700 dark:text-green-300',
      barClass: 'bg-green-500',
    };
  }
  if (probability >= 0.3) {
    return {
      label: s.possible,
      color: 'amber',
      bgClass: 'bg-amber-50 dark:bg-amber-900/20',
      textClass: 'text-amber-700 dark:text-amber-300',
      barClass: 'bg-amber-500',
    };
  }
  if (probability > 0) {
    return {
      label: s.unlikely,
      color: 'rose',
      bgClass: 'bg-rose-50 dark:bg-rose-900/10',
      textClass: 'text-rose-400 dark:text-rose-500',
      barClass: 'bg-rose-300',
    };
  }
  return {
    label: s.ruledOutLabel,
    color: 'gray',
    bgClass: 'bg-gray-50 dark:bg-gray-800',
    textClass: 'text-gray-400 dark:text-gray-600',
    barClass: 'bg-gray-300',
  };
}

export function formatProbability(probability: number): string {
  if (probability >= 1.0) return '100%';
  if (probability <= 0) return '0%';
  return `${Math.round(probability * 100)}%`;
}

export function formatProbabilityLabel(probability: number, lang: Lang): string {
  const tier = getProbabilityTier(probability, lang);
  if (probability >= 1.0) return `${tier.label} \u2713`;
  if (probability <= 0) return tier.label;
  return `${tier.label} \u2014 ${formatProbability(probability)}`;
}

export function seasonSlug(id: string): string {
  return id;
}

export function seasonUrl(id: string): string {
  return `/staffel/${id}/`;
}
