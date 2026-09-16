<script lang="ts">
  import { onMount } from 'svelte';
  import type { Pairing } from '../lib/types';
  import { getProbabilityTier, formatProbability, heatColor, heatRgb } from '../lib/helpers';

  interface Props {
    pairing: Pairing;
    isDoubleMatch?: boolean;
  }

  let { pairing, isDoubleMatch = false }: Props = $props();

  const tier = getProbabilityTier(pairing.probability, 'de');
  const pct = formatProbability(pairing.probability);
  const heat = heatColor(pairing.probability);
  const [hr, hg, hb] = heatRgb(pairing.probability);

  let w = $state(0);
  onMount(() => {
    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (reduce) { w = pairing.probability * 100; return; }
    requestAnimationFrame(() => { w = pairing.probability * 100; });
  });
</script>

<div
  class="group relative rounded-2xl border border-[var(--color-line)] bg-[var(--color-ink-2)] p-4 transition-all duration-300 hover:-translate-y-1 hover:bg-[var(--color-ink-3)] sm:p-5"
  style={`border-left:3px solid ${heat};${pairing.confirmed ? 'box-shadow:0 0 0 1px rgba(255,209,102,0.4),0 0 22px -8px rgba(255,209,102,0.5)' : ''}`}
>
  <!-- heat wash on hover -->
  <div class="pointer-events-none absolute inset-0 opacity-0 transition-opacity duration-300 group-hover:opacity-100"
       style={`background:radial-gradient(120% 100% at 0% 0%, rgba(${hr},${hg},${hb},0.14), transparent 70%)`}></div>

  <div class="relative flex items-start justify-between gap-3">
    <div class="flex min-w-0 items-center gap-1.5 text-sm font-semibold leading-snug">
      <span class="text-[var(--color-him)] truncate">{pairing.male}</span>
      <span class="text-[var(--color-bone-mut)] text-xs">×</span>
      <span class="text-[var(--color-her)] truncate">{pairing.female}</span>
    </div>
    {#if pairing.confirmed}
      <span class="pill shrink-0 border border-[var(--color-gold)]/50 px-2 py-1 text-[var(--color-gold)]">Fix</span>
    {:else if isDoubleMatch}
      <span class="pill shrink-0 border border-[var(--color-match)]/50 px-2 py-1 text-[var(--color-match-hi)]">Doppel</span>
    {/if}
  </div>

  <div class="relative mt-4 flex items-end justify-between gap-3">
    <span class="font-mono text-[0.72rem] font-medium" style={`color:${tier.accent}`}>{tier.label}</span>
    <span class="font-mono text-2xl font-bold leading-none" style={`color:${heat}`}>{pct}</span>
  </div>

  <div class="relative mt-3 h-1.5 w-full overflow-hidden rounded-full bg-[var(--color-line)]">
    <div class="h-full rounded-full transition-[width] duration-700 ease-out"
         style={`width:${w}%;background:linear-gradient(90deg,#3a5fb0,${heat})`}></div>
  </div>
</div>
