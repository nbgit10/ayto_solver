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
  class="group relative p-4 bg-[var(--color-ink-2)] border border-[var(--color-line)] transition-all duration-300 hover:-translate-y-0.5"
  style={`border-left:3px solid ${heat};${pairing.confirmed ? 'box-shadow:0 0 0 1px rgba(255,209,102,0.4),0 0 22px -8px rgba(255,209,102,0.5)' : ''}`}
>
  <!-- heat wash on hover -->
  <div class="pointer-events-none absolute inset-0 opacity-0 transition-opacity duration-300 group-hover:opacity-100"
       style={`background:radial-gradient(120% 100% at 0% 0%, rgba(${hr},${hg},${hb},0.14), transparent 70%)`}></div>

  <div class="relative flex items-center justify-between">
    <div class="flex items-center gap-1.5 text-sm font-semibold min-w-0">
      <span class="text-[var(--color-him)] truncate">{pairing.male}</span>
      <span class="text-[var(--color-bone-mut)] text-xs">×</span>
      <span class="text-[var(--color-her)] truncate">{pairing.female}</span>
    </div>
    {#if pairing.confirmed}
      <span class="font-mono text-[0.55rem] uppercase tracking-wider px-1.5 py-0.5 text-[var(--color-gold)] border border-[var(--color-gold)]/50">fix ✓</span>
    {:else if isDoubleMatch}
      <span class="font-mono text-[0.55rem] uppercase tracking-wider px-1.5 py-0.5 text-[var(--color-match-hi)] border border-[var(--color-match)]/50">2×</span>
    {/if}
  </div>

  <div class="relative mt-3 flex items-end justify-between gap-3">
    <span class="font-mono text-[0.62rem] uppercase tracking-[0.12em]" style={`color:${tier.accent}`}>{tier.label}</span>
    <span class="font-mono font-bold text-2xl leading-none" style={`color:${heat}`}>{pct}</span>
  </div>

  <div class="relative mt-2 h-1.5 w-full bg-[var(--color-line)] overflow-hidden">
    <div class="h-full transition-[width] duration-700 ease-out"
         style={`width:${w}%;background:linear-gradient(90deg,#3a5fb0,${heat})`}></div>
  </div>
</div>
