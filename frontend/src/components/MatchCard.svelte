<script lang="ts">
  import { onMount } from 'svelte';
  import type { Pairing } from '../lib/types';
  import { getProbabilityTier, formatProbability, heatColor } from '../lib/helpers';

  interface Props {
    pairing: Pairing;
    isDoubleMatch?: boolean;
  }

  let { pairing, isDoubleMatch = false }: Props = $props();
  const tier = getProbabilityTier(pairing.probability, 'de');
  const probability = formatProbability(pairing.probability);
  const heat = heatColor(pairing.probability);
  let width = $state(0);

  onMount(() => {
    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (reduce) {
      width = pairing.probability * 100;
      return;
    }
    requestAnimationFrame(() => { width = pairing.probability * 100; });
  });
</script>

<div class="mcard" style={`border-left-color:${heat};${pairing.confirmed ? 'box-shadow:0 0 0 1px rgba(255,209,102,0.4),0 0 22px -8px rgba(255,209,102,0.5)' : ''}`}>
  <div class="between" style="align-items:flex-start">
    <div class="who">
      <span class="him">{pairing.male}</span>
      <i>&amp;</i>
      <span class="her">{pairing.female}</span>
    </div>
    {#if pairing.confirmed}
      <span class="pill gold">Fix</span>
    {:else if isDoubleMatch}
      <span class="pill violet">Doppel</span>
    {/if}
  </div>

  <div class="tier">
    <span style={`color:${heat}`}>{tier.label}</span>
    <b style={`color:${heat}`}>{probability}</b>
  </div>
  <div class="mbar">
    <i style={`width:${width}%;background:linear-gradient(90deg,var(--color-heat-1),${heat})`}></i>
  </div>
</div>
