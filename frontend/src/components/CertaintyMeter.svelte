<script lang="ts">
  import { onMount } from 'svelte';

  interface Props {
    totalSolutions: number;
    solved: boolean;
    confirmedCount: number;
    totalPairs: number;
  }

  let { totalSolutions, solved, confirmedCount, totalPairs }: Props = $props();

  const clarity = solved ? 100 : Math.max(3, Math.round(100 / Math.sqrt(totalSolutions)));
  const radius = 86;
  const length = Math.PI * radius;
  const formatNumber = (value: number) => Math.round(value).toLocaleString('de-DE');

  let offset = $state(length);
  let shownClarity = $state(0);
  let shownSolutions = $state(0);

  onMount(() => {
    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (reduce) {
      offset = length * (1 - clarity / 100);
      shownClarity = clarity;
      shownSolutions = totalSolutions;
      return;
    }

    requestAnimationFrame(() => { offset = length * (1 - clarity / 100); });
    const duration = 1100;
    const start = performance.now();
    const tick = (now: number) => {
      const progress = Math.min(1, (now - start) / duration);
      const eased = 1 - Math.pow(1 - progress, 3);
      shownClarity = Math.round(clarity * eased);
      shownSolutions = Math.round(totalSolutions * eased);
      if (progress < 1) requestAnimationFrame(tick);
      else { shownClarity = clarity; shownSolutions = totalSolutions; }
    };
    requestAnimationFrame(tick);
  });
</script>

<div class="card gauge-card">
  <div class="gauge-wrap">
    <svg viewBox="0 0 200 118" class="gauge" role="img" aria-label={`Klarheit: ${clarity}%`}>
      <defs>
        <linearGradient id="clarityGrad" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%" stop-color="var(--color-heat-1)" />
          <stop offset="55%" stop-color="var(--color-heat-3)" />
          <stop offset="100%" stop-color="var(--color-heat-4)" />
        </linearGradient>
      </defs>
      <path d="M 14 100 A 86 86 0 0 1 186 100" fill="none" stroke="var(--border)" stroke-width="12" stroke-linecap="round" />
      <path
        class="gauge-arc"
        d="M 14 100 A 86 86 0 0 1 186 100"
        fill="none"
        stroke="url(#clarityGrad)"
        stroke-width="12"
        stroke-linecap="round"
        stroke-dasharray={length}
        stroke-dashoffset={offset}
      />
    </svg>
    <div class="gauge-val">
      <div><b>{shownClarity}<span>%</span></b></div>
      <span class="kicker">Klarheit</span>
    </div>
  </div>

  <div>
    <p class="kicker">Der aktuelle Stand</p>
    <p class="clarity-line">
      {#if solved}
        Gelöst. Es gibt nur noch eine <em>Kombination</em>.
      {:else}
        Noch <em>{formatNumber(shownSolutions)}</em> mögliche Kombinationen.
      {/if}
    </p>

    <div class="mini-stats">
      <div class="mini">
        <b style="color:var(--accent-hi)">{formatNumber(shownSolutions)}</b>
        <span>{totalSolutions === 1 ? 'Möglichkeit' : 'Möglichkeiten'}</span>
      </div>
      <div class="mini">
        <b style="color:var(--color-gold)">{confirmedCount} <i>/ {totalPairs}</i></b>
        <span>Fixe Matches</span>
      </div>
    </div>
  </div>
</div>
