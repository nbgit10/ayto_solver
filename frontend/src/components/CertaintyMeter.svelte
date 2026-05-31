<script lang="ts">
  import { onMount } from 'svelte';

  interface Props {
    totalSolutions: number;
    solved: boolean;
    confirmedCount: number;
    totalPairs: number;
  }

  let { totalSolutions, solved, confirmedCount, totalPairs }: Props = $props();

  // "Klarheit" — 1 solution = 100%, more solutions = less clarity.
  const clarity = solved ? 100 : Math.max(3, Math.round(100 / Math.sqrt(totalSolutions)));

  const R = 86;
  const LEN = Math.PI * R; // semicircle arc length
  const fmtDE = (n: number) => n.toLocaleString('de-DE');

  let offset = $state(LEN);         // start empty, animate to filled
  let shownClarity = $state(0);
  let shownSolutions = $state(0);

  onMount(() => {
    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (reduce) {
      offset = LEN * (1 - clarity / 100);
      shownClarity = clarity;
      shownSolutions = totalSolutions;
      return;
    }
    requestAnimationFrame(() => { offset = LEN * (1 - clarity / 100); });
    const dur = 1100, start = performance.now();
    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / dur);
      const e = 1 - Math.pow(1 - t, 3);
      shownClarity = Math.round(clarity * e);
      shownSolutions = Math.round(totalSolutions * e);
      if (t < 1) requestAnimationFrame(tick);
      else { shownClarity = clarity; shownSolutions = totalSolutions; }
    };
    requestAnimationFrame(tick);
  });
</script>

<div class="card p-7 sm:p-9 relative overflow-hidden">
  <div class="pointer-events-none absolute -left-20 -bottom-24 h-72 w-72 rounded-full blur-3xl opacity-50"
       style="background:radial-gradient(circle,rgba(176,38,255,0.4),transparent 65%)"></div>

  <div class="relative grid gap-8 sm:grid-cols-[auto_1fr] sm:items-center">
    <!-- arc gauge -->
    <div class="relative w-[220px] mx-auto sm:mx-0">
      <svg viewBox="0 0 200 118" class="w-full" aria-hidden="true">
        <defs>
          <linearGradient id="clarityGrad" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stop-color="#3a5fb0" />
            <stop offset="55%" stop-color="#b026ff" />
            <stop offset="100%" stop-color="#ff3d8b" />
          </linearGradient>
        </defs>
        <path d="M 14 100 A 86 86 0 0 1 186 100" fill="none"
              stroke="var(--color-line)" stroke-width="12" stroke-linecap="round" />
        <path d="M 14 100 A 86 86 0 0 1 186 100" fill="none"
              stroke="url(#clarityGrad)" stroke-width="12" stroke-linecap="round"
              stroke-dasharray={LEN} stroke-dashoffset={offset}
              style="transition:stroke-dashoffset 1.2s cubic-bezier(0.16,1,0.3,1);filter:drop-shadow(0 0 6px rgba(176,38,255,0.6))" />
      </svg>
      <div class="absolute inset-x-0 bottom-1 text-center">
        <div class="font-mono font-bold text-4xl text-[var(--color-bone)] leading-none">{shownClarity}<span class="text-[var(--color-match-hi)]">%</span></div>
        <div class="kicker mt-1.5">Klarheit</div>
      </div>
    </div>

    <!-- readout -->
    <div>
      <p class="kicker">Das Orakel</p>
      <p class="mt-2 font-display text-2xl sm:text-3xl font-bold text-[var(--color-bone)] leading-tight">
        {#if solved}
          Gelöst. Das Bild ist <span class="italic text-[var(--color-gold)]">eindeutig</span>.
        {:else}
          Noch <span class="italic text-[var(--color-match-hi)]">{fmtDE(shownSolutions)}</span> mögliche Endkonstellationen.
        {/if}
      </p>

      <div class="mt-6 grid grid-cols-2 gap-px bg-[var(--color-line)] border border-[var(--color-line)] max-w-sm">
        <div class="bg-[var(--color-ink-2)] px-4 py-3">
          <div class="font-mono font-bold text-xl text-[var(--color-match-hi)]">{fmtDE(shownSolutions)}</div>
          <div class="font-mono text-[0.58rem] uppercase tracking-[0.16em] text-[var(--color-bone-mut)] mt-0.5">
            {totalSolutions === 1 ? 'Lösung' : 'Lösungen'}
          </div>
        </div>
        <div class="bg-[var(--color-ink-2)] px-4 py-3">
          <div class="font-mono font-bold text-xl text-[var(--color-gold)]">{confirmedCount} <span class="text-[var(--color-bone-mut)]">/ {totalPairs}</span></div>
          <div class="font-mono text-[0.58rem] uppercase tracking-[0.16em] text-[var(--color-bone-mut)] mt-0.5">Fix-Matches</div>
        </div>
      </div>
    </div>
  </div>
</div>
