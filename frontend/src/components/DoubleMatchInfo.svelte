<script lang="ts">
  import { onMount } from 'svelte';
  import type { DoubleMatch } from '../lib/types';
  import { formatProbability } from '../lib/helpers';

  interface Props {
    doubleMatch: DoubleMatch;
  }

  let { doubleMatch }: Props = $props();

  const sorted = [...doubleMatch.candidates].sort((a, b) => b.probability - a.probability);
  const max = Math.max(...sorted.map((c) => c.probability), 0.0001);

  let mounted = $state(false);
  onMount(() => requestAnimationFrame(() => { mounted = true; }));

  const genderColor = (g: string) => (g === 'male' ? 'var(--color-him)' : 'var(--color-her)');
</script>

<div class="card p-5 sm:p-6">
  <p class="mb-6 max-w-xl text-sm leading-relaxed text-[var(--color-bone-dim)]">
    Eine Person kann in dieser Staffel zwei Perfect Matches haben. Wer ist am wahrscheinlichsten dabei?
  </p>

  <div class="space-y-3">
    {#each sorted as c, i}
      <div class="flex items-center gap-3">
        <span class="w-5 text-right font-mono text-[0.65rem] text-[var(--color-bone-mut)]">{String(i + 1).padStart(2, '0')}</span>
        <span class="w-28 truncate text-sm font-semibold" style={`color:${genderColor(c.gender)}`} title={c.name}>{c.name}</span>
        <div class="h-2.5 flex-1 overflow-hidden rounded-full bg-[var(--color-line)]">
          <div class="h-full rounded-full transition-[width] duration-700 ease-out"
               style={`width:${mounted ? (c.probability / max) * 100 : 0}%;background:linear-gradient(90deg,#7b3fd6,#b026ff)`}></div>
        </div>
        <span class="font-mono font-bold text-sm w-12 text-right text-[var(--color-match-hi)]">{formatProbability(c.probability)}</span>
      </div>
    {/each}
  </div>
</div>
