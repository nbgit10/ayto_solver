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

<div class="card p-7">
  <p class="text-sm text-[var(--color-bone-dim)] mb-6 max-w-xl leading-relaxed">
    Die Staffel ist <span class="font-display italic text-[var(--color-bone)]">unbalanciert</span> &mdash; eine Person
    hat <em class="text-[var(--color-match-hi)] not-italic font-semibold">zwei</em> Matches. Wer trägt am wahrscheinlichsten das Doppel-Match?
  </p>

  <div class="space-y-3">
    {#each sorted as c, i}
      <div class="flex items-center gap-4">
        <span class="font-mono text-[0.65rem] text-[var(--color-bone-mut)] w-5 text-right">{String(i + 1).padStart(2, '0')}</span>
        <span class="font-semibold text-sm w-28 truncate" style={`color:${genderColor(c.gender)}`} title={c.name}>{c.name}</span>
        <div class="flex-1 h-2.5 bg-[var(--color-line)] overflow-hidden">
          <div class="h-full transition-[width] duration-700 ease-out"
               style={`width:${mounted ? (c.probability / max) * 100 : 0}%;background:linear-gradient(90deg,#7b3fd6,#b026ff)`}></div>
        </div>
        <span class="font-mono font-bold text-sm w-12 text-right text-[var(--color-match-hi)]">{formatProbability(c.probability)}</span>
      </div>
    {/each}
  </div>
</div>
