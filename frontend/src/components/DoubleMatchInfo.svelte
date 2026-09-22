<script lang="ts">
  import { onMount } from 'svelte';
  import type { DoubleMatch } from '../lib/types';
  import { formatProbability } from '../lib/helpers';

  interface Props {
    doubleMatch: DoubleMatch;
  }

  let { doubleMatch }: Props = $props();
  const sorted = [...doubleMatch.candidates].sort((a, b) => b.probability - a.probability);
  const max = Math.max(...sorted.map((candidate) => candidate.probability), 0.0001);
  let mounted = $state(false);

  onMount(() => requestAnimationFrame(() => { mounted = true; }));
</script>

<div class="card double">
  <p class="lead-sm">Eine Person kann in dieser Staffel zwei Perfect Matches haben. Wer ist am wahrscheinlichsten dabei?</p>
  <div>
    {#each sorted as candidate, index}
      <div class="drow">
        <span class="rank">{String(index + 1).padStart(2, '0')}</span>
        <span class="nm" style={`color:${candidate.gender === 'male' ? 'var(--color-him)' : 'var(--color-her)'}`} title={candidate.name}>{candidate.name}</span>
        <div class="track">
          <i
            style={`width:${mounted ? (candidate.probability / max) * 100 : 0}%;`}
          ></i>
        </div>
        <span class="pc">{formatProbability(candidate.probability)}</span>
      </div>
    {/each}
  </div>
</div>
