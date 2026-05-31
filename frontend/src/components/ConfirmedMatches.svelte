<script lang="ts">
  import type { ConfirmedMatch } from '../lib/types';

  interface Props {
    matches: ConfirmedMatch[];
  }

  let { matches }: Props = $props();
</script>

<div class="flex flex-wrap gap-3">
  {#each matches as match}
    <div class="group relative flex items-center gap-3 px-5 py-4 bg-[var(--color-ink-2)] border border-[var(--color-gold)]/35"
         style="box-shadow:0 0 24px -10px rgba(255,209,102,0.55)">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" class="shrink-0">
        <path d="M7 10V7a5 5 0 0 1 10 0v3" stroke="#ffd166" stroke-width="2" stroke-linecap="round"/>
        <rect x="4" y="10" width="16" height="10" rx="1.5" stroke="#ffd166" stroke-width="2"/>
      </svg>
      <div class="text-base font-semibold">
        <span class="text-[var(--color-him)]">{match.male}</span>
        <span class="text-[var(--color-bone-mut)] mx-1.5 font-display italic">&amp;</span>
        <span class="text-[var(--color-her)]">{match.female}</span>
      </div>
    </div>
  {/each}
</div>
