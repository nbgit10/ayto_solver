<script lang="ts">
  import type { ConfirmedMatch } from '../lib/types';

  interface Props {
    matches: ConfirmedMatch[];
  }

  let { matches }: Props = $props();
</script>

<div class="confirmed">
  {#each matches as match}
    <div class="cmatch">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true">
        <path d="M7 10V7a5 5 0 0 1 10 0v3" />
        <rect x="4" y="10" width="16" height="10" rx="1.5" />
      </svg>
      <div class="who">
        <span class="him">{match.male}</span>
        <i>&amp;</i>
        <span class="her">{match.female}</span>
      </div>
    </div>
  {/each}
</div>
