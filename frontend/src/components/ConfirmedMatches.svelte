<script lang="ts">
  import type { ConfirmedMatch } from '../lib/types';

  interface Props {
    matches: ConfirmedMatch[];
  }

  let { matches }: Props = $props();
</script>

<div class="flex flex-wrap gap-3">
  {#each matches as match}
    <div class="bg-emerald-50 dark:bg-emerald-900/20 border-2 border-emerald-300 dark:border-emerald-700 rounded-xl px-4 py-3 flex items-center gap-3">
      <span class="text-emerald-600 dark:text-emerald-400 text-lg">&#x2713;</span>
      <div>
        <span class="text-blue-700 dark:text-blue-400 font-medium">{match.male}</span>
        <span class="text-gray-400 mx-1">&</span>
        <span class="text-pink-700 dark:text-pink-400 font-medium">{match.female}</span>
      </div>
    </div>
  {/each}
</div>
