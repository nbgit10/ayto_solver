<script lang="ts">
  import type { DoubleMatch } from '../lib/types';
  import { formatProbability } from '../lib/helpers';

  interface Props {
    doubleMatch: DoubleMatch;
  }

  let { doubleMatch }: Props = $props();

  let sorted = $derived(
    [...doubleMatch.candidates].sort((a, b) => b.probability - a.probability)
  );
</script>

<div class="bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-xl p-5">
  <p class="text-sm text-gray-500 dark:text-gray-400 mb-4">
    Bei ungleicher Teilnehmerzahl hat eine Person zwei Matches. Wer ist am wahrscheinlichsten das Doppel-Match?
  </p>
  <div class="space-y-2">
    {#each sorted as candidate}
      <div class="flex items-center gap-3">
        <span class="font-medium text-sm text-gray-900 dark:text-gray-100 w-28 truncate" title={candidate.name}>
          {candidate.name}
        </span>
        <div class="flex-1 h-2.5 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
          <div
            class="h-full rounded-full bg-purple-500 transition-all duration-500"
            style="width: {candidate.probability * 100}%"
          ></div>
        </div>
        <span class="text-sm font-medium text-purple-600 dark:text-purple-400 w-12 text-right">
          {formatProbability(candidate.probability)}
        </span>
      </div>
    {/each}
  </div>
</div>
