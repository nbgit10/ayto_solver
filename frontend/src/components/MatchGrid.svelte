<script lang="ts">
  import type { Pairing } from '../lib/types';
  import { formatProbability } from '../lib/helpers';

  interface Props {
    pairings: Pairing[];
    males: string[];
    females: string[];
  }

  let { pairings, males, females }: Props = $props();

  // Build probability lookup
  let probMap = $derived(() => {
    const map = new Map<string, number>();
    for (const p of pairings) {
      map.set(`${p.male}|${p.female}`, p.probability);
    }
    return map;
  });

  function getProb(male: string, female: string): number {
    return probMap().get(`${male}|${female}`) ?? 0;
  }

  function cellColor(prob: number): string {
    if (prob >= 1.0) return 'bg-emerald-500 text-white font-bold';
    if (prob >= 0.7) return 'bg-green-400 text-white';
    if (prob >= 0.5) return 'bg-green-300 text-green-900';
    if (prob >= 0.3) return 'bg-amber-200 text-amber-900';
    if (prob >= 0.1) return 'bg-rose-100 text-rose-700 dark:bg-rose-900/30 dark:text-rose-400';
    if (prob > 0) return 'bg-rose-50 text-rose-400 dark:bg-rose-950/20 dark:text-rose-600';
    return 'bg-gray-100 text-gray-300 dark:bg-gray-800 dark:text-gray-700';
  }
</script>

<div class="overflow-x-auto -mx-4 px-4">
  <table class="text-xs border-collapse min-w-full">
    <thead>
      <tr>
        <th class="p-1 text-left text-gray-500 dark:text-gray-400 sticky left-0 bg-gray-50 dark:bg-gray-900 z-10"></th>
        {#each females as female}
          <th class="p-1 text-center font-medium text-pink-600 dark:text-pink-400 whitespace-nowrap max-w-16 truncate" title={female}>
            {female.length > 8 ? female.slice(0, 7) + '...' : female}
          </th>
        {/each}
      </tr>
    </thead>
    <tbody>
      {#each males as male}
        <tr>
          <td class="p-1 font-medium text-blue-600 dark:text-blue-400 whitespace-nowrap sticky left-0 bg-gray-50 dark:bg-gray-900 z-10" title={male}>
            {male.length > 10 ? male.slice(0, 9) + '...' : male}
          </td>
          {#each females as female}
            {@const prob = getProb(male, female)}
            <td
              class="p-1 text-center rounded-sm {cellColor(prob)}"
              title="{male} & {female}: {formatProbability(prob)}"
            >
              {prob > 0 ? formatProbability(prob) : '\u2014'}
            </td>
          {/each}
        </tr>
      {/each}
    </tbody>
  </table>
</div>
