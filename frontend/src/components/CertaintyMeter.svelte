<script lang="ts">
  interface Props {
    totalSolutions: number;
    solved: boolean;
    confirmedCount: number;
    totalPairs: number;
  }

  let { totalSolutions, solved, confirmedCount, totalPairs }: Props = $props();

  // Calculate a "certainty" percentage:
  // 1 solution = 100%, more solutions = lower certainty
  let certainty = $derived(
    solved ? 100 : Math.max(5, Math.round(100 / Math.sqrt(totalSolutions)))
  );

  let label = $derived(
    solved ? 'Gelost!' : `${totalSolutions} mogliche Losungen`
  );

  let barColor = $derived(
    certainty >= 100 ? 'bg-emerald-500' :
    certainty >= 70 ? 'bg-green-500' :
    certainty >= 40 ? 'bg-amber-500' :
    'bg-rose-400'
  );
</script>

<div class="bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-xl p-5">
  <div class="flex items-center justify-between mb-3">
    <h3 class="font-semibold text-gray-900 dark:text-gray-100">Wie weit gelost?</h3>
    <span class="text-sm font-medium text-gray-500 dark:text-gray-400">{confirmedCount} / {totalPairs} bestatigt</span>
  </div>
  <div class="w-full h-3 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
    <div
      class="h-full rounded-full transition-all duration-700 ease-out {barColor}"
      style="width: {certainty}%"
    ></div>
  </div>
  <p class="mt-2 text-sm {solved ? 'text-emerald-600 dark:text-emerald-400 font-medium' : 'text-gray-500 dark:text-gray-400'}">
    {label}
  </p>
</div>
