<script lang="ts">
  import type { Pairing } from '../lib/types';
  import { getProbabilityTier, formatProbability } from '../lib/helpers';

  interface Props {
    pairing: Pairing;
    isDoubleMatch?: boolean;
  }

  let { pairing, isDoubleMatch = false }: Props = $props();

  let tier = $derived(getProbabilityTier(pairing.probability, 'de'));
  let pct = $derived(formatProbability(pairing.probability));
</script>

<div class="rounded-lg border p-4 transition-all duration-200 hover:shadow-md {tier.bgClass} {pairing.confirmed ? 'border-emerald-400 dark:border-emerald-600 ring-2 ring-emerald-200 dark:ring-emerald-800' : isDoubleMatch ? 'border-purple-400 dark:border-purple-600 ring-1 ring-purple-200 dark:ring-purple-800' : 'border-gray-200 dark:border-gray-700'}">
  <div class="flex items-center justify-between mb-2">
    <div class="flex items-center gap-2">
      <span class="text-blue-700 dark:text-blue-400 font-medium text-sm">{pairing.male}</span>
      <span class="text-gray-400 text-xs">&</span>
      <span class="text-pink-700 dark:text-pink-400 font-medium text-sm">{pairing.female}</span>
    </div>
    {#if pairing.confirmed}
      <span class="text-emerald-600 dark:text-emerald-400">&#x2713;</span>
    {:else if isDoubleMatch}
      <span class="px-1.5 py-0.5 rounded text-[10px] font-semibold bg-purple-100 text-purple-700 dark:bg-purple-900/40 dark:text-purple-400">2x</span>
    {/if}
  </div>
  <div class="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
    <div
      class="h-full rounded-full transition-all duration-500 {tier.barClass}"
      style="width: {pairing.probability * 100}%"
    ></div>
  </div>
  <div class="flex items-center justify-between mt-1.5">
    <span class="text-xs {tier.textClass}">{tier.label}</span>
    <span class="text-xs font-medium {tier.textClass}">{pct}</span>
  </div>
</div>
