<script lang="ts">
  import type { Pairing } from '../lib/types';
  import { formatProbability, heatColor, heatTextColor, getProbabilityTier } from '../lib/helpers';

  interface Props {
    pairings: Pairing[];
    males: string[];
    females: string[];
  }

  let { pairings, males, females }: Props = $props();

  const probMap = new Map<string, number>();
  const confirmedSet = new Set<string>();
  for (const p of pairings) {
    probMap.set(`${p.male}|${p.female}`, p.probability);
    if (p.confirmed) confirmedSet.add(`${p.male}|${p.female}`);
  }
  const getProb = (m: string, f: string) => probMap.get(`${m}|${f}`) ?? 0;

  let hr = $state(-1); // hovered row (male index)
  let hc = $state(-1); // hovered col (female index)

  let active = $derived(hr >= 0 && hc >= 0);
  let activeProb = $derived(active ? getProb(males[hr], females[hc]) : 0);
  let activeTier = $derived(getProbabilityTier(activeProb, 'de'));

  function enter(r: number, c: number) { hr = r; hc = c; }
  function clear() { hr = -1; hc = -1; }

  const short = (s: string, n: number) => (s.length > n ? s.slice(0, n - 1) + '…' : s);
</script>

<div class="card overflow-hidden">
  <!-- live readout -->
  <div class="flex items-center justify-between gap-4 px-5 py-4 border-b border-[var(--color-line)] min-h-[68px]">
    {#if active}
      <div class="flex items-center gap-2 text-lg font-semibold">
        <span class="text-[var(--color-him)]">{males[hr]}</span>
        <span class="font-display italic text-[var(--color-bone-mut)]">&amp;</span>
        <span class="text-[var(--color-her)]">{females[hc]}</span>
      </div>
      <div class="text-right">
        <div class="font-mono font-bold text-2xl leading-none" style={`color:${heatColor(activeProb)}`}>{formatProbability(activeProb)}</div>
        <div class="font-mono text-[0.6rem] uppercase tracking-[0.12em] mt-1" style={`color:${activeTier.accent}`}>{activeTier.label}</div>
      </div>
    {:else}
      <p class="font-mono text-[0.72rem] uppercase tracking-[0.18em] text-[var(--color-bone-mut)]">
        Fahre über eine Zelle &mdash; Mann <span class="text-[var(--color-him)]">↓</span> trifft Frau <span class="text-[var(--color-her)]">→</span>
      </p>
    {/if}
  </div>

  <div class="overflow-x-auto">
    <table class="border-separate border-spacing-1 p-3" onmouseleave={clear} role="grid">
      <thead>
        <tr>
          <th class="sticky left-0 z-20 bg-[var(--color-ink-2)]"></th>
          {#each females as female, c}
            <th class="px-1 pb-1 align-bottom">
              <div class="font-mono text-[0.62rem] tracking-wide whitespace-nowrap transition-all duration-200 origin-bottom"
                   style={`color:${hc === c ? 'var(--color-her)' : 'var(--color-bone-mut)'};transform:rotate(-45deg) translateX(2px)${hc === c ? ' scale(1.12)' : ''}`}
                   title={female}>
                {short(female, 8)}
              </div>
            </th>
          {/each}
        </tr>
      </thead>
      <tbody>
        {#each males as male, r}
          <tr>
            <th class="sticky left-0 z-10 bg-[var(--color-ink-2)] pr-2 text-right">
              <span class="font-mono text-[0.7rem] whitespace-nowrap transition-all duration-200 inline-block"
                    style={`color:${hr === r ? 'var(--color-him)' : 'var(--color-bone-dim)'}${hr === r ? ';transform:scale(1.08)' : ''}`}
                    title={male}>{short(male, 9)}</span>
            </th>
            {#each females as female, c}
              {@const prob = getProb(male, female)}
              {@const confirmed = confirmedSet.has(`${male}|${female}`)}
              {@const isAxis = hr === r || hc === c}
              {@const isCell = hr === r && hc === c}
              <td class="p-0">
                <button
                  type="button"
                  onmouseenter={() => enter(r, c)}
                  onfocus={() => enter(r, c)}
                  aria-label={`${male} & ${female}: ${formatProbability(prob)}`}
                  class="relative block h-9 w-11 sm:w-12 transition-all duration-150 outline-none"
                  style={`
                    background:${heatColor(prob)};
                    color:${heatTextColor(prob)};
                    opacity:${active && !isAxis ? 0.38 : 1};
                    transform:${isCell ? 'scale(1.18)' : 'scale(1)'};
                    z-index:${isCell ? 30 : 1};
                    box-shadow:${isCell ? '0 0 0 2px var(--color-bone),0 6px 20px -4px rgba(0,0,0,0.7)' : confirmed ? '0 0 0 2px var(--color-gold) inset' : 'none'};
                  `}>
                  <span class="font-mono text-[0.62rem] font-bold">{prob > 0 ? formatProbability(prob) : ''}</span>
                </button>
              </td>
            {/each}
          </tr>
        {/each}
      </tbody>
    </table>
  </div>
</div>
