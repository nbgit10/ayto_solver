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
  <div class="flex min-h-[76px] items-center justify-between gap-4 border-b border-[var(--color-line)] px-4 py-4 sm:px-5">
    {#if active}
      <div class="flex min-w-0 items-center gap-2 text-base font-semibold sm:text-lg">
        <span class="text-[var(--color-him)]">{males[hr]}</span>
        <span class="font-display italic text-[var(--color-bone-mut)]">&amp;</span>
        <span class="text-[var(--color-her)]">{females[hc]}</span>
      </div>
      <div class="text-right">
        <div class="font-mono text-2xl font-bold leading-none" style={`color:${heatColor(activeProb)}`}>{formatProbability(activeProb)}</div>
        <div class="mt-1 font-mono text-[0.62rem] font-medium" style={`color:${activeTier.accent}`}>{activeTier.label}</div>
      </div>
    {:else}
      <p class="font-mono text-[0.68rem] leading-relaxed text-[var(--color-bone-mut)] sm:text-[0.72rem]">
        Tippe auf ein Feld · Männer links, Frauen oben <span class="sm:hidden">· nach rechts wischen</span>
      </p>
    {/if}
  </div>

  <div class="matrix-scroll overflow-x-auto" aria-label="Match-Matrix, horizontal scrollen">
    <table class="min-w-[700px] w-full border-separate border-spacing-1 p-3" on:mouseleave={clear} role="grid">
      <thead>
        <tr>
          <th class="sticky left-0 z-20 w-24 bg-[var(--color-ink-2)]"></th>
          {#each females as female, c}
            <th class="h-20 w-[52px] px-1 pb-1 align-bottom">
              <div class="origin-bottom whitespace-nowrap font-mono text-[0.62rem] tracking-wide transition-all duration-200"
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
            <th class="sticky left-0 z-10 w-24 bg-[var(--color-ink-2)] pr-2 text-right">
              <span class="inline-block whitespace-nowrap font-mono text-[0.7rem] transition-all duration-200"
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
                  on:click={() => enter(r, c)}
                  on:mouseenter={() => enter(r, c)}
                  on:focus={() => enter(r, c)}
                  aria-label={`${male} & ${female}: ${formatProbability(prob)}`}
                  class="relative block h-10 w-full min-w-[52px] rounded-md outline-none transition-all duration-150"
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
