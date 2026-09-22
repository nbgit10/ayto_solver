<script lang="ts">
  import type { Pairing } from '../lib/types';
  import { formatProbability, heatColor, heatTextColor, getProbabilityTier } from '../lib/helpers';

  interface Props {
    pairings: Pairing[];
    males: string[];
    females: string[];
  }

  let { pairings, males, females }: Props = $props();

  const probabilityMap = new Map<string, number>();
  const confirmedSet = new Set<string>();
  for (const pairing of pairings) {
    probabilityMap.set(`${pairing.male}|${pairing.female}`, pairing.probability);
    if (pairing.confirmed) confirmedSet.add(`${pairing.male}|${pairing.female}`);
  }

  const getProbability = (male: string, female: string) => probabilityMap.get(`${male}|${female}`) ?? 0;
  const short = (value: string, length: number) => value.length > length ? `${value.slice(0, length - 1)}…` : value;

  let hoveredRow = $state(-1);
  let hoveredColumn = $state(-1);
  let active = $derived(hoveredRow >= 0 && hoveredColumn >= 0);
  let activeProbability = $derived(active ? getProbability(males[hoveredRow], females[hoveredColumn]) : 0);
  let activeTier = $derived(getProbabilityTier(activeProbability, 'de'));

  function enter(row: number, column: number) {
    hoveredRow = row;
    hoveredColumn = column;
  }

  function clear() {
    hoveredRow = -1;
    hoveredColumn = -1;
  }
</script>

<div class="card matrix">
  <div class="matrix-readout">
    {#if active}
      <div class="ro-names">
        <span class="him">{males[hoveredRow]}</span>
        <i>&amp;</i>
        <span class="her">{females[hoveredColumn]}</span>
      </div>
      <div class="ro-val">
        <b style={`color:${heatColor(activeProbability)}`}>{formatProbability(activeProbability)}</b>
        <span style={activeProbability >= 1 ? 'color:var(--color-gold)' : ''}>{activeTier.label}</span>
      </div>
    {:else}
      <p class="hint">Tippe auf ein Feld · Männer links, Frauen oben <span class="mobile-hint">· nach rechts wischen</span></p>
    {/if}
  </div>

  <div class="matrix-scroll" aria-label="Match-Matrix, horizontal scrollen">
    <table class="mgrid" on:mouseleave={clear} role="grid" aria-label="Match-Chancen Matrix">
      <thead>
        <tr>
          <th class="corner" scope="col"></th>
          {#each females as female, column}
            <th class:is-active={hoveredColumn === column} class="colhead" scope="col">
              <span title={female}>{short(female, 10)}</span>
            </th>
          {/each}
        </tr>
      </thead>
      <tbody>
        {#each males as male, row}
          <tr>
            <th class:is-active={hoveredRow === row} class="rowhead" scope="row">
              <span title={male}>{short(male, 11)}</span>
            </th>
            {#each females as female, column}
              {@const probability = getProbability(male, female)}
              {@const confirmed = confirmedSet.has(`${male}|${female}`)}
              {@const isCell = hoveredRow === row && hoveredColumn === column}
              {@const isDimmed = active && hoveredRow !== row && hoveredColumn !== column}
              <td>
                <button
                  type="button"
                  class:active={isCell}
                  class:dim={isDimmed}
                  class:is-conf={confirmed}
                  class="cell"
                  on:click={() => enter(row, column)}
                  on:mouseenter={() => enter(row, column)}
                  on:focus={() => enter(row, column)}
                  aria-label={`${male} & ${female}: ${formatProbability(probability)}`}
                  style={`background:${heatColor(probability)};color:${heatTextColor(probability)}`}
                >
                  {#if probability > 0}<span>{formatProbability(probability)}</span>{/if}
                </button>
              </td>
            {/each}
          </tr>
        {/each}
      </tbody>
    </table>
  </div>

  <div class="legend">
    <span>0%</span>
    <span class="ramp" aria-hidden="true"></span>
    <span>100%</span>
  </div>
</div>
