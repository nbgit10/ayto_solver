/** Display score based on remaining ambiguity, not a probability. */
export function clarityFromSolutionCount(totalSolutions: number): number {
  if (!Number.isFinite(totalSolutions) || totalSolutions <= 0) return 0;
  return Math.max(0, Math.round(100 - 5 * Math.log2(totalSolutions)));
}
