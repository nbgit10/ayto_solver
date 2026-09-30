import assert from 'node:assert/strict';
import test from 'node:test';
import { clarityFromSolutionCount } from './clarity.ts';

test('clarity score decreases by five points for each doubling', () => {
  assert.equal(clarityFromSolutionCount(1), 100);
  assert.equal(clarityFromSolutionCount(2), 95);
  assert.equal(clarityFromSolutionCount(4), 90);
  assert.equal(clarityFromSolutionCount(1024), 50);
});

test('clarity score is clamped to its valid range', () => {
  assert.equal(clarityFromSolutionCount(0), 0);
  assert.equal(clarityFromSolutionCount(2 ** 21), 0);
});
