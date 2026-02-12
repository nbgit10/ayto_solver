"""Property-based tests using hypothesis for AYTO solver."""
import numpy as np
from hypothesis import given, strategies as st, settings, assume
from ayto_solver.solvers.mip_solver import MIPSolver


@settings(deadline=5000)  # Increase deadline for MIP solver
@given(
    n=st.integers(min_value=2, max_value=6),  # Keep small for performance
    seed=st.integers(min_value=0, max_value=10000)
)
def test_nxn_solution_is_valid_permutation(n, seed):
    """Property: n×n solution should be a permutation matrix."""
    np.random.seed(seed)

    males = [f"M{i}" for i in range(n)]
    females = [f"F{i}" for i in range(n)]

    solver = MIPSolver(males, females)
    solver.solve()

    # Each row sums to 1 (each male has exactly one match)
    row_sums = solver.X_binary.sum(axis=1)
    np.testing.assert_allclose(row_sums, np.ones(n), rtol=1e-5)

    # Each column sums to 1 (each female has exactly one match)
    col_sums = solver.X_binary.sum(axis=0)
    np.testing.assert_allclose(col_sums, np.ones(n), rtol=1e-5)

    # Total matches equals n
    assert abs(solver.X_binary.sum() - n) < 0.01


@settings(deadline=5000)
@given(
    n=st.integers(min_value=2, max_value=5),
    num_truth_booths=st.integers(min_value=1, max_value=3)
)
def test_solution_respects_truth_booth_matches(n, num_truth_booths):
    """Property: Positive truth booth results must appear in solution."""
    assume(num_truth_booths <= n)  # Can't have more truth booths than pairs

    males = [f"M{i}" for i in range(n)]
    females = [f"F{i}" for i in range(n)]

    solver = MIPSolver(males, females)

    # Add truth booth matches (diagonal for simplicity)
    for i in range(num_truth_booths):
        solver.add_truth_booth(f"M{i}", f"F{i}", True)

    solver.solve()

    # Verify each confirmed match appears in solution
    for i in range(num_truth_booths):
        assert solver.X_binary[i, i] > 0.99, f"M{i}-F{i} should be confirmed match"


@settings(deadline=5000)
@given(
    n=st.integers(min_value=3, max_value=5),
    male_idx=st.integers(min_value=0, max_value=4),
    female_idx=st.integers(min_value=0, max_value=4)
)
def test_solution_respects_truth_booth_non_matches(n, male_idx, female_idx):
    """Property: Negative truth booth results must NOT appear in solution."""
    assume(male_idx < n and female_idx < n)

    males = [f"M{i}" for i in range(n)]
    females = [f"F{i}" for i in range(n)]

    solver = MIPSolver(males, females)

    # Add negative truth booth result
    solver.add_truth_booth(f"M{male_idx}", f"F{female_idx}", False)

    solver.solve()

    # Verify this pair does NOT appear in solution
    assert solver.X_binary[male_idx, female_idx] < 0.01, \
        f"M{male_idx}-F{female_idx} should NOT be a match"


@settings(deadline=5000)
@given(
    n=st.integers(min_value=2, max_value=5)
)
def test_solution_is_binary_valued(n):
    """Property: Solution should contain only 0s and 1s."""
    males = [f"M{i}" for i in range(n)]
    females = [f"F{i}" for i in range(n)]

    solver = MIPSolver(males, females)
    solver.solve()

    # All values should be very close to 0 or 1
    for i in range(n):
        for j in range(n):
            val = solver.X_binary[i, j]
            assert (val < 0.01) or (val > 0.99), \
                f"Value at [{i},{j}] = {val} is not binary"


@settings(deadline=5000)
@given(
    n=st.integers(min_value=2, max_value=4),
    num_confirmed=st.integers(min_value=0, max_value=3)
)
def test_confirmed_matches_force_exclusivity(n, num_confirmed):
    """Property: When a match is confirmed, those people can't match others."""
    assume(num_confirmed < n)

    males = [f"M{i}" for i in range(n)]
    females = [f"F{i}" for i in range(n)]

    solver = MIPSolver(males, females)

    # Confirm some matches
    for i in range(num_confirmed):
        solver.add_truth_booth(f"M{i}", f"F{i}", True)

    solver.solve()

    # For each confirmed match, verify exclusivity
    for i in range(num_confirmed):
        # This male should only match this female
        assert solver.X_binary[i, :].sum() > 0.99 and solver.X_binary[i, :].sum() < 1.01
        assert solver.X_binary[i, i] > 0.99

        # This female should only match this male
        assert solver.X_binary[:, i].sum() > 0.99 and solver.X_binary[:, i].sum() < 1.01
        assert solver.X_binary[i, i] > 0.99
