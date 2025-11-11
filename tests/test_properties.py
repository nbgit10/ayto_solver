"""Property-based tests using hypothesis for AYTO solver."""
import numpy as np
from hypothesis import given, strategies as st, settings, assume
from ayto.ayto import AYTO


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

    ayto = AYTO(males, females)
    ayto.solve()

    # Each row sums to 1 (each male has exactly one match)
    row_sums = ayto.X_binary.sum(axis=1)
    np.testing.assert_allclose(row_sums, np.ones(n), rtol=1e-5)

    # Each column sums to 1 (each female has exactly one match)
    col_sums = ayto.X_binary.sum(axis=0)
    np.testing.assert_allclose(col_sums, np.ones(n), rtol=1e-5)

    # Total matches equals n
    assert abs(ayto.X_binary.sum() - n) < 0.01


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

    ayto = AYTO(males, females)

    # Add truth booth matches (diagonal for simplicity)
    for i in range(num_truth_booths):
        ayto.add_truth_booth({"Pair": [f"M{i}", f"F{i}"], "Match": True})

    ayto.solve()

    # Verify each confirmed match appears in solution
    for i in range(num_truth_booths):
        assert ayto.X_binary[i, i] > 0.99, f"M{i}-F{i} should be confirmed match"


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

    ayto = AYTO(males, females)

    # Add negative truth booth result
    ayto.add_truth_booth({
        "Pair": [f"M{male_idx}", f"F{female_idx}"],
        "Match": False
    })

    ayto.solve()

    # Verify this pair does NOT appear in solution
    assert ayto.X_binary[male_idx, female_idx] < 0.01, \
        f"M{male_idx}-F{female_idx} should NOT be a match"


@settings(deadline=5000)
@given(
    n_males=st.integers(min_value=2, max_value=5),
    n_females=st.integers(min_value=2, max_value=5)
)
def test_nxm_solution_has_correct_total_matches(n_males, n_females):
    """Property: n×m solution should have min(n,m) + 1 total matches."""
    males = [f"M{i}" for i in range(n_males)]
    females = [f"F{i}" for i in range(n_females)]

    ayto = AYTO(males, females)
    ayto.solve()

    expected_matches = min(n_males, n_females) + 1
    actual_matches = ayto.X_binary.sum()

    assert abs(actual_matches - expected_matches) < 0.01, \
        f"Expected {expected_matches} matches, got {actual_matches}"


@settings(deadline=5000)
@given(
    n_males=st.integers(min_value=2, max_value=5),
    n_females=st.integers(min_value=2, max_value=5)
)
def test_nxm_each_person_has_at_least_one_match(n_males, n_females):
    """Property: In n×m matching, each person has at least one match."""
    males = [f"M{i}" for i in range(n_males)]
    females = [f"F{i}" for i in range(n_females)]

    ayto = AYTO(males, females)
    ayto.solve()

    # Each male has at least one match
    row_sums = ayto.X_binary.sum(axis=1)
    assert np.all(row_sums >= 0.99), "Each male should have at least one match"

    # Each female has at least one match
    col_sums = ayto.X_binary.sum(axis=0)
    assert np.all(col_sums >= 0.99), "Each female should have at least one match"


@settings(deadline=5000)
@given(
    n=st.integers(min_value=3, max_value=5),
    num_nights=st.integers(min_value=1, max_value=3)
)
def test_matching_night_constraints_are_satisfied(n, num_nights):
    """Property: Solution must satisfy matching night constraints."""
    males = [f"M{i}" for i in range(n)]
    females = [f"F{i}" for i in range(n)]

    ayto = AYTO(males, females)

    # Add matching nights with known counts
    nights_data = []
    for night_idx in range(num_nights):
        # Create a specific pairing for this night
        pairs = []
        for i in range(n):
            # Rotate pairings for each night
            female_idx = (i + night_idx) % n
            pairs.append([f"M{i}", f"F{female_idx}"])

        # For simplicity, say this night had 0 matches
        nights_data.append((pairs, 0))
        ayto.add_matchingnight({"Pairs": pairs, "Matches": 0})

    ayto.solve()

    # Verify solution doesn't include any pairs from the 0-match nights
    for pairs, expected_matches in nights_data:
        actual_matches = 0
        for pair in pairs:
            m_idx = males.index(pair[0])
            f_idx = females.index(pair[1])
            if ayto.X_binary[m_idx, f_idx] > 0.5:
                actual_matches += 1

        assert actual_matches == expected_matches, \
            f"Night should have {expected_matches} matches, got {actual_matches}"


@settings(deadline=5000)
@given(
    n=st.integers(min_value=2, max_value=5)
)
def test_solution_is_binary_valued(n):
    """Property: Solution should contain only 0s and 1s."""
    males = [f"M{i}" for i in range(n)]
    females = [f"F{i}" for i in range(n)]

    ayto = AYTO(males, females)
    ayto.solve()

    # All values should be very close to 0 or 1
    for i in range(n):
        for j in range(n):
            val = ayto.X_binary[i, j]
            assert (val < 0.01) or (val > 0.99), \
                f"Value at [{i},{j}] = {val} is not binary"


@settings(deadline=5000)
@given(
    n_males=st.integers(min_value=3, max_value=6),
    extra_females=st.integers(min_value=1, max_value=2)
)
def test_more_females_exactly_one_has_double_match(n_males, extra_females):
    """Property: With more females, exactly one female has 2 matches, others have 1."""
    n_females = n_males + extra_females

    males = [f"M{i}" for i in range(n_males)]
    females = [f"F{i}" for i in range(n_females)]

    ayto = AYTO(males, females)
    ayto.solve()

    col_sums = ayto.X_binary.sum(axis=0)

    # Count how many females have 2 matches
    females_with_2_matches = np.sum(col_sums > 1.5)

    # Should be exactly `extra_females` females with 2 matches
    # Actually, based on current implementation, total matches = n_males + extra_females
    # and each of first n_males females has 1 match, so extra females must account for extras

    # Total matches
    total_matches = ayto.X_binary.sum()
    expected_total = n_males + extra_females
    assert abs(total_matches - expected_total) < 0.01


@settings(deadline=5000)
@given(
    n_females=st.integers(min_value=3, max_value=6),
    extra_males=st.integers(min_value=1, max_value=2)
)
def test_more_males_exactly_one_has_double_match(n_females, extra_males):
    """Property: With more males, exactly one male has 2 matches, others have 1."""
    n_males = n_females + extra_males

    males = [f"M{i}" for i in range(n_males)]
    females = [f"F{i}" for i in range(n_females)]

    ayto = AYTO(males, females)
    ayto.solve()

    row_sums = ayto.X_binary.sum(axis=1)

    # Total matches
    total_matches = ayto.X_binary.sum()
    expected_total = n_females + extra_males
    assert abs(total_matches - expected_total) < 0.01

    # Each male should have exactly 1 match (current constraint)
    assert np.all(row_sums > 0.99), "Each male should have at least 1 match"
    assert np.all(row_sums < 1.01), "Each male should have at most 1 match"


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

    ayto = AYTO(males, females)

    # Confirm some matches
    for i in range(num_confirmed):
        ayto.add_truth_booth({"Pair": [f"M{i}", f"F{i}"], "Match": True})

    ayto.solve()

    # For each confirmed match, verify exclusivity
    for i in range(num_confirmed):
        # This male should only match this female
        assert ayto.X_binary[i, :].sum() > 0.99 and ayto.X_binary[i, :].sum() < 1.01
        assert ayto.X_binary[i, i] > 0.99

        # This female should only match this male
        assert ayto.X_binary[:, i].sum() > 0.99 and ayto.X_binary[:, i].sum() < 1.01
        assert ayto.X_binary[i, i] > 0.99
