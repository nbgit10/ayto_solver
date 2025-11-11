"""Synthetic test cases with known solutions for AYTO solver."""
import numpy as np
import pytest
from ayto.ayto import AYTO


class TestSimple3x3:
    """Test 3x3 matching with controlled constraints."""

    def test_3x3_unique_solution(self):
        """Test 3x3 case with constraints that lead to unique solution."""
        males = ["M1", "M2", "M3"]
        females = ["F1", "F2", "F3"]

        ayto = AYTO(males, females)

        # Add truth booth: M1-F1 is a match
        ayto.add_truth_booth({"Pair": ["M1", "F1"], "Match": True})

        # Add truth booth: M2-F2 is a match
        ayto.add_truth_booth({"Pair": ["M2", "F2"], "Match": True})

        # This forces M3-F3 to be the third match
        ayto.solve()

        # Expected solution: diagonal matrix
        expected = np.array([
            [1, 0, 0],  # M1-F1
            [0, 1, 0],  # M2-F2
            [0, 0, 1],  # M3-F3
        ])

        np.testing.assert_array_equal(ayto.X_binary, expected)

    def test_3x3_matching_night_constraint(self):
        """Test 3x3 with matching night providing constraints."""
        males = ["M1", "M2", "M3"]
        females = ["F1", "F2", "F3"]

        ayto = AYTO(males, females)

        # Matching night with 0 matches - all pairs are wrong
        ayto.add_matchingnight({
            "Pairs": [["M1", "F1"], ["M2", "F2"], ["M3", "F3"]],
            "Matches": 0
        })

        # Truth booth: M1-F2 is a match
        ayto.add_truth_booth({"Pair": ["M1", "F2"], "Match": True})

        # Truth booth: M2-F3 is a match
        ayto.add_truth_booth({"Pair": ["M2", "F3"], "Match": True})

        ayto.solve()

        # Expected solution: M1-F2, M2-F3, M3-F1
        expected = np.array([
            [0, 1, 0],  # M1-F2
            [0, 0, 1],  # M2-F3
            [1, 0, 0],  # M3-F1
        ])

        np.testing.assert_array_equal(ayto.X_binary, expected)

    def test_3x3_negative_truth_booth(self):
        """Test that negative truth booth results are respected."""
        males = ["M1", "M2", "M3"]
        females = ["F1", "F2", "F3"]

        ayto = AYTO(males, females)

        # M1-F1 is NOT a match
        ayto.add_truth_booth({"Pair": ["M1", "F1"], "Match": False})

        # M1-F2 is NOT a match
        ayto.add_truth_booth({"Pair": ["M1", "F2"], "Match": False})

        # This forces M1-F3
        ayto.add_truth_booth({"Pair": ["M2", "F1"], "Match": True})
        ayto.add_truth_booth({"Pair": ["M3", "F2"], "Match": True})

        ayto.solve()

        # Expected solution
        expected = np.array([
            [0, 0, 1],  # M1-F3 (only option left)
            [1, 0, 0],  # M2-F1
            [0, 1, 0],  # M3-F2
        ])

        np.testing.assert_array_equal(ayto.X_binary, expected)


class TestSimple4x4:
    """Test 4x4 matching with controlled constraints."""

    def test_4x4_unique_solution(self):
        """Test 4x4 case with full truth booth information."""
        males = ["M1", "M2", "M3", "M4"]
        females = ["F1", "F2", "F3", "F4"]

        ayto = AYTO(males, females)

        # Provide 3 matches via truth booth
        ayto.add_truth_booth({"Pair": ["M1", "F2"], "Match": True})
        ayto.add_truth_booth({"Pair": ["M2", "F4"], "Match": True})
        ayto.add_truth_booth({"Pair": ["M3", "F1"], "Match": True})

        # This forces M4-F3
        ayto.solve()

        expected = np.array([
            [0, 1, 0, 0],  # M1-F2
            [0, 0, 0, 1],  # M2-F4
            [1, 0, 0, 0],  # M3-F1
            [0, 0, 1, 0],  # M4-F3
        ])

        np.testing.assert_array_equal(ayto.X_binary, expected)

    def test_4x4_with_matching_nights(self):
        """Test 4x4 with multiple matching nights."""
        males = ["M1", "M2", "M3", "M4"]
        females = ["F1", "F2", "F3", "F4"]

        ayto = AYTO(males, females)

        # First matching night: 1 match
        ayto.add_matchingnight({
            "Pairs": [
                ["M1", "F1"],
                ["M2", "F2"],
                ["M3", "F3"],
                ["M4", "F4"]
            ],
            "Matches": 1
        })

        # Second matching night: 0 matches (all wrong)
        ayto.add_matchingnight({
            "Pairs": [
                ["M1", "F2"],
                ["M2", "F3"],
                ["M3", "F4"],
                ["M4", "F1"]
            ],
            "Matches": 0
        })

        # Truth booth
        ayto.add_truth_booth({"Pair": ["M1", "F1"], "Match": True})

        ayto.solve()

        # M1-F1 is confirmed, others must be from first night but not second
        # Second night ruled out: M2-F3, M3-F4, M4-F1
        # First night had 1 match (M1-F1), so M2-F2, M3-F3, M4-F4 are wrong
        expected = np.array([
            [1, 0, 0, 0],  # M1-F1 (confirmed)
            [0, 0, 0, 1],  # M2-F4 (only valid option)
            [0, 1, 0, 0],  # M3-F2 (only valid option)
            [0, 0, 1, 0],  # M4-F3 (only valid option)
        ])

        np.testing.assert_array_equal(ayto.X_binary, expected)


class TestNxMMatching:
    """Test n×m matching scenarios (unequal numbers)."""

    def test_4x3_more_males(self):
        """Test 4 males with 3 females."""
        males = ["M1", "M2", "M3", "M4"]
        females = ["F1", "F2", "F3"]

        ayto = AYTO(males, females)

        # Add constraints to force specific solution
        ayto.add_truth_booth({"Pair": ["M1", "F1"], "Match": True})
        ayto.add_truth_booth({"Pair": ["M2", "F2"], "Match": True})
        ayto.add_truth_booth({"Pair": ["M3", "F3"], "Match": True})

        ayto.solve()

        # One female must have 2 matches, M4 must match someone
        # Verify constraints are satisfied
        assert ayto.X_binary.sum() == 4  # Total of 4 matches (one double)
        assert (ayto.X_binary.sum(axis=1) == 1).sum() == 4  # Each male has 1 match
        assert (ayto.X_binary.sum(axis=0) <= 2).all()  # Each female has at most 2
        assert (ayto.X_binary.sum(axis=0) >= 1).all()  # Each female has at least 1

        # M1, M2, M3 matches are fixed
        assert ayto.X_binary[0, 0] == 1  # M1-F1
        assert ayto.X_binary[1, 1] == 1  # M2-F2
        assert ayto.X_binary[2, 2] == 1  # M3-F3
        assert ayto.X_binary[3, :].sum() == 1  # M4 has exactly one match

    def test_3x4_more_females(self):
        """Test 3 males with 4 females."""
        males = ["M1", "M2", "M3"]
        females = ["F1", "F2", "F3", "F4"]

        ayto = AYTO(males, females)

        # Add constraints
        ayto.add_truth_booth({"Pair": ["M1", "F1"], "Match": True})
        ayto.add_truth_booth({"Pair": ["M2", "F2"], "Match": True})

        ayto.solve()

        # Verify basic constraints
        assert ayto.X_binary.sum() == 4  # Total of 4 matches
        assert (ayto.X_binary.sum(axis=1) <= 2).all()  # Each male has at most 2
        assert (ayto.X_binary.sum(axis=1) >= 1).all()  # Each male has at least 1
        assert (ayto.X_binary.sum(axis=0) == 1).sum() == 4  # Each female has exactly 1

        # Fixed matches
        assert ayto.X_binary[0, 0] == 1  # M1-F1
        assert ayto.X_binary[1, 1] == 1  # M2-F2


class TestConstraintValidation:
    """Test constraint handling and validation."""

    def test_invalid_male_name(self):
        """Test that invalid male names raise errors."""
        males = ["M1", "M2", "M3"]
        females = ["F1", "F2", "F3"]

        ayto = AYTO(males, females)

        with pytest.raises(ValueError, match="invalid"):
            ayto.add_truth_booth({"Pair": ["M4", "F1"], "Match": True})

    def test_invalid_female_name(self):
        """Test that invalid female names raise errors."""
        males = ["M1", "M2", "M3"]
        females = ["F1", "F2", "F3"]

        ayto = AYTO(males, females)

        with pytest.raises(ValueError, match="invalid"):
            ayto.add_truth_booth({"Pair": ["M1", "F4"], "Match": True})

    def test_missing_matching_night_keys(self):
        """Test that matching night missing required keys raises error."""
        males = ["M1", "M2", "M3"]
        females = ["F1", "F2", "F3"]

        ayto = AYTO(males, females)

        with pytest.raises(KeyError):
            ayto.add_matchingnight({"Pairs": [["M1", "F1"]]})  # Missing "Matches"

        with pytest.raises(KeyError):
            ayto.add_matchingnight({"Matches": 1})  # Missing "Pairs"

    def test_missing_truth_booth_keys(self):
        """Test that truth booth missing required keys raises error."""
        males = ["M1", "M2", "M3"]
        females = ["F1", "F2", "F3"]

        ayto = AYTO(males, females)

        with pytest.raises(KeyError):
            ayto.add_truth_booth({"Pair": ["M1", "F1"]})  # Missing "Match"

        with pytest.raises(KeyError):
            ayto.add_truth_booth({"Match": True})  # Missing "Pair"


class TestSolutionProperties:
    """Test that solutions have expected mathematical properties."""

    def test_solution_is_binary(self):
        """Test that solution contains only 0s and 1s."""
        males = ["M1", "M2", "M3"]
        females = ["F1", "F2", "F3"]

        ayto = AYTO(males, females)
        ayto.add_truth_booth({"Pair": ["M1", "F1"], "Match": True})
        ayto.solve()

        # All values should be 0 or 1 (within floating point tolerance)
        assert np.all((ayto.X_binary == 0) | (ayto.X_binary == 1))

    def test_solution_satisfies_row_constraints(self):
        """Test that each male has correct number of matches."""
        males = ["M1", "M2", "M3"]
        females = ["F1", "F2", "F3"]

        ayto = AYTO(males, females)
        ayto.solve()

        # For n×n, each male should have exactly 1 match
        row_sums = ayto.X_binary.sum(axis=1)
        np.testing.assert_array_equal(row_sums, np.ones(3))

    def test_solution_satisfies_column_constraints(self):
        """Test that each female has correct number of matches."""
        males = ["M1", "M2", "M3"]
        females = ["F1", "F2", "F3"]

        ayto = AYTO(males, females)
        ayto.solve()

        # For n×n, each female should have exactly 1 match
        col_sums = ayto.X_binary.sum(axis=0)
        np.testing.assert_array_equal(col_sums, np.ones(3))

    def test_total_matches_nxn(self):
        """Test that n×n matching has exactly n matches."""
        for n in [2, 3, 4, 5]:
            males = [f"M{i}" for i in range(n)]
            females = [f"F{i}" for i in range(n)]

            ayto = AYTO(males, females)
            ayto.solve()

            assert ayto.X_binary.sum() == n

    def test_total_matches_nxm(self):
        """Test that n×m matching has min(n,m) + 1 matches."""
        # 4 males, 3 females -> 4 total matches
        males = ["M1", "M2", "M3", "M4"]
        females = ["F1", "F2", "F3"]

        ayto = AYTO(males, females)
        ayto.solve()

        assert ayto.X_binary.sum() == 4

        # 3 males, 4 females -> 4 total matches
        males = ["M1", "M2", "M3"]
        females = ["F1", "F2", "F3", "F4"]

        ayto = AYTO(males, females)
        ayto.solve()

        assert ayto.X_binary.sum() == 4
