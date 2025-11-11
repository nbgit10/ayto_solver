"""Synthetic test cases with known solutions for AYTO solver."""
import numpy as np
import pytest
from ayto_solver.solvers.mip_solver import MIPSolver


class TestSimple3x3:
    """Test 3x3 matching with controlled constraints."""

    def test_3x3_unique_solution(self):
        """Test 3x3 case with constraints that lead to unique solution."""
        males = ["M1", "M2", "M3"]
        females = ["F1", "F2", "F3"]

        solver = MIPSolver(males, females)

        # Add truth booth: M1-F1 is a match
        solver.add_truth_booth("M1", "F1", True)

        # Add truth booth: M2-F2 is a match
        solver.add_truth_booth("M2", "F2", True)

        # This forces M3-F3 to be the third match
        solver.solve()

        # Expected solution: diagonal matrix
        expected = np.array([
            [1, 0, 0],  # M1-F1
            [0, 1, 0],  # M2-F2
            [0, 0, 1],  # M3-F3
        ])

        np.testing.assert_array_equal(solver.X_binary, expected)

    def test_3x3_matching_night_constraint(self):
        """Test 3x3 with matching night providing constraints."""
        males = ["M1", "M2", "M3"]
        females = ["F1", "F2", "F3"]

        solver = MIPSolver(males, females)

        # Matching night with 0 matches - all pairs are wrong
        solver.add_matching_night([("M1", "F1"), ("M2", "F2"), ("M3", "F3")], 0)

        # Truth booth: M1-F2 is a match
        solver.add_truth_booth("M1", "F2", True)

        # Truth booth: M2-F3 is a match
        solver.add_truth_booth("M2", "F3", True)

        solver.solve()

        # Expected solution: M1-F2, M2-F3, M3-F1
        expected = np.array([
            [0, 1, 0],  # M1-F2
            [0, 0, 1],  # M2-F3
            [1, 0, 0],  # M3-F1
        ])

        np.testing.assert_array_equal(solver.X_binary, expected)

    def test_3x3_negative_truth_booth(self):
        """Test that negative truth booth results are respected."""
        males = ["M1", "M2", "M3"]
        females = ["F1", "F2", "F3"]

        solver = MIPSolver(males, females)

        # M1-F1 is NOT a match
        solver.add_truth_booth("M1", "F1", False)

        # M1-F2 is NOT a match
        solver.add_truth_booth("M1", "F2", False)

        # This forces M1-F3
        solver.add_truth_booth("M2", "F1", True)
        solver.add_truth_booth("M3", "F2", True)

        solver.solve()

        # Expected solution
        expected = np.array([
            [0, 0, 1],  # M1-F3 (only option left)
            [1, 0, 0],  # M2-F1
            [0, 1, 0],  # M3-F2
        ])

        np.testing.assert_array_equal(solver.X_binary, expected)


class TestSimple4x4:
    """Test 4x4 matching with controlled constraints."""

    def test_4x4_unique_solution(self):
        """Test 4x4 case with full truth booth information."""
        males = ["M1", "M2", "M3", "M4"]
        females = ["F1", "F2", "F3", "F4"]

        solver = MIPSolver(males, females)

        # Provide 3 matches via truth booth
        solver.add_truth_booth("M1", "F2", True)
        solver.add_truth_booth("M2", "F4", True)
        solver.add_truth_booth("M3", "F1", True)

        # This forces M4-F3
        solver.solve()

        expected = np.array([
            [0, 1, 0, 0],  # M1-F2
            [0, 0, 0, 1],  # M2-F4
            [1, 0, 0, 0],  # M3-F1
            [0, 0, 1, 0],  # M4-F3
        ])

        np.testing.assert_array_equal(solver.X_binary, expected)

    def test_4x4_with_matching_nights(self):
        """Test 4x4 with multiple matching nights."""
        males = ["M1", "M2", "M3", "M4"]
        females = ["F1", "F2", "F3", "F4"]

        solver = MIPSolver(males, females)

        # First matching night: 1 match
        solver.add_matching_night([
            ("M1", "F1"),
            ("M2", "F2"),
            ("M3", "F3"),
            ("M4", "F4")
        ], 1)

        # Second matching night: 0 matches (all wrong)
        solver.add_matching_night([
            ("M1", "F2"),
            ("M2", "F3"),
            ("M3", "F4"),
            ("M4", "F1")
        ], 0)

        # Truth booth
        solver.add_truth_booth("M1", "F1", True)

        solver.solve()

        # M1-F1 is confirmed, others must be from first night but not second
        # Second night ruled out: M2-F3, M3-F4, M4-F1
        # First night had 1 match (M1-F1), so M2-F2, M3-F3, M4-F4 are wrong
        expected = np.array([
            [1, 0, 0, 0],  # M1-F1 (confirmed)
            [0, 0, 0, 1],  # M2-F4 (only valid option)
            [0, 1, 0, 0],  # M3-F2 (only valid option)
            [0, 0, 1, 0],  # M4-F3 (only valid option)
        ])

        np.testing.assert_array_equal(solver.X_binary, expected)


class TestNxMMatching:
    """Test n×m matching scenarios (unequal numbers)."""

    def test_4x3_more_males(self):
        """Test 4 males with 3 females."""
        males = ["M1", "M2", "M3", "M4"]
        females = ["F1", "F2", "F3"]

        solver = MIPSolver(males, females)

        # Add constraints to force specific solution
        solver.add_truth_booth("M1", "F1", True)
        solver.add_truth_booth("M2", "F2", True)
        solver.add_truth_booth("M3", "F3", True)

        solver.solve()

        # One female must have 2 matches, M4 must match someone
        # Verify constraints are satisfied
        assert solver.X_binary.sum() == 4  # Total of 4 matches (one double)
        assert (solver.X_binary.sum(axis=1) == 1).sum() == 4  # Each male has 1 match
        assert (solver.X_binary.sum(axis=0) <= 2).all()  # Each female has at most 2
        assert (solver.X_binary.sum(axis=0) >= 1).all()  # Each female has at least 1

        # M1, M2, M3 matches are fixed
        assert solver.X_binary[0, 0] == 1  # M1-F1
        assert solver.X_binary[1, 1] == 1  # M2-F2
        assert solver.X_binary[2, 2] == 1  # M3-F3
        assert solver.X_binary[3, :].sum() == 1  # M4 has exactly one match

    def test_3x4_more_females(self):
        """Test 3 males with 4 females."""
        males = ["M1", "M2", "M3"]
        females = ["F1", "F2", "F3", "F4"]

        solver = MIPSolver(males, females)

        # Add constraints
        solver.add_truth_booth("M1", "F1", True)
        solver.add_truth_booth("M2", "F2", True)

        solver.solve()

        # Verify basic constraints
        assert solver.X_binary.sum() == 4  # Total of 4 matches
        assert (solver.X_binary.sum(axis=1) <= 2).all()  # Each male has at most 2
        assert (solver.X_binary.sum(axis=1) >= 1).all()  # Each male has at least 1
        assert (solver.X_binary.sum(axis=0) == 1).sum() == 4  # Each female has exactly 1

        # Fixed matches
        assert solver.X_binary[0, 0] == 1  # M1-F1
        assert solver.X_binary[1, 1] == 1  # M2-F2


class TestConstraintValidation:
    """Test constraint handling and validation."""

    def test_invalid_male_name(self):
        """Test that invalid male names raise errors."""
        males = ["M1", "M2", "M3"]
        females = ["F1", "F2", "F3"]

        solver = MIPSolver(males, females)

        with pytest.raises(ValueError, match="not in contestants"):
            solver.add_truth_booth("M4", "F1", True)

    def test_invalid_female_name(self):
        """Test that invalid female names raise errors."""
        males = ["M1", "M2", "M3"]
        females = ["F1", "F2", "F3"]

        solver = MIPSolver(males, females)

        with pytest.raises(ValueError, match="not in contestants"):
            solver.add_truth_booth("M1", "F4", True)

    def test_invalid_matching_night_male(self):
        """Test that matching night with invalid male name raises error."""
        males = ["M1", "M2", "M3"]
        females = ["F1", "F2", "F3"]

        solver = MIPSolver(males, females)

        with pytest.raises(ValueError, match="not in contestants"):
            solver.add_matching_night([("M4", "F1")], 1)

    def test_invalid_matching_night_female(self):
        """Test that matching night with invalid female name raises error."""
        males = ["M1", "M2", "M3"]
        females = ["F1", "F2", "F3"]

        solver = MIPSolver(males, females)

        with pytest.raises(ValueError, match="not in contestants"):
            solver.add_matching_night([("M1", "F4")], 1)


class TestSolutionProperties:
    """Test that solutions have expected mathematical properties."""

    def test_solution_is_binary(self):
        """Test that solution contains only 0s and 1s."""
        males = ["M1", "M2", "M3"]
        females = ["F1", "F2", "F3"]

        solver = MIPSolver(males, females)
        solver.add_truth_booth("M1", "F1", True)
        solver.solve()

        # All values should be 0 or 1 (within floating point tolerance)
        assert np.all((solver.X_binary == 0) | (solver.X_binary == 1))

    def test_solution_satisfies_row_constraints(self):
        """Test that each male has correct number of matches."""
        males = ["M1", "M2", "M3"]
        females = ["F1", "F2", "F3"]

        solver = MIPSolver(males, females)
        solver.solve()

        # For n×n, each male should have exactly 1 match
        row_sums = solver.X_binary.sum(axis=1)
        np.testing.assert_array_equal(row_sums, np.ones(3))

    def test_solution_satisfies_column_constraints(self):
        """Test that each female has correct number of matches."""
        males = ["M1", "M2", "M3"]
        females = ["F1", "F2", "F3"]

        solver = MIPSolver(males, females)
        solver.solve()

        # For n×n, each female should have exactly 1 match
        col_sums = solver.X_binary.sum(axis=0)
        np.testing.assert_array_equal(col_sums, np.ones(3))

    def test_total_matches_nxn(self):
        """Test that n×n matching has exactly n matches."""
        for n in [2, 3, 4, 5]:
            males = [f"M{i}" for i in range(n)]
            females = [f"F{i}" for i in range(n)]

            solver = MIPSolver(males, females)
            solver.solve()

            assert solver.X_binary.sum() == n

    def test_total_matches_nxm(self):
        """Test that n×m matching has min(n,m) + 1 matches."""
        # 4 males, 3 females -> 4 total matches
        males = ["M1", "M2", "M3", "M4"]
        females = ["F1", "F2", "F3"]

        solver = MIPSolver(males, females)
        solver.solve()

        assert solver.X_binary.sum() == 4

        # 3 males, 4 females -> 4 total matches
        males = ["M1", "M2", "M3"]
        females = ["F1", "F2", "F3", "F4"]

        solver = MIPSolver(males, females)
        solver.solve()

        assert solver.X_binary.sum() == 4
