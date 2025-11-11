"""Smoke tests for YAML example files to ensure solver still works."""
import yaml
import numpy as np
import pytest
from pathlib import Path
from ayto.ayto import AYTO, AYTO_SEASON4


# Get path to examples directory
EXAMPLES_DIR = Path(__file__).parent.parent / "examples"


def load_yaml_example(filename):
    """Load a YAML example file."""
    filepath = EXAMPLES_DIR / filename
    with open(filepath, "r", encoding="utf-8") as f:
        return yaml.load(f, Loader=yaml.SafeLoader)


def validate_solution(ayto_instance):
    """Validate that a solution satisfies basic properties."""
    X = ayto_instance.X_binary
    n_males = len(ayto_instance.males)
    n_females = len(ayto_instance.females)

    # Solution should be binary (0 or 1)
    assert np.all((X >= -0.01) & (X <= 1.01)), "Solution should be binary"

    # Each value should be close to 0 or 1
    for i in range(n_males):
        for j in range(n_females):
            val = X[i, j]
            assert (val < 0.01) or (val > 0.99), \
                f"Value at [{i},{j}] = {val} should be 0 or 1"

    # For balanced (n×n) cases
    if n_males == n_females:
        # Each male has exactly 1 match
        row_sums = X.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(n_males), atol=0.01,
                                    err_msg="Each male should have exactly 1 match")

        # Each female has exactly 1 match
        col_sums = X.sum(axis=0)
        np.testing.assert_allclose(col_sums, np.ones(n_females), atol=0.01,
                                    err_msg="Each female should have exactly 1 match")

        # Total matches = n
        assert abs(X.sum() - n_males) < 0.01

    # For unbalanced (n×m) cases
    else:
        expected_total = min(n_males, n_females) + 1

        # Each person in smaller set has exactly 1 match
        smaller_size = min(n_males, n_females)

        if n_males < n_females:
            # Each male has at most 2, at least 1
            row_sums = X.sum(axis=1)
            assert np.all(row_sums >= 0.99), "Each male should have at least 1 match"
            assert np.all(row_sums <= 2.01), "Each male should have at most 2 matches"

            # Each female has exactly 1 match
            col_sums = X.sum(axis=0)
            assert np.all(col_sums >= 0.99), "Each female should have at least 1 match"
            assert np.all(col_sums <= 1.01), "Each female should have at most 1 match"
        else:
            # Each female has at most 2, at least 1
            col_sums = X.sum(axis=0)
            assert np.all(col_sums >= 0.99), "Each female should have at least 1 match"
            assert np.all(col_sums <= 2.01), "Each female should have at most 2 matches"

            # Each male has exactly 1 match
            row_sums = X.sum(axis=1)
            assert np.all(row_sums >= 0.99), "Each male should have at least 1 match"
            assert np.all(row_sums <= 1.01), "Each male should have at most 1 match"

        # Total matches
        assert abs(X.sum() - expected_total) < 0.01


class TestSeason2Example:
    """Test Season 2 Germany example."""

    def test_season2_loads_and_solves(self):
        """Test that Season 2 example loads and produces valid solution."""
        data = load_yaml_example("AYTO_Season2_Germany_AfterEp18.yaml")

        ayto = AYTO(data["MALES"], data["FEMALES"])

        for night in data["MATCHING_NIGHTS"]:
            ayto.add_matchingnight(night)

        for tb in data["TRUTH_BOOTH"]:
            ayto.add_truth_booth(tb)

        ayto.solve()

        validate_solution(ayto)
        assert ayto.X_binary is not None


class TestSeason3Example:
    """Test Season 3 Germany example."""

    def test_season3_loads_and_solves(self):
        """Test that Season 3 example loads and produces valid solution."""
        data = load_yaml_example("AYTO_Season3_Germany_AfterEp19.yaml")

        ayto = AYTO(data["MALES"], data["FEMALES"])

        for night in data["MATCHING_NIGHTS"]:
            ayto.add_matchingnight(night)

        for tb in data["TRUTH_BOOTH"]:
            ayto.add_truth_booth(tb)

        ayto.solve()

        validate_solution(ayto)


class TestSeason4Example:
    """Test Season 4 Germany example (uses special AYTO_SEASON4 class)."""

    def test_season4_loads_and_solves(self):
        """Test that Season 4 example loads and produces valid solution."""
        data = load_yaml_example("AYTO_Season4_Germany_AfterEp18.yaml")

        # Season 4 uses special class with inequality constraints
        ayto = AYTO_SEASON4(data["MALES"], data["FEMALES"])

        for night in data["MATCHING_NIGHTS"]:
            ayto.add_matchingnight(night)

        for tb in data["TRUTH_BOOTH"]:
            ayto.add_truth_booth(tb)

        ayto.solve()

        # For Season 4, validation is different (females can have 1-2 matches)
        X = ayto.X_binary
        assert X is not None

        # Basic binary check
        for i in range(len(data["MALES"])):
            for j in range(len(data["FEMALES"])):
                val = X[i, j]
                assert (val < 0.01) or (val > 0.99), f"Value should be binary, got {val}"


class TestSeason5Example:
    """Test Season 5 Germany example."""

    def test_season5_loads_and_solves(self):
        """Test that Season 5 example loads and produces valid solution."""
        data = load_yaml_example("AYTO_Season5_Germany_AfterEP20.yaml")

        ayto = AYTO(data["MALES"], data["FEMALES"])

        for night in data["MATCHING_NIGHTS"]:
            ayto.add_matchingnight(night)

        for tb in data["TRUTH_BOOTH"]:
            ayto.add_truth_booth(tb)

        ayto.solve()

        validate_solution(ayto)


class TestSeasonVIPExample:
    """Test Season VIP Germany example."""

    def test_season_vip_loads_and_solves(self):
        """Test that Season VIP example loads and produces valid solution."""
        data = load_yaml_example("AYTO_SeasonVIP_Germany_AfterEP20.yaml")

        ayto = AYTO(data["MALES"], data["FEMALES"])

        for night in data["MATCHING_NIGHTS"]:
            ayto.add_matchingnight(night)

        for tb in data["TRUTH_BOOTH"]:
            ayto.add_truth_booth(tb)

        ayto.solve()

        validate_solution(ayto)

    def test_season_vip_solution_respects_truth_booths(self):
        """Test that confirmed matches appear in solution."""
        data = load_yaml_example("AYTO_SeasonVIP_Germany_AfterEP20.yaml")

        ayto = AYTO(data["MALES"], data["FEMALES"])

        for night in data["MATCHING_NIGHTS"]:
            ayto.add_matchingnight(night)

        for tb in data["TRUTH_BOOTH"]:
            ayto.add_truth_booth(tb)

        ayto.solve()

        # Check confirmed matches
        for tb in data["TRUTH_BOOTH"]:
            if tb["Match"]:
                male = tb["Pair"][0]
                female = tb["Pair"][1]
                m_idx = data["MALES"].index(male)
                f_idx = data["FEMALES"].index(female)
                assert ayto.X_binary[m_idx, f_idx] > 0.99, \
                    f"{male}-{female} should be confirmed match"


class TestSeasonVIP2Example:
    """Test Season VIP2 Germany example."""

    def test_season_vip2_loads_and_solves(self):
        """Test that Season VIP2 example loads and produces valid solution."""
        data = load_yaml_example("AYTO_SeasonVIP2_Germany_AfterEP20.yaml")

        ayto = AYTO(data["MALES"], data["FEMALES"])

        for night in data["MATCHING_NIGHTS"]:
            ayto.add_matchingnight(night)

        for tb in data["TRUTH_BOOTH"]:
            ayto.add_truth_booth(tb)

        ayto.solve()

        validate_solution(ayto)


class TestSeasonVIP3Example:
    """Test Season VIP3 Germany example."""

    def test_season_vip3_loads_and_solves(self):
        """Test that Season VIP3 example loads and produces valid solution."""
        data = load_yaml_example("AYTO_SeasonVIP3_Germany_AfterEP21.yaml")

        ayto = AYTO(data["MALES"], data["FEMALES"])

        for night in data["MATCHING_NIGHTS"]:
            ayto.add_matchingnight(night)

        for tb in data["TRUTH_BOOTH"]:
            ayto.add_truth_booth(tb)

        ayto.solve()

        validate_solution(ayto)


class TestSeasonVIP4Example:
    """Test Season VIP4 Germany example."""

    def test_season_vip4_loads_and_solves(self):
        """Test that Season VIP4 example loads and produces valid solution."""
        data = load_yaml_example("AYTO_SeasonVIP4_Germany_AfterEP18.yaml")

        ayto = AYTO(data["MALES"], data["FEMALES"])

        for night in data["MATCHING_NIGHTS"]:
            ayto.add_matchingnight(night)

        for tb in data["TRUTH_BOOTH"]:
            ayto.add_truth_booth(tb)

        ayto.solve()

        validate_solution(ayto)


class TestAllExamplesProduceSolutions:
    """Test that all examples produce some solution."""

    @pytest.mark.parametrize("filename", [
        "AYTO_Season2_Germany_AfterEp18.yaml",
        "AYTO_Season3_Germany_AfterEp19.yaml",
        "AYTO_Season5_Germany_AfterEP20.yaml",
        "AYTO_SeasonVIP_Germany_AfterEP20.yaml",
        "AYTO_SeasonVIP2_Germany_AfterEP20.yaml",
        "AYTO_SeasonVIP3_Germany_AfterEP21.yaml",
        "AYTO_SeasonVIP4_Germany_AfterEP18.yaml",
    ])
    def test_example_produces_solution(self, filename):
        """Test that each example file produces a solution."""
        data = load_yaml_example(filename)

        ayto = AYTO(data["MALES"], data["FEMALES"])

        for night in data["MATCHING_NIGHTS"]:
            ayto.add_matchingnight(night)

        for tb in data["TRUTH_BOOTH"]:
            ayto.add_truth_booth(tb)

        ayto.solve()

        # Just verify we got some solution
        assert ayto.X_binary is not None
        assert ayto.X_binary.sum() > 0
