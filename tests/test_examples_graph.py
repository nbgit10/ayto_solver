"""Smoke tests for YAML example files using the Graph solver.

Mirrors test_examples.py but uses GraphSolver instead of MIPSolver.
Can run on any platform (no AMD64/Docker requirement).
"""
import yaml
import pytest
from pathlib import Path
from ayto_solver.solvers.graph_solver import GraphSolver


EXAMPLES_DIR = Path(__file__).parent.parent / "examples"


def load_yaml_example(filename):
    filepath = EXAMPLES_DIR / filename
    with open(filepath, "r", encoding="utf-8") as f:
        return yaml.load(f, Loader=yaml.SafeLoader)


def solve_with_graph(data, max_matchings=10000):
    """Load data into GraphSolver, return (matchings, capped, solver)."""
    solver = GraphSolver(
        data["MALES"],
        data["FEMALES"],
        degree_profile=data.get("DEGREE_PROFILE"),
    )

    for tb in data.get("TRUTH_BOOTH", []):
        if tb["Match"] is not None:
            solver.add_truth_booth(tb["Pair"][0], tb["Pair"][1], tb["Match"])

    for pair in data.get("ADDITIONAL_CONFIRMED_MATCHES", []):
        solver.add_truth_booth(pair[0], pair[1], True)

    for alternative in data.get("ALTERNATIVE_MATCHES", []):
        pairs = alternative.get("Pairs", alternative)
        solver.add_exclusive_alternative(
            [(pair[0], pair[1]) for pair in pairs]
        )

    for night in data.get("MATCHING_NIGHTS", []):
        pairs = [(p[0], p[1]) for p in night["Pairs"]]
        solver.add_matching_night(pairs, night["Matches"])

    matchings, capped = solver.enumerate_all_matchings(max_matchings=max_matchings)
    return matchings, capped, solver


def validate_matching(matching, males, females, degree_profiles=None):
    """Validate a single matching satisfies structural constraints."""
    n_males = len(males)
    n_females = len(females)

    male_counts = {}
    female_counts = {}
    for m, f in matching:
        assert m in males, f"Unknown male: {m}"
        assert f in females, f"Unknown female: {f}"
        male_counts[m] = male_counts.get(m, 0) + 1
        female_counts[f] = female_counts.get(f, 0) + 1

    if degree_profiles is not None:
        actual_male_counts = {
            male: male_counts.get(male, 0) for male in males
        }
        actual_female_counts = {
            female: female_counts.get(female, 0) for female in females
        }
        assert any(
            actual_male_counts == male_degrees
            and actual_female_counts == female_degrees
            for male_degrees, female_degrees in degree_profiles
        ), "Matching does not satisfy any explicit degree profile"
    elif n_males == n_females:
        # Balanced: each person matched exactly once
        assert len(matching) == n_males
        for m in males:
            assert male_counts.get(m, 0) == 1, f"{m} should have exactly 1 match"
        for f in females:
            assert female_counts.get(f, 0) == 1, f"{f} should have exactly 1 match"
    else:
        # Unbalanced: one person from smaller set has double match
        # Total edges = min(n,m) + 1 (the solver's target)
        expected_total = min(n_males, n_females) + 1
        assert len(matching) == expected_total, \
            f"Expected {expected_total} matches, got {len(matching)}"

        if n_males > n_females:
            # Each male has 0 or 1 matches (some may be unmatched if diff > 1)
            for m in males:
                c = male_counts.get(m, 0)
                assert c <= 1, f"{m} has {c} matches, expected 0 or 1"
            # Each female has 1 or 2 matches
            for f in females:
                c = female_counts.get(f, 0)
                assert 1 <= c <= 2, f"{f} has {c} matches, expected 1 or 2"
        else:
            # Each female has 0 or 1 matches (some may be unmatched if diff > 1)
            for f in females:
                c = female_counts.get(f, 0)
                assert c <= 1, f"{f} has {c} matches, expected 0 or 1"
            # Each male has 1 or 2 matches
            for m in males:
                c = male_counts.get(m, 0)
                assert 1 <= c <= 2, f"{m} has {c} matches, expected 1 or 2"


class TestGraphSolverAllSeasons:
    """Test all seasons produce solutions with valid structure."""

    @pytest.mark.parametrize("filename", [
        "AYTO_Season2_Germany_AfterEp18.yaml",
        "AYTO_Season3_Germany_AfterEp19.yaml",
        "AYTO_Season4_Germany_AfterEp18.yaml",
        "AYTO_Season5_Germany_AfterEP20.yaml",
        "AYTO_Season6_Germany_AfterEp20.yaml",
        "AYTO_Season7_Germany_AfterEp10.yaml",
        "AYTO_SeasonVIP_Germany_AfterEP20.yaml",
        "AYTO_SeasonVIP2_Germany_AfterEP20.yaml",
        "AYTO_SeasonVIP3_Germany_AfterEP21.yaml",
        "AYTO_SeasonVIP4_Germany_AfterEP18.yaml",
        "AYTO_SeasonVIP5_Germany_AfterEP20.yaml",
        "AYTO_SeasonVIP6_Germany_AfterEP10.yaml",
    ])
    def test_season_produces_solutions(self, filename):
        data = load_yaml_example(filename)
        matchings, capped, solver = solve_with_graph(data)

        assert len(matchings) > 0, f"{filename}: no solutions found"

        # Validate first matching
        validate_matching(
            matchings[0],
            data["MALES"],
            data["FEMALES"],
            solver.degree_profiles if data.get("DEGREE_PROFILE") else None,
        )

    @pytest.mark.parametrize("filename", [
        "AYTO_Season2_Germany_AfterEp18.yaml",
        "AYTO_Season3_Germany_AfterEp19.yaml",
        "AYTO_Season4_Germany_AfterEp18.yaml",
        "AYTO_Season5_Germany_AfterEP20.yaml",
        "AYTO_Season6_Germany_AfterEp20.yaml",
        "AYTO_Season7_Germany_AfterEp10.yaml",
        "AYTO_SeasonVIP_Germany_AfterEP20.yaml",
        "AYTO_SeasonVIP2_Germany_AfterEP20.yaml",
        "AYTO_SeasonVIP3_Germany_AfterEP21.yaml",
        "AYTO_SeasonVIP4_Germany_AfterEP18.yaml",
        "AYTO_SeasonVIP5_Germany_AfterEP20.yaml",
        "AYTO_SeasonVIP6_Germany_AfterEP10.yaml",
    ])
    def test_confirmed_matches_in_all_solutions(self, filename):
        data = load_yaml_example(filename)
        matchings, _, solver = solve_with_graph(data)

        confirmed = {
            (tb["Pair"][0], tb["Pair"][1])
            for tb in data.get("TRUTH_BOOTH", [])
            if tb["Match"] is True
        }
        confirmed.update(
            (pair[0], pair[1])
            for pair in data.get("ADDITIONAL_CONFIRMED_MATCHES", [])
        )

        for i, matching in enumerate(matchings):
            for pair in confirmed:
                assert pair in matching, \
                    f"{filename} solution {i}: confirmed pair {pair} not found"


class TestSeason4DoubleMatch:
    """Season 4 specific: Caro is the double match (Ken + Max)."""

    def test_caro_double_match(self):
        data = load_yaml_example("AYTO_Season4_Germany_AfterEp18.yaml")
        matchings, capped, solver = solve_with_graph(data)

        assert len(matchings) > 0
        assert not capped

        for matching in matchings:
            caro_partners = {m for m, f in matching if f == "Caro"}
            assert caro_partners == {"Ken", "Max"}, \
                f"Caro should match Ken and Max, got {caro_partners}"

    def test_double_match_probability(self):
        data = load_yaml_example("AYTO_Season4_Germany_AfterEp18.yaml")
        matchings, _, solver = solve_with_graph(data)

        dm_probs = solver.calculate_double_match_probabilities(matchings)
        assert dm_probs.get("Caro", 0) == 1.0, "Caro should be double match in 100% of solutions"


class TestSeason6DoubleMatch:
    """Season 6: Tano has double match (Joanna + Sophia). 3 solutions remain."""

    def test_solution_count(self):
        data = load_yaml_example("AYTO_Season6_Germany_AfterEp20.yaml")
        matchings, capped, _ = solve_with_graph(data)
        assert not capped
        assert len(matchings) == 3

    def test_tano_double_match_in_solutions(self):
        data = load_yaml_example("AYTO_Season6_Germany_AfterEp20.yaml")
        matchings, _, _ = solve_with_graph(data)

        # Tano/Joanna+Sophia should be one of the 3 solutions
        for matching in matchings:
            tano_partners = {f for m, f in matching if m == "Tano"}
            if tano_partners == {"Joanna", "Sophia"}:
                return  # Found it
        pytest.fail("Expected Tano/Joanna+Sophia solution not found")


class TestVIP5:
    """VIP 5: 12 men, 10 women. Jimi Blue latecomer with no match."""

    def test_solution_count(self):
        data = load_yaml_example("AYTO_SeasonVIP5_Germany_AfterEP20.yaml")
        matchings, capped, _ = solve_with_graph(data)
        assert not capped
        assert len(matchings) == 28

    def test_confirmed_pms_present(self):
        """Xander/Elissia, Lennert/Sandra, Calvin O./Nelly in all solutions."""
        data = load_yaml_example("AYTO_SeasonVIP5_Germany_AfterEP20.yaml")
        matchings, _, _ = solve_with_graph(data)

        expected = {
            ("Xander", "Elissia"),
            ("Lennert", "Sandra"),
            ("Calvin O.", "Nelly"),
        }
        for i, matching in enumerate(matchings):
            for pair in expected:
                assert pair in matching, \
                    f"Solution {i}: confirmed pair {pair} not found"

    def test_double_match_is_female(self):
        """Double match should always be on a female (12M > 10F)."""
        data = load_yaml_example("AYTO_SeasonVIP5_Germany_AfterEP20.yaml")
        matchings, _, solver = solve_with_graph(data)

        for i, matching in enumerate(matchings):
            female_counts = {}
            for m, f in matching:
                female_counts[f] = female_counts.get(f, 0) + 1
            doubles = [f for f, c in female_counts.items() if c > 1]
            assert len(doubles) == 1, \
                f"Solution {i}: expected 1 double match, got {len(doubles)}"


class TestVIP6:
    """Current 2026 VIP season data sourced from AYTOFANS."""

    def test_solution_count(self):
        data = load_yaml_example("AYTO_SeasonVIP6_Germany_AfterEP10.yaml")
        matchings, capped, _ = solve_with_graph(data)

        assert not capped
        assert len(matchings) == 56

    def test_double_match_is_preserved(self):
        data = load_yaml_example("AYTO_SeasonVIP6_Germany_AfterEP10.yaml")
        matchings, _, solver = solve_with_graph(data)

        for matching in matchings:
            assert ("Johannes", "Marta") in matching
            assert ("Johannes", "Janice") in matching
            female_counts = {}
            for _, female in matching:
                female_counts[female] = female_counts.get(female, 0) + 1
            double_females = [
                female for female, count in female_counts.items() if count == 2
            ]
            assert len(double_females) == 1
            assert ("Laurenz", double_females[0]) in matching

        double_probs = solver.calculate_double_match_probabilities(matchings)
        assert double_probs["Johannes"] == 1.0

    def test_matchbox_data(self):
        data = load_yaml_example("AYTO_SeasonVIP6_Germany_AfterEP10.yaml")
        boxes = data["TRUTH_BOOTH"]

        assert len(boxes) == 6
        assert boxes[3]["Pair"] == ["Johannes", "Marta"]
        assert boxes[3]["Match"] is True
        assert boxes[5]["Pair"] == ["Laurenz", "Emma"]
        assert boxes[5]["Match"] is None
        assert boxes[5]["Sold"] is True
        assert boxes[5]["SoldPrice"] == 10000

    def test_matching_nights_have_source_counts_and_loners(self):
        data = load_yaml_example("AYTO_SeasonVIP6_Germany_AfterEP10.yaml")

        assert [night["Matches"] for night in data["MATCHING_NIGHTS"]] == [3, 3, 3, 4, 3]
        assert [night["LONERS"] for night in data["MATCHING_NIGHTS"]] == [
            ["Joena"],
            ["Francesca"],
            ["Janice"],
            [],
            ["Fabi"],
        ]
