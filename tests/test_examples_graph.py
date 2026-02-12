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
    solver = GraphSolver(data["MALES"], data["FEMALES"])

    for tb in data.get("TRUTH_BOOTH", []):
        solver.add_truth_booth(tb["Pair"][0], tb["Pair"][1], tb["Match"])

    for night in data.get("MATCHING_NIGHTS", []):
        pairs = [(p[0], p[1]) for p in night["Pairs"]]
        solver.add_matching_night(pairs, night["Matches"])

    matchings, capped = solver.enumerate_all_matchings(max_matchings=max_matchings)
    return matchings, capped, solver


def validate_matching(matching, males, females):
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

    if n_males == n_females:
        # Balanced: each person matched exactly once
        assert len(matching) == n_males
        for m in males:
            assert male_counts.get(m, 0) == 1, f"{m} should have exactly 1 match"
        for f in females:
            assert female_counts.get(f, 0) == 1, f"{f} should have exactly 1 match"
    else:
        # Unbalanced: total = max(n,m), one person from smaller set has 2
        expected_total = max(n_males, n_females)
        assert len(matching) == expected_total, \
            f"Expected {expected_total} matches, got {len(matching)}"

        if n_males > n_females:
            # Each male has exactly 1 match
            for m in males:
                assert male_counts.get(m, 0) == 1, f"{m} should have exactly 1 match"
            # Each female has 1 or 2 matches
            for f in females:
                c = female_counts.get(f, 0)
                assert 1 <= c <= 2, f"{f} has {c} matches, expected 1 or 2"
        else:
            # Each female has exactly 1 match
            for f in females:
                assert female_counts.get(f, 0) == 1, f"{f} should have exactly 1 match"
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
        "AYTO_SeasonVIP_Germany_AfterEP20.yaml",
        "AYTO_SeasonVIP2_Germany_AfterEP20.yaml",
        "AYTO_SeasonVIP3_Germany_AfterEP21.yaml",
        "AYTO_SeasonVIP4_Germany_AfterEP18.yaml",
    ])
    def test_season_produces_solutions(self, filename):
        data = load_yaml_example(filename)
        matchings, capped, solver = solve_with_graph(data)

        assert len(matchings) > 0, f"{filename}: no solutions found"

        # Validate first matching
        validate_matching(matchings[0], data["MALES"], data["FEMALES"])

    @pytest.mark.parametrize("filename", [
        "AYTO_Season2_Germany_AfterEp18.yaml",
        "AYTO_Season3_Germany_AfterEp19.yaml",
        "AYTO_Season4_Germany_AfterEp18.yaml",
        "AYTO_Season5_Germany_AfterEP20.yaml",
        "AYTO_SeasonVIP_Germany_AfterEP20.yaml",
        "AYTO_SeasonVIP2_Germany_AfterEP20.yaml",
        "AYTO_SeasonVIP3_Germany_AfterEP21.yaml",
        "AYTO_SeasonVIP4_Germany_AfterEP18.yaml",
    ])
    def test_confirmed_matches_in_all_solutions(self, filename):
        data = load_yaml_example(filename)
        matchings, _, solver = solve_with_graph(data)

        confirmed = {
            (tb["Pair"][0], tb["Pair"][1])
            for tb in data.get("TRUTH_BOOTH", [])
            if tb["Match"]
        }

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
