"""Graph-solver tests for explicit multi-match degree profiles."""

import pytest

from ayto_solver.solvers.graph_solver import GraphSolver


def degrees(matching, names, position):
    """Return selected edge counts for one side of a matching."""
    counts = {name: 0 for name in names}
    for pair in matching:
        counts[pair[position]] += 1
    return counts


def test_balanced_roster_can_have_a_double_on_each_side():
    """Explicit degrees support two simultaneous double-match people."""
    males = ["A", "B", "C", "D"]
    females = ["W", "X", "Y", "Z"]
    solver = GraphSolver(
        males,
        females,
        degree_profile={
            "males": {"A": 2},
            "females": {"W": 2},
        },
    )
    solver.add_truth_booth("A", "W", True)
    solver.add_truth_booth("A", "X", True)
    solver.add_truth_booth("B", "W", True)

    matchings, capped = solver.enumerate_all_matchings(max_matchings=100)

    assert not capped
    assert matchings
    for matching in matchings:
        assert degrees(matching, males, 0) == {"A": 2, "B": 1, "C": 1, "D": 1}
        assert degrees(matching, females, 1) == {"W": 2, "X": 1, "Y": 1, "Z": 1}

    double_probs = solver.calculate_double_match_probabilities(matchings)
    assert double_probs == {"A": 1.0, "W": 1.0}


def test_explicit_profile_can_leave_a_late_entrant_unmatched():
    """A late entrant can be modeled without abusing roster cardinality."""
    solver = GraphSolver(
        ["A", "B", "Late"],
        ["X", "Y", "Z"],
        degree_profile={
            "males": {"A": 2, "Late": 0},
        },
    )

    matchings, capped = solver.enumerate_all_matchings(max_matchings=100)

    assert not capped
    assert matchings
    for matching in matchings:
        assert degrees(matching, ["A", "B", "Late"], 0) == {
            "A": 2,
            "B": 1,
            "Late": 0,
        }
        assert degrees(matching, ["X", "Y", "Z"], 1) == {
            "X": 1,
            "Y": 1,
            "Z": 1,
        }


def test_alternative_match_group_requires_exactly_one_edge():
    solver = GraphSolver(["A", "B"], ["X", "Y"])
    solver.add_exclusive_alternative([("A", "X"), ("A", "Y")])

    matchings, capped = solver.enumerate_all_matchings(max_matchings=100)

    assert not capped
    assert {tuple(sorted(matching)) for matching in matchings} == {
        (("A", "X"), ("B", "Y")),
        (("A", "Y"), ("B", "X")),
    }


def test_inconsistent_degree_profile_is_rejected():
    with pytest.raises(ValueError, match="degree total"):
        GraphSolver(
            ["A", "B"],
            ["X", "Y"],
            degree_profile={
                "males": {"A": 2, "B": 0},
                "females": {"X": 2},
            },
        )
