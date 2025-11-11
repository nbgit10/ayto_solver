"""Graph-based solver for AYTO matching problem using bipartite graph matching."""
import networkx as nx
from networkx.algorithms import bipartite
import numpy as np
from typing import List, Tuple, Dict, Set
from collections import defaultdict
from itertools import combinations


class GraphSolver:
    """
    Graph-based solver for AYTO matching problem.

    Models the problem as a bipartite graph where:
    - Nodes are contestants (males and females)
    - Edges exist between possible matches (not ruled out by constraints)
    - Find all maximum matchings to enumerate solutions
    """

    def __init__(self, males: List[str], females: List[str]):
        """
        Initialize graph solver.

        Args:
            males: List of male contestant names
            females: List of female contestant names
        """
        self.males = males
        self.females = females
        self.n_males = len(males)
        self.n_females = len(females)

        # Create bipartite graph
        self.graph = nx.Graph()

        # Add nodes
        self.graph.add_nodes_from(
            [(f"M_{m}", {"bipartite": 0}) for m in males]
        )
        self.graph.add_nodes_from(
            [(f"F_{f}", {"bipartite": 1}) for f in females]
        )

        # Initially, all pairs are possible (add all edges)
        for male in males:
            for female in females:
                self.graph.add_edge(f"M_{male}", f"F_{female}")

        # Track ruled out pairs
        self.ruled_out_pairs: Set[Tuple[str, str]] = set()
        self.confirmed_pairs: Set[Tuple[str, str]] = set()

    def add_truth_booth(self, male: str, female: str, is_match: bool):
        """
        Add truth booth constraint.

        Args:
            male: Male contestant name
            female: Female contestant name
            is_match: True if they are a perfect match, False otherwise
        """
        if is_match:
            self.confirmed_pairs.add((male, female))
            # Remove all other edges for these two people
            for other_female in self.females:
                if other_female != female:
                    edge = (f"M_{male}", f"F_{other_female}")
                    if self.graph.has_edge(*edge):
                        self.graph.remove_edge(*edge)
                    self.ruled_out_pairs.add((male, other_female))

            for other_male in self.males:
                if other_male != male:
                    edge = (f"M_{other_male}", f"F_{female}")
                    if self.graph.has_edge(*edge):
                        self.graph.remove_edge(*edge)
                    self.ruled_out_pairs.add((other_male, female))
        else:
            # Remove this edge
            edge = (f"M_{male}", f"F_{female}")
            if self.graph.has_edge(*edge):
                self.graph.remove_edge(*edge)
            self.ruled_out_pairs.add((male, female))

    def add_matching_night(self, pairs: List[Tuple[str, str]], num_matches: int):
        """
        Add matching night constraint.

        For now, we handle special cases:
        - If num_matches == 0, all pairs are ruled out
        - If num_matches == len(pairs), all pairs are confirmed
        - Otherwise, we need more sophisticated constraint handling

        Args:
            pairs: List of (male, female) pairs from that night
            num_matches: Number of correct matches
        """
        if num_matches == 0:
            # All pairs are wrong - remove all edges
            for male, female in pairs:
                edge = (f"M_{male}", f"F_{female}")
                if self.graph.has_edge(*edge):
                    self.graph.remove_edge(*edge)
                self.ruled_out_pairs.add((male, female))

        elif num_matches == len(pairs):
            # All pairs are correct - this determines the solution
            for male, female in pairs:
                self.confirmed_pairs.add((male, female))
                # Remove all other edges
                for other_female in self.females:
                    if other_female != female:
                        edge = (f"M_{male}", f"F_{other_female}")
                        if self.graph.has_edge(*edge):
                            self.graph.remove_edge(*edge)

                for other_male in self.males:
                    if other_male != male:
                        edge = (f"M_{other_male}", f"F_{female}")
                        if self.graph.has_edge(*edge):
                            self.graph.remove_edge(*edge)
        # For intermediate cases, we can't directly prune edges without
        # constraint propagation, so we'll handle this during enumeration

    def enumerate_all_matchings(
        self,
        max_matchings: int = 1000
    ) -> Tuple[List[Set[Tuple[str, str]]], bool]:
        """
        Enumerate all maximum matchings in the bipartite graph.

        Args:
            max_matchings: Maximum number of matchings to find

        Returns:
            Tuple of (list of matchings, whether cap was hit)
        """
        matchings = []
        capped = False

        # For balanced (n×n) graphs, use standard matching enumeration
        if self.n_males == self.n_females:
            matchings = list(self._enumerate_perfect_matchings_balanced(max_matchings))
            if len(matchings) >= max_matchings:
                capped = True
        else:
            # For unbalanced (n×m) graphs, we need to handle double matches
            matchings = list(self._enumerate_matchings_unbalanced(max_matchings))
            if len(matchings) >= max_matchings:
                capped = True

        return matchings, capped

    def _enumerate_perfect_matchings_balanced(self, max_count: int):
        """
        Enumerate perfect matchings for balanced bipartite graph.

        Uses recursive backtracking.
        """
        n = min(self.n_males, self.n_females)
        males_list = list(self.males)

        def backtrack(
            male_idx: int,
            current_matching: Set[Tuple[str, str]],
            used_females: Set[str],
            count: List[int]
        ):
            if count[0] >= max_count:
                return

            if male_idx == n:
                # Found a complete matching
                yield current_matching.copy()
                count[0] += 1
                return

            male = males_list[male_idx]
            male_node = f"M_{male}"

            # Try each female this male can match with
            for female in self.females:
                if female in used_females:
                    continue

                female_node = f"F_{female}"
                if not self.graph.has_edge(male_node, female_node):
                    continue

                # Add this pair to matching
                current_matching.add((male, female))
                used_females.add(female)

                # Recursively match remaining males
                yield from backtrack(male_idx + 1, current_matching, used_females, count)

                # Backtrack
                current_matching.remove((male, female))
                used_females.remove(female)

        count = [0]
        yield from backtrack(0, set(), set(), count)

    def _enumerate_matchings_unbalanced(self, max_count: int):
        """
        Enumerate matchings for unbalanced bipartite graph (n×m case).

        In n×m cases, we have min(n,m)+1 total matches, with one person
        involved in a double match.
        """
        n_smaller = min(self.n_males, self.n_females)
        target_matches = n_smaller + 1

        # Determine which set is larger
        if self.n_females > self.n_males:
            # One male will have 2 matches
            yield from self._enumerate_with_double_match_male(max_count, target_matches)
        else:
            # One female will have 2 matches
            yield from self._enumerate_with_double_match_female(max_count, target_matches)

    def _enumerate_with_double_match_male(self, max_count: int, target_matches: int):
        """Enumerate matchings where one male has 2 matches."""
        count = [0]

        # Try each male as the one with double match
        for double_male in self.males:
            if count[0] >= max_count:
                return

            # Find all pairs of females this male can match with
            possible_females = [
                f for f in self.females
                if self.graph.has_edge(f"M_{double_male}", f"F_{f}")
            ]

            if len(possible_females) < 2:
                continue

            # Try each pair of females for the double match
            for female1, female2 in combinations(possible_females, 2):
                if count[0] >= max_count:
                    return

                # Start matching with this double match
                current_matching = {(double_male, female1), (double_male, female2)}
                used_males = {double_male}
                used_females = {female1, female2}

                # Match remaining males with remaining females (1-to-1)
                remaining_males = [m for m in self.males if m != double_male]
                remaining_females = [f for f in self.females if f not in {female1, female2}]

                # Use backtracking for remaining 1-to-1 matches
                def backtrack_remaining(idx: int):
                    if count[0] >= max_count:
                        return

                    if idx == len(remaining_males):
                        if len(current_matching) == target_matches:
                            yield current_matching.copy()
                            count[0] += 1
                        return

                    male = remaining_males[idx]
                    male_node = f"M_{male}"

                    for female in remaining_females:
                        if female in used_females:
                            continue

                        female_node = f"F_{female}"
                        if not self.graph.has_edge(male_node, female_node):
                            continue

                        current_matching.add((male, female))
                        used_females.add(female)

                        yield from backtrack_remaining(idx + 1)

                        current_matching.remove((male, female))
                        used_females.remove(female)

                yield from backtrack_remaining(0)

    def _enumerate_with_double_match_female(self, max_count: int, target_matches: int):
        """Enumerate matchings where one female has 2 matches."""
        count = [0]

        # Try each female as the one with double match
        for double_female in self.females:
            if count[0] >= max_count:
                return

            # Find all pairs of males this female can match with
            possible_males = [
                m for m in self.males
                if self.graph.has_edge(f"M_{m}", f"F_{double_female}")
            ]

            if len(possible_males) < 2:
                continue

            # Try each pair of males for the double match
            for male1, male2 in combinations(possible_males, 2):
                if count[0] >= max_count:
                    return

                # Start matching with this double match
                current_matching = {(male1, double_female), (male2, double_female)}
                used_males = {male1, male2}
                used_females = {double_female}

                # Match remaining females with remaining males (1-to-1)
                remaining_females = [f for f in self.females if f != double_female]
                remaining_males = [m for m in self.males if m not in {male1, male2}]

                # Use backtracking for remaining 1-to-1 matches
                def backtrack_remaining(idx: int):
                    if count[0] >= max_count:
                        return

                    if idx == len(remaining_females):
                        if len(current_matching) == target_matches:
                            yield current_matching.copy()
                            count[0] += 1
                        return

                    female = remaining_females[idx]
                    female_node = f"F_{female}"

                    for male in remaining_males:
                        if male in used_males:
                            continue

                        male_node = f"M_{male}"
                        if not self.graph.has_edge(male_node, female_node):
                            continue

                        current_matching.add((male, female))
                        used_males.add(male)

                        yield from backtrack_remaining(idx + 1)

                        current_matching.remove((male, female))
                        used_males.remove(male)

                yield from backtrack_remaining(0)

    def calculate_probabilities(
        self,
        matchings: List[Set[Tuple[str, str]]]
    ) -> Dict[Tuple[str, str], float]:
        """Calculate match probabilities from enumerated matchings."""
        if not matchings:
            return {}

        pair_counts = defaultdict(int)
        total = len(matchings)

        for matching in matchings:
            for pair in matching:
                pair_counts[pair] += 1

        return {pair: count / total for pair, count in pair_counts.items()}

    def calculate_double_match_probabilities(
        self,
        matchings: List[Set[Tuple[str, str]]]
    ) -> Dict[str, float]:
        """Calculate probability each person is in double match."""
        if self.n_males == self.n_females or not matchings:
            return {}

        person_counts = defaultdict(int)
        total = len(matchings)

        for matching in matchings:
            # Count matches for each person
            male_match_counts = defaultdict(int)
            female_match_counts = defaultdict(int)

            for male, female in matching:
                male_match_counts[male] += 1
                female_match_counts[female] += 1

            # Find who has double match in this solution
            for person, count in male_match_counts.items():
                if count > 1:
                    person_counts[person] += 1

            for person, count in female_match_counts.items():
                if count > 1:
                    person_counts[person] += 1

        return {person: count / total for person, count in person_counts.items()}
