"""Graph-based solver for AYTO matching problem using bipartite graph matching."""
import random
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

        # Create bipartite graph - start EMPTY, add edges as we learn they're possible
        self.graph = nx.Graph()

        # Add nodes
        self.graph.add_nodes_from(
            [(f"M_{m}", {"bipartite": 0}) for m in males]
        )
        self.graph.add_nodes_from(
            [(f"F_{f}", {"bipartite": 1}) for f in females]
        )

        # Initially, all pairs are possible - we'll track this separately
        # and only add edges to graph after applying initial constraints
        self.ruled_out_pairs: Set[Tuple[str, str]] = set()
        self.confirmed_pairs: Set[Tuple[str, str]] = set()

        # Store matching night constraints for validation
        # Format: List[(pairs, num_matches)]
        self.matching_night_constraints: List[Tuple[List[Tuple[str, str]], int]] = []

        # Flag to track if graph has been finalized
        self._graph_finalized = False

    def add_truth_booth(self, male: str, female: str, is_match: bool):
        """
        Add truth booth constraint.

        Supports double/triple matches: if a person is confirmed with multiple
        partners (e.g., Caro confirmed with both Ken and Max), previously
        confirmed pairs are preserved rather than ruled out.

        Args:
            male: Male contestant name
            female: Female contestant name
            is_match: True if they are a perfect match, False otherwise
        """
        if is_match:
            self.confirmed_pairs.add((male, female))
            # Remove from ruled_out in case a previous confirmation ruled it out
            self.ruled_out_pairs.discard((male, female))

            # Rule out other females for this male, but skip already-confirmed pairs
            for other_female in self.females:
                if other_female != female and (male, other_female) not in self.confirmed_pairs:
                    self.ruled_out_pairs.add((male, other_female))

            # Rule out other males for this female, but skip already-confirmed pairs
            for other_male in self.males:
                if other_male != male and (other_male, female) not in self.confirmed_pairs:
                    self.ruled_out_pairs.add((other_male, female))
        else:
            self.ruled_out_pairs.add((male, female))

    def add_matching_night(self, pairs: List[Tuple[str, str]], num_matches: int):
        """
        Add matching night constraint.

        Special cases for edge pruning:
        - If num_matches == 0, all pairs are ruled out (can prune edges)
        - If num_matches == len(pairs), all pairs are confirmed (can prune edges)
        - Otherwise, store constraint for validation during enumeration

        Args:
            pairs: List of (male, female) pairs from that night
            num_matches: Number of correct matches
        """
        # Always store the constraint for validation
        self.matching_night_constraints.append((pairs, num_matches))

        if num_matches == 0:
            # All pairs are wrong - mark as ruled out
            for male, female in pairs:
                self.ruled_out_pairs.add((male, female))

        elif num_matches == len(pairs):
            # All pairs are correct - mark as confirmed
            for male, female in pairs:
                self.confirmed_pairs.add((male, female))
                # Mark all other pairings as ruled out
                for other_female in self.females:
                    if other_female != female:
                        self.ruled_out_pairs.add((male, other_female))

                for other_male in self.males:
                    if other_male != male:
                        self.ruled_out_pairs.add((other_male, female))

    def _finalize_graph(self):
        """
        Build the graph with only valid edges after all constraints are added.

        This method adds edges for all pairs that are:
        1. Not ruled out by truth booths or matching nights
        2. Part of the feasible solution space
        """
        if self._graph_finalized:
            return

        # Add edges for all pairs that haven't been ruled out
        for male in self.males:
            for female in self.females:
                pair = (male, female)
                if pair not in self.ruled_out_pairs:
                    self.graph.add_edge(f"M_{male}", f"F_{female}")

        self._graph_finalized = True

    def _satisfies_matching_night_constraints(self, matching: Set[Tuple[str, str]]) -> bool:
        """
        Check if a matching satisfies all matching night constraints.

        Args:
            matching: Set of (male, female) pairs

        Returns:
            True if all constraints are satisfied
        """
        for night_pairs, expected_matches in self.matching_night_constraints:
            # Count how many pairs from this night appear in the matching
            # Convert to tuple if needed (in case pairs come as lists from JSON)
            actual_matches = 0
            for pair in night_pairs:
                # Normalize to tuple
                normalized_pair = tuple(pair) if isinstance(pair, list) else pair
                if normalized_pair in matching:
                    actual_matches += 1

            if actual_matches != expected_matches:
                return False

        return True

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
        # Finalize graph by adding only valid edges
        self._finalize_graph()

        matchings = []
        capped = False

        # Generate candidates and filter by matching night constraints
        # We need to generate many more candidates than the target since constraints
        # will filter most of them out. Use a large multiplier.
        candidate_multiplier = 10000 if self.matching_night_constraints else 1

        # For balanced (n×n) graphs, use standard matching enumeration
        if self.n_males == self.n_females:
            all_matchings = self._enumerate_perfect_matchings_balanced(
                max_matchings * candidate_multiplier
            )
            # Filter by matching night constraints
            for matching in all_matchings:
                if self._satisfies_matching_night_constraints(matching):
                    matchings.append(matching)
                    if len(matchings) >= max_matchings:
                        capped = True
                        break
        else:
            # For unbalanced (n×m) graphs, we need to handle double matches
            all_matchings = self._enumerate_matchings_unbalanced(
                max_matchings * candidate_multiplier
            )
            # Filter by matching night constraints
            for matching in all_matchings:
                if self._satisfies_matching_night_constraints(matching):
                    matchings.append(matching)
                    if len(matchings) >= max_matchings:
                        capped = True
                        break

        return matchings, capped

    def _enumerate_perfect_matchings_balanced(self, max_count: int):
        """
        Enumerate perfect matchings for balanced bipartite graph.

        Uses recursive backtracking with randomized candidate ordering
        to avoid systematic bias when capping results.
        """
        n = min(self.n_males, self.n_females)
        males_list = list(self.males)
        random.shuffle(males_list)

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

            # Try each female this male can match with (randomized order)
            candidates = [
                f for f in self.females
                if f not in used_females and self.graph.has_edge(male_node, f"F_{f}")
            ]
            random.shuffle(candidates)

            for female in candidates:
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
        """Enumerate matchings where one male has 2 matches.

        Uses round-robin across double-match candidates to ensure each
        candidate gets fair representation when results are capped.
        """
        males_shuffled = list(self.males)
        random.shuffle(males_shuffled)

        # Build a generator per candidate
        generators = []
        for dm in males_shuffled:
            possible_females = [
                f for f in self.females
                if self.graph.has_edge(f"M_{dm}", f"F_{f}")
            ]
            if len(possible_females) >= 2:
                generators.append(
                    self._gen_matchings_for_double_male(dm, possible_females, target_matches)
                )

        # Round-robin: pull one matching from each candidate in turn
        yield from self._round_robin(generators, max_count)

    def _gen_matchings_for_double_male(
        self, double_male: str, possible_females: List[str], target_matches: int
    ):
        """Yield all matchings for a specific male as double-match candidate."""
        female_pairs = list(combinations(possible_females, 2))
        random.shuffle(female_pairs)

        for female1, female2 in female_pairs:
            current_matching = {(double_male, female1), (double_male, female2)}
            used_males = {double_male}
            used_females = {female1, female2}

            remaining_males = [m for m in self.males if m != double_male]
            random.shuffle(remaining_males)
            remaining_females = [f for f in self.females if f not in {female1, female2}]

            def backtrack(idx: int):
                if idx == len(remaining_males):
                    if len(current_matching) == target_matches:
                        yield current_matching.copy()
                    return

                male = remaining_males[idx]
                male_node = f"M_{male}"
                candidates = [
                    f for f in remaining_females
                    if f not in used_females and self.graph.has_edge(male_node, f"F_{f}")
                ]
                random.shuffle(candidates)

                for female in candidates:
                    current_matching.add((male, female))
                    used_females.add(female)
                    yield from backtrack(idx + 1)
                    current_matching.remove((male, female))
                    used_females.remove(female)

            yield from backtrack(0)

    def _enumerate_with_double_match_female(self, max_count: int, target_matches: int):
        """Enumerate matchings where one female has 2 matches.

        Uses round-robin across double-match candidates to ensure each
        candidate gets fair representation when results are capped.
        """
        females_shuffled = list(self.females)
        random.shuffle(females_shuffled)

        # Build a generator per candidate
        generators = []
        for df in females_shuffled:
            possible_males = [
                m for m in self.males
                if self.graph.has_edge(f"M_{m}", f"F_{df}")
            ]
            if len(possible_males) >= 2:
                generators.append(
                    self._gen_matchings_for_double_female(df, possible_males, target_matches)
                )

        # Round-robin: pull one matching from each candidate in turn
        yield from self._round_robin(generators, max_count)

    def _gen_matchings_for_double_female(
        self, double_female: str, possible_males: List[str], target_matches: int
    ):
        """Yield all matchings for a specific female as double-match candidate."""
        male_pairs = list(combinations(possible_males, 2))
        random.shuffle(male_pairs)

        for male1, male2 in male_pairs:
            current_matching = {(male1, double_female), (male2, double_female)}
            used_males = {male1, male2}
            used_females = {double_female}

            remaining_females = [f for f in self.females if f != double_female]
            random.shuffle(remaining_females)
            remaining_males = [m for m in self.males if m not in {male1, male2}]

            def backtrack(idx: int):
                if idx == len(remaining_females):
                    if len(current_matching) == target_matches:
                        yield current_matching.copy()
                    return

                female = remaining_females[idx]
                female_node = f"F_{female}"
                candidates = [
                    m for m in remaining_males
                    if m not in used_males and self.graph.has_edge(f"M_{m}", female_node)
                ]
                random.shuffle(candidates)

                for male in candidates:
                    current_matching.add((male, female))
                    used_males.add(male)
                    yield from backtrack(idx + 1)
                    current_matching.remove((male, female))
                    used_males.remove(male)

            yield from backtrack(0)

    @staticmethod
    def _round_robin(generators: list, max_count: int):
        """Yield items from generators in round-robin order up to max_count."""
        count = 0
        active = list(range(len(generators)))
        while active and count < max_count:
            still_active = []
            for i in active:
                if count >= max_count:
                    break
                try:
                    yield next(generators[i])
                    count += 1
                    still_active.append(i)
                except StopIteration:
                    pass  # This candidate exhausted
            active = still_active

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
