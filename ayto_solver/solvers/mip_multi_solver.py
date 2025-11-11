"""Enhanced MIP solver that can enumerate multiple solutions."""
import numpy as np
from mip import BINARY, Model, minimize, xsum, OptimizationStatus
from typing import List, Tuple, Dict
from collections import defaultdict

from .mip_solver import MIPSolver


class MIPMultiSolver(MIPSolver):
    """
    MIP solver that can enumerate multiple solutions and calculate probabilities.

    Extends the base MIPSolver to find multiple valid matchings and compute
    match probabilities.
    """

    def enumerate_solutions(
        self,
        max_solutions: int = 1000,
        timeout_per_solution: int = 5
    ) -> Tuple[List[np.ndarray], bool]:
        """
        Enumerate multiple valid solutions.

        Args:
            max_solutions: Maximum number of solutions to find
            timeout_per_solution: Timeout for each solution attempt

        Returns:
            Tuple of (list of solution matrices, whether cap was hit)
        """
        solutions = []
        capped = False

        # Remove redundant constraints once
        self._remove_linearly_dependent_constraints()

        # Flatten constraints for MIP
        A_eq = self.A3D.reshape(-1, self.n_males * self.n_females)
        n_vars = self.n_males * self.n_females

        for iteration in range(max_solutions):
            # Create fresh model for each iteration
            model = Model()
            model.verbose = 0

            # Binary variables
            x = [model.add_var(var_type=BINARY) for _ in range(n_vars)]

            # Objective: minimize L1 norm
            model.objective = minimize(xsum(x[i] for i in range(n_vars)))

            # Add equality constraints
            for i, row in enumerate(A_eq):
                model += xsum(int(row[j]) * x[j] for j in range(n_vars)) == int(self.b[i])

            # Add constraints to exclude previous solutions
            for prev_solution in solutions:
                prev_flat = prev_solution.flatten()
                # At least one variable must be different
                # sum of (x_i XOR prev_i) >= 1
                # This is equivalent to: sum(x_i * (1 - prev_i) + prev_i * (1 - x_i)) >= 1
                # Simplified: sum(x_i) - 2*sum(x_i * prev_i) + sum(prev_i) >= 1
                matching_vars = []
                for j in range(n_vars):
                    if prev_flat[j] > 0.5:
                        matching_vars.append(j)

                # Exclude this exact solution by requiring at least one difference
                # in the matching set
                if matching_vars:
                    model += xsum(x[j] for j in matching_vars) <= len(matching_vars) - 1

            # Solve
            model.emphasis = 2
            status = model.optimize(max_seconds=timeout_per_solution)

            # Check if solution found
            if status in [OptimizationStatus.OPTIMAL, OptimizationStatus.FEASIBLE]:
                # Extract solution
                solution = np.array([x[i].x if x[i].x is not None else 0
                                   for i in range(n_vars)])
                solution_matrix = solution.reshape(self.n_males, self.n_females)

                # Round to binary
                solution_matrix = np.round(solution_matrix)

                # Check if this is a new solution (shouldn't be duplicate, but verify)
                is_duplicate = False
                for prev in solutions:
                    if np.allclose(prev, solution_matrix):
                        is_duplicate = True
                        break

                if not is_duplicate:
                    solutions.append(solution_matrix)
                else:
                    # If we found a duplicate, we've likely exhausted solutions
                    break
            else:
                # No more solutions found
                break

        if len(solutions) >= max_solutions:
            capped = True

        return solutions, capped

    def calculate_probabilities(
        self,
        solutions: List[np.ndarray]
    ) -> Dict[Tuple[str, str], float]:
        """
        Calculate match probabilities from solutions.

        Args:
            solutions: List of solution matrices

        Returns:
            Dictionary mapping (male, female) tuples to probabilities
        """
        if not solutions:
            return {}

        # Count how many times each pair appears
        pair_counts = defaultdict(int)
        total_solutions = len(solutions)

        for solution in solutions:
            for i in range(self.n_males):
                for j in range(self.n_females):
                    if solution[i, j] > 0.5:
                        pair = (self.males[i], self.females[j])
                        pair_counts[pair] += 1

        # Calculate probabilities
        probabilities = {
            pair: count / total_solutions
            for pair, count in pair_counts.items()
        }

        return probabilities

    def calculate_double_match_probabilities(
        self,
        solutions: List[np.ndarray]
    ) -> Dict[str, float]:
        """
        Calculate probability each person is involved in double match.

        Only relevant for n×m scenarios.

        Args:
            solutions: List of solution matrices

        Returns:
            Dictionary mapping person name to probability of double match
        """
        if self.n_males == self.n_females:
            return {}  # No double matches in balanced case

        if not solutions:
            return {}

        total_solutions = len(solutions)
        double_match_counts = defaultdict(int)

        # Determine which is the larger set
        if self.n_females > self.n_males:
            # One or more males will have 2 matches
            for solution in solutions:
                for i in range(self.n_males):
                    num_matches = solution[i, :].sum()
                    if num_matches > 1.5:  # Has 2 matches
                        double_match_counts[self.males[i]] += 1
        else:
            # One or more females will have 2 matches
            for solution in solutions:
                for j in range(self.n_females):
                    num_matches = solution[:, j].sum()
                    if num_matches > 1.5:  # Has 2 matches
                        double_match_counts[self.females[j]] += 1

        probabilities = {
            person: count / total_solutions
            for person, count in double_match_counts.items()
        }

        return probabilities
