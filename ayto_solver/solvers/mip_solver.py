"""MIP-based solver for AYTO matching problem."""
import numpy as np
from mip import BINARY, Model, minimize, xsum
from sympy import Matrix
from typing import List, Tuple, Dict, Optional


class MIPSolver:
    """
    MIP-based solver for Are You The One matching problem.

    Uses Mixed Integer Programming with L1 norm minimization to find
    sparse solutions to the matching constraints.
    """

    def __init__(self, males: List[str], females: List[str]):
        """
        Initialize solver with contestants.

        Args:
            males: List of male contestant names
            females: List of female contestant names
        """
        self.males = males
        self.females = females
        self.n_males = len(males)
        self.n_females = len(females)

        # Initialize constraint matrices
        # A3D shape: (num_constraints, n_males, n_females)
        # Each constraint is represented as a 2D matrix
        self.A3D = np.zeros((0, self.n_males, self.n_females))
        self.b = np.zeros((0,))  # measurement vector

        # Solution will be stored here
        self.X_binary = np.zeros((self.n_males, self.n_females))

        # Add initial structural constraints
        self._add_initial_constraints()

    def _add_initial_constraints(self):
        """Add initial structural constraints based on n×n or n×m matching."""
        n_smaller = min(self.n_males, self.n_females)

        # For the first min(n_males, n_females) people, add 1-to-1 constraints
        for i in range(n_smaller):
            # Each person in smaller set has exactly one match
            if self.n_males <= self.n_females:
                # Each male has exactly 1 match
                constraint = np.zeros((1, self.n_males, self.n_females))
                constraint[0, i, :] = 1
                self.A3D = np.concatenate((self.A3D, constraint), axis=0)
                self.b = np.append(self.b, 1)
            else:
                # Each female has exactly 1 match
                constraint = np.zeros((1, self.n_males, self.n_females))
                constraint[0, :, i] = 1
                self.A3D = np.concatenate((self.A3D, constraint), axis=0)
                self.b = np.append(self.b, 1)

        # Handle n×m cases (unequal numbers)
        if self.n_males != self.n_females:
            # Total matches constraint
            total_constraint = np.ones((1, self.n_males, self.n_females))
            self.A3D = np.concatenate((self.A3D, total_constraint), axis=0)
            self.b = np.append(self.b, n_smaller + 1)  # min(n,m) + 1 total matches

            if self.n_females > self.n_males:
                # More females: each female has exactly 1 match
                for i in range(self.n_females):
                    constraint = np.zeros((1, self.n_males, self.n_females))
                    constraint[0, :, i] = 1
                    self.A3D = np.concatenate((self.A3D, constraint), axis=0)
                    self.b = np.append(self.b, 1)
            else:
                # More males: each male has exactly 1 match
                for i in range(self.n_males):
                    constraint = np.zeros((1, self.n_males, self.n_females))
                    constraint[0, i, :] = 1
                    self.A3D = np.concatenate((self.A3D, constraint), axis=0)
                    self.b = np.append(self.b, 1)

    def add_matching_night(self, pairs: List[Tuple[str, str]], num_matches: int):
        """
        Add constraint from a matching night.

        Args:
            pairs: List of (male, female) pairs that were matched that night
            num_matches: Number of correct matches in that night
        """
        constraint = np.zeros((1, self.n_males, self.n_females))

        for male, female in pairs:
            if male not in self.males:
                raise ValueError(f"Male '{male}' not in contestants list")
            if female not in self.females:
                raise ValueError(f"Female '{female}' not in contestants list")

            m_idx = self.males.index(male)
            f_idx = self.females.index(female)
            constraint[0, m_idx, f_idx] = 1

        self.A3D = np.concatenate((self.A3D, constraint), axis=0)
        self.b = np.append(self.b, num_matches)

    def add_truth_booth(self, male: str, female: str, is_match: bool):
        """
        Add constraint from a truth booth result.

        Args:
            male: Male contestant name
            female: Female contestant name
            is_match: True if they are a perfect match, False otherwise
        """
        if male not in self.males:
            raise ValueError(f"Male '{male}' not in contestants list")
        if female not in self.females:
            raise ValueError(f"Female '{female}' not in contestants list")

        m_idx = self.males.index(male)
        f_idx = self.females.index(female)

        # Add constraint for this specific pair
        constraint = np.zeros((1, self.n_males, self.n_females))
        constraint[0, m_idx, f_idx] = 1
        self.A3D = np.concatenate((self.A3D, constraint), axis=0)
        self.b = np.append(self.b, int(is_match))

        if is_match:
            # If match is confirmed, add exclusivity constraints
            # This male can only match this female
            male_constraint = np.zeros((1, self.n_males, self.n_females))
            male_constraint[0, m_idx, :] = 1
            self.A3D = np.concatenate((self.A3D, male_constraint), axis=0)
            self.b = np.append(self.b, 1)

            # This female can only match this male
            female_constraint = np.zeros((1, self.n_males, self.n_females))
            female_constraint[0, :, f_idx] = 1
            self.A3D = np.concatenate((self.A3D, female_constraint), axis=0)
            self.b = np.append(self.b, 1)

    def _remove_linearly_dependent_constraints(self):
        """Remove linearly dependent constraints to avoid redundancy."""
        # Reshape 3D constraints to 2D for linear independence check
        A2D = self.A3D.reshape((self.A3D.shape[0], -1))

        # Use sympy to find linearly independent rows
        _, independent_indices = Matrix(A2D).T.rref()

        # Keep only linearly independent constraints
        self.A3D = self.A3D[list(independent_indices), :, :]
        self.b = self.b[list(independent_indices)]

    def solve(self, timeout_seconds: int = 5) -> np.ndarray:
        """
        Solve the matching problem using MIP.

        Args:
            timeout_seconds: Maximum time to spend solving

        Returns:
            Binary matrix of shape (n_males, n_females) representing matches
        """
        # Remove redundant constraints
        self._remove_linearly_dependent_constraints()

        # Flatten constraints for MIP
        A_eq = self.A3D.reshape(-1, self.n_males * self.n_females)
        n_vars = self.n_males * self.n_females

        # Create MIP model
        model = Model()
        model.verbose = 0

        # Binary variables for each possible pairing
        x = [model.add_var(var_type=BINARY) for _ in range(n_vars)]

        # Objective: minimize L1 norm (sparsest solution)
        model.objective = minimize(xsum(x[i] for i in range(n_vars)))

        # Add constraints
        for i, row in enumerate(A_eq):
            model += xsum(int(row[j]) * x[j] for j in range(n_vars)) == int(self.b[i])

        # Emphasis on feasibility
        model.emphasis = 2

        # Solve
        model.optimize(max_seconds=timeout_seconds)

        # Extract solution
        solution = np.array([x[i].x if x[i].x is not None else 0
                            for i in range(n_vars)])
        self.X_binary = solution.reshape(self.n_males, self.n_females)

        return self.X_binary

    def get_matches(self) -> List[Tuple[str, str]]:
        """
        Get list of matches from solution.

        Returns:
            List of (male, female) tuples representing matches
        """
        matches = []
        for i in range(self.n_males):
            for j in range(self.n_females):
                if self.X_binary[i, j] > 0.5:
                    matches.append((self.males[i], self.females[j]))
        return matches

    def validate_solution(self) -> Tuple[bool, List[str]]:
        """
        Validate that the solution satisfies all constraints.

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []

        # Check that solution is binary
        if not np.all((self.X_binary >= -0.01) & (self.X_binary <= 1.01)):
            errors.append("Solution contains non-binary values")

        # Check constraints are satisfied
        A_eq = self.A3D.reshape(-1, self.n_males * self.n_females)
        x_flat = self.X_binary.flatten()

        for i in range(len(self.b)):
            constraint_value = np.dot(A_eq[i], x_flat)
            expected_value = self.b[i]
            if abs(constraint_value - expected_value) > 0.01:
                errors.append(
                    f"Constraint {i} violated: got {constraint_value}, "
                    f"expected {expected_value}"
                )

        return (len(errors) == 0, errors)
