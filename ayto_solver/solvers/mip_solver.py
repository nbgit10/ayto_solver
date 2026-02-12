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

        # Track the index of total matches constraint (for n×m cases)
        # This constraint should not be removed by linear dependency check
        self.total_matches_constraint_idx: Optional[int] = None

        # Solution will be stored here
        self.X_binary = np.zeros((self.n_males, self.n_females))

        # Add initial structural constraints
        self._add_initial_constraints()

    def _add_initial_constraints(self):
        """
        Add initial structural constraints based on n×n or n×m matching.

        For n×n: Each person has exactly 1 match (permutation matrix)
        For n×m: One person from smaller set has 2 matches, everyone else has 1
                 Total matches = min(n,m) + 1
        """
        n_smaller = min(self.n_males, self.n_females)

        if self.n_males == self.n_females:
            # n×n case: Perfect 1-to-1 matching (permutation matrix)
            # Each male has exactly 1 match
            for i in range(self.n_males):
                constraint = np.zeros((1, self.n_males, self.n_females))
                constraint[0, i, :] = 1
                self.A3D = np.concatenate((self.A3D, constraint), axis=0)
                self.b = np.append(self.b, 1)

            # Each female has exactly 1 match
            for j in range(self.n_females):
                constraint = np.zeros((1, self.n_males, self.n_females))
                constraint[0, :, j] = 1
                self.A3D = np.concatenate((self.A3D, constraint), axis=0)
                self.b = np.append(self.b, 1)
        else:
            # n×m case: People from smaller set can have double matches
            # Total matches constraint - CRITICAL for n×m cases!
            total_constraint = np.ones((1, self.n_males, self.n_females))
            self.total_matches_constraint_idx = self.A3D.shape[0]
            self.A3D = np.concatenate((self.A3D, total_constraint), axis=0)
            self.b = np.append(self.b, max(self.n_males, self.n_females))  # max(n,m) total matches

            if self.n_females > self.n_males:
                # 10 males × 11 females case:
                # - Each female: exactly 1 match (11 constraints)
                # - One male will have 2 matches, others have 1 (enforced by total=11)
                for j in range(self.n_females):
                    constraint = np.zeros((1, self.n_males, self.n_females))
                    constraint[0, :, j] = 1
                    self.A3D = np.concatenate((self.A3D, constraint), axis=0)
                    self.b = np.append(self.b, 1)

                # Each male: at least 1 match (prevents unmatched males)
                for i in range(self.n_males):
                    constraint = np.zeros((1, self.n_males, self.n_females))
                    constraint[0, i, :] = 1
                    # Note: Using >= 1 constraint, but MIP only supports equality
                    # We encode this as: sum >= 1 by adding it with b=1 and
                    # relying on minimization + total=11 to prevent > 2
                    # Actually, for MIP we need inequalities... let me reconsider
                    # For now, just ensure each male has at least 1 by adding
                    # the constraint that row sum >= 1, but we'll need to handle this
                    # SKIP for now - the total + female constraints imply this
            else:
                # 11 males × 10 females case:
                # - Each male: exactly 1 match (11 constraints)
                # - One female will have 2 matches, others have 1 (enforced by total=11)
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
        """
        Remove linearly dependent constraints to avoid redundancy.

        IMPORTANT: For n×m cases, the total matches constraint is protected
        and will never be removed, even if it appears linearly dependent.
        This constraint is critical for finding solutions with the correct
        number of matches.
        """
        # Reshape 3D constraints to 2D for linear independence check
        A2D = self.A3D.reshape((self.A3D.shape[0], -1))

        # Use sympy to find linearly independent rows
        _, independent_indices = Matrix(A2D).T.rref()

        # For n×m cases, ensure total matches constraint is always kept
        if self.total_matches_constraint_idx is not None:
            independent_indices = set(independent_indices)
            independent_indices.add(self.total_matches_constraint_idx)
            independent_indices = sorted(list(independent_indices))

            # Update the constraint index if its position changed
            new_idx = independent_indices.index(self.total_matches_constraint_idx)
            self.total_matches_constraint_idx = new_idx

        # Keep only linearly independent constraints (+ protected constraints)
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

        # Add equality constraints
        for i, row in enumerate(A_eq):
            model += xsum(int(row[j]) * x[j] for j in range(n_vars)) == int(self.b[i])

        # Add inequality constraints for n×m cases
        if self.n_males != self.n_females:
            # Each person from smaller set must have at least 1 and at most 2 matches
            if self.n_males < self.n_females:
                # More females: constrain males
                for i in range(self.n_males):
                    row_start = i * self.n_females
                    row_end = (i + 1) * self.n_females
                    row_sum = xsum(x[j] for j in range(row_start, row_end))
                    model += row_sum >= 1  # At least 1 match
                    model += row_sum <= 2  # At most 2 matches
            else:
                # More males: constrain females
                for j in range(self.n_females):
                    col_indices = [i * self.n_females + j for i in range(self.n_males)]
                    col_sum = xsum(x[idx] for idx in col_indices)
                    model += col_sum >= 1  # At least 1 match
                    model += col_sum <= 2  # At most 2 matches

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
