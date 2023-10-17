"""Are you the one matching solver."""
import argparse
from pathlib import Path

import numpy as np
import yaml
from mip import BINARY, Model, minimize, xsum
from sympy import Matrix


class AYTO:
    """Class that contains methods to solve and decodes pairs."""

    def __init__(self, MALES, FEMALES):
        """Init."""
        self.males = MALES
        self.females = FEMALES

        self.n_1 = len(self.males)
        self.n_2 = len(self.females)
        m = 0
        self.A3D = np.zeros((m, self.n_1, self.n_2))  # measurement matrix
        self.b = np.zeros((m, 1))  # measurements
        self.X_binary = np.zeros((self.n_1, self.n_2))

        # CONSTRAINT WE HAVE COLUMN/ROWSUM is 1
        # if self.n_2 == self.n_1:
        n3 = min(self.n_1, self.n_2)
        for i in range(n3):
            # every (first set of) females has exactly one match with first set of males
            matches = np.zeros((1, self.n_1, self.n_2))
            matches[0, :n3, i] = 1
            self.A3D = np.concatenate((self.A3D, matches), axis=0)
            self.b = np.append(self.b, 1)

            # Every first set of males has exactly one match with first set of females
            matches = np.zeros((1, self.n_1, self.n_2))
            matches[0, i, :n3] = 1
            self.A3D = np.concatenate((self.A3D, matches), axis=0)
            self.b = np.append(self.b, 1)

        if self.n_2 > self.n_1:
            # add additional constraints for new female
            # new female has one perfect male
            matches = np.zeros((1, self.n_1, self.n_2))
            matches[0, :, -1] = 1
            self.A3D = np.concatenate((self.A3D, matches), axis=0)
            self.b = np.append(self.b, 1)
            # all males have at most two matches
            matches = np.zeros((1, self.n_1, self.n_2))
            matches[0, :, :] = 1
            self.A3D = np.concatenate((self.A3D, matches), axis=0)
            self.b = np.append(self.b, 2)
        if self.n_1 > self.n_2:
            # add additional constraints for new male
            # new male has one perfect match
            matches = np.zeros((1, self.n_1, self.n_2))
            matches[0, -1, :] = 1
            self.A3D = np.concatenate((self.A3D, matches), axis=0)
            self.b = np.append(self.b, 1)
            # all females have two perfect matches at most
            matches = np.zeros((1, self.n_1, self.n_2))
            matches[0, :, :] = 1
            self.A3D = np.concatenate((self.A3D, matches), axis=0)
            self.b = np.append(self.b, 2)

        #     # SILENTLY ASSUMING ONLY ONE MORE FEMALE THAN MALE!
        #     # More females than males, we know each female has one perfect match:
        #     for i in range(self.n_2):
        #         matches = np.zeros((1, self.n_1, self.n_2))
        #         matches[0, :, i] = 1
        #         self.A3D = np.concatenate((self.A3D, matches), axis=0)
        #         self.b = np.append(self.b, 1)
        #     # Each male has one match with the "first set" of females
        #     for i in range(self.n_1):
        #         matches = np.zeros((1, self.n_1, self.n_2))
        #         matches[0, i, :-1] = 1
        #         self.A3D = np.concatenate((self.A3D, matches), axis=0)
        #         self.b = np.append(self.b, 1)
        # else:
        #     # SILENTLY ASSUMING ONLY ONE MORE MALE THAN FEMALE!
        #     # More males than females, we know each male has one perfect match:
        #     for i in range(self.n_1):
        #         matches = np.zeros((1, self.n_1, self.n_2))
        #         matches[0, i, :] = 1
        #         self.A3D = np.concatenate((self.A3D, matches), axis=0)
        #         self.b = np.append(self.b, 1)
        #     # Each female has a match with the "first set" of males
        #     for i in range(self.n_2):
        #         matches = np.zeros((1, self.n_1, self.n_2))
        #         matches[0, :-1, i] = 1
        #         self.A3D_uneq = np.concatenate((self.A3D, matches), axis=0)
        #         self.b_uneq = np.append(self.b, 1)

    def add_matchingnight(self, results: dict):
        """Add constraints derived from matching night."""
        if "Pairs" not in results.keys() or "Matches" not in results.keys():
            raise KeyError("Results need to contain Pairs and Matches")
        matches = np.zeros((1, len(self.males), len(self.females)))
        for pairs in results["Pairs"]:
            m = pairs[0]
            f = pairs[1]
            if m not in self.males or f not in self.females:
                raise ValueError("Check your pairs. One or several names are invalid.")
            matches[0, self.males.index(m), self.females.index(f)] = 1
        # if np.any(np.sum(matches, axis=-2) > 1) or np.any(np.sum(matches, axis=-1) > 1):
        #     raise ValueError("One or many people have multiple matches in this night.")
        self.b = np.append(self.b, results["Matches"])
        self.A3D = np.concatenate((self.A3D, matches), axis=0)

    def add_truth_booth(self, result):
        """Add constraints derived from truth booth."""
        if "Pair" not in result.keys() or "Match" not in result.keys():
            raise KeyError("Results need to contain Pairs and Matches")
        m = result["Pair"][0]
        f = result["Pair"][1]
        if m not in self.males or f not in self.females:
            raise ValueError("Check your truth booth couple. One or several names are invalid.")
        matches = np.zeros((1, len(self.males), len(self.females)))
        matches[0, self.males.index(m), self.females.index(f)] = 1
        self.A3D = np.concatenate((self.A3D, matches), axis=0)
        self.b = np.append(self.b, int(result["Match"]))
        if bool(result["Match"]):
            # If they are a match, they cannot match any others.
            # Explicitly add this in case we have inequal number of males and females
            matches2 = np.zeros((1, len(self.males), len(self.females)))
            matches3 = np.zeros((1, len(self.males), len(self.females)))
            matches2[0, self.males.index(m), :] = 1
            matches3[0, :, self.females.index(f)] = 1
            self.b = np.append(self.b, 1)
            self.A3D = np.concatenate((self.A3D, matches2), axis=0)
            self.b = np.append(self.b, 1)
            self.A3D = np.concatenate((self.A3D, matches3), axis=0)

    def solve(self):
        """Try to solve the problem and identify possible matches."""
        self._check_linear_dependency()
        A_eq = self.A3D.reshape(-1, self.n_1 * self.n_2)
        n = self.n_1 * self.n_2

        # PYTHON MIP:
        model = Model()
        x = [model.add_var(var_type=BINARY) for i in range(n)]
        model.objective = minimize(xsum(x[i] for i in range(n)))
        for i, row in enumerate(A_eq):
            model += xsum(int(row[j]) * x[j] for j in range(n)) == int(self.b[i])
        model.emphasis = 2
        model.verbose = 0
        model.optimize(max_seconds=5)
        self.X_binary = np.asarray([x[i].x for i in range(n)]).reshape(self.n_1, self.n_2)

    def print_matches(self):
        """Pretty print solutions found."""
        print("Current solution proposed:\n")
        print("Males first:")
        for i, row in enumerate(self.X_binary):
            print(f"{self.males[i]} and:")
            for j, cell in enumerate(row):
                if cell > 0.1:
                    print(f"    {self.females[j]}")
        print("\nFemales first:")
        for j, column in enumerate(self.X_binary.T):
            print(f"{self.females[j]} and:")
            for i, cell in enumerate(column):
                if cell > 0.1:
                    print(f"    {self.males[i]}")
        print("\n")

    def _check_linear_dependency(self):
        """Detect and remove linear dependent constraints."""
        A2D = self.A3D.reshape((self.A3D.shape[0], -1))
        _, inds = Matrix(A2D).T.rref()
        self.A3D = self.A3D[inds, :, :]
        self.b = self.b[
            inds,
        ]

    def _add_fixed_pair(self, result):
        """Special constraints for ensuring certain people match the same person."""
        return NotImplementedError


class AYTO_SEASON4(AYTO):
    """Class for German season four."""

    def __init__(self, MALES, FEMALES):  # pylint: disable=super-init-not-called
        """Init."""
        self.males = MALES
        self.females = FEMALES

        self.n_1 = len(self.males)
        self.n_2 = len(self.females)

        # Equality constraints:
        self.A3D = np.zeros((0, self.n_1, self.n_2))  # measurement matrix
        self.b = np.zeros((0, 1))  # measurements

        # Inequality constraints:
        self.A3D_ineq = np.zeros((0, self.n_1, self.n_2))  # measurement matrix
        self.b_ineq = np.zeros((0, 1))  # measurements

        self.X_binary1 = np.zeros((self.n_1, self.n_2))

        # CONSTRAINTS WE HAVE COLUMN/ROWSUM is = 1:

        # every females can have two matches but at least one
        for i in range(self.n_2):
            matches = np.zeros((1, self.n_1, self.n_2))
            matches[0, :, i] = 1
            self.A3D_ineq = np.concatenate((self.A3D_ineq, matches), axis=0)
            self.b_ineq = np.append(self.b_ineq, 2)
            matches = np.zeros((1, self.n_1, self.n_2))
            matches[0, :, i] = -1
            self.A3D_ineq = np.concatenate((self.A3D_ineq, matches), axis=0)
            self.b_ineq = np.append(self.b_ineq, -1)

        for i in range(self.n_1):
            # Every male has one match
            matches = np.zeros((1, self.n_1, self.n_2))
            matches[0, i, :] = 1
            self.A3D = np.concatenate((self.A3D, matches), axis=0)
            self.b = np.append(self.b, 1)

    def solve(self):
        """Try to solve the problem and identify possible matches."""
        self._check_linear_dependency()
        A_eq = self.A3D.reshape(-1, self.n_1 * self.n_2)
        A_ineq = self.A3D_ineq.reshape(-1, self.n_1 * self.n_2)
        n = self.n_1 * self.n_2

        # PYTHON MIP:
        model = Model()
        x = [model.add_var(var_type=BINARY) for i in range(n)]
        model.objective = minimize(xsum(x[i] for i in range(n)))
        for i, row in enumerate(A_eq):
            model += xsum(int(row[j]) * x[j] for j in range(n)) == int(self.b[i])
        for i, row in enumerate(A_ineq):
            model += xsum(int(row[j]) * x[j] for j in range(n)) <= int(self.b_ineq[i])
        model.emphasis = 2
        model.verbose = 0
        model.optimize(max_seconds=5)
        self.X_binary = np.asarray([x[i].x for i in range(n)]).reshape(self.n_1, self.n_2)


def main():
    """Entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_file_path", type=Path)
    args = parser.parse_args()
    with open(args.yaml_file_path, "r", encoding="utf-8") as f:
        progress = yaml.load(f, Loader=yaml.SafeLoader)
    ayto = AYTO(progress["MALES"], progress["FEMALES"])
    for _, val in enumerate(progress["MATCHING_NIGHTS"]):
        ayto.add_matchingnight(val)
    for _, val in enumerate(progress["TRUTH_BOOTH"]):
        ayto.add_truth_booth(val)
    ayto.solve()
    ayto.print_matches()


if __name__ == "__main__":
    main()
