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

        # CONSTRAINT WE HAVE COLUMN/ROWSUME 1s
        if self.n_2 == self.n_1:
            for i in range(self.n_1):
                # Over all columns:
                matches = np.zeros((1, len(self.males), len(self.females)))
                matches[0, :, i] = 1
                self.A3D = np.concatenate((self.A3D, matches), axis=0)
                self.b = np.append(self.b, 1)
                # Over all rows:
                matches = np.zeros((1, len(self.males), len(self.females)))
                matches[0, i, :] = 1
                self.A3D = np.concatenate((self.A3D, matches), axis=0)
                self.b = np.append(self.b, 1)
        elif self.n_2 > self.n_1:
            # SILENTLY ASSUMING ONLY ONE MORE FEMALE THAN MALE!
            # More females than males, we know each female has one perfect match:
            for i in range(self.n_2):
                matches = np.zeros((1, len(self.males), len(self.females)))
                matches[0, :, i] = 1
                self.A3D = np.concatenate((self.A3D, matches), axis=0)
                self.b = np.append(self.b, 1)
            # Each male has one match with the "first set" of females
            for i in range(self.n_1):
                matches = np.zeros((1, len(self.males), len(self.females)))
                matches[0, i, :-1] = 1
                self.A3D = np.concatenate((self.A3D, matches), axis=0)
                self.b = np.append(self.b, 1)
        else:
            # SILENTLY ASSUMING ONLY ONE MORE MALE THAN FEMALE!
            # More males than females, we know each female has one perfect match:
            for i in range(self.n_1):
                matches = np.zeros((1, len(self.males), len(self.females)))
                matches[0, i, 1] = 1
                self.A3D = np.concatenate((self.A3D, matches), axis=0)
                self.b = np.append(self.b, 1)
            # Each female has a match with the "first set" if males
            for i in range(self.n_2):
                matches = np.zeros((1, len(self.males), len(self.females)))
                matches[0, :-1, i] = 1
                self.A3D_uneq = np.concatenate((self.A3D, matches), axis=0)
                self.b_uneq = np.append(self.b, 1)

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
        if np.any(np.sum(matches, axis=-2) > 1) or np.any(np.sum(matches, axis=-1) > 1):
            raise ValueError("One or many people have multiple matches in this night.")
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
        model.optimize(max_seconds=2)
        self.X_binary = np.asarray([x[i].x for i in range(n)]).reshape(self.n_1, self.n_2)

    def print_matches(self):
        """Pretty print solutions found."""
        print("Current solution proposed:\n")
        print("Males first:")
        for i, row in enumerate(self.X_binary):
            print("{} and:".format(self.males[i]))
            for j, cell in enumerate(row):
                if cell > 0.1:
                    print("    {}".format(self.females[j]))
        print("\nFemales first:")
        for j, column in enumerate(self.X_binary.T):
            print("{} and:".format(self.females[j]))
            for i, cell in enumerate(column):
                if cell > 0.1:
                    print("    {}".format(self.males[i]))
        print("\n")

    def check_uniqueness(self):
        """Check if A and b allow for unique optimal solution."""
        # TODO: CHECK UNIQUENESS (RIP?, SPARK?)
        return NotImplementedError

    def _check_linear_dependency(self):
        """Detect and remove linear dependent constraints."""
        A2D = self.A3D.reshape((self.A3D.shape[0], -1))
        _, inds = Matrix(A2D).T.rref()
        self.A3D = self.A3D[inds, :, :]
        self.b = self.b[inds, ]


def main():
    """Entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_file_path", type=Path)
    args = parser.parse_args()
    with open(args.yaml_file_path, "r") as f:
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
