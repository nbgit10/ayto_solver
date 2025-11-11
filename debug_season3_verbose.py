"""Debug script for Season 3 Ep 19 - verbose MIP solve."""
import json
import numpy as np
from mip import BINARY, Model, minimize, xsum, OptimizationStatus
from ayto_solver.solvers.mip_solver import MIPSolver

# Load example
with open('examples/json/AYTO_Season3_Germany_AfterEp19.json', 'r') as f:
    data = json.load(f)

print("=" * 60)
print("Season 3 Ep 19 - Verbose MIP Debug")
print("=" * 60)
print(f"Males: {len(data['males'])}")
print(f"Females: {len(data['females'])}")
print()

solver = MIPSolver(data['males'], data['females'])

# Add constraints
for night in data['matching_nights']:
    solver.add_matching_night(night['pairs'], night['matches'])

for tb in data['truth_booths']:
    solver.add_truth_booth(tb['pair'][0], tb['pair'][1], tb['match'])

# Remove redundant constraints
solver._remove_linearly_dependent_constraints()

# Flatten constraints for MIP
A_eq = solver.A3D.reshape(-1, solver.n_males * solver.n_females)
n_vars = solver.n_males * solver.n_females

print(f"Number of variables: {n_vars}")
print(f"Number of equality constraints: {A_eq.shape[0]}")
print()

# Create MIP model with verbose output
model = Model()
model.verbose = 1  # Enable verbose output

# Binary variables
x = [model.add_var(var_type=BINARY) for _ in range(n_vars)]

# Objective: minimize L1 norm
model.objective = minimize(xsum(x[i] for i in range(n_vars)))

# Add equality constraints
for i, row in enumerate(A_eq):
    model += xsum(int(row[j]) * x[j] for j in range(n_vars)) == int(solver.b[i])

# Add inequality constraints for n×m case
if solver.n_males != solver.n_females:
    print("Adding inequality constraints for n×m case:")
    if solver.n_males < solver.n_females:
        # More females: constrain males
        print(f"  Each male: 1 <= matches <= 2")
        for i in range(solver.n_males):
            row_start = i * solver.n_females
            row_end = (i + 1) * solver.n_females
            row_sum = xsum(x[j] for j in range(row_start, row_end))
            model += row_sum >= 1  # At least 1 match
            model += row_sum <= 2  # At most 2 matches
    print()

# Emphasis on feasibility
model.emphasis = 2

print("Solving...")
print()
status = model.optimize(max_seconds=10)

print()
print("=" * 60)
print(f"Optimization status: {status}")
print(f"Status name: {status.name if hasattr(status, 'name') else 'N/A'}")
print()

# Check if solution found
if status in [OptimizationStatus.OPTIMAL, OptimizationStatus.FEASIBLE]:
    print("Solution found!")
    solution = np.array([x[i].x if x[i].x is not None else 0
                        for i in range(n_vars)])
    solution_matrix = solution.reshape(solver.n_males, solver.n_females)
    print(f"Total matches: {solution_matrix.sum()}")
elif status == OptimizationStatus.INFEASIBLE:
    print("Problem is INFEASIBLE - constraints are contradictory!")
elif status == OptimizationStatus.NO_SOLUTION_FOUND:
    print("No solution found within time limit")
else:
    print(f"Other status: {status}")
