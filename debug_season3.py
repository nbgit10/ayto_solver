"""Debug script for Season 3 Ep 19 MIP solver issue."""
import json
import numpy as np
from ayto_solver.solvers.mip_solver import MIPSolver

# Load example
with open('examples/json/AYTO_Season3_Germany_AfterEp19.json', 'r') as f:
    data = json.load(f)

print("=" * 60)
print("Season 3 Ep 19 Debug")
print("=" * 60)
print(f"Males: {len(data['males'])} - {data['males']}")
print(f"Females: {len(data['females'])} - {data['females']}")
print()

solver = MIPSolver(data['males'], data['females'])

print(f"Initial constraints: {solver.A3D.shape[0]}")
print(f"Initial b values: {solver.b}")
print()

# Add constraints
for i, night in enumerate(data['matching_nights']):
    solver.add_matching_night(night['pairs'], night['matches'])
    print(f"After matching night {i+1}: {solver.A3D.shape[0]} constraints")

print()

for i, tb in enumerate(data['truth_booths']):
    solver.add_truth_booth(tb['pair'][0], tb['pair'][1], tb['match'])
    print(f"After truth booth {i+1} ({tb['pair'][0]}-{tb['pair'][1]}, match={tb['match']}): {solver.A3D.shape[0]} constraints")

print()
print(f"Constraints before linear dependency removal: {solver.A3D.shape[0]}")
print(f"Expected values (b) before: {solver.b[:20]}")  # Show first 20
print()

# Check which constraint is the "total matches" constraint
A_flat_before = solver.A3D.reshape(solver.A3D.shape[0], -1)
for i in range(min(25, len(solver.b))):
    if np.all(A_flat_before[i] == 1):
        print(f"Constraint {i} is TOTAL MATCHES = {solver.b[i]}")
print()

# Solve
solver.solve()

print(f"Constraints after linear dependency removal: {solver.A3D.shape[0]}")
print(f"Expected values (b) after: {solver.b[:20]}")  # Show first 20

# Check if total matches constraint survived
A_flat_after = solver.A3D.reshape(solver.A3D.shape[0], -1)
found_total = False
for i in range(len(solver.b)):
    if np.all(A_flat_after[i] == 1):
        print(f"Constraint {i} is TOTAL MATCHES = {solver.b[i]}")
        found_total = True
if not found_total:
    print("WARNING: Total matches constraint was REMOVED by linear dependency filter!")
print()

# Validate
is_valid, errors = solver.validate_solution()
print(f"Solution valid: {is_valid}")
if not is_valid:
    print("Validation errors:")
    for error in errors:
        print(f"  - {error}")
print()

# Get results
print("Solution matrix:")
print(solver.X_binary)
print()

matches = solver.get_matches()
print(f"Matches found: {len(matches)}")
for male, female in sorted(matches):
    print(f"  {male:15s} - {female}")
print()

# Check row sums (male matches)
print("Male match counts:")
for i, male in enumerate(solver.males):
    count = solver.X_binary[i, :].sum()
    print(f"  {male:15s}: {count:.1f}")
print()

# Check column sums (female matches)
print("Female match counts:")
for j, female in enumerate(solver.females):
    count = solver.X_binary[:, j].sum()
    print(f"  {female:15s}: {count:.1f}")
print()

print(f"Total matches in matrix: {solver.X_binary.sum():.1f}")
if solver.n_males == solver.n_females:
    print(f"Expected for {solver.n_males}×{solver.n_females}: {solver.n_males}")
else:
    n_smaller = min(solver.n_males, solver.n_females)
    print(f"Expected for {solver.n_males}×{solver.n_females}: {n_smaller + 1}")
