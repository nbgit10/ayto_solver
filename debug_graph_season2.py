"""Debug script for Graph solver Season 2 Ep 18."""
import json
from ayto_solver.solvers.graph_solver import GraphSolver

# Load example
with open('examples/json/AYTO_Season2_Germany_AfterEp18.json', 'r') as f:
    data = json.load(f)

print("=" * 60)
print("Graph Solver Season 2 Ep 18 Debug")
print("=" * 60)
print(f"Males: {len(data['males'])}")
print(f"Females: {len(data['females'])}")
print(f"Matching nights: {len(data['matching_nights'])}")
print(f"Truth booths: {len(data['truth_booths'])}")
print()

solver = GraphSolver(data['males'], data['females'])

# Add truth booths
for tb in data['truth_booths']:
    solver.add_truth_booth(tb['pair'][0], tb['pair'][1], tb['match'])

# Add matching nights
for i, night in enumerate(data['matching_nights']):
    solver.add_matching_night(night['pairs'], night['matches'])
    print(f"Night {i+1}: {night['matches']} matches out of {len(night['pairs'])}")

print()
print(f"Matching night constraints stored: {len(solver.matching_night_constraints)}")
print()

# Try enumerating
print("Enumerating matchings...")
matchings, capped = solver.enumerate_all_matchings(max_matchings=10)

print(f"Valid matchings found: {len(matchings)}")
print(f"Capped: {capped}")

if matchings:
    print("\nFirst solution:")
    for pair in sorted(list(matchings[0])):
        print(f"  {pair[0]:15s} - {pair[1]}")
