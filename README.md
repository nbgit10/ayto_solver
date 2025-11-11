# Are You The One - Matching Solver v2.0

A comprehensive solver for the "Are You The One" matching problem, featuring both MIP (Mixed Integer Programming) and Graph-based algorithms with probability calculations.

## Features

### 🔥 Two Solving Algorithms
- **MIP Solver**: Fast single-solution finder using Mixed Integer Programming
- **Graph Solver**: Enumerates all possible solutions (up to 1000) and calculates match probabilities

### 📊 Advanced Analytics
- Match probability calculations for each possible pair
- Double match candidate identification for n×m scenarios (e.g., 11 men + 10 women)
- Solution enumeration with configurable limits
- Handles both balanced (n×n) and unbalanced (n×m) matching scenarios

### 🌐 Modern API & Web UI
- RESTful FastAPI endpoints with OpenAPI (Swagger) documentation
- Interactive web UI for exploring solutions
- JSON input/output format
- Docker support for easy deployment

### 🧪 Comprehensive Testing
- Unit tests with synthetic test cases
- Property-based testing using Hypothesis
- Smoke tests with real-world examples
- Cross-validation between MIP and graph solvers

## Quick Start

### Option 1: Docker (Recommended)

```bash
# Build the Docker image (AMD64 for MIP solver compatibility)
./build.sh

# Run the API server
./run.sh

# Or use docker-compose
docker-compose up
```

The API will be available at `http://localhost:8000`
- Web UI: `http://localhost:8000/`
- API Docs: `http://localhost:8000/docs`

### Option 2: Local Development

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the API server
uvicorn ayto_solver.api.main:app --host 0.0.0.0 --port 8000

# Or run tests
pytest tests/ -v
```

**Note:** The MIP solver requires AMD64 architecture. If you're on ARM64 (Apple Silicon), you must use Docker with `--platform linux/amd64`.

## Architecture

```
ayto_solver/
├── ayto_solver/           # Main package
│   ├── api/              # FastAPI application
│   │   ├── main.py       # API endpoints
│   │   └── static/       # Web UI files
│   ├── models/           # Pydantic schemas
│   │   └── schemas.py    # Request/response models
│   ├── solvers/          # Solving algorithms
│   │   ├── mip_solver.py        # Base MIP solver
│   │   ├── mip_multi_solver.py  # Multi-solution MIP
│   │   └── graph_solver.py      # Graph-based solver
│   └── utils/            # Utility functions
├── ayto/                 # Legacy CLI (backward compatibility)
│   └── ayto.py           # Original solver script
├── examples/             # Example input files
│   ├── *.yaml            # Original YAML format
│   └── json/             # JSON format examples
├── tests/                # Test suite
│   ├── test_synthetic.py # Synthetic test cases
│   ├── test_properties.py # Property-based tests
│   └── test_examples.py  # Real-world examples
└── Dockerfile            # Docker build configuration
```

## API Usage

### 1. Solve with MIP (Single Solution)

**Endpoint:** `POST /solve/mip`

```bash
curl -X POST "http://localhost:8000/solve/mip" \
  -H "Content-Type: application/json" \
  -d @examples/json/AYTO_SeasonVIP_Germany_AfterEP20.json
```

**Response:**
```json
{
  "solution": {
    "matches": [
      ["Alex", "Vanessa"],
      ["Danilo", "Sarah"],
      ...
    ]
  },
  "num_males": 10,
  "num_females": 11,
  "total_matches": 11,
  "solver_type": "mip"
}
```

### 2. Solve with Graph (All Solutions + Probabilities)

**Endpoint:** `POST /solve/graph`

```bash
curl -X POST "http://localhost:8000/solve/graph" \
  -H "Content-Type": application/json" \
  -d @examples/json/AYTO_SeasonVIP_Germany_AfterEP20.json
```

**Response:**
```json
{
  "match_probabilities": [
    {
      "male": "Francesco",
      "female": "Jules",
      "probability": 1.0
    },
    {
      "male": "Alex",
      "female": "Vanessa",
      "probability": 0.75
    },
    ...
  ],
  "double_match_candidates": [
    {
      "name": "Diogo",
      "probability": 0.6,
      "gender": "male"
    },
    ...
  ],
  "total_solutions": 8,
  "solutions_capped": false,
  "num_males": 10,
  "num_females": 11,
  "solver_type": "graph",
  "example_solutions": [...]
}
```

## Input Format (JSON)

```json
{
  "males": ["M1", "M2", "M3"],
  "females": ["F1", "F2", "F3"],
  "matching_nights": [
    {
      "pairs": [
        ["M1", "F1"],
        ["M2", "F2"],
        ["M3", "F3"]
      ],
      "matches": 1
    }
  ],
  "truth_booths": [
    {
      "pair": ["M1", "F2"],
      "match": true
    }
  ]
}
```

## How It Works

### MIP Solver

The MIP solver models the problem as:

```
minimize ||x||₁
subject to: Ax = b, x ∈ {0,1}
```

Where:
- `x` is a binary vector representing all possible pairings
- `A` is the constraint matrix encoding matching nights and truth booths
- `b` is the expected number of matches for each constraint

This compressed sensing approach leverages the sparsity of the solution (only n matches out of n×m possibilities).

### Graph Solver

The graph solver:
1. Models the problem as a bipartite graph (males ↔ females)
2. Edges represent possible matches (not ruled out by constraints)
3. Uses recursive backtracking to enumerate all maximum matchings
4. Handles n×m cases by trying each person as the double-match candidate
5. Calculates probabilities as: `P(pair) = count(solutions with pair) / total solutions`

## n×m Matching (Unbalanced Scenarios)

When there are unequal numbers (e.g., 11 men + 10 women):
- Matching nights still match 10+10 pairs
- One person from the larger set will be involved in a double match
- **New in v2.0**: The solver no longer assumes the last person is the double match
- Both solvers enumerate who could be the double match and calculate probabilities

## Optimization Details

### MIP Approach
- Based on compressed sensing and sparse signal recovery
- Uses L1 norm minimization as a relaxation of L0 norm
- Guarantees finding a valid solution if one exists
- Fast for single solutions (~5 seconds)

### Graph Approach
- Enumerates all possible perfect matchings
- Uses strongly connected components (based on Tamir Tassa's paper)
- Computational cost increases with solution count
- Capped at 1000 solutions for performance

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/test_synthetic.py -v    # Synthetic cases
pytest tests/test_properties.py -v   # Property-based tests
pytest tests/test_examples.py -v     # Real examples

# Run with coverage
pytest tests/ --cov=ayto_solver --cov-report=html
```

**Note:** Tests requiring the MIP solver will fail on ARM64 unless run in Docker.

## Legacy CLI (Backward Compatibility)

The original YAML-based CLI is still available:

```bash
python ayto/ayto.py --yaml_file_path examples/AYTO_SeasonVIP_Germany_AfterEP20.yaml
```

## Known Limitations

### Constraints
- Currently assumes binary gender (males/females matching)
- Cannot enforce specific constraints like "person A and B must match the same person"
- Matching night constraints beyond 0-matches or full-matches require solution enumeration filtering

### Performance
- Graph solver can be slow for highly underconstrained problems
- Solution enumeration capped at 1000 for performance
- MIP solver timeout set to 5-10 seconds per solution

### Platform
- MIP solver requires AMD64 architecture
- Use Docker with `--platform linux/amd64` on ARM64 systems

## Development

### Adding New Solvers

1. Create solver class in `ayto_solver/solvers/`
2. Implement required methods: `__init__`, `add_truth_booth`, `add_matching_night`, `solve`
3. Add endpoint in `ayto_solver/api/main.py`
4. Update schemas if needed in `ayto_solver/models/schemas.py`

### Running in Development Mode

```bash
# Auto-reload on code changes
uvicorn ayto_solver.api.main:app --reload --host 0.0.0.0 --port 8000

# Run with different log level
uvicorn ayto_solver.api.main:app --log-level debug
```

## References

- Original MIP approach: [Compressed Sensing](https://en.wikipedia.org/wiki/Compressed_sensing)
- Graph matching paper: Tassa, T. (2012). "Finding all maximally-matchable edges in a bipartite graph." *Theoretical Computer Science*, 423, 50-58.
- Alternative approaches: [SAS Operations Research Blog](https://blogs.sas.com/content/operations/2018/08/14/are-you-the-one/)

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please:
1. Add tests for new features
2. Run `pytest` and ensure all tests pass
3. Follow existing code style (use `black` formatter)
4. Update documentation

## SPOILER ALERT

<details>
<summary>SPOILER ARE YOU THE ONE SEASON VIP4 GERMANY AFTER EPISODE 18:</summary>

Proposed solution from graph solver (probability > 90%):

- Tim and Linda + Dana ✅ (double match)
- Lucas and Tara
- Nicola and Laura L
- Lars and Jennifer
- Khan and Nadia
- Chris and Emmy ✅
- Alex and Gabriela
- Marc Robin and Laura
- Ozan and Anastassia ✅
- Antonino and Asena

They win the money in the 9th night in episode 20.

</details>

---

**Version 2.0.0** - Complete rewrite with FastAPI, graph solver, and probability calculations
