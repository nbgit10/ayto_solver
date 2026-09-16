# Are You The One - Matching Solver

A solver for the "Are You The One" matching problem with MIP and Graph-based algorithms, probability calculations, and a static frontend for fans.

## Features

- **Two Solving Algorithms**: MIP (fast single solution) and Graph (all solutions + probabilities)
- **Match Probabilities**: Per-pair probability calculations across all valid solutions
- **Double Match Support**: Handles n x m scenarios (e.g., 11 men + 10 women)
- **Explicit Graph Degree Profiles**: Supports late entrants, unmatched people, and multiple simultaneous double matches
- **Static Frontend**: Pre-computed results served as a modern Astro + Svelte site
- **REST API**: FastAPI with Swagger docs for programmatic access
- **12 German Seasons**: Regular + VIP season data included

## Quick Start

### Frontend (Static Site)

```bash
# Generate solver results as JSON
uv sync
uv run python build.py

# Build the static site
cd frontend && npm install && npm run build

# Dev server with hot reload
cd frontend && npm run dev
```

### API Server (Docker)

```bash
docker compose build
docker compose up -d
# API at http://localhost:8000, Swagger at http://localhost:8000/docs
```

### Local Development

```bash
uv sync
uv run uvicorn ayto_solver.api.main:app --reload
```

**Note:** The MIP solver requires AMD64 architecture. Use Docker on ARM Macs. The Graph solver works natively on ARM.

## Architecture

```
ayto_solver/              # Python package
  solvers/
    graph_solver.py       # Graph-based solver (ARM-native, used by build.py)
    mip_solver.py         # MIP solver (AMD64 only)
    mip_multi_solver.py   # MIP multi-solution solver
  models/schemas.py       # Pydantic request/response models
  api/main.py             # FastAPI endpoints

frontend/                 # Astro + Svelte + Tailwind static site
  src/
    pages/
      index.astro         # Homepage with season overview
      staffel/[slug].astro # Per-season detail pages
    components/           # Svelte interactive components
    lib/                  # TypeScript types, i18n, helpers

examples/*.yaml           # Season input data (hand-edited)
seasons.json              # Season registry
build.py                  # Runs solver, outputs JSON to frontend/public/data/
deploy.sh                 # Full build + deploy pipeline
tests/                    # pytest test suite
```

## Frontend Workflow

1. Update season YAML data in `examples/` after each episode
2. Update `seasons.json` if needed (new season, episode count)
3. Run `uv run python build.py` to regenerate JSON
4. Run `cd frontend && npm run build` to build the static site
5. Deploy with `./deploy.sh` (rsync to VPS)

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/solve/mip` | POST | MIP solver (single solution; `?enumerate_solutions=true` for all) |
| `/solve/graph` | POST | Graph solver (all solutions + probabilities) |
| `/health` | GET | Health check |
| `/docs` | GET | Swagger UI |

## How It Works

### Graph Solver

1. Models the problem as a bipartite graph (males <-> females)
2. Edges represent possible matches (not ruled out by constraints)
3. Uses recursive backtracking to enumerate all maximum matchings
4. Handles n x m cases by trying each person as the double-match candidate
5. Calculates probabilities: `P(pair) = count(solutions with pair) / total solutions`

The graph solver also accepts an optional `degree_profile` with exact
per-person degrees. This is required when roster size no longer describes the
matching shape, such as a late entrant preserving a prior double match:

```json
{
  "males": {"Johannes": 2, "Late entrant": 0},
  "females": {"Marta": 2}
}
```

The two sides must have equal degree totals. The MIP endpoint intentionally
rejects this field; use `/solve/graph` for explicit degree profiles.

For an unresolved double-match candidate, `female_double_candidates` or
`male_double_candidates` in the profile enumerates one exact profile per
candidate and combines the resulting solutions for probability calculations.

### MIP Solver

Models the problem as a binary optimization: `minimize ||x||_1 subject to: Ax = b, x in {0,1}` using compressed sensing / sparse signal recovery via python-mip/CBC.

## Testing

```bash
# Docker (required on ARM Macs for MIP solver)
docker compose exec api pytest tests/ -v

# Local
uv run pytest tests/ -v

# Python dependency audit
uv run pip-audit
```

## Platform Constraints

- **MIP solver**: Requires `python-mip` (AMD64 Linux only). Use Docker on ARM Macs.
- **Graph solver**: Pure Python + networkx. Works on any platform. Used by `build.py`.
- The `solvers/__init__.py` lazily imports MIP solvers so the graph solver works without `python-mip`.

## Seasons Included

| Season | Year | Type | Status |
|--------|------|------|--------|
| VIP Staffel 6 | 2026 | VIP | 56 solutions (current) |
| Staffel 7 | 2026 | Regular | 20,399 solutions |
| VIP Staffel 5 | 2025 | VIP | 28 solutions |
| VIP Staffel 4 | 2024 | VIP | Solved (1 solution) |
| VIP Staffel 3 | 2023 | VIP | Solved (1 solution) |
| VIP Staffel 2 | 2022 | VIP | 9 solutions |
| VIP Staffel 1 | 2021 | VIP | Solved (1 solution) |
| Staffel 6 | 2025 | Regular | 3 solutions |
| Staffel 5 | 2023 | Regular | 2 solutions |
| Staffel 4 | 2022 | Regular | 7 solutions |
| Staffel 3 | 2021 | Regular | 11 solutions |
| Staffel 2 | 2021 | Regular | 10 solutions |

## References

- Graph matching: Tassa, T. (2012). "Finding all maximally-matchable edges in a bipartite graph." *Theoretical Computer Science*, 423, 50-58.
- MIP approach: [Compressed Sensing](https://en.wikipedia.org/wiki/Compressed_sensing)

## License

MIT License
