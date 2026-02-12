# Are You The One - Matching Solver

A solver for the "Are You The One" matching problem with MIP and Graph-based algorithms, probability calculations, and a static frontend for fans.

## Features

- **Two Solving Algorithms**: MIP (fast single solution) and Graph (all solutions + probabilities)
- **Match Probabilities**: Per-pair probability calculations across all valid solutions
- **Double Match Support**: Handles n x m scenarios (e.g., 11 men + 10 women)
- **Static Frontend**: Pre-computed results served as a modern Astro + Svelte site
- **REST API**: FastAPI with Swagger docs for programmatic access
- **8 German Seasons**: Regular + VIP season data included

## Quick Start

### Frontend (Static Site)

```bash
# Generate solver results as JSON
.venv/bin/python build.py

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
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn ayto_solver.api.main:app --reload
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
3. Run `.venv/bin/python build.py` to regenerate JSON
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

### MIP Solver

Models the problem as a binary optimization: `minimize ||x||_1 subject to: Ax = b, x in {0,1}` using compressed sensing / sparse signal recovery via python-mip/CBC.

## Testing

```bash
# Docker (required on ARM Macs for MIP solver)
docker compose exec api pytest tests/ -v

# Local (AMD64 only)
.venv/bin/pytest tests/ -v
```

## Platform Constraints

- **MIP solver**: Requires `python-mip` (AMD64 Linux only). Use Docker on ARM Macs.
- **Graph solver**: Pure Python + networkx. Works on any platform. Used by `build.py`.
- The `solvers/__init__.py` lazily imports MIP solvers so the graph solver works without `python-mip`.

## Seasons Included

| Season | Year | Type | Status |
|--------|------|------|--------|
| VIP Staffel 4 | 2024 | VIP | Solved (1 solution) |
| VIP Staffel 3 | 2023 | VIP | Solved (1 solution) |
| VIP Staffel 2 | 2022 | VIP | 9 solutions |
| VIP Staffel 1 | 2021 | VIP | Solved (1 solution) |
| Staffel 5 | 2021 | Regular | Solved (1 solution) |
| Staffel 4 | 2020 | Regular | Infeasible (contradictory data) |
| Staffel 3 | 2019 | Regular | 11 solutions |
| Staffel 2 | 2018 | Regular | 10 solutions |

## References

- Graph matching: Tassa, T. (2012). "Finding all maximally-matchable edges in a bipartite graph." *Theoretical Computer Science*, 423, 50-58.
- MIP approach: [Compressed Sensing](https://en.wikipedia.org/wiki/Compressed_sensing)

## License

MIT License
