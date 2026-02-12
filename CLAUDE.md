# AYTO Solver

"Are You The One" matching problem solver with FastAPI web API and multiple solver backends.

## Architecture

```
ayto_solver/
  api/
    main.py          # FastAPI app (endpoints, CORS, static files)
    static/
      index.html     # Web UI
  models/
    schemas.py       # Pydantic request/response models
  solvers/
    mip_solver.py       # MIP solver (single solution, fast)
    mip_multi_solver.py # MIP solver (enumerates all solutions, probabilities)
    graph_solver.py     # Graph-based solver (networkx, all solutions)
  utils/
    __init__.py
tests/
  test_examples.py   # Smoke tests against real season data in examples/
  test_properties.py # Hypothesis property-based tests
  test_synthetic.py  # Synthetic 3x3/4x4 test cases
examples/
  *.yaml             # Real AYTO season data files
```

## Commands

### Docker (recommended, required on ARM Macs)

```bash
docker compose build
docker compose up -d
docker compose exec api pytest tests/ -v
docker compose down
```

### Local (AMD64 Linux only — MIP/CBC solver needs AMD64)

```bash
pip install -r requirements.txt
pytest tests/ -v
uvicorn ayto_solver.api.main:app --reload
```

## Platform Constraints

The MIP solver depends on `python-mip` which bundles the CBC solver binary for AMD64 Linux only. On ARM Macs, always use Docker (the compose file sets `platform: linux/amd64`).

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/health` | GET | Health check (`{"status": "healthy"}`) |
| `/solve/mip` | POST | MIP solver (single solution; `?enumerate_solutions=true` for all) |
| `/solve/graph` | POST | Graph solver (all solutions + probabilities) |
| `/docs` | GET | Swagger UI |

## Solver Types

- **MIP** (`mip_solver.py`): Mixed Integer Programming via python-mip/CBC. Fast single solution.
- **MIP Multi** (`mip_multi_solver.py`): Enumerates up to 1000 solutions, calculates match probabilities and double-match candidates.
- **Graph** (`graph_solver.py`): NetworkX-based bipartite matching. Enumerates all valid matchings, computes probabilities.

## Testing

Tests require AMD64 (for MIP solver). Run inside Docker on ARM Macs:

```bash
docker compose exec api pytest tests/ -v
```

`pytest.ini` is configured with `pythonpath = .` and `testpaths = tests`.
