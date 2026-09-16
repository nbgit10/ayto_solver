# Use buildx platform selection for AMD64 MIP deployments.
FROM ghcr.io/astral-sh/uv:0.12.15 AS uv
FROM python:3.13-slim

COPY --from=uv /uv /uvx /bin/

WORKDIR /app

# Copy the uv project metadata and install locked runtime dependencies.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

# Copy application code
COPY ayto_solver/ ayto_solver/
COPY examples/ examples/
COPY tests/ tests/
COPY pytest.ini .
RUN uv sync --locked --no-dev

# Expose port for FastAPI
EXPOSE 8000

# Run FastAPI server
CMD ["/app/.venv/bin/uvicorn", "ayto_solver.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
