# Use build arg for platform flexibility (defaults to AMD64 for MIP solver compatibility)
ARG TARGETPLATFORM=linux/amd64
FROM python:3.10-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY ayto_solver/ ayto_solver/
COPY ayto/ ayto/
COPY examples/ examples/

# Expose port for FastAPI
EXPOSE 8000

# Run FastAPI server
CMD ["uvicorn", "ayto_solver.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
