#!/bin/bash
echo "Starting AYTO Solver API server..."
docker run --platform=linux/amd64 -p 8000:8000 ayto-solver-api:latest
