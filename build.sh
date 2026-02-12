#!/bin/bash
echo "Building AYTO Solver API Docker image for AMD64 platform..."
docker buildx build --platform=linux/amd64 -t ayto-solver-api:latest .
echo "Build complete! Image tagged as ayto-solver-api:latest"
