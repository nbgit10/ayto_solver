#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "==> Running build.py (generating JSON data)..."
.venv/bin/python build.py

echo "==> Building Astro frontend..."
cd frontend
npm run build
cd ..

echo "==> Deploying to VPS..."
rsync -avz --delete frontend/dist/ user@vps:/var/www/ayto/

echo "==> Done!"
