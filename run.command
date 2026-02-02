#!/bin/bash
# macOS double-click launcher for Traffic Detector
# Creates venv (if missing), installs requirements, clears screen, runs the app.

set -e
cd "$(dirname "$0")"

# Choose python binary
if command -v python3 >/dev/null 2>&1; then
  PY=python3
else
  PY=python
fi

# Create venv if missing
if [ ! -d ".venv" ]; then
  echo "Creating virtual environment..."
  "$PY" -m venv .venv
fi

# Activate venv
source .venv/bin/activate

# Install/upgrade deps
python -m pip install --upgrade pip >/dev/null || true
python -m pip install -r requirements.txt || true

# Clean view then run
clear
python src/gui_launcher.py

echo
read -p "Done. Press Enter to close this window..."