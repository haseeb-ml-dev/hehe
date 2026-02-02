#!/bin/bash
# Build a macOS .app for the GUI launcher
set -e
cd "$(dirname "$0")"

if command -v python3 >/dev/null 2>&1; then
  PY=python3
else
  PY=python
fi

$PY -m venv .venv || true
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install pyinstaller

# Clean old build artifacts to ensure fresh rebuild
rm -rf dist build "Traffic Detector.spec"

# Build windowed app
# Include config, bicycle classifier models, and fixed YOLO weights to avoid downloads at runtime
pyinstaller --windowed --noconfirm --name "Traffic Detector" --collect-all tk --collect-all tcl --hidden-import parking_detector --add-data "config.yaml:." --add-data "parking_detector.py:." --add-data "src/bicycle_models:src/bicycle_models" --add-data "src/yolov8m.pt:." --add-data "yolov8s-world.pt:." --add-data "rack_detector_best.pt:." --add-data "rack_detector_best.onnx:." src/gui_launcher.py

echo "Built app under dist/Traffic Detector.app"
