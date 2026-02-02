@echo off
setlocal
cd /d "%~dp0"

if not exist .venv (
    echo Creating virtual environment...
    python -m venv .venv
)

call .venv\Scripts\activate.bat
python -m pip install --upgrade pip >nul
python -m pip install -r requirements.txt
python -m pip install pyinstaller

rem Clean old build artifacts to ensure fresh rebuild
if exist dist rmdir /s /q dist
if exist build rmdir /s /q build
if exist "Traffic Detector.spec" del "Traffic Detector.spec"

rem Build a windowed app bundle for GUI launcher
rem Include config, bicycle classifier models, and fixed YOLO weights to avoid downloads at runtime
pyinstaller --windowed --noconfirm --name "Traffic Detector" --collect-all tk --collect-all tcl --hidden-import parking_detector --add-data "config.yaml;." --add-data "parking_detector.py;." --add-data "src\bicycle_models;src\bicycle_models" --add-data "src\yolov8m.pt;src" --add-data "yolov8s-world.pt;." --add-data "rack_detector_best.pt;." --add-data "rack_detector_best.onnx;." src\gui_launcher.py

echo.
echo Built app under dist\Traffic Detector\
echo Done.
