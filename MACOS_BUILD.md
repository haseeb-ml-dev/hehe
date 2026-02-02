# macOS Build Guide (Traffic Detector)

This guide explains how to build the macOS app bundle for Traffic Detector.

## Prerequisites
- macOS 12+ recommended
- Python 3.9+ installed (python3)
- Xcode Command Line Tools (run `xcode-select --install` if needed)

## Step-by-Step Build
1. Open Terminal.
2. Go to the project folder (use the actual macOS path to the project):
   - Example: /Users/YourName/Desktop/traffic_detector
3. Make the build script executable (run this command first to avoid permission issues):
   - chmod +x build_macos.command
4. Run the build script:
   - ./build_macos.command
5. Wait for the build to finish.
6. Confirm the app exists at:
   - dist/Traffic Detector.app

## Common Fixes
- If macOS blocks the app: right‑click the .app, choose Open, then confirm.
- If build fails due to permissions: run `chmod +x build_macos.command` then retry.
- If a model file is missing: place it in the project root and rebuild.

## Notes
- The build script bundles all required models and config files.
- Output CSVs (Power BI-ready) are written to data/output when the app runs.
