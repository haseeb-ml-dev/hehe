# main.py
"""
Traffic Detector - Main Entry Point

This is the primary script for running the traffic detection system.
It loads configuration from config.yaml and processes video files.

For users:
- Configure settings in config.yaml (or follow prompts)
- Run: python main.py
- Results will be saved to the output directory

For developers:
- See config_loader.py for configuration management
- See video_processor.py for core processing logic
"""

import os
import sys
import csv
from datetime import datetime
from config_loader import load_config


def _safe_float(value):
    """
    Safely convert a value to float, returning None on failure.
    
    Handles None, empty strings, and invalid inputs gracefully.
    Useful for parsing CSV and configuration data.
    
    Args:
        value: Any value to convert to float
    
    Returns:
        float: Converted value or None if conversion fails
    """
    try:
        if value is None:
            return None
        s = str(value).strip()
        if not s:
            return None
        return float(s)
    except Exception:
        return None


def _list_gis_files(base_dir):
    """
    List all CSV files in the GIS metadata directory.
    
    Searches for a 'gis' subdirectory in the base directory and returns
    all CSV files found there. Used for location data linked to videos.
    
    Args:
        base_dir: Base project directory
    
    Returns:
        tuple: (gis_directory_path, list_of_csv_files)
               Returns empty list if directory doesn't exist
    """
    gis_dir = os.path.join(base_dir, 'gis')
    if not os.path.isdir(gis_dir):
        return gis_dir, []
    files = [
        f for f in os.listdir(gis_dir)
        if os.path.isfile(os.path.join(gis_dir, f)) and f.lower().endswith('.csv')
    ]
    files.sort()
    return gis_dir, files


def _load_gis_metadata_for_video(gis_csv_path, video_basename):
    """Load location metadata for a given input video filename.

    Expected columns in CSV: video_file, location_name, latitude, longitude
    """
    if not gis_csv_path or not os.path.exists(gis_csv_path):
        return None

    try:
        with open(gis_csv_path, 'r', newline='', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                vf = (row.get('video_file') or row.get('video') or row.get('filename') or '').strip()
                if not vf:
                    continue
                if os.path.basename(vf) != str(video_basename):
                    continue

                location_name = (row.get('location_name') or row.get('location') or '').strip()
                latitude = _safe_float(row.get('latitude') or row.get('lat'))
                longitude = _safe_float(row.get('longitude') or row.get('lon') or row.get('lng'))

                return {
                    'location_name': location_name,
                    'latitude': latitude,
                    'longitude': longitude,
                }
    except Exception:
        return None

    return None

def get_base_directory():
    """
    Return the project root directory (parent of `src/`).
    
    Dynamically calculates the project root relative to this script location.
    Ensures the system works regardless of current working directory.
    
    Returns:
        str: Absolute path to project root directory
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def _clean_path(path_str: str) -> str:
    """Normalize user-entered or drag-and-drop paths by stripping quotes/whitespace."""
    if not path_str:
        return ''
    return path_str.strip().strip('"').strip("'")


def _select_video_file(video_directory: str) -> str:
    """
    Interactive video file selection for non-technical users.
    Searches multiple common locations if initial directory is empty.
    
    Args:
        video_directory: Directory to search for video files
    
    Returns:
        Path to selected video file
    """
    video_files = []
    current_search_dir = video_directory
    
    # Try initial directory
    if os.path.exists(video_directory):
        video_files = [
            f for f in os.listdir(video_directory)
            if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'))
        ]
        video_files.sort()
    
    # If no videos found, search alternative locations
    if not video_files:
        base_dir = get_base_directory()
        alternative_dirs = [
            os.path.join(base_dir, 'videos'),
            os.path.join(base_dir, 'input_videos'),
            os.path.join(base_dir, 'data', 'input_videos'),
        ]
        
        for alt_dir in alternative_dirs:
            if os.path.exists(alt_dir):
                alt_files = [
                    f for f in os.listdir(alt_dir)
                    if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'))
                ]
                if alt_files:
                    video_files = sorted(alt_files)
                    current_search_dir = alt_dir
                    print(f"✅ Found videos in: {alt_dir}")
                    break
    
    if video_files:
        print(f"📂 Found {len(video_files)} video file(s) in: {current_search_dir}")
        print()
        for i, f in enumerate(video_files, 1):
            print(f"   {i}. {f}")
        print()

        # Auto-select when only one video is available
        if len(video_files) == 1:
            only_video = os.path.join(current_search_dir, video_files[0])
            print(f"✅ Using the only video found: {video_files[0]}")
            return only_video
        
        while True:
            choice = _clean_path(input(f"Select video (1-{len(video_files)}) or paste/drag a file path, Enter=use #1: ").strip())
            
            if choice == '':
                return os.path.join(current_search_dir, video_files[0])

            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(video_files):
                    return os.path.join(current_search_dir, video_files[idx])
            
            if os.path.exists(choice):
                return choice
            
            print("❌ Invalid selection. Try again or drag a video file into this window.")
    else:
        print(f"ℹ️  No video files found in default locations.")
        print()
    
    return input("Enter video file path: ").strip()

def _ask_to_override_config(config: dict) -> dict:
    """
    Ask user if they want to override config values interactively.
    If yes, allows user to change specific settings without editing file.
    
    Args:
        config: Current processing configuration dictionary
    
    Returns:
        Updated configuration dictionary
    """
    print("\n✅ Configuration loaded from config.yaml")
    print("Press Enter to keep recommended settings, or type 'y' to change them.")
    choice = input("Change settings? (y/n) [n]: ").strip().lower()
    
    if choice != 'y':
        return config
    
    # Interactive override mode
    print("\n⚙️  INTERACTIVE SETUP (leave blank to keep the suggested value)")
    print("-" * 70)
    
    # Ask for each customizable parameter
    print("Model Settings:")
    yolo_choice = input(f"  YOLO size (n/s/m/l/x) [{config['yolo_model']}]: ").strip().lower()
    if yolo_choice and yolo_choice in {'n', 's', 'm', 'l', 'x'}:
        config['yolo_model'] = yolo_choice
    
    conf_input = input(f"  Confidence threshold (0.1-0.9) [{config['confidence_threshold']}]: ").strip()
    if conf_input:
        try:
            conf_val = float(conf_input)
            config['confidence_threshold'] = max(0.1, min(0.9, conf_val))
        except ValueError:
            print("    ⚠️  Invalid value, keeping current")
    
    print("\nProcessing Settings:")
    frame_input = input(f"  Frame skip (1-10) [{config['frame_skip']}]: ").strip()
    if frame_input:
        try:
            frame_val = int(frame_input)
            config['frame_skip'] = max(1, min(10, frame_val))
        except ValueError:
            print("    ⚠️  Invalid value, keeping current")
    
    pct_input = input(f"  Process percentage (1-100) [{config['process_percentage']}]: ").strip()
    if pct_input:
        try:
            pct_val = int(pct_input)
            config['process_percentage'] = max(1, min(100, pct_val))
        except ValueError:
            print("    ⚠️  Invalid value, keeping current")
    
    print("\nDetection Settings:")
    min_size_input = input(f"  Min object size in pixels [{config['min_object_size']}]: ").strip()
    if min_size_input:
        try:
            size_val = int(min_size_input)
            config['min_object_size'] = max(20, min(200, size_val))
        except ValueError:
            print("    ⚠️  Invalid value, keeping current")
    
    print("\n✅ Configuration updated!\n")
    return config


def _show_config_summary(config: dict) -> None:
    """Display current configuration in user-friendly format."""
    print("\n⚙️  ACTIVE CONFIGURATION")
    print("-" * 60)
    print(f"  YOLO Model Size: {config['yolo_model']}")
    print(f"  Is 360° Video: {'Yes' if config['is_360'] else 'No'}")
    print(f"  Frame Skip: {config['frame_skip']} (process every {config['frame_skip']} frames)")
    print(f"  Process Percentage: {config['process_percentage']}%")
    print(f"  Detection Confidence: {config['confidence_threshold']}")
    print(f"  Min Hits Before Counting: {config['min_hits']}")
    print(f"  Min Object Size: {config['min_object_size']} pixels")
    print(f"  Debug Mode: {'Enabled' if config['debug_mode'] else 'Disabled'}")
    print("-" * 60)

def save_results_to_csv(results, csv_path):
    """
    Save processing results to CSV file with Power BI optimized format.
    
    Power BI Ready Features:
    - Standardized column names with proper casing
    - Clean data types (no empty strings, use NULL for missing)
    - ISO 8601 timestamps for date/time intelligence
    - Calculated metrics for analysis
    - Consistent numeric formatting
    
    Args:
        results: Dictionary containing detection results
        csv_path: Path to save CSV file
    """
    try:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        # Helper to handle empty values
        def clean_value(value, value_type='str'):
            if value is None or value == '':
                return '' if value_type == 'str' else None
            if value_type == 'int':
                return int(value)
            elif value_type == 'float':
                return round(float(value), 2)
            return str(value)

        # Calculate derived metrics
        total_bikes = int(results.get('bicycles_counted', 0))
        parked = int(results.get('parked_bicycles', 0))
        wrongly_parked = int(results.get('moving_bicycles', 0))
        total_parking_analyzed = parked + wrongly_parked
        
        standard = int(results.get('standard_bicycles', 0))
        slightly_non_std = int(results.get('slightly_non_standard', 0))
        highly_non_std = int(results.get('highly_non_standard', 0))
        
        total_detections = int(results.get('bicycle_detections', 0)) + int(results.get('cars_counted', 0))
        filtered = int(results.get('filtered_detections', 0))
        
        # Power BI optimized column structure with consistent naming
        csv_data = {
            # === Primary Keys & Timestamps ===
            'RecordID': datetime.now().strftime("%Y%m%d%H%M%S"),  # Unique identifier
            'ProcessedDateTime': datetime.now().isoformat(),  # ISO 8601 format
            'ProcessedDate': datetime.now().strftime("%Y-%m-%d"),
            'ProcessedTime': datetime.now().strftime("%H:%M:%S"),
            'ProcessedYear': datetime.now().year,
            'ProcessedMonth': datetime.now().month,
            'ProcessedDay': datetime.now().day,
            'ProcessedHour': datetime.now().hour,
            'ProcessedWeekday': datetime.now().strftime("%A"),
            
            # === Video & Location Data ===
            'VideoFile': clean_value(results.get('video_file', ''), 'str'),
            'LocationName': clean_value(results.get('location_name', ''), 'str'),
            'Latitude': clean_value(results.get('latitude'), 'float'),
            'Longitude': clean_value(results.get('longitude'), 'float'),
            
            # === Vehicle Detection Counts ===
            'TotalVehicles': clean_value(results.get('cars_counted', 0), 'int'),
            'TotalBicycles': clean_value(total_bikes, 'int'),
            'TotalVehiclesAndBicycles': clean_value(results.get('cars_counted', 0) + total_bikes, 'int'),
            
            # === Bicycle Type Classification ===
            'Bicycle_Standard': clean_value(standard, 'int'),
            'Bicycle_SlightlyNonStandard': clean_value(slightly_non_std, 'int'),
            'Bicycle_HighlyNonStandard': clean_value(highly_non_std, 'int'),
            'Bicycle_Standard_Pct': round((standard / max(1, total_bikes)) * 100, 1) if total_bikes > 0 else 0,
            'Bicycle_SlightlyNonStandard_Pct': round((slightly_non_std / max(1, total_bikes)) * 100, 1) if total_bikes > 0 else 0,
            'Bicycle_HighlyNonStandard_Pct': round((highly_non_std / max(1, total_bikes)) * 100, 1) if total_bikes > 0 else 0,
            
            # === Parking Compliance Analysis ===
            'Bicycle_Parked': clean_value(parked, 'int'),
            'Bicycle_WronglyParked': clean_value(wrongly_parked, 'int'),
            'Bicycle_ParkingAnalyzed': clean_value(total_parking_analyzed, 'int'),
            'ParkingComplianceRate_Pct': round((parked / max(1, total_parking_analyzed)) * 100, 1) if total_parking_analyzed > 0 else 0,
            'ParkingViolationRate_Pct': round((wrongly_parked / max(1, total_parking_analyzed)) * 100, 1) if total_parking_analyzed > 0 else 0,
            
            # === Signage Detection ===
            'Signs_Parking': clean_value(results.get('parking_sign_detections', 0), 'int'),
            'Signs_NoParking': clean_value(results.get('no_parking_sign_detections', 0), 'int'),
            'Signs_EVCharging': clean_value(results.get('ev_charging_sign_detections', 0), 'int'),
            'Signs_Total': clean_value(
                results.get('parking_sign_detections', 0) + 
                results.get('no_parking_sign_detections', 0) + 
                results.get('ev_charging_sign_detections', 0), 
                'int'
            ),
            
            # === Data Quality Metrics ===
            'Detections_Valid': clean_value(total_detections, 'int'),
            'Detections_Filtered': clean_value(filtered, 'int'),
            'Detections_Total': clean_value(total_detections + filtered, 'int'),
            'DetectionQualityScore_Pct': round((total_detections / max(1, total_detections + filtered)) * 100, 1),
            
            # === Processing Performance ===
            'ProcessingDuration_Sec': clean_value(results.get('processing_time_seconds', 0), 'float'),
            'ProcessingDuration_Min': round(float(results.get('processing_time_seconds', 0)) / 60, 2),
            'ProcessingSpeed_FPS': clean_value(results.get('frames_per_second', 0), 'float'),
            'VideoFrames_Total': clean_value(results.get('total_frames', 0), 'int'),
            'VideoFrames_Analyzed': clean_value(results.get('detection_frames', 0), 'int'),
            'FrameAnalysisRate_Pct': round((results.get('detection_frames', 0) / max(1, results.get('total_frames', 0))) * 100, 1),
            
            # === Metadata for Filtering ===
            'ModelVersion': 'v1.0',
            'ProcessingProfile': 'Production',
            'DataSource': 'TrafficDetector',
        }

        file_exists = os.path.exists(csv_path)

        with open(csv_path, 'a', newline='', encoding='utf-8-sig') as csvfile:  # UTF-8 with BOM for Excel compatibility
            fieldnames = list(csv_data.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            writer.writerow(csv_data)
        
        return True

    except IOError as e:
        print(f"❌ File I/O Error saving CSV: {e}")
        print(f"   Check permissions for: {csv_path}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error saving CSV: {e}")
        import traceback
        traceback.print_exc()
        return False

def save_detailed_counts_to_csv(detailed_counts, csv_path):
    """
    Save detailed detection breakdown to CSV for granular analysis.
    
    Creates a normalized table with one row per detection category,
    perfect for Power BI drill-through and detailed filtering.
    
    Args:
        detailed_counts: Dictionary with tracker and processing_stats data
        csv_path: Path where CSV file should be saved
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        tracker = (detailed_counts or {}).get('tracker', {})
        total_bikes = tracker.get('total_bikes', 0)
        total_cars = tracker.get('total_cars', 0)
        bike_types = tracker.get('bicycle_types', {})
        
        # Create detailed breakdown rows
        rows = []
        timestamp = datetime.now().isoformat()
        
        # Vehicles
        if total_cars > 0:
            rows.append({
                'Timestamp': timestamp,
                'Category': 'Vehicle',
                'SubCategory': 'Car',
                'Count': total_cars,
                'Percentage': 100.0,
                'TotalInCategory': total_cars
            })
        
        # Bicycles by type
        for bike_type in ['standard', 'slightly_non_standard', 'highly_non_standard']:
            count = bike_types.get(bike_type, 0)
            percentage = (count / total_bikes * 100) if total_bikes > 0 else 0
            
            # Convert to display name
            display_name = {
                'standard': 'Standard',
                'slightly_non_standard': 'Slightly Non-Standard',
                'highly_non_standard': 'Highly Non-Standard'
            }.get(bike_type, bike_type)
            
            rows.append({
                'Timestamp': timestamp,
                'Category': 'Bicycle',
                'SubCategory': display_name,
                'Count': count,
                'Percentage': round(percentage, 1),
                'TotalInCategory': total_bikes
            })
        
        # Processing statistics as separate rows
        processing_stats = (detailed_counts or {}).get('processing_stats', {})
        for stat_name, stat_value in processing_stats.items():
            rows.append({
                'Timestamp': timestamp,
                'Category': 'Processing',
                'SubCategory': stat_name.replace('_', ' ').title(),
                'Count': stat_value,
                'Percentage': None,
                'TotalInCategory': None
            })

        # Write to CSV
        file_exists = os.path.exists(csv_path)
        
        with open(csv_path, 'a' if file_exists else 'w', newline='', encoding='utf-8-sig') as csvfile:
            fieldnames = ['Timestamp', 'Category', 'SubCategory', 'Count', 'Percentage', 'TotalInCategory']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerows(rows)
        
        return True

    except Exception as e:
        print(f"❌ Error saving detailed CSV: {e}")
        import traceback
        traceback.print_exc()
        return False

def generate_summary_report(results, video_path, config, output_dir):
    """
    Generate a comprehensive processing summary report (HTML and Text).
    
    Creates human-readable reports including:
    - Processing metadata (date, time, video info)
    - Detection statistics with visualizations in HTML
    - Configuration used
    - Quality metrics
    - Performance analysis
    
    Args:
        results: Dictionary with detection results from processor
        video_path: Path to the processed video
        config: Configuration dictionary used
        output_dir: Directory to save report files
    
    Returns:
        tuple: (text_report_path, html_report_path) or None if error
    """
    try:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Text Report
        text_report_path = os.path.join(output_dir, f"{video_name}_summary_report.txt")
        with open(text_report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("TRAFFIC DETECTION PROCESSING SUMMARY REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Header Info
            f.write("📋 PROCESSING INFORMATION\n")
            f.write("-" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Video File: {os.path.basename(video_path)}\n")
            f.write(f"Video Size: {os.path.getsize(video_path) / (1024*1024):.1f} MB\n\n")
            
            # Detection Results
            f.write("🎯 DETECTION RESULTS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total Cars Detected: {results.get('cars_counted', 0)}\n")
            f.write(f"Total Bicycles Detected: {results.get('bicycles_counted', 0)}\n")
            f.write(f"Total Detections: {results.get('cars_counted', 0) + results.get('bicycles_counted', 0)}\n\n")
            
            # Bicycle Classification
            f.write("🚲 BICYCLE CLASSIFICATION\n")
            f.write("-" * 80 + "\n")
            f.write(f"Standard Bicycles: {results.get('standard_bicycles', 0)}\n")
            f.write(f"Slightly Non-Standard: {results.get('slightly_non_standard', 0)}\n")
            f.write(f"Highly Non-Standard: {results.get('highly_non_standard', 0)}\n\n")
            
            # Parking Analysis
            parked = results.get('parked_bicycles', 0)
            wrongly = results.get('moving_bicycles', 0)
            compliance = (parked / (parked + wrongly) * 100) if (parked + wrongly) > 0 else 0
            
            f.write("🅿️  PARKING ANALYSIS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Properly Parked: {parked}\n")
            f.write(f"Wrongly Parked: {wrongly}\n")
            f.write(f"Parking Compliance Rate: {compliance:.1f}%\n\n")
            
            # Sign Detections
            f.write("🪧  SIGN DETECTIONS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Parking Signs: {results.get('parking_sign_detections', 0)}\n")
            f.write(f"No Parking Signs: {results.get('no_parking_sign_detections', 0)}\n")
            f.write(f"EV Charging Signs: {results.get('ev_charging_sign_detections', 0)}\n\n")
            
            # Quality Metrics
            filtered = results.get('filtered_detections', 0)
            total_detections = results.get('cars_counted', 0) + results.get('bicycles_counted', 0) + filtered
            quality_score = (1 - (filtered / max(1, total_detections))) * 100 if total_detections > 0 else 0
            
            f.write("📊 QUALITY METRICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Filtered Detections: {filtered}\n")
            f.write(f"Detection Quality Score: {quality_score:.1f}%\n")
            f.write(f"Total Frames: {results.get('total_frames', 0)}\n")
            f.write(f"Detection Frames: {results.get('detection_frames', 0)}\n\n")
            
            # Performance Metrics
            f.write("⚡ PERFORMANCE METRICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Processing Time: {results.get('processing_time_seconds', 0):.1f} seconds\n")
            f.write(f"Processing Speed: {results.get('frames_per_second', 0):.1f} FPS\n")
            f.write(f"Frame Skip Ratio: {results.get('detection_frames', 0) / max(1, results.get('total_frames', 0)):.3f}\n\n")
            
            # Configuration Used
            f.write("⚙️  CONFIGURATION USED\n")
            f.write("-" * 80 + "\n")
            f.write(f"YOLO Model Size: {config.get('yolo_model', 'N/A')}\n")
            f.write(f"Confidence Threshold: {config.get('confidence_threshold', 'N/A')}\n")
            f.write(f"Frame Skip: {config.get('frame_skip', 'N/A')}\n")
            f.write(f"Process Percentage: {config.get('process_percentage', 'N/A')}%\n")
            f.write(f"360° Video: {'Yes' if config.get('is_360') else 'No'}\n")
            f.write(f"Debug Mode: {'Enabled' if config.get('debug_mode') else 'Disabled'}\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("End of Report\n")
            f.write("=" * 80 + "\n")
        
        # HTML Report
        html_report_path = os.path.join(output_dir, f"{video_name}_summary_report.html")
        try:
            html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Traffic Detection Summary - {video_name}</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 24px; color: #222; background: #fafafa; }}
    .card {{ background: #fff; border: 1px solid #e6e6e6; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.06); margin-bottom: 20px; }}
    .card h2 {{ margin: 0; padding: 16px 20px; border-bottom: 1px solid #eee; font-size: 20px; }}
    .card .content {{ padding: 16px 20px; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
    .row {{ display: flex; align-items: center; justify-content: space-between; padding: 6px 0; border-bottom: 1px dashed #eee; }}
    .row:last-child {{ border-bottom: none; }}
    .badge {{ display: inline-block; padding: 4px 10px; border-radius: 999px; font-size: 12px; background: #f0f2ff; color: #334; border: 1px solid #dfe4ff; }}
    .kpi {{ font-weight: 600; }}
    .footer {{ margin-top: 12px; font-size: 12px; color: #666; }}
    .bar-wrap {{ background: #f3f3f3; border-radius: 6px; overflow: hidden; height: 10px; }}
    .bar {{ height: 10px; background: linear-gradient(90deg, #5b8cff, #00c2ff); }}
    .muted {{ color: #666; }}
  </style>
</head>
<body>
  <h1 style="margin-bottom: 12px;">Traffic Detection Summary</h1>
  <div class="muted">Video: <span class="badge">{os.path.basename(video_path)}</span> &nbsp; Size: <span class="badge">{os.path.getsize(video_path)/(1024*1024):.1f} MB</span></div>

  <div class="card">
    <h2>Detection Results</h2>
    <div class="content grid">
      <div class="row"><span>Total Cars</span><span class="kpi">{results.get('cars_counted',0)}</span></div>
      <div class="row"><span>Total Bicycles</span><span class="kpi">{results.get('bicycles_counted',0)}</span></div>
    </div>
  </div>

  <div class="card">
    <h2>Bicycle Classification</h2>
    <div class="content">
      <div class="row"><span>Standard</span><span class="kpi">{results.get('standard_bicycles',0)}</span></div>
      <div class="row"><span>Slightly Non-Standard</span><span class="kpi">{results.get('slightly_non_standard',0)}</span></div>
      <div class="row"><span>Highly Non-Standard</span><span class="kpi">{results.get('highly_non_standard',0)}</span></div>
    </div>
  </div>

  <div class="card">
    <h2>Parking Compliance</h2>
    <div class="content">
      <div class="row"><span>Properly Parked</span><span class="kpi">{results.get('parked_bicycles',0)}</span></div>
      <div class="row"><span>Wrongly Parked</span><span class="kpi">{results.get('moving_bicycles',0)}</span></div>
      <div style="margin-top:8px;" class="muted">Compliance Rate</div>
      <div class="bar-wrap">
        <div class="bar" style="width:{(results.get('parked_bicycles',0)/max(1, results.get('parked_bicycles',0)+results.get('moving_bicycles',0)))*100:.1f}%"></div>
      </div>
    </div>
  </div>

  <div class="card">
    <h2>Sign Detections</h2>
    <div class="content grid">
      <div class="row"><span>Parking Signs</span><span class="kpi">{results.get('parking_sign_detections',0)}</span></div>
      <div class="row"><span>No Parking Signs</span><span class="kpi">{results.get('no_parking_sign_detections',0)}</span></div>
      <div class="row"><span>EV Charging Signs</span><span class="kpi">{results.get('ev_charging_sign_detections',0)}</span></div>
    </div>
  </div>

  <div class="card">
    <h2>Processing Performance</h2>
    <div class="content grid">
      <div class="row"><span>Time</span><span class="kpi">{results.get('processing_time_seconds',0):.1f} seconds</span></div>
      <div class="row"><span>Speed</span><span class="kpi">{results.get('frames_per_second',0):.1f} FPS</span></div>
      <div class="row"><span>Frames</span><span class="kpi">{results.get('total_frames',0)}</span></div>
    </div>
    <div class="footer">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} • Model: {config.get('yolo_model','N/A')} • 360°: {'Yes' if config.get('is_360') else 'No'}</div>
  </div>

  <div class="muted">© Traffic Detector</div>
</body>
</html>"""
            with open(html_report_path, 'w', encoding='utf-8') as hf:
                hf.write(html)
        except Exception:
            html_report_path = None

        return (text_report_path, html_report_path) if html_report_path else text_report_path
        
    except Exception as e:
        print(f"❌ Error generating summary report: {e}")
        return None

def main():
    """
    Main entry point for the Traffic Detector.
    
    Streamlined workflow:
    1. Load and validate configuration
    2. Get video input
    3. Process video
    4. Save results and display summary
    
    Error Handling:
    - Configuration errors: exits gracefully with clear messages
    - File I/O errors: specific error context
    - Processing errors: detailed traceback
    """
    print("\n" + "=" * 70)
    print("TRAFFIC DETECTION & BICYCLE COUNTER")
    print("=" * 70 + "\n")
    
    # Load configuration from config.yaml
    try:
        base_dir = get_base_directory()
        config_loader = load_config(base_dir)
        config = config_loader.get_processing_config()
    except FileNotFoundError as e:
        print(f"❌ Configuration file not found: {e}")
        print("   Please ensure config.yaml exists in the project root directory.")
        return
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        print("   Please check config.yaml for syntax errors.")
        return
    except Exception as e:
        print(f"❌ Unexpected error loading configuration: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Show configuration summary
    _show_config_summary(config)
    
    # Ask if user wants to override settings
    config = _ask_to_override_config(config)
    
    # Get video input
    try:
        video_directory = config_loader.get_video_input_path()
        video_path = _select_video_file(video_directory)
    except Exception as e:
        print(f"❌ Error during video selection: {e}")
        return
    
    # Validate video file
    if not video_path or not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return
    
    # Check file size and permissions
    try:
        file_size = os.path.getsize(video_path)
        if file_size == 0:
            print(f"❌ Video file is empty (0 bytes)")
            return
        size_mb = file_size / (1024*1024)
        print(f"Processing: {os.path.basename(video_path)} ({size_mb:.1f} MB)\n")
    except OSError as e:
        print(f"❌ Error accessing video file: {e}")
        return
    
    # Get output settings
    
    try:
        output_cfg = config_loader.get_output_config()
        output_dir = output_cfg['directory']
        
        # Validate output directory
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            print(f"❌ Error creating output directory: {e}")
            return
        
        # Check write permissions
        if not os.access(output_dir, os.W_OK):
            print(f"❌ No write permission for output directory: {output_dir}")
            return
        
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(output_dir, f"{video_name}_processed.avi")
        save_csv = output_cfg['save_csv']
        csv_path = os.path.splitext(output_path)[0] + "_results.csv" if save_csv else None
    
    except Exception as e:
        print(f"❌ Error setting up output configuration: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Process video
    print("=" * 70)
    print()
    
    processor = None
    results = None
    
    try:
        # Import video processor
        try:
            from video_processor import VideoProcessor
        except ImportError as e:
            print(f"❌ Error importing video processor: {e}")
            return
        
        # Initialize processor
        try:
            processor = VideoProcessor(
                is_360_video=config['is_360'],
                frame_skip=config['frame_skip'],
                debug_mode=config['debug_mode'],
                process_percentage=config['process_percentage'],
                min_hits=config['min_hits'],
                model_size=config['yolo_model'],
                min_object_size=config['min_object_size'],
                max_distance_ratio=config['max_distance_ratio'],
            )
            
            processor.confidence_threshold = config['confidence_threshold']
            if hasattr(processor, 'detector') and processor.detector is not None:
                processor.detector.confidence_threshold = config['confidence_threshold']
        
        except Exception as e:
            print(f"❌ Error initializing video processor: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # Process video
        try:
            results = processor.process_video(video_path, output_path)
        except KeyboardInterrupt:
            print("\n\n⚠️  Processing interrupted by user (Ctrl+C)")
            print("   Partial results may be incomplete.")
            return
        except Exception as e:
            print(f"\n❌ Error during video processing: {e}")
            import traceback
            traceback.print_exc()
            return
        
        if results:
            # Save results
            print("\n" + "=" * 70)
            print("✅ PROCESSING COMPLETED SUCCESSFULLY!")
            print("=" * 70)
            
            if save_csv:
                try:
                    success = save_results_to_csv(results, csv_path)
                    if success:
                        print(f"📊 Results saved to: {os.path.basename(csv_path)}")
                    else:
                        print(f"⚠️  Results not saved (check errors above)")
                except Exception as e:
                    print(f"❌ Error saving results: {e}")
                
                if output_cfg['save_detailed_counts']:
                    try:
                        detailed_counts = processor.get_detailed_counts()
                        detailed_csv_path = os.path.splitext(output_path)[0] + "_detailed_counts.csv"
                        save_detailed_counts_to_csv(detailed_counts, detailed_csv_path)
                        print(f"📈 Detailed counts: {os.path.basename(detailed_csv_path)}")
                    except Exception as e:
                        print(f"⚠️  Error saving detailed counts: {e}")
            
            # Generate summary report
            try:
                report_path = generate_summary_report(results, video_path, config, output_dir)
                if report_path:
                    if isinstance(report_path, tuple):
                        text_path, html_path = report_path
                        if text_path:
                            print(f"📄 Summary report: {os.path.basename(text_path)}")
                        if html_path:
                            print(f"🖼️ HTML report: {os.path.basename(html_path)}")
                    else:
                        print(f"📄 Summary report: {os.path.basename(report_path)}")

                    # Auto-open report and output folder for convenience (Windows-only)
                    try:
                        to_open = None
                        if isinstance(report_path, tuple):
                            to_open = report_path[1] or report_path[0]
                        else:
                            to_open = report_path
                        if os.name == 'nt':
                            if to_open and os.path.exists(to_open):
                                os.startfile(to_open)
                            if os.path.exists(output_dir):
                                os.startfile(output_dir)
                    except Exception:
                        pass
            except Exception as e:
                print(f"⚠️  Error generating summary report: {e}")
            
            # Display results
            print("=" * 70)
            print("DETECTION RESULTS")
            print("=" * 70)
            print()
            print(f"Total Cars:      {results['cars_counted']}")
            print(f"Total Bicycles:  {results['bicycles_counted']}")
            print()
            
            # Bicycle breakdown
            standard = results.get('standard_bicycles', 0)
            slightly_non = results.get('slightly_non_standard', 0)
            highly_non = results.get('highly_non_standard', 0)
            if standard + slightly_non + highly_non > 0:
                print("Bicycle Classification:")
                print(f"  Standard:              {standard}")
                print(f"  Slightly Non-Standard: {slightly_non}")
                print(f"  Highly Non-Standard:   {highly_non}")
                print()
            
            # Parking analysis
            parked_bikes = results.get('parked_bicycles', 0)
            wrongly_parked = results.get('moving_bicycles', 0)
            if parked_bikes + wrongly_parked > 0:
                compliance = (parked_bikes / (parked_bikes + wrongly_parked)) * 100
                print("Parking Compliance:")
                print(f"  Properly Parked:  {parked_bikes}")
                print(f"  Wrongly Parked:   {wrongly_parked}")
                print(f"  Compliance Rate:  {compliance:.1f}%")
                print()
            
            # Sign detections
            parking_signs = results.get('parking_sign_detections', 0)
            no_parking_signs = results.get('no_parking_sign_detections', 0)
            ev_signs = results.get('ev_charging_sign_detections', 0)
            if parking_signs + no_parking_signs + ev_signs > 0:
                print("Signs Detected:")
                print(f"  Parking Signs:        {parking_signs}")
                print(f"  No Parking Signs:     {no_parking_signs}")
                print(f"  EV Charging Signs:    {ev_signs}")
                print()

            # Performance
            print("Processing Performance:")
            print(f"  Time:     {results.get('processing_time_seconds', 0):.1f} seconds")
            print(f"  Speed:    {results.get('frames_per_second', 0):.1f} FPS")
            print(f"  Frames:   {results.get('total_frames', 0)}")
            print()
            
            print("=" * 70)
        else:
            print("=" * 70)
            print("❌ Processing failed - no results returned")
            print("=" * 70)
    
    except ImportError as e:
        print(f"❌ Import error: {e}")
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()