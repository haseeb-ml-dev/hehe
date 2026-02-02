# 📖 Traffic Detector - User Guide

## Overview

The Traffic Detector is an AI-powered system that automatically:
- ✅ Detects and counts vehicles (cars)
- ✅ Detects and counts bicycles
- ✅ Classifies bicycles by type (standard, non-standard, etc.)
- ✅ Detects parking violations
- ✅ Analyzes parking behavior
- ✅ Handles 360° video footage
- ✅ Filters out distant/small objects to reduce clutter

**All without needing any coding knowledge!**

---

## How It Works

### **Simple 3-Step Process**

```
1. CONFIGURE          2. PROVIDE VIDEO       3. GET RESULTS
   ↓                     ↓                      ↓
config.yaml    +   video file         →   CSV reports
               +   Output settings        Processed video
```

---

## Getting Started

### **Step 1: Install (One Time Only)**

If you haven't done this yet:

```bash
pip install -r requirements.txt
```

This installs all the AI models and libraries needed.

---

### **Step 2: Configure Settings**

Open `config.yaml` in any text editor. You'll see clearly labeled sections:

**Example config.yaml excerpt:**
```yaml
# Video Input Settings
video:
  input_directory: "./data/input_videos"
  is_360_degree: false

# Model Settings  
model:
  yolo_size: "m"              # m = medium (balanced)
  confidence_threshold: 0.4   # 0.1-0.9 range

# Processing
processing:
  frame_skip: 3               # Check every 3rd frame
  process_percentage: 100     # Process all frames
```

**Most common settings to adjust:**

| If you want... | Change this... | To this value |
|---|---|---|
| Faster processing | `frame_skip` | 5 or 10 |
| Slower but more accurate | `frame_skip` | 1 |
| Fewer false alarms | `confidence_threshold` | 0.5 or 0.6 |
| Catch more objects | `confidence_threshold` | 0.3 or 0.35 |

---

### **Step 3: Run the Program**

```bash
python src/main.py
```

The program will:
1. ✅ Load your settings from `config.yaml`
2. ✅ Show you a list of videos in `data/input_videos/`
3. ✅ Ask you to select which video to process
4. ✅ Start processing
5. ✅ Save results to `data/output/`

---

## Understanding Your Results

### **What You Get**

After processing, you'll find in `data/output/`:

| File | Contains |
|------|----------|
| `video_name_processed.avi` | Video with detection boxes drawn |
| `video_name_results.csv` | Summary of all detections |
| `video_name_detailed_counts.csv` | Breakdown by category |

---

### **Reading the Results CSV**

```
video_file,cars_counted,bicycles_counted,standard_bicycles,...
test.mp4,15,42,28,8,6,1200,0.85,...
         ↑  ↑   ↑  ↑ ↑ ↑
         |  |   |  | | └─ Processing time
         |  |   |  | └──  Non-standard bicycles
         |  |   |  └───   Slightly non-standard
         |  |   └────      Standard bicycles
         |  └─────         Total bicycles
         └──────           Total cars
```

---

### **Example Output Interpretation**

```
🚗 Total Vehicles: 15
🚲 Total Bicycles: 42

🅿️  PARKING ANALYSIS:
  Parked Bicycles: 38
  Wrongly Parked: 4
  Parking Rate: 90.5%

🪧 SIGN DETECTIONS:
  Parking Signs: 5
  No Parking Signs: 3
```

**What this means:**
- 15 cars detected in the video
- 42 bicycles detected
- 38 are parked correctly (90.5% parking compliance)
- 4 are parked incorrectly (9.5%)
- 5 parking permit signs and 3 no-parking signs found

---

## Common Scenarios & Settings

### **Scenario 1: Quick Preview (Get Results Fast)**
```yaml
frame_skip: 10              # Check every 10th frame
process_percentage: 50      # Only process 50% of frames
yolo_size: "s"             # Use smaller model
```
**Result:** 5-10x faster, less accurate

### **Scenario 2: Normal Processing (Balanced)**
```yaml
frame_skip: 3              # ← Recommended
process_percentage: 100
yolo_size: "m"            # ← Recommended
```
**Result:** Good speed & accuracy

### **Scenario 3: Highest Accuracy**
```yaml
frame_skip: 1              # Check every frame
process_percentage: 100
yolo_size: "l"            # Use larger model
confidence_threshold: 0.3  # Catch more objects
```
**Result:** Slowest, most accurate

---

## Video Quality Tips

For best results, ensure:

✅ **Clear visibility** - Objects should be clearly visible  
✅ **Adequate lighting** - Not too dark or overexposed  
✅ **Reasonable resolution** - At least 720p recommended  
✅ **Steady camera** - Minimal blur or instability  
✅ **Known locations** - Same area, similar conditions = better results  

---

## Troubleshooting

### ❌ "Error: Video file not found"

**Solution:**
1. Check file is in `data/input_videos/`
2. Verify filename has correct extension (.mp4, .avi, etc.)
3. Make sure file isn't corrupted

---

### ❌ Processing is extremely slow

**Solutions (try in order):**
1. Increase `frame_skip` to 5 or 10
2. Reduce `process_percentage` to 50 or 25
3. Use smaller model: change `yolo_size` to "s"
4. Lower video resolution in your video file

---

### ❌ Missing bicycles (low count)

**Solutions:**
1. Lower `confidence_threshold` to 0.3
2. Lower `min_object_size` to 30
3. Check if bicycles are very small or blurry in video
4. Ensure good lighting in video

---

### ❌ Too many false detections

**Solutions:**
1. Raise `confidence_threshold` to 0.5 or 0.6
2. Raise `min_object_size` to 50 or 60
3. Raise `min_hits_before_counting` to 2

---

### ❌ Output video not created

**Possible causes:**
1. Not enough disk space in `data/output/`
2. Permission issue - check folder permissions
3. Output path invalid - check `config.yaml`

**Solution:** Check that `data/output/` exists and is writable

---

## Advanced Features

### **360° Video Support**

If your video is 360-degree equirectangular format:

```yaml
video:
  is_360_degree: true  # Enable 360 processing
```

The system will intelligently process the spherical video.

---

### **GIS/Location Tagging (Optional)**

To tag detections with location data:

1. Create `gis/metadata.csv`:
   ```
   video_file,location_name,latitude,longitude
   video1.mp4,Downtown Park,40.7128,-74.0060
   video2.mp4,Main Street,40.7614,-73.9776
   ```

2. Set in `config.yaml`:
   ```yaml
   gis:
     metadata_file: "gis/metadata.csv"
   ```

3. Run the program - location data will be included in results

---

## Performance Expectations

| Video Length | Default Settings | Fast Mode |
|---|---|---|
| 1 minute | 2-3 minutes | 20-30 seconds |
| 5 minutes | 10-15 minutes | 1-2 minutes |
| 10 minutes | 20-30 minutes | 2-4 minutes |
| 60 minutes | 2-3 hours | 15-25 minutes |

*Times vary based on system CPU/GPU and video resolution*

---

## Accuracy Expectations

- **Vehicle detection:** ~85-95% accuracy
- **Bicycle detection:** ~75-90% accuracy  
- **Parking classification:** ~80-90% accuracy
- **Sign detection:** ~70-85% accuracy

Accuracy depends on:
- Video quality and resolution
- Lighting conditions
- Object size and visibility
- Configuration settings

---

## Privacy & Data

- ✅ All processing happens on your computer
- ✅ No data is sent to the cloud
- ✅ No personal data collection
- ✅ You control what gets recorded in results

---

## Tips & Tricks

### **💡 Test with a short clip first**
Before processing a long video, test with a 30-second clip to verify settings work well.

### **💡 Keep default settings**
Don't change everything at once. Adjust one setting and re-test.

### **💡 Save important results**
Make backups of important CSV files before running new processing.

### **💡 Check the output video**
Open the processed video to visually verify detection quality.

### **💡 Use consistent settings**
For comparing multiple videos, use the same configuration.

---

## Frequently Asked Questions

**Q: Do I need GPU/CUDA?**
A: No, the system works on CPU. GPU makes it faster, but not required.

**Q: Can I run multiple videos at once?**
A: Run the program once per video. For batch processing, contact support.

**Q: What formats work?**
A: MP4, AVI, MOV, MKV, FLV, WMV

**Q: Can I edit config while processing?**
A: No, finish current processing first, then edit and run again.

**Q: Why is accuracy sometimes lower?**
A: Usually due to video quality, lighting, small objects, or blur.

**Q: Where do I find old results?**
A: All files are saved in `data/output/` with timestamps.

---

## Next Steps

1. ✅ Install: `pip install -r requirements.txt`
2. ✅ Configure: Edit `config.yaml` (or use defaults)
3. ✅ Add video: Place file in `data/input_videos/`
4. ✅ Run: `python src/main.py`
5. ✅ Check results: Open CSV and video in `data/output/`

---

## Support

If you encounter issues:
1. Check **Troubleshooting** section above
2. Review `config.yaml` comments
3. Check console output for error messages
4. Verify video file integrity

---

**Happy detecting! 🚗🚲**
