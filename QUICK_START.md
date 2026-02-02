# 🚗 Traffic Detector - Quick Start Guide

## For Non-Technical Users

Welcome! This guide will help you get started with the Traffic Detection system.

---

## ⚡ Quick Start (3 Steps)

### 1. **Prepare Your Video**
   - Place your video file in: `data/input_videos/`
   - Supported formats: MP4, AVI, MOV, MKV, FLV, WMV

### 2. **Run the Program**
   ```bash
   python src/main.py
   ```

### 3. **Get Your Results**
   - Processed video: `data/output/`
   - Results CSV: `data/output/*_results.csv`
   - Detailed counts: `data/output/*_detailed_counts.csv`

---

## 🎛️ Configuration (Optional)

### **Easy Way: Edit `config.yaml`**

The `config.yaml` file is your control center. It has clear descriptions and helpful tips.

**Common adjustments:**

| Scenario | Setting | Value |
|----------|---------|-------|
| Processing too slow | `frame_skip` | 5 or 10 |
| Too many false detections | `confidence_threshold` | 0.5 or 0.6 |
| Missing some objects | `confidence_threshold` | 0.3 |
| Reduce visual clutter | `min_object_size` | 50 or 60 |

---

## 📊 Understanding Results

### **results.csv** contains:
- Total cars detected
- Total bicycles detected
- Bicycle classifications (standard, non-standard)
- Processing statistics
- Parking detection results

### **detailed_counts.csv** contains:
- Breakdown by category
- Percentages for each type
- Processing performance metrics

---

## ⚙️ Settings Explained

### **Frame Skip**
- `frame_skip: 1` = Check every frame (slowest, most accurate)
- `frame_skip: 3` = Check every 3rd frame (balanced - **recommended**)
- `frame_skip: 5` = Check every 5th frame (fastest, less accurate)

### **Confidence Threshold**
- `0.3-0.4` = More detections (might include false positives)
- `0.5-0.6` = Better accuracy (fewer detections)
- **Default: 0.4** (balanced)

### **Detection Sensitivity**
- `1.0` = Very strict (closest objects only)
- `1.5` = Moderate (balanced - **recommended**)
- `2.0` = Loose (include distant objects)

---

## 🆘 Troubleshooting

### **"Video file not found"**
- Check that your video is in `data/input_videos/`
- Verify the file format is supported

### **Processing is very slow**
- Increase `frame_skip` to 5 or 10
- Reduce `process_percentage` to 50
- Use smaller `yolo_size` (e.g., 's' instead of 'm')

### **Missing bicycles in results**
- Lower `confidence_threshold` to 0.3
- Lower `min_object_size` to 30
- Check if bicycles are very small in the video

### **Too many false positives**
- Increase `confidence_threshold` to 0.5 or 0.6
- Increase `min_object_size` to 50 or 60
- Set `min_hits_before_counting` to 2 or 3

---

## 📁 Folder Structure

```
traffic_detector/
├── config.yaml              ← Edit this for settings
├── requirements.txt         ← Package dependencies
├── src/
│   ├── main.py             ← Main program
│   ├── config_loader.py    ← Configuration system
│   └── ...other files
├── data/
│   ├── input_videos/       ← Put videos here
│   └── output/             ← Results saved here
└── gis/                    ← Location data (optional)
```

---

## 🔧 Installation

If you haven't installed dependencies yet:

```bash
pip install -r requirements.txt
```

---

## 💡 Tips for Best Results

1. **Use default settings** - They work well for most scenarios
2. **Test with a short video first** - Get familiar with the system
3. **Consistent lighting** - Videos with consistent lighting work better
4. **Clear objects** - Make sure bicycles/cars are visible and not too blurry
5. **Check the output video** - Helps you understand detection quality

---

## 📞 Common Questions

**Q: How long does processing take?**
A: Depends on video length and settings. A 5-minute video typically takes 2-10 minutes with default settings.

**Q: Can I process multiple videos?**
A: Yes, run `python src/main.py` again for each video.

**Q: What if I want to change settings?**
A: Edit `config.yaml` before running the program. Changes apply automatically.

**Q: Are the results accurate?**
A: Results are ~80-90% accurate depending on video quality, lighting, and object visibility.

---

## 🚀 Next Steps

1. Place a video in `data/input_videos/`
2. Run: `python src/main.py`
3. Select your video when prompted
4. Wait for processing to complete
5. Check `data/output/` for results!

---

**Need help?** Check `config.yaml` - it has detailed descriptions for every setting!
