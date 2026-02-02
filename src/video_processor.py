import sys
import os

# Ensure local imports work in bundled apps
BASE_DIR = os.path.dirname(__file__)
PARENT_DIR = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
for path in (BASE_DIR, PARENT_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

import cv2
import os
from tqdm import tqdm
from detector import ParkingDetector
from sign_detector import SignDetector
from tracker import Tracker
from motion_estimator import MotionEstimator
from equirectangular_processor import EquirectangularProcessor
import numpy as np
import time
import sys


def _bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0

class VideoProcessor:
    def __init__(
        self,
        is_360_video=False,
        frame_skip=1,
        debug_mode=False,
        process_percentage=100,
        min_hits=3,
        model_size='m',
        show_progress=True,
        progress_callback=None,
        progress_every=30,
        # Add distance filtering parameters
        min_object_size=40,
        max_distance_ratio=0.0015,
    ):
        # Initialize detector with distance filtering
        self.detector = ParkingDetector(
            model_size=model_size,
            min_object_size=min_object_size,
            max_distance_ratio=max_distance_ratio
        )

        # Optional sign detector
        self.sign_detector = SignDetector(conf=0.15, imgsz=960)
        if is_360_video:
            self.sign_detection_interval = max(15, int(frame_skip) * 8)
        else:
            self.sign_detection_interval = max(10, int(frame_skip) * 5)
        self._last_sign_detections = []

        # Tracker settings
        iou_threshold = 0.2 if is_360_video else 0.3
        # How long to keep tracks without a successful match (in *detection* updates).
        # Keeping this too large causes "ghost" boxes to linger when objects leave the frame.
        fs = max(1, int(frame_skip))
        if is_360_video:
            max_age = max(18, 4 * fs)
        else:
            max_age = max(12, 3 * fs)
        self.tracker = Tracker(
            is_360=is_360_video,
            max_age=max_age,
            min_hits=int(min_hits),
            iou_threshold=iou_threshold,
        )
        self.motion_estimator = MotionEstimator()

        self.is_360_video = is_360_video
        self.frame_skip = max(1, int(frame_skip))
        self.debug_mode = debug_mode
        self.process_percentage = max(1, min(100, int(process_percentage)))
        self.show_progress = bool(show_progress)
        self.progress_callback = progress_callback
        self.progress_every = max(1, int(progress_every))
        
        # Store distance filtering parameters
        self.min_object_size = min_object_size
        self.max_distance_ratio = max_distance_ratio

        if self.debug_mode:
            self.detector.set_debug_mode(True)

        if is_360_video:
            self.equi_processor = EquirectangularProcessor()

        self.confidence_threshold = 0.4

        # Statistics
        self.processing_stats = {
            'total_frames': 0,
            'processed_frames': 0,
            'detection_frames': 0,
            'total_detections': 0,
            'filtered_detections': 0,
            'bicycle_detections': 0,
            'parking_sign_detections': 0,
            'no_parking_sign_detections': 0,
            'ev_charging_sign_detections': 0,
        }

        # Unique sign counting across the whole video (prevents per-frame overcounting)
        self._unique_signs = []  # list of dicts: {'bbox': [x1,y1,x2,y2], 'category': str}
        self._unique_sign_counts = {
            'parking_sign': 0,
            'no_parking_sign': 0,
            'ev_charging_sign': 0,
        }
        self._sign_dedupe_iou = 0.55

    def _ingest_sign_detections(self, sign_detections):
        """Update unique sign counts from the latest detections."""
        if not sign_detections:
            return

        for sd in sign_detections:
            try:
                cat = getattr(sd, 'category', None)
                bbox = getattr(sd, 'bbox', None)
                if not cat or not bbox or len(bbox) != 4:
                    continue

                bbox_f = [float(b) for b in bbox]
                matched = False

                for entry in self._unique_signs:
                    if entry.get('category') != cat:
                        continue
                    iou = _bbox_iou(bbox_f, entry.get('bbox', bbox_f))
                    if iou >= float(self._sign_dedupe_iou):
                        # Update stored bbox to the latest (keeps boxes roughly current)
                        entry['bbox'] = bbox_f
                        matched = True
                        break

                if not matched:
                    self._unique_signs.append({'bbox': bbox_f, 'category': str(cat)})
                    if cat in self._unique_sign_counts:
                        self._unique_sign_counts[cat] += 1
            except Exception:
                continue

    def process_video(self, video_path, output_path=None):
        def _safe_print(*args, **kwargs):
            try:
                print(*args, **kwargs)
            except Exception:
                pass
        
        _safe_print(f"📹 Processing: {os.path.basename(video_path)}")
        _safe_print(f"⚡ Frame skip: {self.frame_skip} | 🌐 360°: {'Yes' if self.is_360_video else 'No'}")
        _safe_print(f"🔧 Debug mode: {'On' if self.debug_mode else 'Off'}")
        _safe_print(f"📏 Distance filtering: Objects < {self.min_object_size}px or < {self.max_distance_ratio*100:.2f}% of frame are filtered")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            _safe_print(f"❌ Error: Could not open video: {video_path}")
            return None

        reported_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = float(fps) if fps and fps > 0 else 30.0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        # Auto-enable 360 processing
        if not self.is_360_video and width > 0 and height > 0:
            ratio = float(width) / float(height)
            if 1.8 <= ratio <= 2.2:
                self.is_360_video = True
                self.tracker.is_360 = True
                self.tracker.iou_threshold = 0.2
                if not hasattr(self, 'equi_processor'):
                    self.equi_processor = EquirectangularProcessor()
                _safe_print("🌐 Auto-detected 360° equirectangular video (2:1); enabling 360 handling.")

        if output_path is None:
            base, _ = os.path.splitext(video_path)
            output_path = base + "_processed.avi"

        # MJPG is most compatible as an AVI container; auto-fix if user chose mp4.
        if isinstance(output_path, str) and output_path.lower().endswith('.mp4'):
            output_path = os.path.splitext(output_path)[0] + '.avi'
            _safe_print(f"⚠️  Switching output to AVI for MJPG compatibility: {output_path}")

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        # MJPG is widely supported on Windows and avoids mp4v playback issues.
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not out.isOpened():
            _safe_print(f"❌ Error: Could not open output writer: {output_path}")
            cap.release()
            return None

        frames_to_process = None
        if reported_total_frames > 0:
            frames_to_process = max(1, int(reported_total_frames * (self.process_percentage / 100.0)))

        start_time = time.time()
        processed_count = 0
        frame_idx = 0

        pbar_total = frames_to_process if frames_to_process is not None else None
        disable_bar = (not self.show_progress) or (getattr(sys, 'stderr', None) is None)
        bar_file = None
        if getattr(sys, 'stderr', None) is not None:
            bar_file = sys.stderr
        elif getattr(sys, 'stdout', None) is not None:
            bar_file = sys.stdout
        else:
            class _NullWriter:
                def write(self, *_):
                    return 0
                def flush(self):
                    return None
            bar_file = _NullWriter()

        with tqdm(total=pbar_total, desc="Processing", unit="frame", disable=disable_bar, file=bar_file) as pbar:
            while True:
                if frames_to_process is not None and frame_idx >= frames_to_process:
                    break

                ret, frame = cap.read()
                if not ret or frame is None or frame.size == 0:
                    break

                self.processing_stats['total_frames'] += 1
                height, width = frame.shape[:2]

                # 1. Estimate camera movement
                cam_dx, cam_dy = self.motion_estimator.estimate_camera_movement(frame)

                # Decide whether to run detection on this frame
                should_detect = (processed_count % self.frame_skip == 0)
                detections = []

                if should_detect:
                    self.processing_stats['detection_frames'] += 1

                    # 2. Detect vehicles (with distance filtering)
                    detections = self.detector.detect_vehicles(frame)

                    # Update filter stats
                    filter_stats = self.detector.get_distance_filter_stats()
                    self.processing_stats['total_detections'] += filter_stats['total_detections']
                    self.processing_stats['filtered_detections'] += (
                        filter_stats['filtered_distant'] + filter_stats['filtered_small']
                    )
                    
                    self.processing_stats['bicycle_detections'] += len(
                        [d for d in detections if 'bicycle' in d.get('class_name', '')]
                    )

                    # 3. Handle 360 duplicates
                    if self.is_360_video:
                        detections = self.equi_processor.detect_360_duplicates(detections, width)

                    # 4. Detect signs periodically
                    if self.sign_detector and self.sign_detector.available:
                        if (processed_count % self.sign_detection_interval) == 0:
                            self._last_sign_detections = self.sign_detector.detect(frame)
                            self._ingest_sign_detections(self._last_sign_detections)
                            # Keep processing_stats reflecting UNIQUE totals
                            self.processing_stats['parking_sign_detections'] = int(self._unique_sign_counts.get('parking_sign', 0))
                            self.processing_stats['no_parking_sign_detections'] = int(self._unique_sign_counts.get('no_parking_sign', 0))
                            self.processing_stats['ev_charging_sign_detections'] = int(self._unique_sign_counts.get('ev_charging_sign', 0))

                # 5. Update tracker every frame
                tracks = self.tracker.update(detections, cam_dx, cam_dy, width, height, detection_ran=should_detect)

                # 6. Visualization
                vis_frame = self._draw_enhanced_hud(
                    frame,
                    tracks,
                    cam_dx,
                    cam_dy,
                    frame_idx,
                    width,
                    height,
                    detection_ran=should_detect,
                )
                out.write(vis_frame)

                processed_count += 1
                self.processing_stats['processed_frames'] += 1
                frame_idx += 1

                # GUI progress callback (safe for windowed builds)
                if self.progress_callback is not None and frames_to_process:
                    try:
                        percent = int(min(100, max(0, (processed_count / float(frames_to_process)) * 100)))
                        fps_current = processed_count / max(1e-6, (time.time() - start_time))
                        if (processed_count % self.progress_every) == 0:
                            self.progress_callback({
                                'percent': percent,
                                'processed': processed_count,
                                'total': int(frames_to_process),
                                'fps': fps_current,
                            })
                    except Exception:
                        pass

                if self.show_progress and (processed_count % 30 == 0):
                    counts = self.tracker.get_counts()
                    filter_stats = self.detector.get_distance_filter_stats()
                    detection_frames = self.processing_stats.get('detection_frames', 0)
                    fps_current = processed_count / max(1, time.time() - start_time)
                    pbar.set_postfix({
                        'cars': counts.get('total_cars', 0),
                        'bikes': counts.get('total_bikes', 0),
                        'filtered': filter_stats.get('filtered_distant', 0) + filter_stats.get('filtered_small', 0),
                        'detections': detection_frames,
                        'fps': f'{fps_current:.1f}'
                    })

                pbar.update(1)

            if self.show_progress:
                final_counts = self.tracker.get_counts()
                filter_stats = self.detector.get_distance_filter_stats()
                final_filtered = filter_stats.get('filtered_distant', 0) + filter_stats.get('filtered_small', 0)
                pbar.set_postfix({
                    'cars': final_counts.get('total_cars', 0),
                    'bikes': final_counts.get('total_bikes', 0),
                    'filtered': final_filtered,
                    'detections': self.processing_stats.get('detection_frames', 0)
                })
                try:
                    pbar.refresh()
                except Exception:
                    pass

        # Ensure a final progress callback at 100%
        if self.progress_callback is not None and frames_to_process:
            try:
                self.progress_callback({
                    'percent': 100,
                    'processed': int(processed_count),
                    'total': int(frames_to_process),
                    'fps': processed_count / max(1e-6, (time.time() - start_time)),
                })
            except Exception:
                pass

        cap.release()
        out.release()

        processing_time = time.time() - start_time
        counts = self.tracker.get_counts()

        # Print filtering statistics
        total_detected = self.processing_stats.get('total_detections', 0)
        total_filtered = self.processing_stats.get('filtered_detections', 0)
        if total_detected > 0:
            filter_percentage = (total_filtered / total_detected) * 100
            _safe_print(f"\n📊 Distance filtering: {total_filtered}/{total_detected} objects filtered ({filter_percentage:.1f}%)")

        result = {
            'video_file': os.path.basename(video_path),
            'cars_counted': counts.get('total_cars', 0),
            'bicycles_counted': counts.get('total_bikes', 0),
            'parked_bicycles': counts.get('parked_bikes', 0),
            'moving_bicycles': counts.get('moving_bikes', 0),
            'parking_sign_detections': int(self.processing_stats.get('parking_sign_detections', 0)),
            'no_parking_sign_detections': int(self.processing_stats.get('no_parking_sign_detections', 0)),
            'ev_charging_sign_detections': int(self.processing_stats.get('ev_charging_sign_detections', 0)),
            'standard_bicycles': counts.get('bicycle_types', {}).get('standard', 0),
            'slightly_non_standard': counts.get('bicycle_types', {}).get('slightly_non_standard', 0),
            'highly_non_standard': counts.get('bicycle_types', {}).get('highly_non_standard', 0),
            'processing_time_seconds': round(processing_time, 2),
            'frames_per_second': round(self.processing_stats['processed_frames'] / processing_time, 2) if processing_time > 0 else 0,
            'total_frames': self.processing_stats['total_frames'],
            'detection_frames': self.processing_stats['detection_frames'],
            'bicycle_detections': self.processing_stats['bicycle_detections'],
            'filtered_detections': total_filtered,
        }

        # Detailed summary now handled by main display to avoid duplicate console output
        return result

    def _draw_enhanced_hud(self, frame, tracks, dx, dy, frame_idx, frame_width, frame_height, detection_ran=True):
        img = frame.copy()
        height, width = img.shape[:2]

        # Draw sign detections
        if getattr(self, 'sign_detector', None) is not None and getattr(self.sign_detector, 'available', False):
            for sd in getattr(self, '_last_sign_detections', []) or []:
                try:
                    x1, y1, x2, y2 = map(int, sd.bbox)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width, x2), min(height, y2)
                    if sd.category == 'ev_charging_sign':
                        color = (255, 0, 255)
                        label = f"EV SIGN {sd.confidence:.2f}"
                    elif sd.category == 'no_parking_sign':
                        color = (0, 0, 255)
                        label = f"NO P {sd.confidence:.2f}"
                    else:
                        color = (0, 165, 255)
                        label = f"P SIGN {sd.confidence:.2f}"

                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        img, label, (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                    )
                except Exception:
                    continue
        
        # Draw vehicle tracks.
        # IMPORTANT: Only draw boxes on frames where detection ran.
        # On skipped frames we do not have fresh object positions, and camera-motion
        # prediction can cause boxes to "float" away from objects as they leave view.
        tracks_to_draw = tracks if detection_ran else []

        for t in tracks_to_draw:
            # If detection ran on this frame and the track was not matched, hide it.
            # This prevents stale boxes from lingering when an object leaves the frame.
            try:
                if detection_ran and int(getattr(t, 'misses', 0)) > 0:
                    continue
            except Exception:
                pass

            x1, y1, x2, y2 = map(int, t.bbox)
            
            # Ensure coordinates are within frame bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)

            # If the bbox is fully out of frame or invalid, don't draw it.
            if x2 <= x1 or y2 <= y1:
                continue
            
            # Determine bicycle subtype and color
            if 'bicycle' in t.class_name:
                # Check parking status
                t_class = getattr(t, 'class_name', '')
                t_detail = getattr(t, 'detailed_type', '')
                is_parked = (
                    (isinstance(t_class, str) and '_parked' in t_class) or
                    (isinstance(t_detail, str) and '_parked' in t_detail)
                )

                bike_type = 'standard'
                if hasattr(t, 'detailed_type'):
                    dt = t.detailed_type
                    if isinstance(dt, str) and dt.startswith('bicycle_'):
                        bike_type = dt[len('bicycle_'):]
                    else:
                        bike_type = dt
                
                # Colors based on bicycle type
                if bike_type == 'standard':
                    color = (255, 0, 0)
                    thickness = 2
                    label_color = (200, 200, 255)
                elif bike_type == 'slightly_non_standard':
                    color = (0, 255, 255)
                    thickness = 2
                    label_color = (255, 255, 200)
                elif bike_type == 'highly_non_standard':
                    color = (0, 0, 255)
                    thickness = 3
                    label_color = (255, 200, 200)
                else:
                    color = (128, 128, 128)
                    thickness = 1
                    label_color = (200, 200, 200)
                
                # Create label
                label = f"ID:{t.id}"
                if bike_type != 'standard':
                    if bike_type == 'slightly_non_standard':
                        label += " (SNS)"
                    elif bike_type == 'highly_non_standard':
                        label += " (HNS)"
                    else:
                        label += f" {bike_type[:3]}"

                # Add parking status
                if is_parked:
                    label += " 🅿️"
            else:
                color = (0, 255, 0)
                thickness = 2
                label_color = (200, 255, 200)
                label = f"ID:{t.id} CAR"
            
            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label with background
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            cv2.rectangle(
                img, (x1, y1 - text_height - 10),
                (x1 + text_width + 10, y1), color, -1
            )
            
            cv2.putText(
                img, label, (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 1
            )
            
            # Draw history trail for bicycles
            if 'bicycle' in t.class_name and len(t.positions) > 1:
                pts = np.array([(int(p[0]), int(p[1])) for p in t.positions], np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(img, [pts], False, color, 1)
                
                if len(t.positions) > 0:
                    last_pos = t.positions[-1]
                    cv2.circle(img, (int(last_pos[0]), int(last_pos[1])), 3, color, -1)

        # Professional HUD overlay
        counts = self.tracker.get_counts()
        height, width = img.shape[:2]

        BG_COLOR = (18, 18, 24)
        BORDER = (0, 180, 255)
        TEXT = (255, 255, 255)
        GOOD = (0, 220, 120)
        WARN = (255, 180, 0)
        BAD = (0, 80, 255)

        def draw_panel(x1, y1, x2, y2, title):
            overlay = img.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), BG_COLOR, -1)
            cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)
            cv2.rectangle(img, (x1, y1), (x2, y2), BORDER, 2)
            cv2.putText(img, title, (x1 + 12, y1 + 28), cv2.FONT_HERSHEY_DUPLEX, 0.9, BORDER, 2)
            cv2.line(img, (x1 + 10, y1 + 36), (x2 - 10, y1 + 36), BORDER, 1)

        # Panel 1: Summary
        draw_panel(15, 15, 380, 190, "SUMMARY")
        cv2.putText(img, f"Cars: {counts['total_cars']}", (30, 75), cv2.FONT_HERSHEY_DUPLEX, 1.1, GOOD, 2)
        cv2.putText(img, f"Bikes: {counts['total_bikes']}", (30, 115), cv2.FONT_HERSHEY_DUPLEX, 1.1, GOOD, 2)

        # Panel 2: Bicycle Types
        draw_panel(width - 380, 15, width - 15, 190, "BICYCLE TYPES")
        bike_type_info = [
            ('standard', 'Standard'),
            ('slightly_non_standard', 'Slightly Non-Std'),
            ('highly_non_standard', 'Highly Non-Std'),
        ]
        y = 75
        for key, label in bike_type_info:
            count = counts['bicycle_types'].get(key, 0)
            pct = (count / counts['total_bikes'] * 100) if counts['total_bikes'] > 0 else 0
            cv2.putText(img, f"{label}: {count} ({pct:.0f}%)", (width - 360, y),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, TEXT, 2)
            y += 32

        # Panel 3: Parking
        draw_panel(15, height - 170, 380, height - 15, "PARKING")
        parked = int(counts.get('parked_bikes', 0))
        moving = int(counts.get('moving_bikes', 0))
        cv2.putText(img, f"Parked: {parked}", (30, height - 115), cv2.FONT_HERSHEY_DUPLEX, 1.0, GOOD, 2)
        cv2.putText(img, f"Wrongly Parked: {moving}", (30, height - 75), cv2.FONT_HERSHEY_DUPLEX, 1.0, BAD, 2)

        # Panel 4: Processing
        draw_panel(width - 380, height - 170, width - 15, height - 15, "PROCESSING")
        filter_stats = self.detector.get_distance_filter_stats()
        filtered_total = filter_stats.get('filtered_distant', 0) + filter_stats.get('filtered_small', 0)
        cv2.putText(img, f"Frame: {frame_idx}", (width - 360, height - 115), cv2.FONT_HERSHEY_DUPLEX, 0.9, TEXT, 2)
        cv2.putText(img, f"Filtered: {filtered_total}", (width - 360, height - 75), cv2.FONT_HERSHEY_DUPLEX, 0.9, WARN, 2)

        # Draw distance threshold line (visual aid)
        min_size_px = self.min_object_size
        cv2.line(img, (0, height - min_size_px), (width, height - min_size_px), (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img, f"Min size: {min_size_px}px", (width - 220, height - min_size_px - 6),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        return img

    def _print_detailed_summary(self, counts, processing_time):
        print("\n" + "=" * 60)
        print("📊 PROCESSING COMPLETE - DETAILED SUMMARY")
        print("=" * 60)
        print("\n" + "-" * 40)
        print(f"🚗 Total Cars: {counts['total_cars']}")
        print(f"🚲 Total Bicycles: {counts['total_bikes']}")

        # Parking statistics
        parked_bikes = counts.get('parked_bikes', 0)
        moving_bikes = counts.get('moving_bikes', 0)
        print("\n🅿️  PARKING STATISTICS:")
        print("-" * 40)
        print(f"   Parked Bicycles: {parked_bikes}")
        print(f"   Wrongly Parked Bicycles: {moving_bikes}")
        if parked_bikes + moving_bikes > 0:
            parked_percentage = (parked_bikes / (parked_bikes + moving_bikes)) * 100
            print(f"   Parking Rate: {parked_percentage:.1f}%")
        
        if counts['total_bikes'] > 0:
            print("\n🔍 BICYCLE CLASSIFICATION BREAKDOWN:")
            print("-" * 40)
            
            bike_types = [
                ('standard', 'Standard Bicycles'),
                ('slightly_non_standard', 'Slightly Non-Standard'),
                ('highly_non_standard', 'Highly Non-Standard'),
            ]
            
            for key, display_name in bike_types:
                count = counts['bicycle_types'].get(key, 0)
                percentage = (count / counts['total_bikes']) * 100
                print(f"{display_name:30} {count:3d} ({percentage:5.1f}%)")
        
        # Distance filtering stats
        total_detected = self.processing_stats.get('total_detections', 0)
        total_filtered = self.processing_stats.get('filtered_detections', 0)
        if total_detected > 0:
            print("\n📏 DISTANCE FILTERING STATS:")
            print("-" * 40)
            print(f"   Total detections: {total_detected}")
            print(f"   Filtered (distant/small): {total_filtered}")
            filter_percentage = (total_filtered / total_detected) * 100
            print(f"   Filtering rate: {filter_percentage:.1f}%")
        
        print("=" * 60)
    
    def get_detailed_counts(self):
        """Get comprehensive count data"""
        tracker_counts = self.tracker.get_counts()
        detector_counts = self.detector.get_bicycle_type_counts()
        
        result = {
            'tracker': tracker_counts,
            'detector': detector_counts,
            'processing_stats': self.processing_stats
        }
        return result