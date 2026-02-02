"""
Vehicle and bicycle detection module.
"""
import cv2
import numpy as np
import os
import sys
import logging
from ultralytics import YOLO

logger = logging.getLogger(__name__)


def get_resource_path(relative_path: str) -> str:
    """Resolve resource path for bundled or dev environments."""
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# Import bicycle classifier
try:
    from bicycle_classifier import BicycleClassifier
    BICYCLE_CLASSIFIER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Bicycle classifier not available: {e}")
    BICYCLE_CLASSIFIER_AVAILABLE = False

# Import parking detector
try:
    from parking_detector import ParkingDetector as ExternalParkingDetector  # type: ignore
    PARKING_DETECTOR_AVAILABLE = True
except ImportError as e:
    try:
        import sys
        _base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        if _base_dir not in sys.path:
            sys.path.insert(0, _base_dir)
        from parking_detector import ParkingDetector as ExternalParkingDetector  # type: ignore
        PARKING_DETECTOR_AVAILABLE = True
    except ImportError as e2:
        logger.warning(f"Parking detector not available: {e2}")
        PARKING_DETECTOR_AVAILABLE = False

class ParkingDetector:
    def __init__(self, model_size='m', confidence_threshold=0.5, device='cpu', 
                 model_path=None, min_object_size=40, max_distance_ratio=0.0015):
        # Confidence threshold for filtering detections
        self.confidence_threshold = confidence_threshold
        
        # DISTANCE FILTERING PARAMETERS
        self.min_object_size = int(min_object_size)  # Minimum dimension in pixels
        self.max_distance_ratio = float(max_distance_ratio)  # Minimum area ratio
        self.min_confidence_for_distant = 0.65  # Higher confidence for smaller objects
        
        self._default_imgsz = 640
        
        # Initialize enhanced bicycle classifier if available
        if BICYCLE_CLASSIFIER_AVAILABLE:
            try:
                print("[DETECTOR] Initializing bicycle classifier...")
                # The classifier will handle path resolution internally
                self.bicycle_classifier = BicycleClassifier(dataset_path='bicycle_dataset', model_save_path='src/bicycle_models')
                print("✅ Enhanced bicycle classifier loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load bicycle classifier: {e}")
                print(f"[DETECTOR] ERROR: {e}")
                self.bicycle_classifier = None
        else:
            self.bicycle_classifier = None
            print("⚠️  Bicycle classifier not available")
        
        # YOLO model file handling
        if model_path:
            model_file = model_path
        else:
            possible_paths = [
                f'yolov8{model_size}.pt',
                os.path.join(os.path.dirname(__file__), f'yolov8{model_size}.pt'),
                os.path.join(os.path.dirname(os.path.dirname(__file__)), f'yolov8{model_size}.pt'),
                get_resource_path(os.path.join('src', f'yolov8{model_size}.pt')),  # PyInstaller bundle in src/
                get_resource_path(f'yolov8{model_size}.pt'),  # PyInstaller bundle root
            ]
            
            model_file = None
            print(f"[DETECTOR] Searching for yolov8{model_size}.pt...")
            for candidate in possible_paths:
                exists = os.path.exists(candidate)
                print(f"  {'✓' if exists else '✗'} {candidate}")
                if exists:
                    model_file = candidate
                    break
            
            if not model_file:
                model_file = f'yolov8{model_size}.pt'
                print(f"⚠️  Local YOLO model not found, will download: {model_file}")

        try:
            self.vehicle_model = YOLO(model_file)
            print(f"✅ Vehicle detection model loaded: {model_file}")
        except Exception as e:
            logger.exception(f"Failed to load YOLO model '{model_file}': {e}")
            raise
        
        # Move model to device
        try:
            if device and device.lower() != 'cpu':
                if hasattr(self.vehicle_model, 'to'):
                    try:
                        self.vehicle_model.to(device)
                        print(f"✅ Vehicle model moved to device: {device}")
                    except Exception as e:
                        print(f"⚠️  Could not move vehicle model to {device}: {e}")
        except Exception as e:
            logger.exception("Unexpected error while setting device '%s': %s", device, e)

        # Initialize parking detector
        if PARKING_DETECTOR_AVAILABLE:
            try:
                self.parking_detector = ExternalParkingDetector()
                print("✅ Parking detector loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load parking detector: {e}")
                self.parking_detector = None
        else:
            self.parking_detector = None
            print("⚠️ Parking detector not available")

        # Class map
        self.class_map = {
            1: 'bicycle',
            3: 'bicycle',
            2: 'car',
            5: 'car',
            7: 'car'
        }
        
        # Detailed classification counts
        self.bicycle_type_counts = {
            'standard': 0,
            'slightly_non_standard': 0,
            'highly_non_standard': 0
        }
        
        # For debugging and analysis
        self.classification_history = []
        self.debug_mode = False
        self.parking_status = {}
        
        # Distance filtering statistics
        self.distance_filter_stats = {
            'total_detections': 0,
            'filtered_distant': 0,
            'filtered_small': 0,
            'passed_filters': 0
        }
    
    def _is_object_too_distant(self, bbox, frame_width, frame_height):
        """Check if object is too far away to process reliably"""
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        area = width * height
        
        # 1. Absolute size check
        if width < self.min_object_size or height < self.min_object_size:
            if self.debug_mode:
                print(f"📏 Filtered: Too small ({int(width)}x{int(height)} < {self.min_object_size}px)")
            self.distance_filter_stats['filtered_small'] += 1
            return True
        
        # 2. Relative size check (area ratio)
        frame_area = frame_width * frame_height
        area_ratio = area / frame_area
        
        if area_ratio < self.max_distance_ratio:
            if self.debug_mode:
                print(f"📐 Filtered: Too distant (ratio: {area_ratio:.6f} < {self.max_distance_ratio})")
            self.distance_filter_stats['filtered_distant'] += 1
            return True
        
        # 3. Aspect ratio sanity check (too thin objects are often false positives)
        aspect_ratio = width / height if height > 0 else 0
        if aspect_ratio > 5.0 or aspect_ratio < 0.2:
            if self.debug_mode:
                print(f"📐 Filtered: Unusual aspect ratio ({aspect_ratio:.2f})")
            self.distance_filter_stats['filtered_distant'] += 1
            return True
        
        self.distance_filter_stats['passed_filters'] += 1
        return False
    
    def detect_vehicles(self, image):
        """Detect vehicles with distance filtering"""
        frame_height, frame_width = image.shape[:2]
        
        # Reset filter stats for this frame
        self.distance_filter_stats['total_detections'] = 0
        
        # Run inference with dynamic size
        max_dim = max(int(frame_height), int(frame_width)) if hasattr(image, 'shape') else 0
        if max_dim >= 3500:
            imgsz = 1280
        elif max_dim >= 2000:
            imgsz = 960
        else:
            imgsz = self._default_imgsz

        vehicle_results = self.vehicle_model(image, verbose=False, conf=self.confidence_threshold, imgsz=imgsz)
        detections = []
        
        for r in vehicle_results:
            if r.boxes is not None:
                for box in r.boxes:
                    self.distance_filter_stats['total_detections'] += 1
                    
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # Map to our desired class_name
                    class_name = self.class_map.get(cls_id, None)
                    if class_name is None:
                        continue
                    
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Apply distance filtering
                    if self._is_object_too_distant([x1, y1, x2, y2], frame_width, frame_height):
                        continue
                    
                    # Ensure bbox coordinates are valid
                    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame_width, x2), min(frame_height, y2)
                    
                    detailed_class = class_name
                    bike_type = "standard"
                    bike_confidence = 0.5
                    is_parked = False
                    park_confidence = 0.0
                    is_motor_scooter = (cls_id == 3)

                    if class_name == 'bicycle' and self.bicycle_classifier and not is_motor_scooter:
                        # Adaptive padding based on object size
                        width = x2 - x1
                        height = y2 - y1
                        padding_factor = 0.2  # 20% padding
                        pad_w = int(width * padding_factor)
                        pad_h = int(height * padding_factor)
                        
                        x1_pad = max(0, int(x1) - pad_w)
                        y1_pad = max(0, int(y1) - pad_h)
                        x2_pad = min(frame_width, int(x2) + pad_w)
                        y2_pad = min(frame_height, int(y2) + pad_h)
                        
                        bicycle_roi = image[y1_pad:y2_pad, x1_pad:x2_pad]
                        
                        if bicycle_roi.size > 0 and bicycle_roi.shape[0] > 20 and bicycle_roi.shape[1] > 20:
                            try:
                                bike_type, bike_confidence = self.bicycle_classifier.classify_bicycle(bicycle_roi)
                                
                                if self.debug_mode:
                                    print(f"🔄 Bicycle classification: {bike_type} (conf: {bike_confidence:.2f})")
                                    self.classification_history.append({
                                        'type': bike_type,
                                        'confidence': bike_confidence,
                                        'size': bicycle_roi.shape
                                    })
                                
                                detailed_class = f"bicycle_{bike_type}"
                                
                            except Exception as e:
                                logger.error(f"Error classifying bicycle: {e}")
                                detailed_class = "bicycle_standard"
                                bike_type = "standard"
                        else:
                            detailed_class = "bicycle_standard"
                            bike_type = "standard"

                    if class_name == 'bicycle' and is_motor_scooter:
                        bike_type = "standard"

                    parking_status_suffix = ""
                    if class_name == 'bicycle' and self.parking_detector:
                        try:
                            is_parked_from_detector, park_confidence, _ = self.parking_detector.is_bicycle_parked(
                                image,
                                [x1, y1, x2, y2],
                                bike_confidence
                            )
                            is_parked = bool(is_parked_from_detector)
                            park_confidence = max(0.0, min(1.0, park_confidence))

                            # Only apply parking status if confidence is reliable
                            if park_confidence >= 0.4:
                                parking_status_suffix = "_parked" if is_parked else "_moving"

                            if self.debug_mode:
                                status = "PARKED" if is_parked else "WRONGLY PARKED"
                                print(f"🅿️  Parking detection: {status} (conf: {park_confidence:.2f})")

                        except Exception as e:
                            logger.error(f"Error in parking detection: {e}")

                    if class_name == 'bicycle':
                        base_bike_label = f"bicycle_{bike_type}"
                        detailed_class = base_bike_label + parking_status_suffix
                    
                    detections.append({
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'confidence': conf,
                        'class_name': detailed_class,
                        'class_id': cls_id,
                        'detailed_type': (f"bicycle_{bike_type}" if class_name == 'bicycle' else class_name),
                        'bike_type': bike_type,
                        'bike_confidence': bike_confidence,
                        **({
                            'is_parked': is_parked,
                            'park_confidence': park_confidence,
                        } if class_name == 'bicycle' else {})
                    })
        
        if self.debug_mode and self.distance_filter_stats['total_detections'] > 0:
            filtered = self.distance_filter_stats['filtered_distant'] + self.distance_filter_stats['filtered_small']
            print(f"📊 Distance filtering: {filtered}/{self.distance_filter_stats['total_detections']} objects filtered")
        
        return detections
    
    def draw_detections(self, image, detections):
        """
        Draw all detections on image
        
        Args:
            image: Input image
            detections: Vehicle detections
        Returns:
            Image with all detections drawn
        """
        img = image.copy()
        
        # Draw vehicle detections
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            
            # Color coding for vehicles
            if 'bicycle' in det['class_name']:
                bike_type = det.get('bike_type', 'standard')
                
                if bike_type == 'standard':
                    color = (255, 0, 0)  # Blue for standard
                    label_color = (200, 200, 255)
                elif bike_type == 'slightly_non_standard':
                    color = (0, 255, 255)  # Cyan for slightly non-standard
                    label_color = (255, 255, 200)
                elif bike_type == 'highly_non_standard':
                    color = (0, 0, 255)  # Red for highly non-standard
                    label_color = (255, 200, 200)
                else:
                    color = (200, 200, 200)
                    label_color = (200, 200, 200)
                
                confidence = det.get('bike_confidence', 0.0)
                label = f"{bike_type} {confidence:.2f}"
            else:
                color = (0, 255, 0)  # Green for cars
                label_color = (200, 255, 200)
                label = f"{det['class_name']} {det['confidence']:.2f}"
            
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with background
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            
            cv2.rectangle(
                img, 
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                color,
                -1
            )
            
            cv2.putText(
                img, label, 
                (x1, y1 - baseline - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                label_color, 
                2
            )
        
        return img
    
    @staticmethod
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

    def get_bicycle_type_counts(self):
        """Get counts of each bicycle type detected"""
        return self.bicycle_type_counts.copy()
    
    def get_classification_stats(self):
        """Get classification statistics for debugging"""
        if self.classification_history:
            total = len(self.classification_history)
            types = {}
            for item in self.classification_history:
                t = item['type']
                types[t] = types.get(t, 0) + 1
            
            stats = {
                'total_classifications': total,
                'type_distribution': types,
                'avg_confidence': np.mean([item['confidence'] for item in self.classification_history])
            }
            return stats
        return None
    
    def get_distance_filter_stats(self):
        """Get distance filtering statistics"""
        return self.distance_filter_stats.copy()
    
    def set_debug_mode(self, enabled=True):
        """Enable/disable debug mode"""
        self.debug_mode = enabled
        
    def set_distance_filter_params(self, min_object_size=None, max_distance_ratio=None):
        """Update distance filtering parameters"""
        if min_object_size is not None:
            self.min_object_size = int(min_object_size)
        if max_distance_ratio is not None:
            self.max_distance_ratio = float(max_distance_ratio)
        print(f"📏 Updated distance filters: min_size={self.min_object_size}px, max_ratio={self.max_distance_ratio}")