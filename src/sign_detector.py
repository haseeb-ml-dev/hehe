import os
import sys
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def get_resource_path(relative_path: str) -> str:
    """Get absolute path to resource, works for dev and PyInstaller bundle."""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except AttributeError:
        # Running in normal Python environment
        base_path = os.path.abspath(".")
    
    return os.path.join(base_path, relative_path)

try:
    from ultralytics import YOLO

    ULTRALYTICS_AVAILABLE = True
except Exception as e:
    YOLO = None
    ULTRALYTICS_AVAILABLE = False
    logger.warning("Ultralytics not available: %s", e)


@dataclass
class SignDetection:
    bbox: List[float]  # [x1,y1,x2,y2]
    confidence: float
    label: str
    category: str  # 'parking_sign' | 'no_parking_sign' | 'ev_charging_sign'


class SignDetector:
    """YOLO-World sign detector for parking and EV charging signage.

    Uses open-vocabulary prompts via YOLO-World (`yolov8s-world.pt` by default).
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf: float = 0.15,
        imgsz: int = 640,
    ) -> None:
        self.conf = float(conf)
        self.imgsz = int(imgsz)

        # Tiling (critical for tiny signs in 360° footage)
        self.tiling_enabled = True
        self.tile_overlap = 0.15  # fraction overlap between adjacent tiles
        self.min_frame_for_tiling = 900  # min(width,height) threshold
        self.nms_iou = 0.55

        self.model = None
        self.available = False

        if not ULTRALYTICS_AVAILABLE or YOLO is None:
            print("⚠️  Ultralytics not available; sign detection disabled")
            return

        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

        candidate_paths: List[str] = []
        if model_path:
            candidate_paths.append(model_path)
        # Prefer local weights in repo root
        candidate_paths.extend(
            [
                os.path.join(base_dir, "yolov8s-world.pt"),
                os.path.join(os.path.dirname(__file__), "yolov8s-world.pt"),
                get_resource_path("yolov8s-world.pt"),  # PyInstaller bundle root
                get_resource_path(os.path.join("src", "yolov8s-world.pt")),  # PyInstaller bundle in src/
            ]
        )

        weights = next((p for p in candidate_paths if os.path.exists(p)), None)
        if not weights:
            # Fall back to the canonical Ultralytics model name.
            # If the environment has internet/cache access, Ultralytics will fetch it.
            weights = "yolov8s-world.pt"
            msg = "⚠️  YOLO-World weights not found locally; attempting to load/download yolov8s-world.pt"
            print(msg)
            logger.warning(msg)

        try:
            self.model = YOLO(weights)
            # Require YOLO-World prompt support; otherwise we'd be running a closed-vocab model
            # and silently getting zero relevant detections.
            if not hasattr(self.model, "set_classes"):
                msg = "⚠️  Loaded model does not support YOLO-World prompts; sign detection disabled"
                print(msg)
                logger.warning(msg)
                self.model = None
                self.available = False
                return

            self._set_prompts()
            self.available = True
            print(f"✅ Sign detector loaded: {os.path.basename(weights)}")
            logger.info("Loaded sign detector weights: %s", weights)
        except Exception as e:
            logger.warning("Failed to load YOLO-World model '%s': %s", weights, e)
            self.model = None
            self.available = False
            print(f"⚠️  Failed to load YOLO-World sign detector: {e}")

    def _set_prompts(self) -> None:
        if not self.model:
            return

        # Detailed prompts. Keep short but specific (better precision, fewer false positives).
        # Split parking vs no-parking into separate prompt sets so we can distinguish them.
        self.parking_prompts = [
            "parking sign",
            "P parking sign",
            "public parking sign",
            "parking area sign",
            "parking zone sign",
            "parking allowed sign",
        ]

        self.no_parking_prompts = [
            "no parking sign",
            "no parking symbol sign",
            "no stopping sign",
            "no standing sign",
        ]

        self.ev_prompts = [
            "EV charging sign",
            "electric vehicle charging sign",
            "charging station sign",
            "EV parking sign",
            "electric vehicle only parking sign",
            "charging point sign",
        ]

        # YOLO-World supports `set_classes(list[str])`
        prompts = self.parking_prompts + self.no_parking_prompts + self.ev_prompts
        if hasattr(self.model, "set_classes"):
            try:
                self.model.set_classes(prompts)
            except Exception as e:
                logger.warning("Could not set YOLO-World prompts: %s", e)

        self._prompt_to_category: Dict[str, str] = {}
        for p in self.parking_prompts:
            self._prompt_to_category[p] = "parking_sign"
        for p in self.no_parking_prompts:
            self._prompt_to_category[p] = "no_parking_sign"
        for p in self.ev_prompts:
            self._prompt_to_category[p] = "ev_charging_sign"

    def detect(self, frame) -> List[SignDetection]:
        if not self.available or self.model is None:
            return []

        h, w = frame.shape[:2]

        use_tiling = bool(self.tiling_enabled and min(h, w) >= self.min_frame_for_tiling)

        # For wide 360-ish frames, use more columns (better zoom for distant signs)
        if use_tiling:
            aspect = (w / float(h)) if h else 1.0
            if aspect >= 1.8:
                rows, cols = 2, 3
            else:
                rows, cols = 2, 2
        else:
            rows, cols = 1, 1

        detections = self._detect_tiled(frame, rows=rows, cols=cols) if (rows * cols) > 1 else self._detect_single(frame)
        return self._nms(detections, iou_thresh=self.nms_iou)

    def _detect_single(self, frame) -> List[SignDetection]:
        try:
            results = self.model(frame, verbose=False, conf=self.conf, imgsz=self.imgsz)
        except Exception as e:
            logger.warning("Sign detection failed: %s", e)
            return []

        return self._parse_results(results, x_offset=0, y_offset=0)

    def _detect_tiled(self, frame, rows: int, cols: int) -> List[SignDetection]:
        h, w = frame.shape[:2]
        if h <= 0 or w <= 0:
            return []

        overlap = float(self.tile_overlap)
        tile_w = max(1, int(np.ceil(w / cols)))
        tile_h = max(1, int(np.ceil(h / rows)))

        step_w = max(1, int(tile_w * (1.0 - overlap)))
        step_h = max(1, int(tile_h * (1.0 - overlap)))

        detections: List[SignDetection] = []
        y_starts = list(range(0, max(1, h - tile_h + 1), step_h))
        x_starts = list(range(0, max(1, w - tile_w + 1), step_w))

        # Ensure last tile covers the end
        if not y_starts or y_starts[-1] != (h - tile_h):
            y_starts.append(max(0, h - tile_h))
        if not x_starts or x_starts[-1] != (w - tile_w):
            x_starts.append(max(0, w - tile_w))

        for y0 in y_starts:
            for x0 in x_starts:
                tile = frame[y0 : y0 + tile_h, x0 : x0 + tile_w]
                if tile.size == 0:
                    continue
                try:
                    results = self.model(tile, verbose=False, conf=self.conf, imgsz=self.imgsz)
                except Exception as e:
                    logger.warning("Tiled sign detection failed: %s", e)
                    continue

                detections.extend(self._parse_results(results, x_offset=x0, y_offset=y0))

        return detections

    def _parse_results(self, results, x_offset: int, y_offset: int) -> List[SignDetection]:
        out: List[SignDetection] = []

        for r in results:
            boxes = getattr(r, "boxes", None)
            if boxes is None:
                continue

            names = getattr(r, "names", None) or getattr(self.model, "names", None)

            for b in boxes:
                try:
                    conf = float(b.conf[0])
                    x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().tolist()
                    cls_id = int(b.cls[0]) if getattr(b, "cls", None) is not None else None

                    label = "sign"
                    if isinstance(names, dict) and cls_id is not None and cls_id in names:
                        label = str(names[cls_id])

                    norm_label = label.strip()
                    category = self._prompt_to_category.get(norm_label)
                    if not category:
                        ll = norm_label.lower()
                        if "ev" in ll or "charging" in ll or "electric" in ll:
                            category = "ev_charging_sign"
                        elif "no parking" in ll or "no-parking" in ll or "no stopping" in ll or "no standing" in ll:
                            category = "no_parking_sign"
                        elif "parking" in ll or ll == "p":
                            category = "parking_sign"
                        else:
                            continue

                    out.append(
                        SignDetection(
                            bbox=[
                                float(x1 + x_offset),
                                float(y1 + y_offset),
                                float(x2 + x_offset),
                                float(y2 + y_offset),
                            ],
                            confidence=conf,
                            label=norm_label,
                            category=category,
                        )
                    )
                except Exception:
                    continue

        return out

    @staticmethod
    def _iou_xyxy(a: List[float], b: List[float]) -> float:
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

    def _nms(self, detections: List[SignDetection], iou_thresh: float) -> List[SignDetection]:
        # Simple category-wise NMS
        kept: List[SignDetection] = []
        by_cat: Dict[str, List[SignDetection]] = {}
        for d in detections:
            by_cat.setdefault(d.category, []).append(d)

        for cat, ds in by_cat.items():
            ds_sorted = sorted(ds, key=lambda x: x.confidence, reverse=True)
            while ds_sorted:
                best = ds_sorted.pop(0)
                kept.append(best)
                remaining: List[SignDetection] = []
                for other in ds_sorted:
                    if self._iou_xyxy(best.bbox, other.bbox) < float(iou_thresh):
                        remaining.append(other)
                ds_sorted = remaining

        return kept

    @staticmethod
    def summarize(detections: List[SignDetection]) -> Dict[str, int]:
        out = {
            "parking_sign_detections": 0,
            "no_parking_sign_detections": 0,
            "ev_charging_sign_detections": 0,
        }
        for d in detections:
            if d.category == "parking_sign":
                out["parking_sign_detections"] += 1
            elif d.category == "no_parking_sign":
                out["no_parking_sign_detections"] += 1
            elif d.category == "ev_charging_sign":
                out["ev_charging_sign_detections"] += 1
        return out
