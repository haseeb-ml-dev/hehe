"""
Rack-based bicycle parking detector.
Determines if a bicycle is parked by checking for rack-like objects
within a small region around the bicycle bounding box.
"""
import os
import sys
import logging
from typing import Tuple, Optional, List

import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)


def get_resource_path(relative_path: str) -> str:
	"""Resolve resource path for bundled or dev environments."""
	try:
		base_path = sys._MEIPASS  # type: ignore[attr-defined]
	except Exception:
		base_path = os.path.abspath(".")
	return os.path.join(base_path, relative_path)


class ParkingDetector:
	"""Detect racks near bicycle bbox to determine parked vs not parked."""

	def __init__(self, model_path: Optional[str] = None, conf: float = 0.25, imgsz: int = 640):
		self.conf = float(conf)
		self.imgsz = int(imgsz)

		if model_path:
			weights = model_path
		else:
			candidates = [
				"rack_detector_best.pt",
				os.path.join(os.path.dirname(__file__), "rack_detector_best.pt"),
				os.path.join(os.path.dirname(os.path.dirname(__file__)), "rack_detector_best.pt"),
				get_resource_path("rack_detector_best.pt"),
				get_resource_path(os.path.join("src", "rack_detector_best.pt")),
			]
			weights = None
			for cand in candidates:
				if os.path.exists(cand):
					weights = cand
					break
			if not weights:
				weights = "rack_detector_best.pt"
				logger.warning("Rack detector weights not found locally; will attempt download: %s", weights)

		self.model = YOLO(weights)
		logger.info("Rack detector loaded: %s", weights)

	def _expand_bbox(self, bbox: List[float], img_w: int, img_h: int) -> List[int]:
		"""Expand bbox by a small margin to include nearby racks."""
		x1, y1, x2, y2 = bbox
		w = max(1.0, x2 - x1)
		h = max(1.0, y2 - y1)

		pad_x = max(8.0, 0.20 * w)
		pad_y = max(8.0, 0.20 * h)
		pad_x = min(pad_x, 60.0)
		pad_y = min(pad_y, 60.0)

		nx1 = int(max(0, x1 - pad_x))
		ny1 = int(max(0, y1 - pad_y))
		nx2 = int(min(img_w - 1, x2 + pad_x))
		ny2 = int(min(img_h - 1, y2 + pad_y))

		return [nx1, ny1, nx2, ny2]

	def is_bicycle_parked(
		self,
		image: np.ndarray,
		bbox: List[float],
		bike_confidence: float = 0.5,
	) -> Tuple[bool, float, Optional[List[float]]]:
		"""Return (is_parked, rack_confidence, rack_bbox)."""
		try:
			if image is None or not hasattr(image, "shape"):
				return False, 0.0, None

			img_h, img_w = image.shape[:2]
			if img_h < 2 or img_w < 2:
				return False, 0.0, None

			crop_bbox = self._expand_bbox(bbox, img_w, img_h)
			x1, y1, x2, y2 = crop_bbox
			if x2 <= x1 or y2 <= y1:
				return False, 0.0, None

			crop = image[y1:y2, x1:x2]
			if crop.size == 0:
				return False, 0.0, None

			results = self.model(crop, verbose=False, conf=self.conf, imgsz=self.imgsz)
			best_conf = 0.0
			best_box = None

			for r in results:
				if r.boxes is None:
					continue
				for box in r.boxes:
					conf = float(box.conf[0])
					if conf > best_conf:
						best_conf = conf
						bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy().tolist()
						best_box = [
							float(bx1 + x1),
							float(by1 + y1),
							float(bx2 + x1),
							float(by2 + y1),
						]

			is_parked = best_conf >= self.conf
			return is_parked, float(best_conf), best_box
		except Exception as e:
			logger.error("Rack parking detection error: %s", e)
			return False, 0.0, None
