# tracker.py (UPDATED - fixed bicycle type normalization)
import numpy as np
from scipy.optimize import linear_sum_assignment
from utils_360 import Utils360

class Track:
    def __init__(self, track_id, bbox, class_name, frame_width, detailed_type=None, is_360=False):
        self.id = track_id
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.class_name = class_name  # Main class (bicycle, car)
        self.detailed_type = detailed_type if detailed_type else class_name  # Subclass for bicycles
        self.hits = 1
        self.misses = 0
        self.age = 0
        self.frame_width = frame_width
        self.is_360 = bool(is_360)
        self.positions = [self.get_center()] # History for smoothing
        self.type_history = []  # Track type changes

        # Residual motion after camera-compensated prediction.
        # If an object is stationary in the world, after applying camera_dx/dy
        # the detection should land close to the predicted position.
        self.motion_residuals = []  # list of float magnitudes (px)
        
        # Add detailed type to history if provided
        if detailed_type:
            self.type_history.append(detailed_type)

    def get_center(self):
        return ((self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2)

    def predict(self, camera_dx, camera_dy, is_360=False):
        """
        Shift the track's expected position based on camera movement.
        camera_dx/camera_dy are the estimated translation of image content from prev->current.
        To predict where an object moves on-screen, apply this content shift directly.
        is_360: whether to wrap x coordinate for 360 video
        """
        w = self.bbox[2] - self.bbox[0]
        h = self.bbox[3] - self.bbox[1]

        cx, cy = self.get_center()
        cx = cx + camera_dx
        cy = cy + camera_dy
        
        # Wrap X coordinate only if 360 video
        if is_360:
            cx = Utils360.wrap_x(cx, self.frame_width)
        
        # Update bbox
        self.bbox = [cx - w/2, cy - h/2, cx + w/2, cy + h/2]
        self.age += 1

    def update(self, new_bbox, new_detailed_type=None):
        pred_center = self.get_center()
        det_center = (
            (float(new_bbox[0]) + float(new_bbox[2])) / 2.0,
            (float(new_bbox[1]) + float(new_bbox[3])) / 2.0,
        ) if new_bbox is not None and len(new_bbox) == 4 else pred_center

        dx = float(det_center[0] - pred_center[0])
        if self.is_360 and self.frame_width:
            # seam-aware horizontal residual
            dx = float(Utils360.shortest_distance_x(det_center[0], pred_center[0], self.frame_width))
        else:
            dx = abs(dx)
        dy = det_center[1] - pred_center[1]
        self.motion_residuals.append(float(np.hypot(dx, dy)))
        if len(self.motion_residuals) > 30:
            self.motion_residuals.pop(0)

        # Smooth bbox updates to reduce jitter from detector noise.
        # Keep it simple: EMA between previous bbox and new detection bbox.
        try:
            alpha = 0.7
            if self.bbox is not None and len(self.bbox) == 4 and new_bbox is not None and len(new_bbox) == 4:
                self.bbox = [
                    alpha * float(new_bbox[0]) + (1.0 - alpha) * float(self.bbox[0]),
                    alpha * float(new_bbox[1]) + (1.0 - alpha) * float(self.bbox[1]),
                    alpha * float(new_bbox[2]) + (1.0 - alpha) * float(self.bbox[2]),
                    alpha * float(new_bbox[3]) + (1.0 - alpha) * float(self.bbox[3]),
                ]
            else:
                self.bbox = new_bbox
        except Exception:
            self.bbox = new_bbox
        self.hits += 1
        self.misses = 0
        
        if new_detailed_type:
            self.detailed_type = new_detailed_type
            self.type_history.append(new_detailed_type)
            if len(self.type_history) > 10:  # Keep last 10 type observations
                self.type_history.pop(0)
        
        self.positions.append(self.get_center())
        if len(self.positions) > 30:
            self.positions.pop(0)

    def is_stationary(self, threshold_px=8.0, min_samples=5):
        if len(self.motion_residuals) < int(min_samples):
            return False
        recent = self.motion_residuals[-int(min_samples):]
        return float(np.median(recent)) <= float(threshold_px)

    def get_most_common_type(self):
        """Get the most frequently observed type in history"""
        if not self.type_history:
            return self.detailed_type
        
        from collections import Counter
        return Counter(self.type_history).most_common(1)[0][0]

class Tracker:
    def __init__(self, is_360=False, max_age=30, min_hits=3, iou_threshold=0.3):
        self.tracks = []
        self.frame_count = 0
        self.next_id = 1
        self.is_360 = is_360
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        
        # Detailed counters - ONLY THREE CATEGORIES
        self.total_cars = 0
        self.total_bikes = 0
        self.total_parked_bikes = 0
        self.total_moving_bikes = 0
        self.bicycle_type_counts = {
            'standard': 0,
            'slightly_non_standard': 0,
            'highly_non_standard': 0
        }
        self.counted_ids = set()
        
        # For bicycle type tracking
        self.bicycle_track_types = {}  # track_id -> bicycle type

        # For bicycle parking status tracking
        self.bicycle_track_parking_status = {}  # track_id -> 'parked' | 'moving'

        # Recent counted positions
        # Recent counted boxes to suppress duplicate tracks of the *same* object.
        # Store (bbox, frame_idx, class_key).
        self.counted_positions = []
        # Keep a longer suppression window; track continuity should prevent recounts,
        # but this helps when detections drop out briefly.
        self.recount_cooldown_frames = 900
        self.recount_distance = 80

        # Count only when objects are sufficiently visible (prevents early far detections
        # being counted and then counted again when the camera gets closer).
        # Fractions are relative to the full frame area.
        self.min_area_frac_bicycle = 0.00025
        self.min_area_frac_car = 0.0020

    @staticmethod
    def _normalize_bicycle_type(label):
        """Normalize labels into one of the three bicycle type categories."""
        if not label or not isinstance(label, str):
            return 'standard'

        # Strip parking suffix if present
        if label.endswith('_parked'):
            label = label[:-7]
        elif label.endswith('_moving'):
            label = label[:-7]

        # Remove 'bicycle_' prefix if present
        if label.startswith('bicycle_'):
            label = label[8:]  # Remove 'bicycle_'
        
        # Map common variations to our three categories
        label_lower = label.lower()
        
        # IMPORTANT: check 'slightly'/'highly' before 'standard'
        # because both 'slightly_non_standard' and 'highly_non_standard'
        # contain the substring 'standard'.
        if 'slightly' in label_lower:
            return 'slightly_non_standard'
        if 'highly' in label_lower:
            return 'highly_non_standard'

        # Exact/near-exact matches for standard
        if label_lower in {'standard', 'bicycle'}:
            return 'standard'
        if label_lower.endswith('_standard') or label_lower.startswith('standard'):
            return 'standard'
        
        # Default mapping for unknown types
        if label_lower in ['standard', 'slightly_non_standard', 'highly_non_standard']:
            return label_lower
        
        # Default to standard if unknown
        return 'standard'

    @staticmethod
    def _count_class_key(class_name):
        """Normalize class labels for de-duplication.

        If parking status is appended (e.g., bicycle_standard_parked/moving),
        strip that suffix so parked/moving doesn't cause double-counting.
        """
        if not class_name or not isinstance(class_name, str):
            return class_name

        if 'bicycle' in class_name:
            if class_name.endswith('_parked'):
                return class_name[:-7]
            if class_name.endswith('_moving'):
                return class_name[:-7]
        return class_name

    @staticmethod
    def _parse_parking_status(label):
        """Return 'parked'/'moving' if label encodes parking status."""
        if not label or not isinstance(label, str):
            return None
        if label.endswith('_parked'):
            return 'parked'
        if label.endswith('_moving'):
            return 'moving'
        return None

    @staticmethod
    def _status_from_detection(det):
        """Infer parking status from detection metadata or suffix."""
        status = Tracker._parse_parking_status(det.get('class_name'))
        if status in {'parked', 'moving'}:
            return status
        park_conf = det.get('park_confidence')
        if isinstance(park_conf, (int, float)) and float(park_conf) >= 0.4:
            is_parked = bool(det.get('is_parked'))
            return 'parked' if is_parked else 'moving'
        return None

    def update(self, detections, camera_dx, camera_dy, frame_width, frame_height, detection_ran=True):
        """
        detections: list of dicts {'bbox': [x1,y1,x2,y2], 'class_name': str, 'confidence': float, 'detailed_type': str}
        This method should be called every frame. If no detection on a frame, pass detections=[]
        """
        self.frame_count += 1
        # 1. Predict new positions of existing tracks
        for track in self.tracks:
            track.predict(camera_dx, camera_dy, is_360=self.is_360)

        # If detection didn't run for this frame (frame skipping), don't penalize tracks.
        # We already predicted positions using camera motion, so just keep tracks alive.
        if not detection_ran:
            return self.tracks

        # 2. Match detections to tracks using Hungarian Algorithm
        unmatched_tracks, unmatched_detections, matches = self._match(detections, frame_width)

        # 3. Update matched tracks
        for t_idx, d_idx in matches:
            det = detections[d_idx]
            self.tracks[t_idx].update(det['bbox'], det.get('detailed_type'))
            
            # Store bicycle type if applicable
            if 'bicycle' in det['class_name']:
                # Use detailed_type if available, otherwise use class_name
                bike_type_label = det.get('detailed_type', det['class_name'])
                bike_type = self._normalize_bicycle_type(bike_type_label)
                self.bicycle_track_types[self.tracks[t_idx].id] = bike_type

                # Store parking status if present
                status = self._status_from_detection(det)
                if status:
                    self.bicycle_track_parking_status[self.tracks[t_idx].id] = status

        # 4. Create new tracks for unmatched detections
        for d_idx in unmatched_detections:
            det = detections[d_idx]
            new_track = Track(
                track_id=self.next_id,
                bbox=det['bbox'],
                class_name=det['class_name'],
                frame_width=frame_width,
                detailed_type=det.get('detailed_type'),
                is_360=self.is_360,
            )
            self.tracks.append(new_track)
            
            # Store bicycle type if applicable
            if 'bicycle' in det['class_name']:
                bike_type_label = det.get('detailed_type', det['class_name'])
                bike_type = self._normalize_bicycle_type(bike_type_label)
                self.bicycle_track_types[self.next_id] = bike_type

                status = self._status_from_detection(det)
                if status:
                    self.bicycle_track_parking_status[self.next_id] = status
            
            self.next_id += 1

        # 5. Handle lost tracks
        for t_idx in unmatched_tracks:
            if 0 <= t_idx < len(self.tracks):
                self.tracks[t_idx].misses += 1

        # 6. Remove dead tracks
        self.tracks = [t for t in self.tracks if t.misses < self.max_age]

        # 7. Update counts (only count stable tracks)
        self._update_counts(frame_width, frame_height)

        return self.tracks

    def _match(self, detections, frame_width):
        if len(self.tracks) == 0:
            return [], list(range(len(detections))), []
        if len(detections) == 0:
            return list(range(len(self.tracks))), [], []

        # Cost matrix combining IoU and center distance (more robust than IoU alone).
        # This reduces track fragmentation when an object grows/shrinks (far -> near).
        cost_matrix = np.zeros((len(self.tracks), len(detections)), dtype=np.float32)
        iou_matrix = np.zeros((len(self.tracks), len(detections)), dtype=np.float32)
        dist_norm_matrix = np.zeros((len(self.tracks), len(detections)), dtype=np.float32)

        for t, track in enumerate(self.tracks):
            for d, det in enumerate(detections):
                tb = track.bbox
                db = det['bbox']
                if self.is_360:
                    iou = Utils360.get_iou_360(tb, db, frame_width)
                else:
                    iou = Utils360._bbox_iou(tb, db)

                t_cx, t_cy = track.get_center()
                d_cx = (float(db[0]) + float(db[2])) / 2.0
                d_cy = (float(db[1]) + float(db[3])) / 2.0

                if self.is_360:
                    dist = Utils360.calculate_360_distance((t_cx, t_cy), (d_cx, d_cy), frame_width)
                else:
                    dist = float(np.hypot(d_cx - t_cx, d_cy - t_cy))

                tw = max(1.0, float(tb[2] - tb[0]))
                th = max(1.0, float(tb[3] - tb[1]))
                norm = max(tw, th)
                dist_norm = min(2.0, float(dist) / float(norm))

                iou_matrix[t, d] = float(iou)
                dist_norm_matrix[t, d] = float(dist_norm)

                # Weighted combination; IoU dominates but distance helps keep identity.
                cost_matrix[t, d] = (1.0 - float(iou)) + (0.25 * float(dist_norm))

        # Hungarian Algorithm
        try:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
        except Exception:
            # Fallback greedy matching
            return self._greedy_match(cost_matrix)

        unmatched_tracks = []
        unmatched_detections = []
        matches = []

        # Filter bad matches.
        # Accept if IoU is good OR the center distance is small enough.
        for t, d in zip(row_ind, col_ind):
            iou = float(iou_matrix[t, d])
            dist_norm = float(dist_norm_matrix[t, d])
            if (iou >= float(self.iou_threshold)) or (iou >= 0.05 and dist_norm <= 0.35):
                matches.append((t, d))
            else:
                unmatched_tracks.append(t)
                unmatched_detections.append(d)

        # Find completely unmatched
        for t in range(len(self.tracks)):
            if t not in row_ind:
                unmatched_tracks.append(t)
        
        for d in range(len(detections)):
            if d not in col_ind:
                unmatched_detections.append(d)

        return unmatched_tracks, unmatched_detections, matches

    def _greedy_match(self, cost_matrix):
        """
        Simple greedy matcher used as fallback if scipy is not available.
        Returns: unmatched_tracks, unmatched_detections, matches
        """
        n_tracks, n_dets = cost_matrix.shape
        cost = cost_matrix.copy()
        unmatched_tracks = list(range(n_tracks))
        unmatched_detections = list(range(n_dets))
        matches = []

        # threshold in cost space (larger cost = worse). Accept matches with cost <= (1 - iou_threshold)
        accept_cost_threshold = (1.0 - self.iou_threshold)

        # Greedily pick the smallest cost pair until no acceptable pairs remain
        while True:
            min_idx = np.unravel_index(np.argmin(cost), cost.shape)
            min_val = cost[min_idx]
            if min_val > accept_cost_threshold:
                break
            t_idx, d_idx = min_idx
            matches.append((t_idx, d_idx))

            # mark row and column as used by setting to large value
            cost[t_idx, :] = 1e6
            cost[:, d_idx] = 1e6

        # Recompute unmatched lists
        used_t = {m[0] for m in matches}
        used_d = {m[1] for m in matches}
        unmatched_tracks = [t for t in range(n_tracks) if t not in used_t]
        unmatched_detections = [d for d in range(n_dets) if d not in used_d]

        return unmatched_tracks, unmatched_detections, matches

    def _update_counts(self, frame_width, frame_height):
        """
        Count tracks only once, and avoid recounting by checking recent counted positions.
        Uses Euclidean distance for non-360, and Utils360.calculate_360_distance for 360 videos.
        """
        frame_area = float(max(1, int(frame_width) * int(frame_height)))

        for track in self.tracks:
            # Only count if it's been seen consistently (min_hits)
            # and hasn't been counted yet
            if track.hits >= self.min_hits and track.id not in self.counted_ids:
                class_key = self._count_class_key(track.class_name)

                w = float(track.bbox[2] - track.bbox[0])
                h = float(track.bbox[3] - track.bbox[1])
                area_frac = float(max(0.0, w) * max(0.0, h)) / frame_area

                # Size-based maturity gate (prevents far->near double counts).
                if 'bicycle' in track.class_name:
                    if area_frac < float(self.min_area_frac_bicycle):
                        continue
                else:
                    if area_frac < float(self.min_area_frac_car):
                        continue

                # For cars: additionally require stationarity so moving traffic isn't counted.
                if 'bicycle' not in track.class_name:
                    if track.age < 10:
                        continue
                    size = max(1.0, max(w, h))
                    stationary_threshold = max(10.0, 0.10 * size)
                    if not track.is_stationary(
                        threshold_px=stationary_threshold,
                        min_samples=max(8, self.min_hits),
                    ):
                        continue

                # Check if this center is near a recently-counted position (to avoid duplicates)
                if self._is_near_recent_count(track.bbox, class_key, frame_width):
                    # mark as counted to avoid repeated checks, but do NOT increment totals
                    self.counted_ids.add(track.id)
                    continue

                # Not a recent duplicate -> count it
                if 'bicycle' in track.class_name:
                    self.total_bikes += 1

                    # Determine parked vs wrongly-parked.
                    # Prefer the detector-provided parking status (rack-near-bicycle inference),
                    # falling back to stationarity only if the detector did not provide a status.
                    status = self.bicycle_track_parking_status.get(track.id)
                    if status not in {'parked', 'moving'}:
                        size = max(1.0, max(w, h))
                        stationary_threshold = max(12.0, 0.12 * size)
                        is_stationary = track.is_stationary(
                            threshold_px=stationary_threshold,
                            min_samples=max(5, self.min_hits),
                        )
                        status = 'parked' if is_stationary else 'moving'
                        self.bicycle_track_parking_status[track.id] = status

                    if status == 'parked':
                        self.total_parked_bikes += 1
                    else:
                        # This bucket is displayed as "Wrongly Parked" in the UI.
                        self.total_moving_bikes += 1
                    
                    # Determine bicycle subtype
                    bike_type = 'standard'  # Default to standard
                    
                    # First check if we have stored type for this track
                    if track.id in self.bicycle_track_types:
                        bike_type = self.bicycle_track_types[track.id]
                    # Then check track's detailed type
                    elif hasattr(track, 'detailed_type') and track.detailed_type:
                        bike_type = self._normalize_bicycle_type(track.detailed_type)
                    # Finally check type history
                    elif hasattr(track, 'get_most_common_type'):
                        bike_type = self._normalize_bicycle_type(track.get_most_common_type())
                    
                    # Update subtype count
                    if bike_type in self.bicycle_type_counts:
                        self.bicycle_type_counts[bike_type] += 1
                    else:
                        # If unknown type, count as standard
                        self.bicycle_type_counts['standard'] += 1
                
                else:
                    # group cars, trucks, buses as "cars"
                    self.total_cars += 1

                # register counted id and position
                self.counted_ids.add(track.id)
                self.counted_positions.append((list(track.bbox), self.frame_count, class_key))

        # Cleanup old counted_positions to keep memory bounded
        cutoff = self.frame_count - self.recount_cooldown_frames
        # counted_positions entries: (bbox, frame_idx, class_key)
        self.counted_positions = [p for p in self.counted_positions if p[1] >= cutoff]

    def _is_near_recent_count(self, bbox, class_key, frame_width):
        """Return True if bbox overlaps a recently-counted bbox of same class.

        Uses IoU-based suppression so multiple distinct objects that are close together
        (e.g., many parked bicycles) are NOT accidentally collapsed.
        """
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return False

        current_bbox = list(bbox)

        # Current center/area
        cx = (float(current_bbox[0]) + float(current_bbox[2])) / 2.0
        cy = (float(current_bbox[1]) + float(current_bbox[3])) / 2.0
        cw = max(0.0, float(current_bbox[2]) - float(current_bbox[0]))
        ch = max(0.0, float(current_bbox[3]) - float(current_bbox[1]))
        c_area = cw * ch

        for prev_bbox, prev_frame_idx, pclass in self.counted_positions:
            if pclass != class_key:
                continue

            if self.is_360:
                iou = Utils360.get_iou_360(current_bbox, prev_bbox, frame_width)
            else:
                iou = Utils360._bbox_iou(current_bbox, prev_bbox)

            if iou >= 0.6:
                return True

            # If IoU fails (common after brief occlusion / jitter), add an extra guard
            # for bicycles only: treat a new track as a duplicate if it reappears close
            # to a recently-counted bicycle with similar size.
            if isinstance(class_key, str) and 'bicycle' in class_key:
                try:
                    # Only treat as a re-count if it happens shortly after the original count.
                    # This targets occlusions / tracker resets, but avoids suppressing a genuinely
                    # new bicycle that appears later in roughly the same place.
                    recent_window = 180
                    if (self.frame_count - int(prev_frame_idx)) > int(recent_window):
                        continue

                    pcx = (float(prev_bbox[0]) + float(prev_bbox[2])) / 2.0
                    pcy = (float(prev_bbox[1]) + float(prev_bbox[3])) / 2.0
                    pw = max(0.0, float(prev_bbox[2]) - float(prev_bbox[0]))
                    ph = max(0.0, float(prev_bbox[3]) - float(prev_bbox[1]))
                    p_area = pw * ph

                    if self.is_360:
                        dist = float(Utils360.calculate_360_distance((cx, cy), (pcx, pcy), frame_width))
                    else:
                        dist = float(np.hypot(cx - pcx, cy - pcy))

                    # Size similarity gate: prevents collapsing adjacent parked bikes.
                    area_denom = max(1.0, float(max(c_area, p_area)))
                    area_diff = abs(float(c_area) - float(p_area)) / area_denom

                    # Distance threshold scales with bbox size, capped by recount_distance.
                    size_ref = max(1.0, float(max(pw, ph)))
                    dist_thresh = float(min(self.recount_distance, max(20.0, 0.25 * size_ref)))

                    if area_diff <= 0.55 and dist <= dist_thresh:
                        return True
                except Exception:
                    pass

        return False

    def get_counts(self):
        """Get all counts including bicycle subtypes"""
        return {
            'total_cars': self.total_cars,
            'total_bikes': self.total_bikes,
            'parked_bikes': self.total_parked_bikes,
            'moving_bikes': self.total_moving_bikes,
            'bicycle_types': self.bicycle_type_counts.copy()
        }
    
    def get_detailed_counts(self):
        """Get counts in a format compatible with older code"""
        return self.total_cars, self.total_bikes