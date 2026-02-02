import cv2
import numpy as np

class EquirectangularProcessor:
    def __init__(self):
        # Thresholds tuned for seam-only duplicate removal.
        # NOTE: duplicates should only happen across the left/right seam edges.
        self.seam_edge_margin = 120
        self.duplicate_threshold = 120
        self.vertical_threshold = 60
        
    def detect_360_duplicates(self, detections, frame_width):
        """
        Removes detections that are actually the same object split across the seam.
        """
        if not detections:
            return detections
            
        unique_detections = []
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        edge_margin = max(10, min(int(self.seam_edge_margin), int(frame_width // 4)))

        for det in detections:
            is_duplicate = False
            bbox = det['bbox']
            center_x = (bbox[0] + bbox[2]) / 2

            near_left = bbox[0] <= edge_margin
            near_right = bbox[2] >= (frame_width - edge_margin)

            # If it's not near the seam edges, it can't be a seam-duplicate.
            if not (near_left or near_right):
                unique_detections.append(det)
                continue
            
            for unique_det in unique_detections:
                u_bbox = unique_det['bbox']
                u_center_x = (u_bbox[0] + u_bbox[2]) / 2

                u_near_left = u_bbox[0] <= edge_margin
                u_near_right = u_bbox[2] >= (frame_width - edge_margin)

                # Seam duplicates should appear on opposite sides only.
                if not ((near_left and u_near_right) or (near_right and u_near_left)):
                    continue
                
                # Across-seam circular distance is the wrap distance.
                dist = abs(center_x - u_center_x)
                wrap_dist = frame_width - dist
                
                # Check vertical distance
                y_dist = abs(((bbox[1]+bbox[3])/2) - ((u_bbox[1]+u_bbox[3])/2))
                
                if wrap_dist < self.duplicate_threshold and y_dist < self.vertical_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_detections.append(det)
        
        return unique_detections