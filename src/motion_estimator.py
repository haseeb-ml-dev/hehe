import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

class MotionEstimator:
    def __init__(self, smoothing_alpha=0.6):
        self.prev_gray = None
        self.feature_params = dict(maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)
        self.lk_params = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        # Smoothed camera motion values for stability
        self.smooth_dx = 0.0
        self.smooth_dy = 0.0
        self.smoothing_alpha = float(smoothing_alpha)

        # To keep motion estimation fast on 4K/360 footage, run optical flow on a
        # downscaled grayscale image and scale the resulting translation back up.
        self.max_flow_width = 960

    def estimate_camera_movement(self, frame):
        """
        Returns (dx, dy) representing the global camera shift.
        Positive dx means camera moved RIGHT (so image content moves LEFT).

        Uses RANSAC-based affine estimation on matched feature points when possible,
        with a median optical-flow fallback. Result is smoothed with an EMA.
        """
        frame_gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        scale = 1.0
        if self.max_flow_width and frame_gray_full.shape[1] > int(self.max_flow_width):
            scale = float(self.max_flow_width) / float(frame_gray_full.shape[1])
            new_w = int(round(frame_gray_full.shape[1] * scale))
            new_h = int(round(frame_gray_full.shape[0] * scale))
            frame_gray = cv2.resize(frame_gray_full, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            frame_gray = frame_gray_full

        # First frame -> initialize
        if self.prev_gray is None:
            self.prev_gray = frame_gray
            return 0.0, 0.0

        # 1. Find features in previous frame
        p0 = cv2.goodFeaturesToTrack(self.prev_gray, mask=None, **self.feature_params)

        dx, dy = 0.0, 0.0

        if p0 is not None:
            # 2. Calculate Optical Flow to current frame
            p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, frame_gray, p0, None, **self.lk_params)

            if p1 is not None:
                good_new = p1[st.flatten() == 1]
                good_old = p0[st.flatten() == 1]

                if len(good_new) >= 6:
                    try:
                        # Estimate a robust affine transform from old -> new
                        M, inliers = cv2.estimateAffinePartial2D(good_old.reshape(-1,2), good_new.reshape(-1,2), method=cv2.RANSAC, ransacReprojThreshold=3.0)
                        if M is not None:
                            # M is 2x3: [a b tx; c d ty] where tx,ty are translation in image
                            dx = float(M[0,2])
                            dy = float(M[1,2])
                        else:
                            # Fallback to median of flow vectors
                            movement_vectors = good_new - good_old
                            dx = float(np.median(movement_vectors[:, 0]))
                            dy = float(np.median(movement_vectors[:, 1]))
                    except Exception:
                        logger.exception("Affine estimation failed; falling back to median flow")
                        movement_vectors = good_new - good_old
                        dx = float(np.median(movement_vectors[:, 0]))
                        dy = float(np.median(movement_vectors[:, 1]))
                elif len(good_new) > 0:
                    movement_vectors = good_new - good_old
                    dx = float(np.median(movement_vectors[:, 0]))
                    dy = float(np.median(movement_vectors[:, 1]))

        # Smooth values to reduce jitter
        # Scale translation back to full-resolution coordinates.
        if scale != 1.0 and scale > 0:
            dx = dx / scale
            dy = dy / scale

        self.smooth_dx = self.smoothing_alpha * dx + (1.0 - self.smoothing_alpha) * self.smooth_dx
        self.smooth_dy = self.smoothing_alpha * dy + (1.0 - self.smoothing_alpha) * self.smooth_dy

        # Clamp extreme motion to avoid occasional outliers causing tracks to 'jump'
        # (common in very wide/360 frames when feature matching gets confused).
        max_shift = 80.0
        if self.smooth_dx > max_shift:
            self.smooth_dx = max_shift
        elif self.smooth_dx < -max_shift:
            self.smooth_dx = -max_shift

        if self.smooth_dy > max_shift:
            self.smooth_dy = max_shift
        elif self.smooth_dy < -max_shift:
            self.smooth_dy = -max_shift

        # Update previous frame
        self.prev_gray = frame_gray

        return self.smooth_dx, self.smooth_dy