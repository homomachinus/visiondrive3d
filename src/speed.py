import cv2
import numpy as np
from collections import deque

# Sedan dimensions (m): width x length x height
SEDAN_W = 1.8
SEDAN_L = 4.5
SEDAN_H = 1.5

# Optical flow parameters
_LK_PARAMS = dict(
    winSize  = (15, 15),
    maxLevel = 2,
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01),
)

_FEAT_PARAMS = dict(
    maxCorners   = 150,
    qualityLevel = 0.01,
    minDistance  = 20,
    blockSize    = 7,
)

# Speed scaling: 1 optical-flow pixel/frame -> X m/s  (tunable)
SPEED_SCALE    = 2.3
MIN_FLOW_MAG   = 0.5
MAX_FLOW_MAG   = 80.0
REDETECT_EVERY = 20       # frames between forced re-detection
MIN_POINTS     = 20
SMOOTH_WINDOW  = 20       # frame window for running-mean smoothing

# ROI: top portion of frame (sky/trees/buildings move cleanest with ego motion)
ROI_TOP    = 0.00
ROI_BOTTOM = 0.40


class SpeedEstimator:
    """Lucas-Kanade optical flow speed estimator.

    Feed each BGR video frame to .update(frame) and read .speed_ms
    (metres per second) or .speed_kmh.

    The estimator tracks feature points in the upper 40% of the frame
    (distant stationary scene) and converts median optical-flow magnitude
    to an ego speed estimate.
    """

    def __init__(self):
        self._prev_gray  = None
        self._prev_pts   = None
        self._frame_idx  = 0
        self._hist       = deque(maxlen=SMOOTH_WINDOW)
        self.speed_ms    = 0.0   # current smoothed speed (m/s)
        self.speed_kmh   = 0.0   # convenience km/h

    def reset(self):
        self._prev_gray = None
        self._prev_pts  = None
        self._hist.clear()
        self.speed_ms  = 0.0
        self.speed_kmh = 0.0

    def update(self, frame):
        """Process one BGR frame and update self.speed_ms / self.speed_kmh."""
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        roi_mask = np.zeros((h, w), dtype=np.uint8)
        y0 = int(h * ROI_TOP)
        y1 = int(h * ROI_BOTTOM)
        roi_mask[y0:y1, :] = 255

        need_detect = (
            self._prev_gray is None
            or self._prev_pts is None
            or len(self._prev_pts) < MIN_POINTS
            or self._frame_idx % REDETECT_EVERY == 0
        )

        if need_detect:
            self._prev_pts  = cv2.goodFeaturesToTrack(gray, mask=roi_mask, **_FEAT_PARAMS)
            self._prev_gray = gray
            self._frame_idx += 1
            return self.speed_ms

        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, self._prev_pts, None, **_LK_PARAMS
        )

        if next_pts is None or status is None:
            self._prev_gray = gray
            self._prev_pts  = None
            self._frame_idx += 1
            return self.speed_ms

        ok   = status.ravel() == 1
        p0   = self._prev_pts[ok].reshape(-1, 2)
        p1   = next_pts[ok].reshape(-1, 2)

        if len(p0) > 0:
            diff  = p1 - p0
            mags  = np.linalg.norm(diff, axis=1)
            valid = (mags >= MIN_FLOW_MAG) & (mags <= MAX_FLOW_MAG)
            mags_v = mags[valid]
            raw = float(np.median(mags_v)) * SPEED_SCALE if len(mags_v) > 0 else 0.0
            self._hist.append(raw)

        self.speed_ms  = float(np.mean(self._hist)) if self._hist else 0.0
        self.speed_kmh = self.speed_ms * 3.6

        self._prev_gray = gray
        self._prev_pts  = p1.reshape(-1, 1, 2)
        self._frame_idx += 1
        return self.speed_ms
