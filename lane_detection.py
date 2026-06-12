import cv2
import numpy as np

VIDEO = "input.mp4"

# ============================================================
# TUNABLE PARAMETERS
# ============================================================
CANNY_LOW       = 50        # Canny edge lower threshold
CANNY_HIGH      = 150       # Canny edge upper threshold
HOUGH_THRESHOLD = 40        # Min votes for a line to count
HOUGH_MIN_LEN   = 40        # Min line length (px)
HOUGH_MAX_GAP   = 20        # Max gap between line segments

ANGLE_MIN       = 20        # Ignore lines flatter than this (deg)
ANGLE_MAX       = 75        # Ignore lines steeper than this (deg)

# ROI: trapezium vertices as fraction of (width, height)
# Bottom-left, Bottom-right, Top-right, Top-left
ROI_BL = (0.05, 1.00)
ROI_BR = (0.95, 1.00)
ROI_TR = (0.60, 0.50)
ROI_TL = (0.40, 0.50)

SMOOTH_ALPHA    = 0.15      # EMA smoothing (0=frozen, 1=no smooth)
# ============================================================


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(img, mask)


def make_roi_vertices(h, w):
    return np.array([[
        (int(ROI_BL[0] * w), int(ROI_BL[1] * h)),
        (int(ROI_BR[0] * w), int(ROI_BR[1] * h)),
        (int(ROI_TR[0] * w), int(ROI_TR[1] * h)),
        (int(ROI_TL[0] * w), int(ROI_TL[1] * h)),
    ]], dtype=np.int32)


def fit_lane(lines, h, side):
    """
    Least-squares fit of a single line through all detected
    segments on one side. Returns (x_bottom, x_top) or None.
    """
    pts_x, pts_y = [], []
    for x1, y1, x2, y2 in lines:
        pts_x += [x1, x2]
        pts_y += [y1, y2]

    if len(pts_x) < 2:
        return None

    fit = np.polyfit(pts_y, pts_x, 1)  # x = f(y)
    poly = np.poly1d(fit)

    y_bottom = h
    y_top    = int(h * ROI_TL[1]) + 10   # just below ROI top

    x_bottom = int(poly(y_bottom))
    x_top    = int(poly(y_top))

    return (x_bottom, y_bottom, x_top, y_top)


def filter_lines(lines, h, w):
    left_lines, right_lines = [], []

    if lines is None:
        return left_lines, right_lines

    mid_x = w // 2

    for line in lines:
        x1, y1, x2, y2 = line[0]

        # Skip near-horizontal lines
        dx = x2 - x1
        dy = y2 - y1
        if dy == 0:
            continue

        angle = abs(np.degrees(np.arctan2(abs(dy), abs(dx))))
        if not (ANGLE_MIN <= angle <= ANGLE_MAX):
            continue

        slope = dy / (dx + 1e-6)

        # Left lane: negative slope, mostly left of center
        if slope < 0 and max(x1, x2) < mid_x + w * 0.15:
            left_lines.append((x1, y1, x2, y2))

        # Right lane: positive slope, mostly right of center
        elif slope > 0 and min(x1, x2) > mid_x - w * 0.15:
            right_lines.append((x1, y1, x2, y2))

    return left_lines, right_lines


def draw_lane_overlay(frame, left_line, right_line):
    """Draw a filled polygon between the two lane lines."""
    overlay = frame.copy()

    if left_line and right_line:
        lx_b, ly_b, lx_t, ly_t = left_line
        rx_b, ry_b, rx_t, ry_t = right_line

        poly_pts = np.array([
            [lx_b, ly_b],
            [lx_t, ly_t],
            [rx_t, ry_t],
            [rx_b, ry_b],
        ], dtype=np.int32)

        cv2.fillPoly(overlay, [poly_pts], (0, 180, 0))
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

    def draw_line(ln, color):
        if ln:
            x_b, y_b, x_t, y_t = ln
            cv2.line(frame, (x_b, y_b), (x_t, y_t), color, 6, cv2.LINE_AA)

    draw_line(left_line,  (0, 255,   0))
    draw_line(right_line, (0, 255,   0))

    return frame


# ── EMA state ──────────────────────────────────────────────
prev_left  = None
prev_right = None


def smooth(prev, cur):
    if prev is None or cur is None:
        return cur
    return tuple(int((1 - SMOOTH_ALPHA) * p + SMOOTH_ALPHA * c)
                 for p, c in zip(prev, cur))


# ── Main loop ──────────────────────────────────────────────
cap = cv2.VideoCapture(VIDEO)

cv2.namedWindow("Lane Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Lane Detection", 960, 540)

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # loop video
        continue

    h, w = frame.shape[:2]
    vertices = make_roi_vertices(h, w)

    # 1. Grayscale + blur
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (5, 5), 0)

    # 2. Canny edges
    edges = cv2.Canny(blur, CANNY_LOW, CANNY_HIGH)

    # 3. Mask to ROI
    roi   = region_of_interest(edges, vertices)

    # 4. Hough lines
    lines = cv2.HoughLinesP(
        roi,
        rho=1,
        theta=np.pi / 180,
        threshold=HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LEN,
        maxLineGap=HOUGH_MAX_GAP,
    )

    # 5. Filter & fit
    left_segs, right_segs = filter_lines(lines, h, w)
    left_fit  = fit_lane(left_segs,  h, "left")
    right_fit = fit_lane(right_segs, h, "right")

    # 6. EMA smooth to reduce jitter
    prev_left, prev_right
    left_fit  = smooth(prev_left,  left_fit)
    right_fit = smooth(prev_right, right_fit)
    prev_left  = left_fit
    prev_right = right_fit

    # 7. Draw
    result = draw_lane_overlay(frame.copy(), left_fit, right_fit)

    # Debug: show ROI boundary
    cv2.polylines(result, vertices, True, (255, 100, 0), 1)

    # Status text
    status = []
    if left_fit:  status.append("L:OK")
    if right_fit: status.append("R:OK")
    cv2.putText(result, "  ".join(status) or "No lane detected",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (0, 255, 0) if status else (0, 0, 255), 2)

    cv2.imshow("Lane Detection", result)

    key = cv2.waitKey(1)
    if key == 27:   # ESC
        break

    # Live-tune threshold with +/- keys
    if key == ord('+'):
        CANNY_LOW  = min(CANNY_LOW  + 5, 200)
        CANNY_HIGH = min(CANNY_HIGH + 5, 250)
        print(f"Canny: {CANNY_LOW}/{CANNY_HIGH}")
    if key == ord('-'):
        CANNY_LOW  = max(CANNY_LOW  - 5, 10)
        CANNY_HIGH = max(CANNY_HIGH - 5, 30)
        print(f"Canny: {CANNY_LOW}/{CANNY_HIGH}")

cap.release()
cv2.destroyAllWindows()