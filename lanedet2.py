import cv2
import numpy as np

VIDEO = "input.mp4"

# =====================================
# TUNING
# =====================================

THRESH_WHITE = 180

CANNY_LOW = 50
CANNY_HIGH = 150

HOUGH_THRESHOLD = 25
MIN_LINE_LENGTH = 40
MAX_LINE_GAP = 25

MIN_SLOPE = 0.25
MIN_LENGTH = 50

BOTTOM_TOUCH_RATIO = 0.65

X_CLUSTER_THRESHOLD = 60

# ROI jalan
ROI_BL = (0.00, 1.00)   # full kiri bawah
ROI_BR = (1.00, 1.00)   # full kanan bawah
ROI_TR = (0.85, 0.50)   # sedikit lebih lebar kanan atas
ROI_TL = (0.15, 0.50)   # sedikit lebih lebar kiri atas


# tracking
TRACK_X_TOLERANCE = 120
TRACK_ANGLE_TOLERANCE = 20

EMA_ALPHA = 0.80

cap = cv2.VideoCapture(VIDEO)

if not cap.isOpened():
    raise RuntimeError(f"Cannot open {VIDEO}")

cv2.namedWindow(
    "Lane Detection",
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    "Lane Detection",
    960,
    540
)

prev_tracks = []


def line_angle(line):
    x1, y1, x2, y2 = line
    return np.degrees(
        np.arctan2(
            y2 - y1,
            x2 - x1
        )
    )


def smooth_line(old_line, new_line):

    old_line = np.array(old_line, dtype=np.float32)
    new_line = np.array(new_line, dtype=np.float32)

    out = (
        EMA_ALPHA * old_line +
        (1.0 - EMA_ALPHA) * new_line
    )

    return out.astype(np.int32)


while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    h, w = gray.shape

    split_y = h // 2

    top = gray[:split_y]
    bottom = gray[split_y:]

    bottom_h, bottom_w = bottom.shape

    # =====================================
    # ROI TRAPESIUM
    # =====================================

    roi_mask = np.zeros_like(bottom)

    polygon = np.array([[
        (int(ROI_TL[0] * bottom_w), int(ROI_TL[1] * bottom_h)),  # top-left
        (int(ROI_TR[0] * bottom_w), int(ROI_TR[1] * bottom_h)),  # top-right
        (int(ROI_BR[0] * bottom_w), int(ROI_BR[1] * bottom_h)),  # bottom-right
        (int(ROI_BL[0] * bottom_w), int(ROI_BL[1] * bottom_h)),  # bottom-left
    ]], dtype=np.int32)



    cv2.fillPoly(
        roi_mask,
        polygon,
        255
    )

    # =====================================
    # WHITE THRESHOLD
    # =====================================

    _, white_mask = cv2.threshold(
        bottom,
        THRESH_WHITE,
        255,
        cv2.THRESH_BINARY
    )

    white_mask = cv2.bitwise_and(
        white_mask,
        roi_mask
    )

    kernel = np.ones((3, 3), np.uint8)

    white_mask = cv2.morphologyEx(
        white_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    white_mask = cv2.morphologyEx(
        white_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # =====================================
    # EDGE
    # =====================================

    edges = cv2.Canny(
        white_mask,
        CANNY_LOW,
        CANNY_HIGH
    )

    # =====================================
    # HOUGH
    # =====================================

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=HOUGH_THRESHOLD,
        minLineLength=MIN_LINE_LENGTH,
        maxLineGap=MAX_LINE_GAP
    )

    candidates = []

    if lines is not None:

        for l in lines:

            x1, y1, x2, y2 = l[0]

            dx = x2 - x1
            dy = y2 - y1

            length = np.sqrt(
                dx * dx +
                dy * dy
            )

            if length < MIN_LENGTH:
                continue

            if dx == 0:
                slope = 999
            else:
                slope = dy / dx

            if abs(slope) < MIN_SLOPE:
                continue

            bottom_touch = max(y1, y2)

            if bottom_touch < bottom_h * BOTTOM_TOUCH_RATIO:
                continue

            center_x = (x1 + x2) / 2.0
            angle = line_angle(
                [x1, y1, x2, y2]
            )

            score = length

            # =====================================
            # TRACKING BONUS
            # =====================================

            for prev in prev_tracks:

                px = prev["center_x"]
                pa = prev["angle"]

                if abs(center_x - px) < TRACK_X_TOLERANCE:
                    score += 300

                if abs(angle - pa) < TRACK_ANGLE_TOLERANCE:
                    score += 200

            candidates.append(
                (
                    score,
                    [x1, y1, x2, y2]
                )
            )

    # =====================================
    # CLUSTER BERDASARKAN X
    # =====================================

    clusters = []

    for score, line in candidates:

        x1, y1, x2, y2 = line

        center_x = (x1 + x2) / 2

        assigned = False

        for cluster in clusters:

            if abs(
                cluster["center_x"] -
                center_x
            ) < X_CLUSTER_THRESHOLD:

                cluster["lines"].append(line)

                assigned = True
                break

        if not assigned:

            clusters.append({
                "center_x": center_x,
                "lines": [line]
            })

    # =====================================
    # MERGE
    # =====================================

    merged_lines = []

    for cluster in clusters:

        lines_cluster = cluster["lines"]

        xs1 = []
        ys1 = []
        xs2 = []
        ys2 = []

        for line in lines_cluster:

            x1, y1, x2, y2 = line

            if y1 > y2:
                x1, x2 = x2, x1
                y1, y2 = y2, y1

            xs1.append(x1)
            ys1.append(y1)

            xs2.append(x2)
            ys2.append(y2)

        merged_lines.append([
            int(np.mean(xs1)),
            int(np.mean(ys1)),
            int(np.mean(xs2)),
            int(np.mean(ys2))
        ])

    # =====================================
    # TRACK UPDATE
    # =====================================

    new_tracks = []

    for line in merged_lines:

        x1, y1, x2, y2 = line

        center_x = (
            x1 + x2
        ) / 2

        angle = line_angle(line)

        new_tracks.append({
            "center_x": center_x,
            "angle": angle
        })

    prev_tracks = new_tracks

    # =====================================
    # DRAW
    # =====================================

    bottom_bgr = cv2.cvtColor(
        bottom,
        cv2.COLOR_GRAY2BGR
    )

    cv2.polylines(
        bottom_bgr,
        polygon,
        True,
        (255, 0, 0),
        2
    )

    for line in merged_lines:

        x1, y1, x2, y2 = line

        cv2.line(
            bottom_bgr,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            4
        )

    top_bgr = cv2.cvtColor(
        top,
        cv2.COLOR_GRAY2BGR
    )

    result = np.vstack([
        top_bgr,
        bottom_bgr
    ])

    cv2.putText(
        result,
        f"Lines: {len(merged_lines)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow(
        "Lane Detection",
        result
    )

    key = cv2.waitKey(1)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()