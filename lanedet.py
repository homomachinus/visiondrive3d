import cv2
import numpy as np

VIDEO = "input.mp4"

cap = cv2.VideoCapture(VIDEO)

cv2.namedWindow(
    "White Detection",
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    "White Detection",
    960,
    540
)

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

    top = gray[:split_y, :]
    bottom = gray[split_y:, :]

    # ====================================
    # WHITE PIXEL DETECTION
    # ====================================

    _, white_mask = cv2.threshold(
        bottom,
        180,     # nanti kita tuning
        255,
        cv2.THRESH_BINARY
    )

    # hilangkan noise kecil
    kernel = np.ones((3,3), np.uint8)

    white_mask = cv2.morphologyEx(
        white_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    # ====================================
    # OVERLAY
    # ====================================

    bottom_bgr = cv2.cvtColor(
        bottom,
        cv2.COLOR_GRAY2BGR
    )

    bottom_bgr[white_mask > 0] = (0,255,0)

    top_bgr = cv2.cvtColor(
        top,
        cv2.COLOR_GRAY2BGR
    )

    result = np.vstack([
        top_bgr,
        bottom_bgr
    ])

    cv2.imshow(
        "White Detection",
        result
    )

    key = cv2.waitKey(1)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()