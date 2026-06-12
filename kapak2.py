import os
import cv2
import numpy as np

# =========================================================
# CONFIG
# =========================================================

IMAGE_DIR = "img"
LABEL_DIR = "labels5"
OUTPUT_DIR = "output_overlay7"

# =========================================================
# CLASS NAMES
# =========================================================

CLASS_NAMES = {
    0: "sky",
    1: "building",
}

# =========================================================
# CLASS COLORS
# =========================================================

CLASS_COLORS = {
    0: (255, 0, 0),
    1: (0, 255, 0),
}

# =========================================================
# CREATE OUTPUT
# =========================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
# IMAGE FILES
# =========================================================

image_files = [
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

print(f"\nFound {len(image_files)} images\n")

# =========================================================
# PROCESS
# =========================================================

for image_name in image_files:

    # =====================================================
    # SUPPORT:
    # image.jpg -> image.jpg.txt
    # =====================================================

    label_name = image_name + ".txt"

    image_path = os.path.join(
        IMAGE_DIR,
        image_name
    )

    label_path = os.path.join(
        LABEL_DIR,
        label_name
    )

    # =====================================================
    # CHECK LABEL EXIST
    # =====================================================

    if not os.path.exists(label_path):

        print(f"[WARNING] Label not found: {label_name}")
        continue

    # =====================================================
    # LOAD IMAGE
    # =====================================================

    image = cv2.imread(image_path)

    if image is None:
        print(f"[ERROR] Cannot read image: {image_name}")
        continue

    h, w = image.shape[:2]

    # =====================================================
    # READ LABEL
    # =====================================================

    with open(label_path, "r") as f:
        lines = f.readlines()

    # =====================================================
    # DRAW OBJECTS
    # =====================================================

    for line in lines:

        line = line.strip()

        if line == "":
            continue

        parts = line.split()

        # minimal polygon
        if len(parts) < 7:
            continue

        # =================================================
        # CLASS
        # =================================================

        class_id = int(parts[0])

        coords = list(map(float, parts[1:]))

        # =================================================
        # YOLO -> PIXEL
        # =================================================

        points = []

        for i in range(0, len(coords), 2):

            x = int(coords[i] * w)
            y = int(coords[i + 1] * h)

            points.append([x, y])

        pts = np.array(points, np.int32)

        # =================================================
        # COLOR
        # =================================================

        color = CLASS_COLORS.get(
            class_id,
            (255, 255, 0)
        )

        class_name = CLASS_NAMES.get(
            class_id,
            f"class_{class_id}"
        )

        # =================================================
        # OVERLAY
        # =================================================

        overlay = image.copy()

        cv2.fillPoly(
            overlay,
            [pts],
            color
        )

        alpha = 0.35

        image = cv2.addWeighted(
            overlay,
            alpha,
            image,
            1 - alpha,
            0
        )

        # =================================================
        # POLYGON LINE
        # =================================================

        cv2.polylines(
            image,
            [pts],
            isClosed=True,
            color=color,
            thickness=2
        )

        # =================================================
        # LABEL TEXT
        # =================================================

        x, y = pts[0][0], pts[0][1]

        cv2.putText(
            image,
            class_name,
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

    # =====================================================
    # SAVE OUTPUT
    # =====================================================

    output_path = os.path.join(
        OUTPUT_DIR,
        image_name
    )

    cv2.imwrite(output_path, image)

    print(f"[OK] Saved: {output_path}")

print("\nDONE!")