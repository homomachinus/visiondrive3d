from ultralytics import YOLO
import cv2
import numpy as np

# =========================
# LOAD MODEL
# =========================
model = YOLO(r"D:\mobil\runs\segment\my-seg2\weights\best.pt")

# =========================
# VIDEO INPUT
# =========================
video_path = "v3.mp4"
cap = cv2.VideoCapture(video_path)

# =========================
# THRESHOLD SETTINGS
# =========================
CONF_THRESHOLD = 0.10
IOU_THRESHOLD = 0.25
MASK_THRESHOLD = 0.35

# HSV tolerance
H_TOLERANCE = 70
S_TOLERANCE = 100
V_TOLERANCE = 60

# =========================
# CLASS COLORS
# =========================
class_colors = {
    0: (0, 255, 0),
    1: (255, 0, 0),
    2: (0, 0, 255),
    3: (255, 255, 0),
}

# =========================
# VIDEO WRITER
# =========================
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

out = cv2.VideoWriter(
    'output.mp4',
    fourcc,
    fps,
    (width, height)
)

# =========================
# MORPH KERNEL
# =========================
kernel = np.ones((7, 7), np.uint8)

# =========================
# LOOP
# =========================
while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    # =========================
    # YOLO INFERENCE
    # =========================
    results = model.track(
        frame,
        persist=True,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        verbose=False
    )

    result = results[0]

    # =========================
    # COPY FRAME
    # =========================
    overlay = frame.copy()

    if result.masks is not None:

        boxes = result.boxes
        masks = result.masks.data.cpu().numpy()

        # Convert ke HSV sekali saja
        hsv_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2HSV
        )

        for i, box in enumerate(boxes):

            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            color = class_colors.get(
                cls_id,
                (255, 255, 255)
            )

            # =========================
            # YOLO MASK
            # =========================
            yolo_mask = masks[i]

            yolo_mask = cv2.resize(
                yolo_mask,
                (frame.shape[1], frame.shape[0])
            )

            yolo_mask = (
                yolo_mask > MASK_THRESHOLD
            ).astype(np.uint8)

            # =========================
            # AMBIL WARNA RATA-RATA
            # DARI AREA YOLO
            # =========================
            hsv_pixels = hsv_frame[
                yolo_mask == 1
            ]

            # Skip jika kosong
            if len(hsv_pixels) == 0:
                continue

            avg_hsv = np.mean(
                hsv_pixels,
                axis=0
            )

            h, s, v = avg_hsv

            # =========================
            # HSV RANGE
            # =========================
            lower = np.array([
                max(h - H_TOLERANCE, 0),
                max(s - S_TOLERANCE, 0),
                max(v - V_TOLERANCE, 0)
            ], dtype=np.uint8)

            upper = np.array([
                min(h + H_TOLERANCE, 179),
                min(s + S_TOLERANCE, 255),
                min(v + V_TOLERANCE, 255)
            ], dtype=np.uint8)

            # =========================
            # HSV SEGMENTATION
            # =========================
            hsv_mask = cv2.inRange(
                hsv_frame,
                lower,
                upper
            )

            # =========================
            # LIMIT AREA KE BBOX
            # =========================
            bbox_mask = np.zeros_like(hsv_mask)

            bbox_mask[
                y1:y2,
                x1:x2
            ] = 255

            hsv_mask = cv2.bitwise_and(
                hsv_mask,
                bbox_mask
            )

            # =========================
            # GABUNG YOLO + HSV
            # =========================
            combined_mask = cv2.bitwise_or(
                yolo_mask * 255,
                hsv_mask
            )

            # =========================
            # MORPHOLOGY
            # =========================
            combined_mask = cv2.morphologyEx(
                combined_mask,
                cv2.MORPH_CLOSE,
                kernel
            )

            combined_mask = cv2.morphologyEx(
                combined_mask,
                cv2.MORPH_OPEN,
                kernel
            )

            combined_mask = cv2.GaussianBlur(
                combined_mask,
                (7, 7),
                0
            )

            combined_mask = (
                combined_mask > 127
            ).astype(np.uint8)

            # =========================
            # FILL SEGMENTATION
            # =========================
            colored_mask = np.zeros_like(frame)

            colored_mask[
                combined_mask == 1
            ] = color

            overlay = cv2.addWeighted(
                overlay,
                1.0,
                colored_mask,
                0.5,
                0
            )

            # =========================
            # DRAW BBOX
            # =========================
            cv2.rectangle(
                overlay,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            # =========================
            # LABEL
            # =========================
            label = f"{cls_name} {conf:.2f}"

            cv2.putText(
                overlay,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

    # =========================
    # SHOW
    # =========================
    cv2.imshow(
        "YOLO + HSV Hybrid Segmentation",
        overlay
    )

    # =========================
    # SAVE
    # =========================
    out.write(overlay)

    # =========================
    # EXIT
    # =========================
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# =========================
# RELEASE
# =========================
cap.release()
out.release()

cv2.destroyAllWindows()