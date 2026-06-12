from ultralytics import YOLO
import cv2
import pygame
import time
import sys
import numpy as np

try:
    import src as v
except ImportError:
    print("pastikan folder src/ ada dan lengkap.")
    sys.exit(1)


# warna per tipe kendaraan
CLASS_COLORS = {
    "car":        (0, 200, 210),
    "motorcycle": (210, 100, 0),
    "bus":        (100, 210, 50),
    "truck":      (200, 50, 100),
}

# ukuran nyata kendaraan (lebar, panjang, tinggi) dalam meter
CLASS_DIMS = {
    "motorcycle": (0.8, 2.0, 1.5),
    "bus":        (2.8, 10.0, 3.5),
    "truck":      (2.5, 8.0, 3.0),
}
DEFAULT_DIM = (2.0, 4.5, 1.5)  # default: mobil sedan


def get_color(cls_name):
    """ambil warna berdasarkan kelas kendaraan"""
    return CLASS_COLORS.get(cls_name, v.C_OTHER)


def get_dimensions(cls_name):
    """ambil dimensi nyata kendaraan berdasarkan kelasnya"""
    return CLASS_DIMS.get(cls_name, DEFAULT_DIM)


def filter_occlusion(vehicles):
    """buang kendaraan yg tertutup kendaraan lain > 50% area"""
    filtered = []
    for i, obj in enumerate(vehicles):
        is_occluded = False
        x1_1, y1_1, x2_1, y2_1 = obj['box']
        for j, other in enumerate(vehicles):
            if i == j:
                continue
            x1_2, y1_2, x2_2, y2_2 = other['box']
            ix1, iy1 = max(x1_1, x1_2), max(y1_1, y1_2)
            ix2, iy2 = min(x2_1, x2_2), min(y2_1, y2_2)
            if ix2 > ix1 and iy2 > iy1:
                inter_area = (ix2 - ix1) * (iy2 - iy1)
                area1 = max(1, (x2_1 - x1_1) * (y2_1 - y1_1))
                area2 = max(1, (x2_2 - x1_2) * (y2_2 - y1_2))
                # jika overlap > 50% dan objek lain lebih dekat → tertutup
                if inter_area / min(area1, area2) > 0.5 and other['depth'] < obj['depth']:
                    is_occluded = True
                    break
        if not is_occluded:
            filtered.append(obj)
    return filtered


def frame_to_pip(annotated_frame, vehicles, pip_w=320, pip_h=180):
    """konversi frame opencv ke surface pygame ukuran pip"""
    # tulis label jarak di frame
    for obj in vehicles:
        x1, y1, x2, y2 = obj['box']
        cv2.putText(
            annotated_frame,
            f"{obj['depth']:.1f}m",
            (x1 + 5, max(30, y2 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA
        )
    small = cv2.resize(annotated_frame, (pip_w, pip_h))
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    return pygame.surfarray.make_surface(np.swapaxes(rgb, 0, 1))


class Tracker:
    """kelola history posisi kendaraan antar-frame (ema smoothing)"""

    def __init__(self, alpha=0.15):
        self.history = {}
        self.alpha = alpha

    def cleanup(self, current_ids):
        """hapus id yg sudah tidak terdeteksi"""
        for old_id in list(self.history.keys()):
            if old_id not in current_ids:
                del self.history[old_id]

    def get_locked_class(self, obj_id):
        """kunci kelas kendaraan supaya tidak ganti-ganti"""
        val = self.history.get(obj_id)
        if val is None:
            return None
        return val['cls'] if isinstance(val, dict) else val[2]

    def smooth(self, obj_id, wx, depth):
        """haluskan posisi pakai ema"""
        if obj_id in self.history and not isinstance(self.history[obj_id], dict):
            prev_wx, prev_depth, _ = self.history[obj_id]
            wx    = self.alpha * wx    + (1 - self.alpha) * prev_wx
            depth = self.alpha * depth + (1 - self.alpha) * prev_depth
        return wx, depth

    def save(self, obj_id, wx, depth, cls_name):
        self.history[obj_id] = (wx, depth, cls_name)


class InferVisualizer:
    """loop utama: deteksi yolo + visualisasi 3d"""

    TARGET_CLASSES = ["car", "motorcycle"]

    def __init__(self, video_path):
        pygame.init()
        self.screen = pygame.display.set_mode((v.W, v.H))
        pygame.display.set_caption("fsd yolo visualizer")
        self.clock = pygame.time.Clock()

        self.model = YOLO("yolo11n.pt")
        self.cap   = cv2.VideoCapture(video_path)
        self.video_w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.video_h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        self.ego    = v.EgoVehicle()
        self.sim_t  = 0.0
        self.tick   = 0
        self.paused = False

        self.tracker = Tracker(alpha=0.15)
        self.speed_estimator = v.SpeedEstimator()

    def _estimate_depth(self, bbox_h, y2, wh, focal_px, horizon_y):
        """hitung estimasi jarak dari tinggi bbox & posisi y"""
        d_height = (wh * focal_px) / max(1, bbox_h)
        d_y      = ((self.video_h - horizon_y) * 4.0) / max(1, y2 - horizon_y)
        # bobot: 70% dari tinggi, 30% dari posisi y
        depth = d_height * 0.7 + d_y * 0.3
        return max(2.0, min(depth, 150.0))

    def _process_frame(self, frame, focal_px, horizon_y):
        """jalankan yolo dan kumpulkan data kendaraan"""
        results = self.model.track(frame, persist=True, verbose=False, conf=0.45)[0]

        # update tracker: buang id yg hilang
        current_ids = set()
        if results.boxes.id is not None:
            current_ids = set(results.boxes.id.cpu().numpy().astype(int))
        self.tracker.cleanup(current_ids)

        vehicles = []
        for box in results.boxes:
            cls_id   = int(box.cls[0])
            cls_name = self.model.names[cls_id]

            # kunci kelas supaya konsisten
            obj_id = int(box.id[0]) if box.id is not None else None
            if obj_id is not None:
                locked = self.tracker.get_locked_class(obj_id)
                if locked:
                    cls_name = locked

            if cls_name not in self.TARGET_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            ww, wl, wh = get_dimensions(cls_name)
            depth = self._estimate_depth(y2 - y1, y2, wh, focal_px, horizon_y)

            # posisi lateral (sumbu x)
            cx = (x1 + x2) / 2.0
            wx = self.ego.x + ((cx - self.video_w / 2.0) / focal_px) * depth

            # smoothing ema
            if obj_id is not None:
                wx, depth = self.tracker.smooth(obj_id, wx, depth)
                self.tracker.save(obj_id, wx, depth, cls_name)

            wy = self.ego.y + depth
            selected = abs(wx - self.ego.x) < (v.ROAD_W / 3) * 0.8 and depth < 35.0

            # hanya tampilkan yg di atas aspal
            if abs(wx - self.ego.x) <= v.ROAD_W / 2.0:
                vehicles.append({
                    'wx': wx, 'wy': wy, 'wz': 0.0,
                    'ww': ww, 'wl': wl, 'wh': wh,
                    'color': get_color(cls_name),
                    'selected': selected,
                    'depth': depth,
                    'box': (x1, y1, x2, y2),
                })

        return results, filter_occlusion(vehicles)

    def _update_speed(self, frame):
        """Run speed estimator on current frame and sync ego speed."""
        speed_ms = self.speed_estimator.update(frame)
        self.ego.speed = max(0.0, speed_ms)
        return speed_ms

    def run(self):
        last_t  = time.time()
        running = True

        # focal kamera dashcam & garis horizon
        focal_px  = self.video_h * 1.2
        horizon_y = self.video_h * 0.45

        detected_vehicles = []
        cam_surf = None

        while running and self.cap.isOpened():
            now   = time.time()
            dt    = min(now - last_t, 0.05)
            last_t = now

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        self.paused = not self.paused

            if not self.paused:
                ret, frame = self.cap.read()
                if not ret:
                    break

                self.sim_t += dt
                self.tick  += 1
                self.ego.y += self.ego.speed * dt

                results, detected_vehicles = self._process_frame(frame, focal_px, horizon_y)
                speed_ms = self._update_speed(frame)

                # buat pip (picture-in-picture) dari frame yolo
                cam_surf = frame_to_pip(results.plot(), detected_vehicles)

            # --- render scene ---
            self.screen.fill(v.C_BG)
            cam_x = self.ego.x
            cam_y = self.ego.y + v.CAM_Y_OFFSET

            speed_ms = getattr(self, '_last_speed_ms', 0.0)
            v.draw_road(self.screen, cam_x, cam_y,
                        speed_ms=self.ego.speed, elapsed=self.sim_t)

            for obj in detected_vehicles:
                v.draw_box(
                    self.screen, cam_x, cam_y,
                    obj['wx'], obj['wy'], obj['ww'], obj['wl'], obj['wh'],
                    obj['color'], selected=obj['selected'], is_ego=False, wz=obj['wz']
                )

            v.draw_box(
                self.screen, cam_x, cam_y,
                self.ego.x, self.ego.y, self.ego.w, self.ego.l, self.ego.h,
                v.C_EGO, is_ego=True
            )

            v.draw_hud(self.screen, self.ego, self.tick, self.sim_t)

            # tempel pip di pojok kanan bawah
            if cam_surf is not None:
                pip_w, pip_h, pad = 320, 180, 20
                pip_x = v.W - pip_w - pad
                pip_y = v.H - pip_h - pad
                pygame.draw.rect(self.screen, (200, 200, 200), (pip_x-2, pip_y-2, pip_w+4, pip_h+4), 2)
                self.screen.blit(cam_surf, (pip_x, pip_y))

            pygame.display.flip()
            self.clock.tick(v.FPS)

        self.cap.release()
        pygame.quit()


if __name__ == "__main__":
    app = InferVisualizer("vx1.mp4")
    app.run()