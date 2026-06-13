"""
FSD Autonomous Driving Visualizer & 3D Traffic Scene Reconstruction
Menggunakan YOLOv11 & Video Input (Implementasi Pinhole Depth & SORT 3D)
"""

import pygame
import math
import random
import time
import numpy as np
import cv2

from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG & VISUAL CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
W, H   = 1280, 760
FPS    = 60

C_BG         = (12,  14,  18)
C_TANAH      = (16,  18,  22)
C_ASPAL      = (22,  24,  32)
C_TROTOAR    = (32,  35,  45)
C_TROTOAR_ED = (44,  50,  65)
C_EGO        = (30,  140, 255)
C_OTHER      = (160, 170, 185)

ROAD_W     = 14.0
TROTOAR_W  =  3.5
LANE_W     = ROAD_W / 3

CAM_Y_OFFSET = -2.0
CAM_Z        =  9.0
FOCAL        = 600.0  # Konstanta kamera Z_c (sesuaikan dengan FOV video Anda)

# ══════════════════════════════════════════════════════════════════════════════
#  PAPER IMPLEMENTATION: PERCEPTION & MATH PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
class PaperMath:
    @staticmethod
    def estimate_depth(bbox_h, camera_constant=FOCAL, avg_car_height=1.6):
        """ Formula Kedalaman Pinhole: d = s * Z_c * (1/h) """
        if bbox_h <= 0: return 0.1
        return avg_car_height * camera_constant * (1.0 / bbox_h)

class VehicleTracker3D:
    """ SORT Tracking Framework untuk Kendaraan 3D (Constant Velocity Model) """
    _id_count = 0
    
    def __init__(self, start_state):
        self.kf = KalmanFilter(dim_x=11, dim_z=6)
        self.kf.F = np.eye(11)
        dt = 1.0 / 30.0 
        for i in range(5):
            self.kf.F[i, i+6] = dt 
            
        self.kf.H = np.zeros((6, 11))
        for i in range(6):
            self.kf.H[i, i] = 1.0
            
        self.kf.x[:6, 0] = start_state
        self.kf.P *= 10.0
        self.kf.R *= 10.0 # Measurement noise
        self.kf.Q *= 0.1  # Process noise
        
        self.time_since_update = 0
        VehicleTracker3D._id_count += 1
        self.id = VehicleTracker3D._id_count
        self.hits = 0

    def update(self, measurement):
        self.time_since_update = 0
        self.hits += 1
        self.kf.update(measurement)

    def predict(self):
        self.kf.predict()
        self.time_since_update += 1
        return self.kf.x

class SORTManager:
    def __init__(self, max_age=15, min_hits=3):
        self.trackers = []
        self.max_age = max_age
        self.min_hits = min_hits

    def update(self, measurements):
        # 1. Prediksi posisi baru untuk tracker yang ada
        for trk in self.trackers:
            trk.predict()
            
        if len(measurements) == 0:
            self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]
            return []

        # 2. Hitung cost matrix berdasarkan jarak Euclidean (X, Z)
        cost_matrix = np.zeros((len(self.trackers), len(measurements)))
        for t, trk in enumerate(self.trackers):
            for m, meas in enumerate(measurements):
                dist = np.linalg.norm(trk.kf.x[:2, 0] - meas[:2])
                cost_matrix[t, m] = dist

        # 3. Hungarian Algorithm untuk Assignment
        if len(self.trackers) > 0:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
        else:
            row_ind, col_ind = [], []

        assigned_tracks = []
        assigned_meas = []
        
        # 4. Update tracker yang mendapat assignment
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < 10.0: # Threshold jarak
                self.trackers[r].update(measurements[c])
                assigned_tracks.append(r)
                assigned_meas.append(c)

        # 5. Buat tracker baru untuk deteksi yang tidak ter-assign
        unassigned_meas = [m for m in range(len(measurements)) if m not in assigned_meas]
        for m in unassigned_meas:
            trk = VehicleTracker3D(measurements[m])
            self.trackers.append(trk)

        # 6. Hapus tracker lama
        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]
        
        # 7. Kembalikan tracker yang valid
        return [t for t in self.trackers if t.hits >= self.min_hits or t.time_since_update == 0]


# ══════════════════════════════════════════════════════════════════════════════
#  PROJECTION & PYGAME VISUALS
# ══════════════════════════════════════════════════════════════════════════════
def project(wx, wy, wz, cam_x, cam_y):
    dx = wx - cam_x
    dy = wy - cam_y
    dz = wz - CAM_Z
    if dy < 0.1:
        return None
    px = W / 2 + (dx / dy) * FOCAL
    py = H / 2 - (dz / dy) * FOCAL * 0.72
    return (int(px), int(py))

def draw_box(surf, cam_x, cam_y, wx, wy, ww, wl, wh, color, is_ego=False, wz=0.0):
    hw, hl = ww / 2, wl / 2
    corners = [project(wx+sx, wy+sy, wz+sz, cam_x, cam_y)
               for sx, sy, sz in [
                   (-hw,-hl,0),(hw,-hl,0),(hw,hl,0),(-hw,hl,0),
                   (-hw,-hl,wh),(hw,-hl,wh),(hw,hl,wh),(-hw,hl,wh),
               ]]
    if len([c for c in corners if c]) < 4: return
    r, g, b  = C_EGO if is_ego else color
    edge_c   = C_EGO if is_ego else (min(r+40, 255), min(g+40, 255), min(b+40, 255))
    lw       = 2 if is_ego else 1
    fs = pygame.Surface((W, H), pygame.SRCALPHA)
    
    def poly(idxs, c):
        pts = [corners[i] for i in idxs if corners[i]]
        if len(pts) >= 3:
            try: pygame.draw.polygon(fs, c, pts)
            except: pass
            
    poly([4,5,6,7], (r, g, b, 70))
    poly([3,0,4,7], (int(r*.55), int(g*.55), int(b*.55), 150))
    poly([1,2,6,5], (int(r*.55), int(g*.55), int(b*.55), 150))
    poly([0,1,5,4], (r, g, b, 130))
    poly([2,3,7,6], (r, g, b, 130))
    surf.blit(fs, (0, 0))
    
    for a, b in [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]:
        if corners[a] and corners[b]:
            pygame.draw.line(surf, edge_c, corners[a], corners[b], lw)

def draw_road(surf, cam_x, cam_y):
    y_near, y_far = cam_y + 0.5, cam_y + 150
    def road_pts(x_left, x_right, y_n, y_f):
        p = [project(x_left, y_n, 0, cam_x, cam_y), project(x_right, y_n, 0, cam_x, cam_y),
             project(x_right, y_f, 0, cam_x, cam_y), project(x_left,  y_f, 0, cam_x, cam_y)]
        return [pt for pt in p if pt]

    rw, tw = ROAD_W / 2, TROTOAR_W
    for sign in (-1, 1):
        pts = road_pts(sign*(rw+tw), sign*(rw+tw+300), y_near, y_far)
        if len(pts) == 4: pygame.draw.polygon(surf, C_TANAH, pts)

    for sign in (-1, 1):
        inner, outer = sign * rw, sign * (rw + tw)
        pts = road_pts(min(inner,outer), max(inner,outer), y_near, y_far)
        if len(pts) == 4:
            pygame.draw.polygon(surf, C_TROTOAR, pts)
            pygame.draw.polygon(surf, C_TROTOAR_ED, pts, 1)

    pts = road_pts(-rw, rw, y_near, y_far)
    if len(pts) == 4: pygame.draw.polygon(surf, C_ASPAL, pts)

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN PROCESSING LOOP
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("Memuat Model YOLOv11...")
    # Gunakan yolo11n.pt, ultralytics akan mendownload otomatis jika tidak ada
    model = YOLO('yolo11n.pt') 
    
    print("Membuka Video input.mp4...")
    cap = cv2.VideoCapture('input.mp4')
    if not cap.isOpened():
        print("Error: Tidak dapat membuka input.mp4. Pastikan file ada di direktori ini.")
        return

    # Inisialisasi Visualizer 3D (Pygame)
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("3D Traffic Scene Reconstruction - YOLOv11 Realtime")
    clock = pygame.time.Clock()

    tracker_manager = SORTManager()
    
    # Ego vehicle statis di koordinat (0,0) di dunia 3D
    cam_x, cam_y = 0.0, CAM_Y_OFFSET
    ego_w, ego_l, ego_h = 2.0, 4.6, 1.55

    vid_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    running = True
    while running and cap.isOpened():
        # Handle Pygame Events
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                running = False

        ret, frame = cap.read()
        if not ret:
            print("Video selesai.")
            break

        # 1. Deteksi Kendaraan menggunakan YOLOv11
        # Class 2: car, 5: bus, 7: truck (COCO dataset)
        results = model(frame, classes=[2, 5, 7], verbose=False)
        
        measurements = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cls = int(box.cls[0])
            
            bbox_w = x2 - x1
            bbox_h = y2 - y1
            center_x = x1 + bbox_w / 2
            
            # 2. Pinhole Depth Estimation
            z_depth = PaperMath.estimate_depth(bbox_h, FOCAL=400.0, avg_car_height=1.6)
            
            # 3. Konversi dari Piksel ke Koordinat Dunia 3D
            # (center_x - vid_width/2) merepresentasikan posisi lateral di gambar
            x_world = ((center_x - vid_width / 2) / FOCAL) * z_depth
            
            # Estimasi kasaran w, l, h untuk 3D Bounding Box (mocking GPLVM)
            if cls == 5 or cls == 7: # Bus/Truck
                v_w, v_l, v_h = 2.5, 8.0, 3.0
            else: # Car
                v_w, v_l, v_h = 1.9, 4.2, 1.4
                
            scale = v_w * v_h
            ratio = v_w / v_h
            yaw = 0.0 # Diasumsikan lurus (karena tidak ada 3D Deepbox)
            
            measurements.append(np.array([x_world, z_depth, 0.0, yaw, scale, ratio, v_w, v_l, v_h]))
            
            # Gambar 2D Bounding Box di OpenCV
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # 4. Update Tracker (SORT)
        measurements_state = [m[:6] for m in measurements]
        active_tracks = tracker_manager.update(measurements_state)

        # 5. Render Pygame (3D Reconstruction)
        screen.fill(C_BG)
        draw_road(screen, cam_x, cam_y)
        draw_box(screen, cam_x, cam_y, 0.0, 0.0, ego_w, ego_l, ego_h, C_EGO, is_ego=True)

        for trk in active_tracks:
            t_x, t_y = trk.kf.x[0, 0], trk.kf.x[1, 0]
            # Karena pengukuran dimensi tidak dimasukkan ke dalam state KF secara langsung,
            # kita ambil dimensi standar mobil untuk divisualisasikan
            draw_box(screen, cam_x, cam_y, t_x, t_y, 1.9, 4.2, 1.4, C_OTHER)

        pygame.display.flip()
        
        # 6. Tampilkan Video Asli OpenCV
        cv2.imshow("YOLOv11 2D Detection", cv2.resize(frame, (640, 360)))
        
        if cv2.waitKey(1) & 0xFF == 27: # Tekan ESC di window OpenCV untuk keluar
            break
            
        clock.tick(FPS)

    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()

if __name__ == "__main__":
    main()