from ultralytics import YOLO
import cv2
import pygame
import math
import time
import sys
import numpy as np

# Import dari visual.py yang baru
try:
    import visual as v
except ImportError:
    print("Pastikan file visual.py tersimpan dan berada di folder yang sama.")
    sys.exit(1)

class InferVisualizer:
    def __init__(self, video_path):
        pygame.init()
        self.screen = pygame.display.set_mode((v.W, v.H))
        pygame.display.set_caption("FSD YOLO Visualizer (Precise Depth)")
        self.clock = pygame.time.Clock()
        
        # Load YOLO model
        # Load YOLO model
        self.model = YOLO("yolo11n.pt")
        # Objek yang difilter
        self.target_classes = ["car", "motorcycle"]
        
        self.cap = cv2.VideoCapture(video_path)
        self.video_w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.video_h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        # Ego vehicle state
        self.ego = v.EgoVehicle()
        self.ego.speed = 10.0 # m/s (sekitar 36 km/h)
        self.sim_t = 0.0
        self.tick = 0
        self.paused = False
        
        # State untuk smoothing/tracking objek
        self.track_history = {}
        self.alpha = 0.15 # Semakin kecil semakin mulus, tapi sedikit ada delay

    def get_color(self, cls_name):
        if cls_name == "car": return (0, 200, 210)
        if cls_name == "motorcycle": return (210, 100, 0)
        if cls_name == "bus": return (100, 210, 50)
        if cls_name == "truck": return (200, 50, 100)
        return v.C_OTHER

    def get_dimensions(self, cls_name):
        # Mengembalikan ukuran asli dunia nyata: (lebar, panjang, tinggi) dalam meter
        if cls_name == "motorcycle": return 0.8, 2.0, 1.5
        if cls_name == "bus": return 2.8, 10.0, 3.5
        if cls_name == "truck": return 2.5, 8.0, 3.0
        return 2.0, 4.5, 1.5 # default car

    def run(self):
        last_t = time.time()
        running = True
        
        # Asumsi focal length kamera (biasanya ~1.0 sampai 1.5 x tinggi frame untuk dashcam)
        focal_px = self.video_h * 1.2 
        horizon_y = self.video_h * 0.45

        detected_vehicles = []
        cam_surf = None

        while running and self.cap.isOpened():
            now = time.time()
            dt = min(now - last_t, 0.05)
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
                self.tick += 1
                self.ego.y += self.ego.speed * dt
                
                # Inferensi YOLO menggunakan Tracker (persist=True) dengan batas threshold 0.45
                results = self.model.track(frame, persist=True, verbose=False, conf=0.45)[0]
                
                detected_vehicles = []
                
                # Cleanup memory untuk ID yang sudah tidak terdeteksi
                current_ids = set()
                if results.boxes.id is not None:
                    current_ids = set(results.boxes.id.cpu().numpy().astype(int))
                for old_id in list(self.track_history.keys()):
                    if old_id not in current_ids:
                        del self.track_history[old_id]

                for box in results.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = self.model.names[cls_id]
                    
                    # KUNCI CLASS: Jika objek ini sudah pernah terlacak, paksakan class lamanya
                    # agar tidak berubah-ubah (misal dari car jadi truck)
                    obj_id = None
                    if box.id is not None:
                        obj_id = int(box.id[0])
                        if obj_id in self.track_history:
                            # Jika struktur sebelumnya dictionary, abaikan (reset via del nanti atau pakai fallback)
                            val = self.track_history[obj_id]
                            if isinstance(val, dict):
                                cls_name = val['cls']
                            else:
                                cls_name = val[2]
                    
                    if cls_name in self.target_classes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        
                        bbox_w = max(1, x2 - x1)
                        bbox_h = max(1, y2 - y1)
                        
                        ww, wl, wh = self.get_dimensions(cls_name)
                        
                        # ==========================================
                        # PRESISI DEPTH ESTIMATION & POSITIONING
                        # ==========================================
                        # Menggunakan tinggi bounding box dibandingkan dengan tinggi asli objek
                        # Rumus perspektif dasar: Depth = (Real_Height * Focal_Length) / BoundingBox_Height
                        depth_from_height = (wh * focal_px) / bbox_h
                        
                        # Alternatif menggunakan bottom point (y2) jika objek terpotong
                        y_diff = max(1, y2 - horizon_y)
                        depth_from_y = ((self.video_h - horizon_y) * 4.0) / y_diff
                        
                        wz = 0.0 # elevasi dasar dari aspal
                        
                        # Kombinasi bobot untuk kendaraan di aspal
                        final_depth = (depth_from_height * 0.7) + (depth_from_y * 0.3)
                        final_depth = max(2.0, min(final_depth, 150.0))
                        
                        # Menghitung posisi lateral (sumbu X)
                        cx = (x1 + x2) / 2.0
                        offset_x = cx - self.video_w / 2.0
                        wx = self.ego.x + (offset_x / focal_px) * final_depth
                        
                        # SMOOTHING (Menghaluskan gerakan antar-frame menggunakan EMA)
                        if box.id is not None:
                            obj_id = int(box.id[0])
                            
                            if obj_id in self.track_history and not isinstance(self.track_history[obj_id], dict):
                                prev_wx, prev_depth, _ = self.track_history[obj_id]
                                wx = (self.alpha * wx) + ((1 - self.alpha) * prev_wx)
                                final_depth = (self.alpha * final_depth) + ((1 - self.alpha) * prev_depth)
                            
                            # Simpan state terbaru beserta nama class-nya untuk di-lock
                            self.track_history[obj_id] = (wx, final_depth, cls_name)

                        # Posisi Y (panjang/maju) setelah di-smooth
                        wy = self.ego.y + final_depth
                        
                        col = self.get_color(cls_name)
                        
                        # Objek ditandai (selected) jika cukup dekat dan berada di lajur kita
                        selected = (abs(wx - self.ego.x) < (v.ROAD_W/3)*0.8 and final_depth < 35.0)
                        
                        # Filter: Hanya tampilkan kendaraan yang ada di atas aspal jalan
                        # Lebar aspal jalan (v.ROAD_W), jadi batasnya adalah v.ROAD_W / 2 ke kiri dan kanan.
                        if abs(wx - self.ego.x) <= v.ROAD_W / 2.0:
                            detected_vehicles.append({
                                'wx': wx, 'wy': wy, 'wz': wz, 'ww': ww, 'wl': wl, 'wh': wh,
                                'color': col, 'selected': selected,
                                'depth': final_depth,
                                'box': (int(x1), int(y1), int(x2), int(y2))
                            })

                # ==========================================
                # FILTER OCCLUSION (OVERLAP)
                # Jika dua box saling menimpa > 50%, ambil yang jaraknya lebih dekat
                # ==========================================
                filtered_vehicles = []
                for i, obj in enumerate(detected_vehicles):
                    is_occluded = False
                    for j, other in enumerate(detected_vehicles):
                        if i == j: continue
                        
                        x1_1, y1_1, x2_1, y2_1 = obj['box']
                        x1_2, y1_2, x2_2, y2_2 = other['box']
                        
                        ix1, iy1 = max(x1_1, x1_2), max(y1_1, y1_2)
                        ix2, iy2 = min(x2_1, x2_2), min(y2_1, y2_2)
                        
                        if ix2 > ix1 and iy2 > iy1:
                            inter_area = (ix2 - ix1) * (iy2 - iy1)
                            area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
                            area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
                            
                            # Intersection over Minimum Area
                            overlap_ratio = inter_area / min(area1, area2)
                            
                            if overlap_ratio > 0.5:
                                # Jika objek 'other' lebih dekat, maka 'obj' ini tertutup (occluded)
                                if other['depth'] < obj['depth']:
                                    is_occluded = True
                                    break
                                    
                    if not is_occluded:
                        filtered_vehicles.append(obj)
                        
                detected_vehicles = filtered_vehicles

                # ==========================================
                # VIDEO ASLI OVERLAY (Picture-in-Picture)
                # ==========================================
                # Menggambar bounding box deteksi YOLO langsung ke frame
                annotated_frame = results.plot()
                
                # Menambahkan label Depth Estimation (jarak) di atas frame
                for obj in detected_vehicles:
                    x1, y1, x2, y2 = obj['box']
                    depth = obj['depth']
                    # Tuliskan jarak (meter) di bagian bawah dalam bounding box
                    text = f"{depth:.1f}m"
                    cv2.putText(annotated_frame, text, (x1 + 5, max(30, y2 - 10)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
                
                # Resize video agar tidak memenuhi layar (misal 320x180 atau 1/4 ukuran)
                pip_w, pip_h = 320, 180
                small_frame = cv2.resize(annotated_frame, (pip_w, pip_h))
                
                # Konversi format OpenCV BGR ke RGB Pygame
                small_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                
                # Rotasi matrix agar sesuai dengan sumbu Pygame (lebar, tinggi)
                surf_rgb = np.swapaxes(small_rgb, 0, 1)
                cam_surf = pygame.surfarray.make_surface(surf_rgb)
                
            # ==========================================
            # RENDER MENGGUNAKAN MATERIAL visual.py (Selalu Render, Walaupun Pause)
            # ==========================================
            self.screen.fill(v.C_BG)
            
            cam_y = self.ego.y + v.CAM_Y_OFFSET
            cam_x = self.ego.x
            
            # Gambar aspal & trotoar
            if hasattr(v, 'draw_road'):
                v.draw_road(self.screen, cam_x, cam_y, self.tick)
            
            # Gambar setiap kendaraan hasil YOLO
            for obj in detected_vehicles:
                v.draw_box(self.screen, cam_x, cam_y, 
                           obj['wx'], obj['wy'], obj['ww'], obj['wl'], obj['wh'], 
                           obj['color'], selected=obj['selected'], is_ego=False, wz=obj['wz'])
            
            # Gambar Ego Vehicle
            v.draw_box(self.screen, cam_x, cam_y,
                       self.ego.x, self.ego.y, self.ego.w, self.ego.l, self.ego.h,
                       v.C_EGO, is_ego=True)
            
            # Gambar HUD / status bar (opsional jika tersedia)
            if hasattr(v, 'draw_hud'):
                v.draw_hud(self.screen, self.ego, self.tick, self.sim_t)
            
            # Tempelkan jendela PiP video asli jika sudah ada
            if cam_surf is not None:
                pip_w, pip_h = 320, 180
                pad = 20
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