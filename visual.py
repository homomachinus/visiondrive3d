"""
FSD Tesla-style Visualizer
- Jalan lurus, POV dari belakang-atas (seperti Tesla FSD v14)
- Bounding box kendaraan memanjang ke depan (sumbu Y = maju)
- Tidak ada belokan, perempatan, atau pejalan kaki
- Elemen: aspal, trotoar, mobil kotak, HUD
"""

import pygame
import math
import random
import time

# ── DIMENSI ──────────────────────────────────────────────────────────────────
W, H   = 1280, 760
FPS    = 60

# ── WARNA ────────────────────────────────────────────────────────────────────
C_BG         = (10, 11, 16)
C_TANAH      = (16, 17, 22)          # lahan kosong di luar jalan
C_ASPAL      = (22, 24, 32)
C_ASPAL_LINE = (38, 42, 55)
C_TROTOAR    = (32, 35, 45)
C_TROTOAR_ED = (44, 50, 65)
C_LANE       = (55, 60, 45)          # marka kuning redup
C_EGO        = (30, 140, 255)        # biru (ego = Tesla biru)
C_EGO_GLOW   = (0,  80, 200)
C_OTHER      = (160, 170, 185)       # kendaraan lain abu terang
C_OTHER_DIM  = (90, 100, 115)
C_PATH       = (30, 200, 100)        # jalur hijau (opsional, tipis)
C_HUD_BG     = (8, 10, 16)
C_TEXT       = (220, 225, 235)
C_DIM        = (100, 110, 130)
C_GREEN      = (60, 220, 90)
C_BLUE_LIGHT = (80, 160, 255)
C_RED_LIGHT  = (220, 60, 60)
C_YELLOW     = (220, 180, 50)

# ── PROYEKSI PSEUDO-3D ────────────────────────────────────────────────────────
# Dunia: X = kanan, Y = maju (jauh), Z = atas
# POV: kamera di belakang-atas ego, sedikit miring ke bawah

CAM_Y_OFFSET  = -14.0   # kamera mundur dari ego
CAM_Z         =  9.0    # ketinggian kamera
FOCAL         = 420.0   # focal length proyeksi

def project(wx, wy, wz, cam_x, cam_y):
    """World → screen pixel."""
    dx = wx - cam_x
    dy = wy - cam_y          # dy positif = lebih jauh
    dz = wz - CAM_Z

    if dy < 0.5:             # di belakang kamera
        return None

    # Perspektif sederhana
    px = W / 2 + (dx / dy) * FOCAL
    py = H / 2 - (dz / dy) * FOCAL * 0.72   # sedikit squish vertikal
    return (int(px), int(py))

def proj_scale(wy, cam_y):
    """Skala objek berdasarkan jarak."""
    dy = wy - cam_y
    if dy < 0.5:
        return 0.0
    return FOCAL / dy

# ── FONT ─────────────────────────────────────────────────────────────────────
pygame.init()
try:
    fn = pygame.font.match_font('dejavusansmono,consolas,couriernew,monospace')
    F_L  = pygame.font.Font(fn, 22)
    F_M  = pygame.font.Font(fn, 15)
    F_S  = pygame.font.Font(fn, 12)
    F_XS = pygame.font.Font(fn, 10)
except:
    F_L  = pygame.font.SysFont('monospace', 22)
    F_M  = pygame.font.SysFont('monospace', 15)
    F_S  = pygame.font.SysFont('monospace', 12)
    F_XS = pygame.font.SysFont('monospace', 10)

# ── BOX 3D ───────────────────────────────────────────────────────────────────
def draw_box(surf, cam_x, cam_y,
             wx, wy, ww, wl, wh,
             color, selected=False, is_ego=False, wz=0.0):
    """
    Gambar kotak 3D.
    wx,wy = center bawah kotak di dunia
    ww = lebar (sumbu X), wl = panjang (sumbu Y/maju), wh = tinggi
    """
    hw = ww / 2
    hl = wl / 2

    # 8 sudut
    corners = []
    for (sx, sy, sz) in [
        (-hw, -hl, wz), ( hw, -hl, wz), ( hw,  hl, wz), (-hw,  hl, wz),   # bawah
        (-hw, -hl, wz+wh),( hw, -hl, wz+wh),( hw,  hl, wz+wh),(-hw,  hl, wz+wh),  # atas
    ]:
        p = project(wx + sx, wy + sy, sz, cam_x, cam_y)
        corners.append(p)

    # cek minimal 4 titik valid
    valid = [c for c in corners if c is not None]
    if len(valid) < 4:
        return

    # Warna face
    if is_ego:
        top_c   = (*C_EGO,       80)
        side_c  = (*C_EGO_GLOW,  60)
        front_c = (*C_EGO,       90)
        edge_c  = C_EGO
        glow_c  = C_EGO
    else:
        dim = 1.0
        top_c   = (*color,       50)
        side_c  = (int(color[0]*0.6), int(color[1]*0.6), int(color[2]*0.6), 45)
        front_c = (*color,       65)
        edge_c  = color if selected else C_OTHER_DIM
        glow_c  = color

    fs = pygame.Surface((W, H), pygame.SRCALPHA)

    def face(idxs, c):
        pts = [corners[i] for i in idxs if corners[i]]
        if len(pts) >= 3:
            try:
                pygame.draw.polygon(fs, c, pts)
            except:
                pass

    # Top, front (belakang = dekat kamera), right, left
    face([4,5,6,7], top_c)
    face([0,1,5,4], front_c)   # sisi dekat kamera
    face([1,2,6,5], side_c)
    face([3,0,4,7], side_c)

    surf.blit(fs, (0, 0))

    # Edge lines
    lw = 2 if (selected or is_ego) else 1
    pairs = [
        (0,1),(1,2),(2,3),(3,0),   # bawah
        (4,5),(5,6),(6,7),(7,4),   # atas
        (0,4),(1,5),(2,6),(3,7),   # vertikal
    ]
    for a, b in pairs:
        if corners[a] and corners[b]:
            pygame.draw.line(surf, edge_c, corners[a], corners[b], lw)

    # Glow ring dasar untuk ego
    if is_ego and corners[0] and corners[1] and corners[2] and corners[3]:
        gsurf = pygame.Surface((W, H), pygame.SRCALPHA)
        pts_base = [corners[i] for i in [0,1,2,3] if corners[i]]
        if len(pts_base) == 4:
            pygame.draw.polygon(gsurf, (*C_EGO, 30), pts_base)
            pygame.draw.polygon(gsurf, (*C_EGO, 120), pts_base, 2)
        surf.blit(gsurf, (0,0))

    # Label jarak (bukan ego) - Dihilangkan sesuai permintaan
    # if not is_ego and corners[6]:
    #     cx_s, cy_s = corners[6]
    #     if 50 < cx_s < W-50 and 50 < cy_s < H-100:
    #         dy = wy - cam_y
    #         dist_m = dy
    #         lbl = F_XS.render(f"{dist_m:.0f}m", True, C_DIM)
    #         bg = pygame.Surface((lbl.get_width()+6, lbl.get_height()+4), pygame.SRCALPHA)
    #         pygame.draw.rect(bg, (8,10,16,160), bg.get_rect(), border_radius=3)
    #         surf.blit(bg, (cx_s - lbl.get_width()//2 - 3, cy_s - lbl.get_height() - 8))
    #         surf.blit(lbl, (cx_s - lbl.get_width()//2, cy_s - lbl.get_height() - 6))


# ── JALAN ─────────────────────────────────────────────────────────────────────
ROAD_W     = 14.0   # total lebar jalan (3 lane kira-kira)
TROTOAR_W  =  3.5
LANE_W     = ROAD_W / 3

def draw_road(surf, cam_x, cam_y, tick):
    """Gambar aspal lurus + trotoar kiri-kanan + marka putus-putus."""
    # Jarak render
    y_near = cam_y + 1
    y_far  = cam_y + 120

    # Titik 4 pojok aspal
    def road_pts(x_left, x_right, y_n, y_f):
        p = [
            project(x_left,  y_n, 0, cam_x, cam_y),
            project(x_right, y_n, 0, cam_x, cam_y),
            project(x_right, y_f, 0, cam_x, cam_y),
            project(x_left,  y_f, 0, cam_x, cam_y),
        ]
        return [pt for pt in p if pt]

    rw = ROAD_W / 2
    tw = TROTOAR_W

    # ── Lahan kosong (Tanah) Kiri & Kanan ──
    # Mengisi bagian luar dari trotoar hingga kejauhan (tanah luas)
    for sign in [-1, 1]:
        inner_t = sign * (rw + tw)
        outer_t = sign * (rw + tw + 200.0)  # Tanah ditarik sejauh 200 meter ke luar
        pts = road_pts(min(inner_t, outer_t), max(inner_t, outer_t), y_near, y_far)
        if len(pts) == 4:
            pygame.draw.polygon(surf, C_TANAH, pts)

    # ── Trotoar kiri & kanan ──
    for sign in [-1, 1]:
        inner = sign * rw
        outer = sign * (rw + tw)
        pts = road_pts(min(inner, outer), max(inner, outer), y_near, y_far)
        if len(pts) == 4:
            pygame.draw.polygon(surf, C_TROTOAR, pts)
            pygame.draw.polygon(surf, C_TROTOAR_ED, pts, 1)

        # Tepi trotoar (kerb line)
        pa = project(inner, y_near, 0.18, cam_x, cam_y)
        pb = project(inner, y_far,  0.18, cam_x, cam_y)
        if pa and pb:
            pygame.draw.line(surf, C_TROTOAR_ED, pa, pb, 2)

    # ── Aspal utama ──
    pts = road_pts(-rw, rw, y_near, y_far)
    if len(pts) == 4:
        pygame.draw.polygon(surf, C_ASPAL, pts)




# ── KENDARAAN LAIN ────────────────────────────────────────────────────────────
class Vehicle:
    # Dimensi kotak: W=lebar, L=panjang ke depan, H=tinggi
    TYPES = [
        {'w': 2.0, 'l': 4.5, 'h': 1.5},  # sedan
        {'w': 2.2, 'l': 4.8, 'h': 1.7},  # SUV
        {'w': 2.4, 'l': 5.2, 'h': 2.0},  # MPV
        {'w': 2.5, 'l': 8.0, 'h': 2.5},  # van / truck kecil
    ]
    LANES = [-LANE_W, 0.0, LANE_W]        # center X tiap lane

    def __init__(self, y, lane_idx, speed, rng):
        t = rng.choice(self.TYPES)
        self.w = t['w']
        self.l = t['l']
        self.h = t['h']
        self.x = self.LANES[lane_idx] + rng.uniform(-0.3, 0.3)
        self.y = float(y)
        self.speed = speed
        self.lane  = lane_idx
        self.color = C_OTHER

    def update(self, dt):
        self.y += self.speed * dt

    @property
    def selected(self):
        return abs(self.x) < LANE_W * 0.7 and self.y > 0  # di lane tengah depan


class EgoVehicle:
    def __init__(self):
        self.x     = 0.0
        self.y     = 0.0
        self.speed = 0.0
        self.accel = 0.0
        self.steer = 0.0
        self.w, self.l, self.h = 2.0, 4.6, 1.6


# ── HUD ───────────────────────────────────────────────────────────────────────
def draw_hud(surf, ego, tick, sim_t):
    pass


# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("FSD Autonomous Driving Visualizer")
    clock  = pygame.time.Clock()

    ego = EgoVehicle()
    ego.speed = 8.0   # m/s awal

    rng = random.Random(7)

    # Spawn kendaraan di depan ego
    vehicles = []
    spawn_positions = []
    for lane in range(3):
        for j in range(5):
            y = 10 + j * 18 + rng.uniform(-3, 3)
            spd = rng.uniform(4.5, 9.0)
            vehicles.append(Vehicle(y, lane, spd, rng))

    sim_t    = 0.0
    tick     = 0
    paused   = False
    last_t   = time.time()

    # Kamera mengikuti ego (ego selalu di Y=0)
    # Dunia bergerak, ego diam di Y=0
    world_offset = 0.0  # seberapa jauh dunia sudah "digulir"

    while True:
        now = time.time()
        dt  = min(now - last_t, 0.05)
        last_t = now

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); return
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    pygame.quit(); return
                elif ev.key == pygame.K_SPACE:
                    paused = not paused
                elif ev.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    ego.speed = min(ego.speed + 1, 20)
                elif ev.key == pygame.K_MINUS:
                    ego.speed = max(ego.speed - 1, 1)

        if not paused:
            tick   += 1
            sim_t  += dt

            # Variasi kecepatan ringan
            target_spd = 8.0 + 2.5 * math.sin(sim_t * 0.18)
            ego.accel  = (target_spd - ego.speed) * 1.8
            ego.accel  = max(-3.5, min(2.5, ego.accel))
            ego.speed  = max(1.0, min(14.0, ego.speed + ego.accel * dt))
            ego.steer  = math.sin(sim_t * 0.08) * 1.2   # sedikit goyang

            # Dunia bergerak mundur relatif ke ego
            world_offset += ego.speed * dt

            # Update Y semua kendaraan (mereka juga maju, lebih lambat)
            for v in vehicles:
                # gerak relatif: kendaraan maju dg speed sendiri
                # ego maju dg ego.speed → relative = v.speed - ego.speed
                rel_spd = v.speed - ego.speed
                v.y -= rel_spd * dt   # jika lambat → mendekat ke ego

            # Respawn kendaraan yang sudah lewat ego (y < -15) atau terlalu jauh (y > 130)
            for v in vehicles:
                if v.y < -15:
                    v.y = 100 + rng.uniform(0, 30)
                    v.speed = rng.uniform(4.5, 9.5)
                    v.x = Vehicle.LANES[v.lane] + rng.uniform(-0.3, 0.3)
                elif v.y > 130:
                    v.y = 10 + rng.uniform(0, 10)
                    v.speed = rng.uniform(4.5, 9.5)

        # ── RENDER ──────────────────────────────────────────────────────────
        screen.fill(C_BG)

        cam_x = ego.x
        cam_y = ego.y + CAM_Y_OFFSET   # kamera di belakang ego

        # Jalan
        draw_road(screen, cam_x, cam_y, tick)

        # Sort kendaraan jauh → dekat agar overlap benar
        sorted_v = sorted(vehicles, key=lambda v: -v.y)

        for v in sorted_v:
            selected = (abs(v.x - ego.x) < LANE_W * 1.2 and 2 < v.y < 60)
            col = C_OTHER if not selected else (200, 210, 230)
            draw_box(screen, cam_x, cam_y,
                     v.x, v.y, v.w, v.l, v.h,
                     col, selected=selected)

        # Ego vehicle (y = 0 selalu)
        draw_box(screen, cam_x, cam_y,
                 ego.x, ego.y, ego.w, ego.l, ego.h,
                 C_EGO, is_ego=True)

        # HUD
        draw_hud(screen, ego, tick, sim_t)

        # Pause overlay
        if paused:
            po = pygame.Surface((W, H), pygame.SRCALPHA)
            pygame.draw.rect(po, (0,0,0,80), (0,0,W,H))
            pt = F_L.render("[ PAUSED — SPACE to resume ]", True, C_YELLOW)
            screen.blit(po, (0,0))
            screen.blit(pt, (W//2 - pt.get_width()//2, H//2 - 15))

        pygame.display.flip()
        clock.tick(FPS)


if __name__ == "__main__":
    print("FSD Visualizer — Kontrol:")
    print("  SPACE     Pause/Resume")
    print("  +/-       Naikkan/turunkan kecepatan")
    print("  ESC       Keluar")
    main()