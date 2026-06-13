"""
FSD Autonomous Driving Visualizer — standalone single file
(Dynamic Scale & Fixed Z-Sorting / Overlap Bug)
"""

import pygame
import math
import random
import time

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════
W, H   = 1280, 760
FPS    = 60

C_BG         = (10,  11,  16)
C_TANAH      = (16,  18,  22)
C_ASPAL      = (22,  24,  32)
C_TROTOAR    = (32,  35,  45)
C_TROTOAR_ED = (44,  50,  65)

C_EGO        = (30,  140, 255)
C_EGO_GLOW   = (0,   80,  200)
C_OTHER      = (160, 170, 185)
C_OTHER_DIM  = (80,  90,  105)

ROAD_W     = 14.0
TROTOAR_W  =  3.5
LANE_W     = ROAD_W / 3

CAM_Y_OFFSET = -14.0
CAM_Z        =   9.0
FOCAL        = 420.0

# ══════════════════════════════════════════════════════════════════════════════
#  PROJECTION
# ══════════════════════════════════════════════════════════════════════════════
def project(wx, wy, wz, cam_x, cam_y):
    dx = wx - cam_x
    dy = wy - cam_y
    dz = wz - CAM_Z
    if dy < 0.5:
        return None
    px = W / 2 + (dx / dy) * FOCAL
    py = H / 2 - (dz / dy) * FOCAL * 0.72
    return (int(px), int(py))

# ══════════════════════════════════════════════════════════════════════════════
#  DRAW BOX
# ══════════════════════════════════════════════════════════════════════════════
def draw_box(surf, cam_x, cam_y, wx, wy, ww, wl, wh, color,
             selected=False, is_ego=False, wz=0.0):
    hw = ww / 2
    hl = wl / 2
    corners = [project(wx+sx, wy+sy, wz+sz, cam_x, cam_y)
               for sx, sy, sz in [
                   (-hw,-hl,0),(hw,-hl,0),(hw,hl,0),(-hw,hl,0),
                   (-hw,-hl,wh),(hw,-hl,wh),(hw,hl,wh),(-hw,hl,wh),
               ]]
    if len([c for c in corners if c]) < 4:
        return
    r, g, b  = C_EGO if is_ego else color
    edge_c   = C_EGO if is_ego else (color if selected else C_OTHER_DIM)
    lw       = 2 if (is_ego or selected) else 1
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

# ══════════════════════════════════════════════════════════════════════════════
#  DRAW ROAD
# ══════════════════════════════════════════════════════════════════════════════
def draw_road(surf, cam_x, cam_y, speed_ms=0.0, elapsed=0.0):
    y_near = cam_y + 1
    y_far  = cam_y + 120

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

    for sign in (-1, 1):
        pts = road_pts(sign*(rw+tw), sign*(rw+tw+200), y_near, y_far)
        if len(pts) == 4:
            pygame.draw.polygon(surf, C_TANAH, pts)

    for sign in (-1, 1):
        inner = sign * rw
        outer = sign * (rw + tw)
        pts = road_pts(min(inner,outer), max(inner,outer), y_near, y_far)
        if len(pts) == 4:
            pygame.draw.polygon(surf, C_TROTOAR, pts)
            pygame.draw.polygon(surf, C_TROTOAR_ED, pts, 1)
        pa = project(inner, y_near, 0.18, cam_x, cam_y)
        pb = project(inner, y_far,  0.18, cam_x, cam_y)
        if pa and pb:
            pygame.draw.line(surf, C_TROTOAR_ED, pa, pb, 2)

    pts = road_pts(-rw, rw, y_near, y_far)
    if len(pts) == 4:
        pygame.draw.polygon(surf, C_ASPAL, pts)

    _draw_accessories(surf, cam_x, cam_y, speed_ms, elapsed, rw, y_near, y_far)

# ══════════════════════════════════════════════════════════════════════════════
#  TREE SYSTEM & ACCESSORIES
# ══════════════════════════════════════════════════════════════════════════════
TREE_STEP  = 12.0  # Diperlebar agar penataan lebih rapi dan elegan
LAMP_STEP  = 24.0

TREE_VARIANTS = [
    {'trunk_h': 2.2, 'trunk_w': 0.30,
     'canopy': [(0.0,1.6,(28,95,38)),(0.9,1.9,(35,110,45)),(1.8,1.5,(42,125,50)),(2.5,0.9,(50,140,55))]},
    {'trunk_h': 4.0, 'trunk_w': 0.22,
     'canopy': [(0.0,1.1,(25,85,35)),(1.1,1.4,(32,100,42)),(2.2,1.2,(38,115,48)),(3.1,0.9,(44,128,52)),(3.9,0.5,(52,140,58))]},
    {'trunk_h': 3.0, 'trunk_w': 0.38,
     'canopy': [(-0.3,2.2,(22,80,32)),(0.8,2.6,(30,100,40)),(2.0,2.4,(36,112,46)),(3.0,1.8,(42,124,50)),(3.9,1.1,(50,136,56))]},
    {'trunk_h': 2.8, 'trunk_w': 0.30,
     'canopy': [(0.0,1.4,(26,88,36)),(1.0,1.8,(33,105,44)),(2.1,1.6,(40,118,50)),(3.0,1.0,(48,132,54))]},
    {'trunk_h': 1.6, 'trunk_w': 0.18,
     'canopy': [(0.0,1.0,(30,100,40)),(0.8,1.2,(38,115,48)),(1.5,0.8,(46,130,54))]},
    {'trunk_h': 3.5, 'trunk_w': 0.26,
     'canopy': [(0.0,1.3,(24,82,34)),(1.2,1.7,(31,98,42)),(2.4,1.5,(38,112,48)),(3.3,1.1,(45,126,53)),(4.1,0.6,(52,138,57))]},
]

SPRITE_REF_DY = 30.0
_tree_sprites_raw:   dict = {}

def _build_tree_sprite(vi: int) -> pygame.Surface:
    v   = TREE_VARIANTS[vi]
    th  = v['trunk_h']
    tw  = v['trunk_w']

    px_per_wu = FOCAL / SPRITE_REF_DY
    pz_per_wu = px_per_wu * 0.72

    max_z = th + max(zo + r for zo, r, _ in v['canopy'])
    max_r = max(r for _, r, _ in v['canopy'])

    pw = int((tw + max_r * 2 + 0.5) * px_per_wu) + 8
    ph = int(max_z * pz_per_wu) + 8
    pw = max(pw, 12);  ph = max(ph, 12)

    spr = pygame.Surface((pw, ph), pygame.SRCALPHA)
    ox = pw // 2;  oy = ph

    def s2p(wu_x, wu_z):
        return (int(ox + wu_x * px_per_wu), int(oy - wu_z * pz_per_wu))

    trunk_c = (65, 42, 22);  trunk_d = (45, 28, 12)
    hw = tw / 2
    bl = s2p(-hw,0); br = s2p(hw,0); tl = s2p(-hw,th); tr = s2p(hw,th)
    if all([bl,br,tl,tr]):
        pygame.draw.polygon(spr, (*trunk_c, 230), [bl, br, tr, tl])
        pygame.draw.polygon(spr, (*trunk_d, 180), [bl, br, tr, tl], 1)

    for z_off, r_wu, base_col in v['canopy']:
        z = th + z_off
        cx, cy = s2p(0, z)
        rx = max(3, int(r_wu * px_per_wu))
        ry = max(2, int(r_wu * pz_per_wu * 0.75))
        shadow = (max(0,base_col[0]-20), max(0,base_col[1]-25), max(0,base_col[2]-20))
        hi     = (min(255,base_col[0]+22), min(255,base_col[1]+28), min(255,base_col[2]+20))
        pygame.draw.ellipse(spr, (*shadow, 180), (cx-rx, cy-ry+ry//3, rx*2, ry+ry//2))
        pygame.draw.ellipse(spr, (*base_col, 235), (cx-rx, cy-ry, rx*2, ry*2))
        hi_r = max(2, rx//3)
        pygame.draw.ellipse(spr, (*hi, 90), (cx-rx//3-hi_r, cy-ry//3-hi_r//2, hi_r*2, hi_r))

    return spr

def _variant_for_slot(world_slot: int, lane_seed: int) -> int:
    h = (world_slot * 1_000_003 + lane_seed * 999_983) & 0x7FFF_FFFF
    return h % len(TREE_VARIANTS)

def _draw_tree_at(surf, cam_x, cam_y, wx, wy, vi: int):
    dy = wy - cam_y
    if dy < 0.5:
        return
        
    base = project(wx, wy, 0, cam_x, cam_y)
    if not base:
        return

    if vi not in _tree_sprites_raw:
        _tree_sprites_raw[vi] = _build_tree_sprite(vi)
    raw_spr = _tree_sprites_raw[vi]

    scale = SPRITE_REF_DY / dy
    if scale > 20.0:
        return

    sw = max(2, int(raw_spr.get_width() * scale))
    sh = max(2, int(raw_spr.get_height() * scale))

    spr = pygame.transform.scale(raw_spr, (sw, sh))
    surf.blit(spr, (base[0] - sw // 2, base[1] - sh))

def _draw_lamp_at(surf, cam_x, cam_y, x_lamp, y, sign):
    pole_b = project(x_lamp, y, 0.0,  cam_x, cam_y)
    pole_t = project(x_lamp, y, 3.8,  cam_x, cam_y)
    arm_e  = project(x_lamp + sign*0.65, y, 4.05, cam_x, cam_y)
    bulb   = project(x_lamp + sign*0.65, y, 4.10, cam_x, cam_y)
    
    if pole_b and pole_t:
        pygame.draw.line(surf, (72, 78, 95), pole_b, pole_t, 2)
    if pole_t and arm_e:
        pygame.draw.line(surf, (82, 88, 108), pole_t, arm_e, 2)
    if bulb:
        pygame.draw.circle(surf, (255, 218, 135), bulb, 3)
        gs = pygame.Surface((12, 12), pygame.SRCALPHA)
        pygame.draw.circle(gs, (255, 200, 80, 40), (6, 6), 6)
        surf.blit(gs, (bulb[0]-6, bulb[1]-6))

def _draw_accessories(surf, cam_x, cam_y, speed_ms, elapsed, rw, y_near, y_far):
    total_dist = speed_ms * elapsed
    draw_items = []

    # ── Kumpulkan Posisi Pohon ───────────────────────────────────────────────
    for side_idx, sign in enumerate((-1, 1)):
        for row, x_base in enumerate([
            sign * (rw + TROTOAR_W + 1.2),
            sign * (rw + TROTOAR_W + 5.5), # Digeser sedikit lebih jauh ke samping
        ]):
            row_dist    = total_dist + (TREE_STEP * 0.5 if row == 1 else 0.0)
            tree_offset = row_dist % TREE_STEP
            base_slot   = int(row_dist // TREE_STEP)
            lane_seed = side_idx * 2 + row

            y = y_near - tree_offset
            s = 0
            while y < y_far:
                world_slot = base_slot + s
                vi = _variant_for_slot(world_slot, lane_seed)
                draw_items.append(('tree', y, x_base, vi))
                y += TREE_STEP
                s += 1

    # ── Kumpulkan Posisi Lampu ───────────────────────────────────────────────
    lamp_offset = total_dist % LAMP_STEP
    for sign in (-1, 1):
        x_lamp = sign * (rw + TROTOAR_W * 0.5)
        y = y_near - lamp_offset
        while y < y_far:
            draw_items.append(('lamp', y, x_lamp, sign))
            y += LAMP_STEP

    # ── FIX: Z-Sorting (Urutkan objek paling jauh / y tertinggi, ke terdekat) 
    draw_items.sort(key=lambda item: item[1], reverse=True)

    # ── Gambar Semua Objek secara Berurutan ──────────────────────────────────
    for item in draw_items:
        if item[0] == 'tree':
            _, y, x_base, vi = item
            _draw_tree_at(surf, cam_x, cam_y, x_base, y, vi)
        elif item[0] == 'lamp':
            _, y, x_lamp, sign = item
            _draw_lamp_at(surf, cam_x, cam_y, x_lamp, y, sign)

# ══════════════════════════════════════════════════════════════════════════════
#  VEHICLES
# ══════════════════════════════════════════════════════════════════════════════
VEHICLE_TYPES = [
    {'w': 1.9, 'l': 4.2, 'h': 1.35, 'color': (150,160,175)},
    {'w': 2.0, 'l': 4.6, 'h': 1.60, 'color': (140,150,165)},
    {'w': 2.1, 'l': 4.8, 'h': 1.75, 'color': (130,140,155)},
    {'w': 2.3, 'l': 7.5, 'h': 2.40, 'color': (110,120,135)},
    {'w': 1.8, 'l': 3.8, 'h': 1.20, 'color': (160,165,180)},
]
LANES_X = [-LANE_W, 0.0, LANE_W]

class Vehicle:
    def __init__(self, y, lane, speed, rng):
        t = rng.choice(VEHICLE_TYPES)
        self.w, self.l, self.h = t['w'], t['l'], t['h']
        self.color = t['color']
        self.x     = LANES_X[lane] + rng.uniform(-0.2, 0.2)
        self.y     = float(y)
        self.speed = float(speed)
        self.lane  = lane

class EgoVehicle:
    def __init__(self):
        self.x, self.y = 0.0, 0.0
        self.speed = 8.0
        self.accel = 0.0
        self.w, self.l, self.h = 2.0, 4.6, 1.55

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    pygame.init()
    try:
        fn = pygame.font.match_font('dejavusansmono,consolas,couriernew,monospace')
        F_XS = pygame.font.Font(fn, 10)
    except:
        F_XS = pygame.font.SysFont('monospace', 10)

    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("FSD Autonomous Driving Visualizer - Fixed Z-Sorting")
    clock  = pygame.time.Clock()

    ego = EgoVehicle()
    rng = random.Random(7)

    vehicles = []
    for lane in range(3):
        for j in range(6):
            y   = 10 + j * 16 + rng.uniform(-3, 3)
            spd = rng.uniform(4.5, 9.0)
            vehicles.append(Vehicle(y, lane, spd, rng))

    elapsed = 0.0
    paused  = False
    last_t  = time.time()

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
            elapsed += dt
            target = 8.0 + 2.5 * math.sin(elapsed * 0.18)
            ego.accel = max(-3.5, min(2.5, (target - ego.speed) * 1.8))
            ego.speed = max(1.0, min(18.0, ego.speed + ego.accel * dt))

            for v in vehicles:
                rel = v.speed - ego.speed
                v.y -= rel * dt
                if v.y < -18:
                    v.y = 95 + rng.uniform(0, 30)
                    v.speed = rng.uniform(4.5, 10.0)
                    v.x = LANES_X[v.lane] + rng.uniform(-0.2, 0.2)
                elif v.y > 130:
                    v.y = 10 + rng.uniform(0, 10)
                    v.speed = rng.uniform(4.5, 10.0)

        screen.fill(C_BG)
        cam_x = ego.x
        cam_y = ego.y + CAM_Y_OFFSET

        draw_road(screen, cam_x, cam_y, ego.speed, elapsed)

        for v in sorted(vehicles, key=lambda v: -v.y):
            selected = abs(v.x - ego.x) < LANE_W * 1.1 and 1.5 < v.y < 60
            draw_box(screen, cam_x, cam_y,
                     v.x, v.y, v.w, v.l, v.h,
                     v.color, selected=selected)

        draw_box(screen, cam_x, cam_y,
                 ego.x, ego.y, ego.w, ego.l, ego.h,
                 C_EGO, is_ego=True)

        if paused:
            po = pygame.Surface((W, H), pygame.SRCALPHA)
            pygame.draw.rect(po, (0,0,0,80), (0,0,W,H))
            f  = pygame.font.SysFont('monospace', 22)
            pt = f.render("[ PAUSED — SPACE to resume ]", True, (220,180,50))
            screen.blit(po, (0,0))
            screen.blit(pt, (W//2 - pt.get_width()//2, H//2 - 15))

        pygame.display.flip()
        clock.tick(FPS)

if __name__ == "__main__":
    print("FSD Visualizer — Kontrol: SPACE=pause  +/-=kecepatan  ESC=keluar")
    main()