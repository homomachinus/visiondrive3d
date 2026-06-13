"""
FSD Autonomous Driving Visualizer — standalone single file
(Dense Cityscape, AR HUD, All Traffic Lights ON, Forward Motion)
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

C_BG         = (12,  14,  18)
C_TANAH      = (16,  18,  22)
C_ASPAL      = (22,  24,  32)
C_TROTOAR    = (32,  35,  45)
C_TROTOAR_ED = (44,  50,  65)

C_EGO        = (30,  140, 255)
C_OTHER      = (160, 170, 185)
C_OTHER_DIM  = (80,  90,  105)

ROAD_W     = 14.0
TROTOAR_W  =  3.5
LANE_W     = ROAD_W / 3

CAM_Y_OFFSET = -14.0
CAM_Z        =   9.0
FOCAL        = 420.0

# ══════════════════════════════════════════════════════════════════════════════
#  PROJECTION (Kendaraan & Jalan)
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

# ══════════════════════════════════════════════════════════════════════════════
#  DRAW VEHICLE BOX
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
#  DRAW SOLID BOX (Gedung)
# ══════════════════════════════════════════════════════════════════════════════
def draw_solid_box(surf, cam_x, cam_y, wx, wy, ww, wl, wh, color):
    hw = ww / 2
    hl = wl / 2
    
    pts3d = [
        (wx - hw, wy - hl, 0),  
        (wx + hw, wy - hl, 0),  
        (wx + hw, wy + hl, 0),  
        (wx - hw, wy + hl, 0),  
        (wx - hw, wy - hl, wh), 
        (wx + hw, wy - hl, wh), 
        (wx + hw, wy + hl, wh), 
        (wx - hw, wy + hl, wh), 
    ]

    def clip_face(indices):
        face_pts = [pts3d[i] for i in indices]
        min_y = cam_y + 0.5
        clipped = []
        for i in range(len(face_pts)):
            p1 = face_pts[i]
            p2 = face_pts[(i + 1) % len(face_pts)]
            
            if p1[1] >= min_y:
                clipped.append(p1)
                
            if (p1[1] >= min_y > p2[1]) or (p1[1] < min_y <= p2[1]):
                dy_diff = p2[1] - p1[1]
                if abs(dy_diff) > 0.0001:
                    t = (min_y - p1[1]) / dy_diff
                    ix = p1[0] + t * (p2[0] - p1[0])
                    iy = min_y
                    iz = p1[2] + t * (p2[2] - p1[2])
                    clipped.append((ix, iy, iz))
        return clipped

    def proj(p):
        dx = p[0] - cam_x
        dy = p[1] - cam_y
        dz = p[2] - CAM_Z
        px = W / 2 + (dx / dy) * FOCAL
        py = H / 2 - (dz / dy) * FOCAL * 0.72
        return (int(px), int(py))

    def draw_face(indices, shade):
        clipped_pts3d = clip_face(indices)
        if len(clipped_pts3d) >= 3:
            screen_pts = [proj(p) for p in clipped_pts3d]
            r, g, b = color
            c = (int(r * shade), int(g * shade), int(b * shade))
            pygame.draw.polygon(surf, c, screen_pts)
            pygame.draw.polygon(surf, (20, 22, 28), screen_pts, 1)

    if wx < cam_x:
        draw_face([1, 2, 6, 5], 0.60) # Sisi Kanan
    else:
        draw_face([0, 3, 7, 4], 0.60) # Sisi Kiri
        
    draw_face([0, 1, 5, 4], 0.80)     # Sisi Depan
    
    if CAM_Z > wh:
        draw_face([4, 5, 6, 7], 1.0)  # Sisi Atap (Hanya jika kamera lebih tinggi)

# ══════════════════════════════════════════════════════════════════════════════
#  DRAW TRAFFIC LIGHT
# ══════════════════════════════════════════════════════════════════════════════
def draw_traffic_light(surf, cam_x, cam_y, tl_x, tl_y, sign):
    if tl_y - cam_y < 0.5:
        return

    # Proyeksi tiang utama dan penyangga
    p_base = project(tl_x, tl_y, 0.0, cam_x, cam_y)
    p_top  = project(tl_x, tl_y, 6.0, cam_x, cam_y)
    
    arm_x = tl_x - sign * (ROAD_W * 0.45) # Tiang menjulur ke tengah jalan
    p_arm  = project(arm_x, tl_y, 6.0, cam_x, cam_y)

    # Gambar tiang
    if p_base and p_top:
        pygame.draw.line(surf, (50, 55, 65), p_base, p_top, 4)
    if p_top and p_arm:
        pygame.draw.line(surf, (50, 55, 65), p_top, p_arm, 3)

    # Box Lampu
    bw = 0.35
    bh_top = 5.8
    bh_bot = 4.2
    
    box_pts3d = [
        (arm_x - bw, tl_y, bh_top),
        (arm_x + bw, tl_y, bh_top),
        (arm_x + bw, tl_y, bh_bot),
        (arm_x - bw, tl_y, bh_bot)
    ]
    
    box_proj = [project(px, py, pz, cam_x, cam_y) for px, py, pz in box_pts3d]
    if len([p for p in box_proj if p]) == 4:
        # Gambar kotak hitam
        pygame.draw.polygon(surf, (15, 15, 15), box_proj)
        pygame.draw.polygon(surf, (40, 40, 40), box_proj, 1)

        # SEMUA LAMPU MENYALA TERANG (Mode Placeholder)
        c_red = (255, 40, 40)
        c_yel = (255, 200, 20)
        c_grn = (40, 255, 80)
        
        # Posisi ketinggian masing-masing lampu (Z-axis)
        lz_red = 5.45
        lz_yel = 5.00
        lz_grn = 4.55

        # Render ketiga lampu
        for c, lz in [(c_red, lz_red), (c_yel, lz_yel), (c_grn, lz_grn)]:
            lp = project(arm_x, tl_y, lz, cam_x, cam_y)
            if lp:
                dy = tl_y - cam_y
                radius = max(2, int(FOCAL * 0.15 / dy))
                
                # Gambar inti lampu yang solid
                pygame.draw.circle(surf, c, lp, radius)
                
                # Efek pendaran (Glow) selalu aktif untuk semuanya
                glow_surf = pygame.Surface((radius*4, radius*4), pygame.SRCALPHA)
                pygame.draw.circle(glow_surf, (*c, 60), (radius*2, radius*2), radius*2)
                surf.blit(glow_surf, (lp[0] - radius*2, lp[1] - radius*2))

# ══════════════════════════════════════════════════════════════════════════════
#  DRAW ROAD
# ══════════════════════════════════════════════════════════════════════════════
def draw_road(surf, cam_x, cam_y):
    y_near = cam_y + 0.5
    y_far  = cam_y + 150

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
        pts = road_pts(sign*(rw+tw), sign*(rw+tw+300), y_near, y_far)
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

# ══════════════════════════════════════════════════════════════════════════════
#  BUILDING SYSTEM
# ══════════════════════════════════════════════════════════════════════════════
BUILDING_STEP = 12.0

def _get_building_props(world_slot, side, row):
    h1 = (world_slot * 73856093 ^ side * 19349663 ^ row * 827419) & 0xFFFFFFFF
    rng = random.Random(h1)
    
    b_type = rng.random()
    
    if row == 0:
        if b_type < 0.65:
            w = rng.uniform(5.0, 10.0)
            l = rng.uniform(11.5, 12.0)
            h = rng.uniform(4.0, 9.0)
        elif b_type < 0.90:
            w = rng.uniform(8.0, 15.0)
            l = rng.uniform(11.5, 12.0)
            h = rng.uniform(12.0, 25.0) 
        else:
            w = rng.uniform(10.0, 18.0)
            l = rng.uniform(11.5, 12.0)
            h = rng.uniform(25.0, 45.0) 
        gap = rng.uniform(0.5, 2.5)
    else:
        if b_type < 0.30:
            w = rng.uniform(10.0, 18.0)
            l = rng.uniform(11.5, 12.0)
            h = rng.uniform(15.0, 30.0) 
        elif b_type < 0.80:
            w = rng.uniform(15.0, 25.0)
            l = rng.uniform(11.5, 12.0)
            h = rng.uniform(40.0, 70.0)
        else:
            w = rng.uniform(20.0, 35.0)
            l = rng.uniform(11.5, 12.0)
            h = rng.uniform(70.0, 120.0)
        gap = rng.uniform(14.0, 18.0)
    
    c_base = int(rng.uniform(60, 160))
    if row == 1: 
        c_base = max(30, c_base - 25)
    color = (c_base, c_base, c_base)
    
    return w, l, h, color, gap

def get_buildings(speed_ms, elapsed, rw, y_near, y_far):
    buildings = []
    total_dist = speed_ms * elapsed

    for side in (-1, 1):
        for row in (1, 0):
            b_offset  = total_dist % BUILDING_STEP
            base_slot = int(total_dist // BUILDING_STEP)
            
            y = y_near - b_offset
            s = 0
            while y < y_far:
                world_slot = base_slot + s
                w, l, h, color, gap = _get_building_props(world_slot, side, row)
                x_center = side * (rw + TROTOAR_W + gap + w / 2)
                
                buildings.append({
                    'type': 'building',
                    'x': x_center,
                    'y': y,
                    'w': w, 'l': l, 'h': h,
                    'color': color,
                    'row': row
                })
                y += BUILDING_STEP
                s += 1
            
    return buildings

# ══════════════════════════════════════════════════════════════════════════════
#  TRAFFIC LIGHT SYSTEM
# ══════════════════════════════════════════════════════════════════════════════
TL_STEP = 150.0  # Traffic light muncul setiap 150 meter

def get_traffic_lights(speed_ms, elapsed, rw, y_near, y_far):
    tls = []
    total_dist = speed_ms * elapsed
    offset = total_dist % TL_STEP
    base_slot = int(total_dist // TL_STEP)
    
    y = y_near - offset
    s = 0
    while y < y_far:
        world_slot = base_slot + s
        
        for sign in (-1, 1):
            tl_x = sign * (rw + TROTOAR_W * 0.5)
            tls.append({
                'type': 'traffic_light',
                'x': tl_x,
                'y': y,
                'sign': sign
            })
        
        y += TL_STEP
        s += 1
        
    return tls

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
        self.speed = 12.0
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
    pygame.display.set_caption("FSD Autonomous Driving Visualizer - AR HUD & All Traffic Lights ON")
    clock  = pygame.time.Clock()

    ego = EgoVehicle()
    rng = random.Random(7)

    vehicles = []
    for lane in range(3):
        for j in range(6):
            y   = 10 + j * 16 + rng.uniform(-3, 3)
            spd = rng.uniform(6.0, 10.0)
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
                    ego.speed = min(ego.speed + 2.0, 25.0)
                elif ev.key == pygame.K_MINUS:
                    ego.speed = max(ego.speed - 2.0, 1.0)

        if not paused:
            elapsed += dt
            
            for v in vehicles:
                rel = v.speed - ego.speed
                v.y += rel * dt 
                
                if v.y < -20: 
                    v.y = 150 + rng.uniform(0, 30)
                    v.speed = rng.uniform(5.0, 10.0)
                    v.x = LANES_X[v.lane] + rng.uniform(-0.2, 0.2)
                elif v.y > 200:
                    v.y = -10 - rng.uniform(0, 20)
                    v.speed = rng.uniform(12.0, 16.0)
                    v.x = LANES_X[v.lane] + rng.uniform(-0.2, 0.2)

        screen.fill(C_BG)
        cam_x = ego.x
        cam_y = ego.y + CAM_Y_OFFSET

        draw_road(screen, cam_x, cam_y)

        # Kumpulkan semua elemen dunia
        buildings = get_buildings(ego.speed, elapsed, ROAD_W/2, cam_y - 20, cam_y + 300)
        traffic_lights = get_traffic_lights(ego.speed, elapsed, ROAD_W/2, cam_y - 20, cam_y + 300)

        render_list = []
        render_list.append({'type': 'ego', 'obj': ego, 'y_near': ego.y - ego.l/2, 'abs_x': abs(ego.x)})
        
        for v in vehicles:
            render_list.append({'type': 'vehicle', 'obj': v, 'y_near': v.y - v.l/2, 'abs_x': abs(v.x)})
            
        for b in buildings:
            render_list.append({'type': 'building', 'obj': b, 'y_near': b['y'] - b['l']/2, 'abs_x': abs(b['x'])})
            
        for tl in traffic_lights:
            render_list.append({'type': 'traffic_light', 'obj': tl, 'y_near': tl['y'], 'abs_x': abs(tl['x'])})

        # Z-Sorting tingkat lanjut berdasarkan sumbu Y (jarak ke depan) dan X (posisi terluar)
        render_list.sort(key=lambda item: (item['y_near'], item['abs_x']), reverse=True)

        for item in render_list:
            if item['type'] == 'vehicle':
                v = item['obj']
                selected = abs(v.x - ego.x) < LANE_W * 1.1 and 1.5 < v.y < 60
                draw_box(screen, cam_x, cam_y, v.x, v.y, v.w, v.l, v.h, v.color, selected=selected)
                
            elif item['type'] == 'ego':
                draw_box(screen, cam_x, cam_y, ego.x, ego.y, ego.w, ego.l, ego.h, C_EGO, is_ego=True)
                
            elif item['type'] == 'building':
                b = item['obj']
                draw_solid_box(screen, cam_x, cam_y, b['x'], b['y'], b['w'], b['l'], b['h'], b['color'])
                
                # AR HUD TEKS DIMENSI GEDUNG
                dy = b['y'] - cam_y
                if dy > 0.5:
                    top_pt = project(b['x'], b['y'], b['h'] + 1.5, cam_x, cam_y)
                    if top_pt:
                        dim_text = f"{b['l']:.1f}x{b['w']:.1f}x{b['h']:.1f}m"
                        txt_surf = F_XS.render(dim_text, True, (160, 170, 185)) 
                        tx, ty = top_pt
                        if -50 <= tx <= W + 50 and -50 <= ty <= H + 50:
                            screen.blit(txt_surf, (tx - txt_surf.get_width() // 2, ty - txt_surf.get_height()))

            elif item['type'] == 'traffic_light':
                tl = item['obj']
                draw_traffic_light(screen, cam_x, cam_y, tl['x'], tl['y'], tl['sign'])

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