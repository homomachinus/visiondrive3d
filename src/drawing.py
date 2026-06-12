import pygame
from .config import (
    W, H,
    C_EGO, C_EGO_GLOW, C_OTHER_DIM,
    C_TANAH, C_ASPAL, C_TROTOAR, C_TROTOAR_ED,
    ROAD_W, TROTOAR_W, LANE_W,
)
from .projection import project


def draw_box(surf, cam_x, cam_y, wx, wy, ww, wl, wh, color,
             selected=False, is_ego=False, wz=0.0):
    """gambar kotak 3d satu kendaraan di scene"""
    hw = ww / 2
    hl = wl / 2

    # 8 sudut kotak (bawah dulu, lalu atas)
    corners = []
    for sx, sy, sz in [
        (-hw, -hl, wz),    (hw, -hl, wz),    (hw,  hl, wz),    (-hw,  hl, wz),
        (-hw, -hl, wz+wh), (hw, -hl, wz+wh), (hw,  hl, wz+wh), (-hw,  hl, wz+wh),
    ]:
        corners.append(project(wx + sx, wy + sy, sz, cam_x, cam_y))

    if len([c for c in corners if c]) < 4:
        return

    if is_ego:
        top_c   = (*C_EGO,      80)
        side_c  = (*C_EGO_GLOW, 60)
        front_c = (*C_EGO,      90)
        edge_c  = C_EGO
    else:
        top_c   = (*color, 50)
        side_c  = (int(color[0]*0.6), int(color[1]*0.6), int(color[2]*0.6), 45)
        front_c = (*color, 65)
        edge_c  = color if selected else C_OTHER_DIM

    # gambar face transparan
    fs = pygame.Surface((W, H), pygame.SRCALPHA)

    def face(idxs, c):
        pts = [corners[i] for i in idxs if corners[i]]
        if len(pts) >= 3:
            try:
                pygame.draw.polygon(fs, c, pts)
            except Exception:
                pass

    face([4, 5, 6, 7], top_c)    # atas
    face([0, 1, 5, 4], front_c)  # sisi dekat kamera
    face([1, 2, 6, 5], side_c)   # kanan
    face([3, 0, 4, 7], side_c)   # kiri
    surf.blit(fs, (0, 0))

    # gambar edge lines
    lw = 2 if (selected or is_ego) else 1
    for a, b in [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]:
        if corners[a] and corners[b]:
            pygame.draw.line(surf, edge_c, corners[a], corners[b], lw)

    # glow ring untuk ego
    if is_ego and all(corners[i] for i in [0, 1, 2, 3]):
        gsurf = pygame.Surface((W, H), pygame.SRCALPHA)
        pts_base = [corners[i] for i in [0, 1, 2, 3]]
        pygame.draw.polygon(gsurf, (*C_EGO, 30), pts_base)
        pygame.draw.polygon(gsurf, (*C_EGO, 120), pts_base, 2)
        surf.blit(gsurf, (0, 0))


def draw_road(surf, cam_x, cam_y, tick):
    """gambar aspal, trotoar, dan tanah di kiri-kanan"""
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

    # tanah luar kiri & kanan
    for sign in (-1, 1):
        inner_t = sign * (rw + tw)
        outer_t = sign * (rw + tw + 200.0)
        pts = road_pts(min(inner_t, outer_t), max(inner_t, outer_t), y_near, y_far)
        if len(pts) == 4:
            pygame.draw.polygon(surf, C_TANAH, pts)

    # trotoar kiri & kanan
    for sign in (-1, 1):
        inner = sign * rw
        outer = sign * (rw + tw)
        pts = road_pts(min(inner, outer), max(inner, outer), y_near, y_far)
        if len(pts) == 4:
            pygame.draw.polygon(surf, C_TROTOAR, pts)
            pygame.draw.polygon(surf, C_TROTOAR_ED, pts, 1)
        # garis tepi kerb
        pa = project(inner, y_near, 0.18, cam_x, cam_y)
        pb = project(inner, y_far,  0.18, cam_x, cam_y)
        if pa and pb:
            pygame.draw.line(surf, C_TROTOAR_ED, pa, pb, 2)

    # aspal utama
    pts = road_pts(-rw, rw, y_near, y_far)
    if len(pts) == 4:
        pygame.draw.polygon(surf, C_ASPAL, pts)


def draw_hud(surf, ego, tick, sim_t):
    """placeholder hud, belum diimplementasi"""
    pass
