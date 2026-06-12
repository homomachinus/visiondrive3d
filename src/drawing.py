import pygame
from .config import (
    W, H,
    C_EGO, C_EGO_GLOW, C_OTHER_DIM,
    C_TANAH, C_ASPAL, C_TROTOAR, C_TROTOAR_ED,
    ROAD_W, TROTOAR_W, LANE_W,
)
import math
from .projection import project

# Fixed lane marking dimensions (meters)
DASH_LEN       = 1.0    # length of each dashed line segment
DASH_GAP       = 1.5    # gap between dashes
ACCESSORY_STEP = 20.0   # spacing between roadside accessories (meters)


def draw_box(surf, cam_x, cam_y, wx, wy, ww, wl, wh, color,
             selected=False, is_ego=False, wz=0.0):
    hw = ww / 2
    hl = wl / 2

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

    fs = pygame.Surface((W, H), pygame.SRCALPHA)

    def face(idxs, c):
        pts = [corners[i] for i in idxs if corners[i]]
        if len(pts) >= 3:
            try:
                pygame.draw.polygon(fs, c, pts)
            except Exception:
                pass

    face([4, 5, 6, 7], top_c)
    face([0, 1, 5, 4], front_c)
    face([1, 2, 6, 5], side_c)
    face([3, 0, 4, 7], side_c)
    surf.blit(fs, (0, 0))

    lw = 2 if (selected or is_ego) else 1
    for a, b in [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]:
        if corners[a] and corners[b]:
            pygame.draw.line(surf, edge_c, corners[a], corners[b], lw)

    if is_ego and all(corners[i] for i in [0, 1, 2, 3]):
        gsurf = pygame.Surface((W, H), pygame.SRCALPHA)
        pts_base = [corners[i] for i in [0, 1, 2, 3]]
        pygame.draw.polygon(gsurf, (*C_EGO, 30), pts_base)
        pygame.draw.polygon(gsurf, (*C_EGO, 120), pts_base, 2)
        surf.blit(gsurf, (0, 0))


def draw_road(surf, cam_x, cam_y, speed_ms=0.0, elapsed=0.0):
    """Draw asphalt, kerb, pavement, center dashes, and roadside accessories.

    Args:
        surf      : pygame Surface to draw on
        cam_x/y   : camera world position
        speed_ms  : current ego speed in m/s (drives animation offset)
        elapsed   : total elapsed simulation time in seconds
    """
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

    # outer ground strips
    for sign in (-1, 1):
        inner_t = sign * (rw + tw)
        outer_t = sign * (rw + tw + 200.0)
        pts = road_pts(min(inner_t, outer_t), max(inner_t, outer_t), y_near, y_far)
        if len(pts) == 4:
            pygame.draw.polygon(surf, C_TANAH, pts)

    # kerb / pavement
    for sign in (-1, 1):
        inner = sign * rw
        outer = sign * (rw + tw)
        pts = road_pts(min(inner, outer), max(inner, outer), y_near, y_far)
        if len(pts) == 4:
            pygame.draw.polygon(surf, C_TROTOAR, pts)
            pygame.draw.polygon(surf, C_TROTOAR_ED, pts, 1)
        pa = project(inner, y_near, 0.18, cam_x, cam_y)
        pb = project(inner, y_far,  0.18, cam_x, cam_y)
        if pa and pb:
            pygame.draw.line(surf, C_TROTOAR_ED, pa, pb, 2)

    # road surface
    pts = road_pts(-rw, rw, y_near, y_far)
    if len(pts) == 4:
        pygame.draw.polygon(surf, C_ASPAL, pts)
        _draw_center_dashed(surf, cam_x, cam_y, speed_ms, elapsed, rw, y_near, y_far)
        _draw_side_accessories(surf, cam_x, cam_y, speed_ms, elapsed, rw, y_near, y_far)


def draw_hud(surf, ego, speed_ms, sim_t):
    """placeholder hud"""
    pass


def _draw_center_dashed(surf, cam_x, cam_y, speed_ms, elapsed, road_half_width, y_near, y_far):
    """Animate center dashed lane markings.

    offset = distance (m) the ego has traveled modulo one dash+gap cycle.
    This makes the dashes appear to stream backward at the correct real-world speed.
    """
    cycle  = DASH_LEN + DASH_GAP
    offset = (speed_ms * elapsed) % cycle   # metres the world has scrolled

    y = y_near - offset
    while y < y_far:
        p0 = project(0, y,            0, cam_x, cam_y)
        p1 = project(0, y + DASH_LEN, 0, cam_x, cam_y)
        if p0 and p1:
            pygame.draw.line(surf, (255, 255, 255), p0, p1, 2)
        y += cycle


def _draw_side_accessories(surf, cam_x, cam_y, speed_ms, elapsed, road_half_width, y_near, y_far):
    """Animate roadside lamps (right) and trees (left).

    Accessories are spaced ACCESSORY_STEP metres apart.
    offset scrolls at the same speed as the dashes so both are in sync.
    """
    offset = (speed_ms * elapsed) % ACCESSORY_STEP

    for side in (-1, 1):
        x_pos = side * (road_half_width + 5)
        y = y_near - offset
        while y < y_far:
            top    = project(x_pos, y,       0.0, cam_x, cam_y)
            bottom = project(x_pos, y + 2.0, 0.0, cam_x, cam_y)
            if top and bottom:
                pygame.draw.line(surf, (150, 150, 150), top, bottom, 2)
                lamp = project(x_pos, y + 2.2, 0.0, cam_x, cam_y)
                if lamp:
                    pygame.draw.circle(surf, (255, 200, 0), (int(lamp[0]), int(lamp[1])), 4)
            if side == -1:
                tx = x_pos - 2.5
                tt = project(tx, y,       0.0, cam_x, cam_y)
                tb = project(tx, y + 1.5, 0.0, cam_x, cam_y)
                if tt and tb:
                    pygame.draw.line(surf, (101, 67, 33), tt, tb, 2)
                    c = project(tx, y + 1.8, 0.0, cam_x, cam_y)
                    if c:
                        pygame.draw.circle(surf, (34, 139, 34), (int(c[0]), int(c[1])), 5)
            y += ACCESSORY_STEP
