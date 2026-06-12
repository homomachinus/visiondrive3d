from .config import FOCAL, W, H, CAM_Z


def project(wx, wy, wz, cam_x, cam_y):
    """world → pixel layar"""
    dx = wx - cam_x
    dy = wy - cam_y   # positif = makin jauh
    dz = wz - CAM_Z
    if dy < 0.5:      # di belakang kamera, skip
        return None
    px = W / 2 + (dx / dy) * FOCAL
    py = H / 2 - (dz / dy) * FOCAL * 0.72
    return int(px), int(py)


def proj_scale(wy, cam_y):
    """skala perspektif berdasarkan jarak"""
    dy = wy - cam_y
    if dy < 0.5:
        return 0.0
    return FOCAL / dy
