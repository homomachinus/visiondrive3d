from .config import ROAD_W, LANE_W, C_OTHER


class EgoVehicle:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.speed = 0.0
        self.accel = 0.0
        self.steer = 0.0
        self.w, self.l, self.h = 2.0, 4.6, 1.6


class Vehicle:
    # tipe kendaraan yg tersedia (lebar, panjang, tinggi)
    TYPES = [
        {'w': 2.0, 'l': 4.5, 'h': 1.5},  # sedan
        {'w': 2.2, 'l': 4.8, 'h': 1.7},  # suv
        {'w': 2.4, 'l': 5.2, 'h': 2.0},  # mpv
        {'w': 2.5, 'l': 8.0, 'h': 2.5},  # van / truck kecil
    ]
    LANES = [-LANE_W, 0.0, LANE_W]  # posisi tengah tiap lajur

    def __init__(self, y, lane_idx, speed, rng):
        t = rng.choice(self.TYPES)
        self.w = t['w']
        self.l = t['l']
        self.h = t['h']
        self.x = self.LANES[lane_idx] + rng.uniform(-0.3, 0.3)
        self.y = float(y)
        self.speed = speed
        self.lane = lane_idx
        self.color = C_OTHER

    def update(self, dt):
        self.y += self.speed * dt

    @property
    def selected(self):
        # kendaraan di lajur tengah dan di depan ego
        return abs(self.x) < LANE_W * 0.7 and self.y > 0
