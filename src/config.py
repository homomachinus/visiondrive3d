# ukuran layar
W, H = 1280, 760
FPS  = 60

# warna tema gelap
C_BG         = (10,  11,  16)
C_TANAH      = (16,  17,  22)
C_ASPAL      = (22,  24,  32)
C_ASPAL_LINE = (38,  42,  55)
C_TROTOAR    = (32,  35,  45)
C_TROTOAR_ED = (44,  50,  65)
C_LANE       = (55,  60,  45)
C_EGO        = (30,  140, 255)
C_EGO_GLOW   = (0,   80,  200)
C_OTHER      = (160, 170, 185)
C_OTHER_DIM  = (90,  100, 115)
C_PATH       = (30,  200, 100)
C_HUD_BG     = (8,   10,  16)
C_TEXT       = (220, 225, 235)
C_DIM        = (100, 110, 130)
C_GREEN      = (60,  220, 90)
C_BLUE_LIGHT = (80,  160, 255)
C_RED_LIGHT  = (220, 60,  60)
C_YELLOW     = (220, 180, 50)

# geometri jalan (meter)
ROAD_W    = 14.0   # total lebar jalan 3 lajur
TROTOAR_W =  3.5
LANE_W    = ROAD_W / 3

# parameter kamera pseudo-3d
CAM_Y_OFFSET = -14.0   # kamera mundur dari ego
CAM_Z        =   9.0   # ketinggian kamera
FOCAL        = 420.0   # focal length proyeksi
