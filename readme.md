# VisionDrive 3D

Sistem visualisasi mengemudi otonom berbasis YOLO dengan tampilan pseudo-3D gaya Tesla FSD.
Program membaca video dashcam, mendeteksi kendaraan di sekitar, lalu memproyeksikannya ke tampilan bird's-eye view 3D secara real-time.

---

## Cara Kerja

1. Video dashcam dibaca frame per frame menggunakan OpenCV.
2. Setiap frame diproses oleh model YOLO untuk mendeteksi kendaraan (mobil, motor).
3. Jarak kendaraan diestimasi menggunakan dua metode perspektif:
   - tinggi bounding box dibanding tinggi nyata objek
   - posisi vertikal bounding box relatif terhadap garis horizon
4. Posisi 3D setiap kendaraan dihaluskan antar-frame menggunakan Exponential Moving Average (EMA).
5. Kendaraan yang saling tumpang tindih lebih dari 50% area difilter menggunakan occlusion filter.
6. Hasil deteksi dirender ke tampilan pseudo-3D menggunakan Pygame.
7. Video asli dengan anotasi ditampilkan sebagai Picture-in-Picture di sudut layar.

---

## Struktur Proyek

```
visiondrive3d/
|
|-- inf.py              entri utama, jalankan file ini
|
|-- src/
|   |-- __init__.py     re-export semua modul
|   |-- config.py       konstanta: warna, ukuran layar, geometri jalan, kamera
|   |-- projection.py   fungsi proyeksi koordinat 3d ke pixel layar
|   |-- vehicles.py     class EgoVehicle dan Vehicle
|   |-- drawing.py      fungsi render: draw_box, draw_road, draw_hud
|
|-- yolo11n.pt          model YOLO yang digunakan untuk deteksi
```

---

## Dependensi

```
ultralytics
opencv-python
pygame
numpy
```

Install sekaligus:

```
pip install ultralytics opencv-python pygame numpy
```

---

## Menjalankan

Siapkan file video dengan nama `vx1.mp4` di folder yang sama, lalu:

```
python inf.py
```

Untuk mengganti file video, ubah baris terakhir di `inf.py`:

```python
app = InferVisualizer("nama_video.mp4")
```

---

## Kontrol

| Tombol | Fungsi             |
|--------|--------------------|
| SPACE  | pause / resume     |
| ESC    | keluar             |

---

## Parameter Tuning

Semua parameter utama bisa diubah di `src/config.py` (ukuran layar, warna, geometri jalan) dan di bagian atas `inf.py` (kelas target, dimensi kendaraan, alpha EMA).

| Parameter         | Lokasi          | Keterangan                                      |
|-------------------|-----------------|-------------------------------------------------|
| `TARGET_CLASSES`  | `inf.py`        | kelas kendaraan yang dideteksi                  |
| `CLASS_DIMS`      | `inf.py`        | dimensi nyata tiap tipe kendaraan (meter)       |
| `alpha`           | `Tracker`       | faktor EMA smoothing (lebih kecil = lebih halus)|
| `ROAD_W`          | `src/config.py` | lebar total jalan dalam meter                   |
| `FOCAL`           | `src/config.py` | focal length proyeksi pseudo-3d                 |
| `CAM_Y_OFFSET`    | `src/config.py` | jarak kamera mundur dari ego vehicle            |

---

## Catatan

- Model default `yolo11n.pt` adalah versi nano, cepat tapi akurasi sedang. Ganti ke `yolo11s.pt` atau lebih besar untuk akurasi lebih tinggi.
- Estimasi kedalaman menggunakan asumsi kamera dashcam standar (`focal = tinggi_frame * 1.2`). Sesuaikan jika kamera berbeda.
- File `3d.py` adalah eksperimen rendering OpenGL terpisah dan tidak dipakai di workflow utama.
