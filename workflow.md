# DriveVerse
## Real-Time 3D Traffic Scene Reconstruction using Monocular Vision

> Updated Architecture:
> Dual-Model Pipeline (YOLOv11 + Segmentation Model)
> Inspired by GPLVM workflow paper but redesigned using modern perception stack.

Reference workflow adaptation from:
"3D Traffic Scenes Reconstruction for Autonomous Vehicles using GPLVM" :contentReference[oaicite:0]{index=0}

---

# 1. PROJECT OVERVIEW

## Objective

Membangun sistem real-time 3D traffic scene reconstruction menggunakan:
- monocular dashcam camera
- object detection
- semantic / instance segmentation
- monocular depth estimation
- multi-object tracking
- Bird's Eye View transformation
- occupancy reconstruction
- realtime 3D visualization

---

# 2. KEY DIFFERENCE FROM ORIGINAL GPLVM PAPER

| Original Paper | Updated System |
|---|---|
| YOLOv9 | YOLOv11 |
| GPLVM reconstruction | Dense geometry reconstruction |
| Shape prior fitting | Direct scene reconstruction |
| Hough lane detection | Deep lane segmentation |
| Depth from bbox size | Dense monocular depth |
| SORT | ByteTrack |
| Vehicle template interpolation | Scene-level occupancy representation |
| 3.1 sec/frame | Realtime-oriented architecture |

---

# 3. FINAL SYSTEM STACK

| Module | Technology |
|---|---|
| Object Detection | YOLOv11 |
| Segmentation | SAM2 / SegFormer |
| Depth Estimation | Depth Anything V2 |
| Multi Object Tracking | ByteTrack |
| Lane Detection | LaneATT |
| Geometry Projection | OpenCV |
| BEV Transformation | Homography |
| 3D Reconstruction | Occupancy Grid / Point Cloud |
| Visualization | Open3D / Unity |
| Framework | PyTorch |
| GPU Acceleration | CUDA / TensorRT |

---

# 4. SYSTEM ARCHITECTURE

```text
Dashcam Video
    ↓
Frame Extraction
    ↓
YOLOv11 Detection
    ↓
Bounding Box Extraction
    ↓
ROI Cropping
    ↓
Segmentation Model
    ↓
Instance Mask Generation
    ↓
Depth Anything V2
    ↓
Dense Depth Map
    ↓
Object-Depth Association
    ↓
2D → 3D Projection
    ↓
ByteTrack
    ↓
Trajectory Estimation
    ↓
Lane Detection
    ↓
Bird's Eye View Transformation
    ↓
3D Occupancy Reconstruction
    ↓
Open3D / Unity Visualization
```

---

# 5. COMPLETE WORKFLOW

---

# STAGE 1 — VIDEO ACQUISITION

## Objective

Mengambil input video realtime dari dashcam.

---

## Input

```text
Dashcam Video Stream
```

Format:
- MP4
- AVI
- Webcam Stream

---

## Process

### 1. Video Capture
Menggunakan OpenCV:
- cv2.VideoCapture()

---

### 2. Frame Extraction

Extract frame:
- 15–30 FPS

---

### 3. Resize & Normalize

Resize:
```python
1280x720
```

Normalization:
```python
0-1 float normalization
```

---

## Output

```python
frame.shape = (720,1280,3)
```

---

# STAGE 2 — OBJECT DETECTION

# MODEL:
# YOLOv11

---

## Objective

Mendeteksi object traffic:
- car
- truck
- bus
- motorcycle
- bicycle
- pedestrian

---

## Why Separate Detection Model?

YOLOv11 digunakan khusus untuk:
- localization
- fast inference
- realtime bbox extraction

Karena:
- lebih ringan
- lebih cepat
- detection lebih stabil

dibanding multitask segmentation model.

---

## Input

```python
RGB Frame
```

---

## Internal Process

### 1. Backbone

Feature extraction:
- CNN backbone
- multi-scale feature map

---

### 2. Neck

Feature fusion:
- FPN/PAN architecture

---

### 3. Detection Head

Generate:
- bbox coordinate
- confidence
- class probability

---

## Output

```python
[
 {
   "bbox":[x1,y1,x2,y2],
   "confidence":0.94,
   "class":"car"
 }
]
```

---

## Important Notes

Detection dilakukan full-frame:
- seluruh image diproses sekali.

---

# STAGE 3 — ROI EXTRACTION

## Objective

Mengambil region kendaraan untuk segmentation.

---

## Process

Setiap bbox dari YOLOv11:
- di-crop dari frame asli.

---

## Formula

```python
roi = frame[y1:y2, x1:x2]
```

---

## Why ROI Cropping?

Karena segmentation model:
- lebih fokus
- lebih cepat
- noise lebih kecil

---

## Output

```python
vehicle_crop
```

---

# STAGE 4 — INSTANCE SEGMENTATION

# MODEL:
# SAM2 / SegFormer

---

## Objective

Menghasilkan object mask presisi tinggi.

---

## Why Separate Segmentation Model?

Karena:
- segmentation lebih presisi
- contour kendaraan lebih akurat
- shape extraction lebih bagus
- depth association lebih stabil

---

## Input

```python
vehicle_crop
```

atau:
```python
bbox prompt
```

untuk SAM2.

---

## Internal Process

### 1. Feature Encoding

Encoder memahami:
- edge
- contour
- semantic region

---

### 2. Mask Decoding

Generate:
- binary mask
- object contour

---

## Output

```python
binary_mask
```

Contoh:

```python
mask.shape = (256,256)
```

---

# STAGE 5 — MASK POSTPROCESSING

## Objective

Membersihkan noise hasil segmentation.

---

## Process

### 1. Morphological Opening

Remove:
- isolated pixels

---

### 2. Morphological Closing

Fill:
- mask holes

---

### 3. Largest Contour Selection

Ambil contour utama kendaraan.

---

## Output

```python
clean_vehicle_mask
```

---

# STAGE 6 — DEPTH ESTIMATION

# MODEL:
# Depth Anything V2

---

## Objective

Mengestimasi kedalaman tiap pixel.

---

## Input

```python
RGB Frame
```

---

## Internal Process

### 1. Vision Transformer Encoder

Extract:
- semantic geometry
- scene structure

---

### 2. Depth Decoder

Generate:
- dense depth map

---

## Output

```python
depth_map[h,w]
```

Contoh:

```python
depth_map[400,300] = 14.2 meter
```

---

# STAGE 7 — OBJECT DEPTH ASSOCIATION

## Objective

Menghubungkan segmentation mask dengan depth map.

---

## Why Important?

Karena:
- bbox centroid sering tidak akurat
- object bisa overlap
- depth object harus lebih stabil

---

## Process

### 1. Ambil semua pixel mask

```python
masked_pixels = depth_map[mask==1]
```

---

### 2. Compute Representative Depth

Gunakan:
- median depth

lebih stabil dibanding mean.

---

## Formula

```python
Z = median(masked_depth)
```

---

## Output

```python
vehicle_depth = 13.7 meter
```

---

# STAGE 8 — CAMERA CALIBRATION

## Objective

Menghitung intrinsic camera parameter.

---

## Calibration Method

### Chessboard Calibration

Menggunakan:
- OpenCV calibration

---

## Parameters

```text
fx = focal length x
fy = focal length y
cx = principal point x
cy = principal point y
```

---

## Intrinsic Matrix

```text
K =
[ fx  0 cx ]
[ 0 fy cy ]
[ 0  0  1 ]
```

---

## Output

```python
camera_matrix
distortion_coefficients
```

---

# STAGE 9 — 2D TO 3D PROJECTION

## Objective

Mengubah object pixel menjadi koordinat dunia.

---

## Input

- bbox centroid
- segmentation centroid
- object depth
- intrinsic matrix

---

## Formula

:contentReference[oaicite:1]{index=1}

---

## Process

### 1. Compute centroid

```python
u,v = object_center
```

---

### 2. Convert to camera coordinate

Generate:
```python
(X,Y,Z)
```

---

## Output

```python
car_position_3d
```

Contoh:

```python
(2.4,-0.8,13.7)
```

---

# STAGE 10 — MULTI OBJECT TRACKING

# MODEL:
# ByteTrack

---

## Objective

Tracking kendaraan antar frame.

---

## Why ByteTrack?

Karena:
- lebih stabil dibanding SORT
- better occlusion handling
- minim ID switching

---

## Input

```python
detections
+
confidence
```

---

## Process

### 1. Detection Association

Matching object:
- frame t
- frame t+1

---

### 2. Motion Prediction

Trajectory estimation.

---

### 3. ID Persistence

Maintain:
```python
Vehicle ID
```

---

## Output

```python
tracked_vehicle
```

---

# STAGE 11 — LANE DETECTION

# MODEL:
# LaneATT

---

## Objective

Mendeteksi:
- lane line
- drivable area
- road boundary

---

## Input

```python
RGB Frame
```

---

## Output

```python
lane_polylines
```

---

## Improvement from GPLVM Paper

Original paper:
- Hough Transform

Updated:
- Deep lane estimation

Better for:
- curved road
- missing lane
- shadow
- night condition

---

# STAGE 12 — BIRD'S EYE VIEW (BEV)

## Objective

Mengubah perspective camera menjadi top-view.

---

## Process

### 1. Perspective Transformation

Menggunakan:
- homography matrix

---

### 2. Inverse Perspective Mapping

Transform:
- lane
- object position
- road area

ke coordinate top-view.

---

## Output

```python
BEV_map
```

---

# STAGE 13 — OCCUPANCY RECONSTRUCTION

## Objective

Membangun representasi 3D traffic scene.

---

## Input

- object coordinate
- lane coordinate
- segmentation mask
- depth map

---

## Process

### 1. Point Cloud Generation

Generate:
```python
(x,y,z)
```

untuk object & road.

---

### 2. Occupancy Grid

Voxelize:
- occupied space
- free space

---

### 3. Scene Graph Construction

Build:
- vehicle nodes
- lane nodes
- trajectory

---

## Output

```python
3D Occupancy Scene
```

---

# STAGE 14 — VISUALIZATION

# TOOL:
# Open3D / Unity

---

## Objective

Menampilkan realtime 3D traffic visualization.

---

## Visualization Components

### Vehicle
- 3D cube
- trajectory line
- ID label

---

### Road
- lane boundary
- drivable plane

---

### Environment
- occupancy point cloud

---

## Final Output

```text
Realtime 3D Traffic Scene
```

---

# 6. TEAM DIVISION

# PERSON 1 — AI PERCEPTION ENGINEER

---

## MAIN RESPONSIBILITY

Semua AI model:
- detection
- segmentation
- tracking
- lane detection

---

# TASK 1 — YOLOv11 DETECTION

## Jobdesk

### Setup YOLOv11
- environment
- inference
- CUDA optimization

---

### Detection Pipeline
- bbox extraction
- class filtering
- confidence threshold

---

### Evaluation
- FPS benchmark
- detection accuracy

---

## Deliverables

```text
bbox detection pipeline
```

---

# TASK 2 — SEGMENTATION MODEL

## Jobdesk

### Setup SAM2 / SegFormer

---

### ROI Segmentation

Input:
- vehicle crop
- bbox prompt

---

### Mask Refinement

- morphology
- contour smoothing

---

## Deliverables

```text
instance segmentation pipeline
```

---

# TASK 3 — BYTE TRACKING

## Jobdesk

### Tracking Integration

- object association
- ID persistence
- trajectory smoothing

---

## Deliverables

```text
tracking pipeline
```

---

# TASK 4 — LANE DETECTION

## Jobdesk

### LaneATT setup

---

### Lane postprocessing

- curve smoothing
- lane filtering

---

## Deliverables

```text
lane detection pipeline
```

---

# PERSON 2 — 3D RECONSTRUCTION ENGINEER

---

## MAIN RESPONSIBILITY

Semua:
- geometry
- depth
- reconstruction
- visualization

---

# TASK 1 — DEPTH ESTIMATION

## Jobdesk

### Setup Depth Anything V2

---

### Depth normalization

---

### Depth stabilization

---

## Deliverables

```text
dense depth estimation pipeline
```

---

# TASK 2 — CAMERA CALIBRATION

## Jobdesk

### Intrinsic calibration

---

### Distortion correction

---

## Deliverables

```text
camera parameter
```

---

# TASK 3 — 2D → 3D PROJECTION

## Jobdesk

### Pixel-to-world conversion

---

### Coordinate transformation

---

### Depth association

---

## Deliverables

```text
3D object coordinate
```

---

# TASK 4 — BEV TRANSFORMATION

## Jobdesk

### Homography

---

### Perspective mapping

---

## Deliverables

```text
BEV map
```

---

# TASK 5 — OCCUPANCY RECONSTRUCTION

## Jobdesk

### Point cloud generation

---

### Voxel occupancy

---

### Scene graph

---

## Deliverables

```text
3D occupancy scene
```

---

# TASK 6 — VISUALIZATION

## Jobdesk

### Open3D rendering

---

### Unity integration

---

### Realtime rendering

---

## Deliverables

```text
final visualization system
```

---

# TASK 7 — FINAL INTEGRATION

## Jobdesk

### Merge:
- detection
- segmentation
- depth
- tracking
- reconstruction

---

### Pipeline synchronization

---

### Final testing

---

# 7. DATASET

| Dataset | Usage |
|---|---|
| BDD100K | detection |
| KITTI | depth validation |
| Cityscapes | segmentation |
| nuScenes | BEV testing |

---

# 8. PERFORMANCE TARGET

| Metric | Target |
|---|---|
| FPS | 15–30 FPS |
| Detection Accuracy | >90% |
| Tracking Stability | High |
| Visualization Latency | <100ms |
| Depth Stability | Consistent |

---

# 9. FUTURE IMPROVEMENT

## Possible Upgrade

### 1.
BEVFormer

### 2.
Occupancy Network

### 3.
Gaussian Splatting

### 4.
SLAM Integration

### 5.
TensorRT Optimization

### 6.
Multi-Camera Fusion

---

# 10. FINAL EXPECTED OUTPUT

## System mampu:

- detect kendaraan realtime
- segmentasi kendaraan presisi tinggi
- estimate dense monocular depth
- reconstruct 3D scene
- generate BEV occupancy map
- visualize realtime traffic scene

---
