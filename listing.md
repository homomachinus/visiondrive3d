```text
DriveVerse/
│
├── README.md
├── requirements.txt
├── environment.yml
├── main.py
├── config.py
├── .gitignore
│
├── configs/
│   │
│   ├── detection.yaml
│   ├── segmentation.yaml
│   ├── depth.yaml
│   ├── tracking.yaml
│   ├── lane.yaml
│   ├── bev.yaml
│   ├── visualization.yaml
│   └── camera.yaml
│
├── data/
│   │
│   ├── raw/
│   │   ├── videos/
│   │   │   ├── dashcam_01.mp4
│   │   │   ├── dashcam_02.mp4
│   │   │   └── test_drive.mp4
│   │   │
│   │   ├── images/
│   │   └── calibration/
│   │       ├── chessboard_01.jpg
│   │       ├── chessboard_02.jpg
│   │       └── chessboard_03.jpg
│   │
│   ├── processed/
│   │   ├── frames/
│   │   ├── masks/
│   │   ├── depth/
│   │   ├── bev/
│   │   └── pointcloud/
│   │
│   ├── annotations/
│   │   ├── detection/
│   │   └── segmentation/
│   │
│   └── outputs/
│       ├── videos/
│       ├── visualizations/
│       ├── logs/
│       └── metrics/
│
├── models/
│   │
│   ├── yolov11/
│   │   ├── yolov11.pt
│   │   ├── inference.py
│   │   ├── detector.py
│   │   ├── preprocess.py
│   │   └── postprocess.py
│   │
│   ├── segmentation/
│   │   ├── sam2/
│   │   │   ├── sam2.pt
│   │   │   ├── predictor.py
│   │   │   └── mask_generator.py
│   │   │
│   │   └── segformer/
│   │       ├── segformer.pth
│   │       └── segmentor.py
│   │
│   ├── depth_anything/
│   │   ├── depth_anything_v2.pth
│   │   ├── depth_estimator.py
│   │   ├── preprocess.py
│   │   └── postprocess.py
│   │
│   ├── bytetrack/
│   │   ├── tracker.py
│   │   ├── kalman_filter.py
│   │   └── matching.py
│   │
│   └── laneatt/
│       ├── laneatt.pth
│       ├── lane_detector.py
│       └── postprocess.py
│
├── src/
│   │
│   ├── detection/
│   │   ├── run_detection.py
│   │   ├── bbox_filter.py
│   │   ├── confidence_filter.py
│   │   └── class_mapper.py
│   │
│   ├── segmentation/
│   │   ├── run_segmentation.py
│   │   ├── roi_cropper.py
│   │   ├── mask_refinement.py
│   │   ├── contour_extractor.py
│   │   └── morphology.py
│   │
│   ├── depth/
│   │   ├── run_depth.py
│   │   ├── depth_filter.py
│   │   ├── depth_normalization.py
│   │   ├── depth_smoothing.py
│   │   └── depth_association.py
│   │
│   ├── tracking/
│   │   ├── run_tracking.py
│   │   ├── trajectory.py
│   │   ├── id_manager.py
│   │   └── motion_estimator.py
│   │
│   ├── lane/
│   │   ├── run_lane_detection.py
│   │   ├── lane_filter.py
│   │   ├── lane_smoothing.py
│   │   └── drivable_area.py
│   │
│   ├── calibration/
│   │   ├── calibrate_camera.py
│   │   ├── intrinsic.py
│   │   ├── distortion.py
│   │   └── chessboard_detector.py
│   │
│   ├── projection/
│   │   ├── pixel_to_world.py
│   │   ├── coordinate_transform.py
│   │   ├── projection_math.py
│   │   └── object_position.py
│   │
│   ├── bev/
│   │   ├── homography.py
│   │   ├── inverse_perspective.py
│   │   ├── bev_generator.py
│   │   └── occupancy_map.py
│   │
│   ├── reconstruction/
│   │   ├── pointcloud_generator.py
│   │   ├── voxelization.py
│   │   ├── scene_graph.py
│   │   ├── occupancy_grid.py
│   │   └── mesh_builder.py
│   │
│   ├── visualization/
│   │   ├── open3d_renderer.py
│   │   ├── unity_bridge.py
│   │   ├── trajectory_visualizer.py
│   │   ├── lane_visualizer.py
│   │   ├── object_visualizer.py
│   │   └── realtime_viewer.py
│   │
│   ├── utils/
│   │   ├── logger.py
│   │   ├── fps_counter.py
│   │   ├── video_writer.py
│   │   ├── image_utils.py
│   │   ├── geometry_utils.py
│   │   └── config_loader.py
│   │
│   └── pipeline/
│       ├── full_pipeline.py
│       ├── synchronize.py
│       ├── frame_manager.py
│       └── pipeline_controller.py
│
├── notebooks/
│   │
│   ├── detection_test.ipynb
│   ├── segmentation_test.ipynb
│   ├── depth_test.ipynb
│   ├── tracking_test.ipynb
│   ├── bev_test.ipynb
│   ├── reconstruction_test.ipynb
│   └── visualization_test.ipynb
│
├── scripts/
│   │
│   ├── download_models.sh
│   ├── run_pipeline.sh
│   ├── benchmark.sh
│   ├── export_tensorrt.sh
│   └── train_segmentation.sh
│
├── docs/
│   │
│   ├── architecture.md
│   ├── workflow.md
│   ├── installation.md
│   ├── api_reference.md
│   └── experiment_results.md
│
├── experiments/
│   │
│   ├── exp_01/
│   ├── exp_02/
│   ├── exp_03/
│   └── logs/
│
└── tests/
    │
    ├── test_detection.py
    ├── test_segmentation.py
    ├── test_depth.py
    ├── test_tracking.py
    ├── test_projection.py
    ├── test_bev.py
    └── test_visualization.py
```

### FLOW ANTAR FILE

```text
main.py
│
├── run_detection.py
│       ↓
│   YOLOv11 Detector
│       ↓
│   bbox output
│
├── run_segmentation.py
│       ↓
│   SAM2 / SegFormer
│       ↓
│   segmentation mask
│
├── run_depth.py
│       ↓
│   Depth Anything V2
│       ↓
│   dense depth map
│
├── run_tracking.py
│       ↓
│   ByteTrack
│       ↓
│   tracked object
│
├── pixel_to_world.py
│       ↓
│   2D → 3D coordinate
│
├── bev_generator.py
│       ↓
│   top-view occupancy map
│
├── pointcloud_generator.py
│       ↓
│   3D reconstruction
│
└── realtime_viewer.py
        ↓
    Open3D / Unity visualization
```

### PEMBAGIAN FOLDER BERDASARKAN PERSON

# PERSON 1 — AI PERCEPTION

```text
models/yolov11/
models/segmentation/
models/bytetrack/
models/laneatt/

src/detection/
src/segmentation/
src/tracking/
src/lane/
```

---

# PERSON 2 — RECONSTRUCTION & VISUALIZATION

```text
models/depth_anything/

src/depth/
src/calibration/
src/projection/
src/bev/
src/reconstruction/
src/visualization/
```

---

# FILE PALING PENTING

| File                | Fungsi                 |
| ------------------- | ---------------------- |
| main.py             | entry point            |
| full_pipeline.py    | pipeline utama         |
| run_detection.py    | YOLOv11 inference      |
| run_segmentation.py | segmentation inference |
| run_depth.py        | depth estimation       |
| pixel_to_world.py   | 2D→3D conversion       |
| occupancy_grid.py   | reconstruction         |
| realtime_viewer.py  | visualization          |

---

# ALUR EKSEKUSI REALTIME

```text
video frame
↓
YOLOv11
↓
bbox
↓
crop ROI
↓
segmentation model
↓
mask
↓
Depth Anything
↓
depth map
↓
depth association
↓
3D coordinate
↓
ByteTrack
↓
BEV mapping
↓
occupancy reconstruction
↓
Open3D visualization
```
