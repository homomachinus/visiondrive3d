from ultralytics import YOLO

def main():
    model = YOLO("yolo11n-seg.pt")
    model.train(
        data="datasets/segme/data.yaml",
        epochs=100,
        imgsz=640,
        batch=4,
        workers=0
    )

if __name__ == "__main__":
    main()