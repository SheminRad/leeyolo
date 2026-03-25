from ultralytics import YOLO


def main():
    print("1. Initializing Custom LEE-YOLO Architecture...")
    model = YOLO("lee-yolov8n.yaml")

    print("2. Starting Training on China_Drone Dataset with Nvidia RTX...")
    results = model.train(
        data="china_drone.yaml",
        epochs=300,
        batch=16,  # <-- FIX 1: Drop batch size from 32 to 16
        workers=2,  # <-- FIX 2: Drop workers from 8 to 2 (This saves MASSIVE System RAM)
        imgsz=640,
        device=0,
        optimizer="AdamW",
        name="lee_yolo_rdd2022",
        deterministic=False
    )

    print("✅ Training Complete!")


if __name__ == '__main__':
    main()