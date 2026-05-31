import os
os.environ["LEE_YOLO_ROBUST_MODE"] = "True"

from ultralytics import YOLO


def main():
    print("1. Initializing LEE-YOLO Architecture...")
    # Load your custom architecture (ensure your loss.py has the new weights saved!)
    model = YOLO("lee-yolov8n.yaml")

    print("2. Starting ROBUST Training (Domain Randomization) on Mac M4...")
    results = model.train(
        data="japan.yaml",  # We train on the Source Domain
        epochs=30,
        batch=-1,
        imgsz=640,
        # patience=5,

        # ---> HARDWARE ACCELERATION <---
        # device="mps",  # Utilizes the Apple Silicon M4 Neural Engine

        # ---> COVARIATE SHIFT MITIGATION (Simulating Corruptions) <---
        # Color & Lighting shifts (Different asphalt, weather, sensors)
        hsv_h=0.1,  # Hue variance
        hsv_s=0.7,  # Saturation variance (simulates wet vs dry asphalt)
        hsv_v=0.6,  # Value/Brightness variance (simulates harsh shadows/sunlight)

        # Geometric shifts (Different UAV flight angles and camera lenses)
        degrees=15.0,  # Rotation (simulates UAV banking)
        perspective=0.001,  # Perspective distortion (simulates UAV pitch/roll)
        scale=0.3,  # Zooming in/out (simulates altitude changes)
        fliplr=0.5,  # 50% chance to flip left-right

        # Corruption simulation
        # blur_p=0.1,  # 10% chance to apply motion blur
        # -------------------------------------------------------------

        optimizer="AdamW",
        project="robust/detect",
        name="lee_yolo_robust"
    )

    print("✅ Robust Training Complete! Ready for Edge-Constraint Validation.")


if __name__ == '__main__':
    main()