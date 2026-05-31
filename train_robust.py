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

        # ---> RECALIBRATED CAR-TO-CAR COVARIATE SHIFT <---
        # Subtle lighting shifts (realistic dashcam differences)
        hsv_h=0.015,  # Very slight hue shift (different camera sensors)
        hsv_s=0.3,  # 30% saturation (simulates slightly wet vs dry roads)
        hsv_v=0.3,  # 30% brightness (simulates overcast vs sunny, without erasing cracks)

        # Subtle geometric shifts (Dashboard cameras don't fly)
        degrees=0.0,  # Cars don't bank like drones. Keep this 0.
        perspective=0.0,  # Keep dashboard perspective locked.
        scale=0.1,  # 10% zoom (simulates slightly different camera focal lengths)
        fliplr=0.5,  # 50% chance to flip left-right (Still great for training)

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