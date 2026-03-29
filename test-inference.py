from ultralytics import YOLO


def main():
    print("1. Loading your trained LEE-YOLO brain...")
    # IMPORTANT: Make sure this path points exactly to where your best.pt is located!
    # Based on your screenshots, it should be in the 'lee_yolo_rdd20224' folder.
    model = YOLO("/home/khan/Shemin/Pycharm/leeyolo/runs/detect/lee_yolo_rdd2022/weights/best.pt")

    print("2. Running inference on a test image...")
    # Point 'source' to an image or video you want to test.
    # Let's use one of the images from your China_Drone dataset for now.
    results = model.predict(
        source="RDD2022/Czech/test/images/",  # <-- You can point this to a folder or a specific .jpg file
        conf=0.25,  # Confidence threshold: Only draw boxes if the model is at least 25% sure
        save=True,  # This tells the engine to save a copy of the image with the boxes drawn on it
        device=0  # Use your RTX 3070 Ti for lightning-fast inference
    )

    print("✅ Inference complete! Check the new 'runs/detect/predict' folder.")


if __name__ == '__main__':
    main()