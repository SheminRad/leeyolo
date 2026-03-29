from ultralytics import YOLO


def main():
    print("1. Loading your China-trained LEE-YOLO brain...")
    model = YOLO("runs/detect/lee_yolo_rdd2022/weights/best.pt")

    print("2. Running official metrics on the unseen Czech dataset...")
    # This will NOT train the model. It will only grade it.
    metrics = model.val(
        data="czech.yaml",  # The unseen dataset
        split="val",  # Only look at the validation images
        device=0  # Use your RTX 3070 Ti
    )

    print("✅ Evaluation Complete!")
    # Print out the exact numbers for your thesis table
    print(f"mAP50 on Czech Dataset: {metrics.box.map50:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")


if __name__ == '__main__':
    main()