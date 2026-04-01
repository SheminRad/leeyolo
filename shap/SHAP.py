import os
import cv2
import numpy as np
import shap
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Import the extraction function from your new file
from find_images import get_image_lists

MODEL_PATH = "/home/shemin/PycharmProjects/leeyolo/runs/detect/lee_yolo_rdd2022/weights/best.pt"
BASE_IMAGE_DIR = "/home/shemin/PycharmProjects/leeyolo/RDD2022/China_Drone/train/images"
CLASS_NAMES = ["longitudinal_crack", "transverse_crack", "alligator_crack", "pothole"]

print(f"Loading Model: {MODEL_PATH}...")
model = YOLO(MODEL_PATH)


def yolo_predict_wrapper(images):
    # Hypersensitive wrapper logic initialized (summed confidences, NMS disabled)
    batch_scores = []
    for img in images:
        img_uint8 = (img * 255).astype(np.uint8)
        results = model.predict(img_uint8, imgsz=640, verbose=False, conf=0.001, iou=0.99)
        class_scores = np.zeros(len(CLASS_NAMES))

        if len(results[0].boxes) > 0:
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            confs = results[0].boxes.conf.cpu().numpy()
            for cls_id in range(len(CLASS_NAMES)):
                if cls_id in classes:
                    class_scores[cls_id] = np.sum(confs[classes == cls_id])
        batch_scores.append(class_scores)
    return np.array(batch_scores)


def main():
    # Fetch the 4 lists populated with 5 images each
    lists = get_image_lists()

    for target_class_id, image_list in enumerate(lists):
        class_name = CLASS_NAMES[target_class_id]

        # Creates the folder (e.g., 'pothole/') if it does not already exist
        os.makedirs(class_name, exist_ok=True)

        print(f"\n--- Starting Batch Analysis for: {class_name.upper()} ---")

        for image_name in image_list:
            full_image_path = os.path.join(BASE_IMAGE_DIR, image_name)
            print(f"Processing: {image_name}")

            img = cv2.imread(full_image_path)
            if img is None:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (640, 640))
            img_normalized = img.astype(np.float32) / 255.0

            masker = shap.maskers.Image("blur(20,20)", img_normalized.shape)
            explainer = shap.Explainer(yolo_predict_wrapper, masker, output_names=CLASS_NAMES)

            # Execute SHAP partition
            shap_values = explainer(np.expand_dims(img_normalized, axis=0), max_evals=500, batch_size=10)

            # Target plotting sequence
            target_shap_array = shap_values.values[..., target_class_id]
            shap.image_plot(
                shap_values=[target_shap_array],
                pixel_values=np.expand_dims(img_normalized, axis=0),
                show=False
            )

            # Format requested: e.g., pothole/pothole_China_Drone_000123.png
            # Replaced .jpg with .png in the output string to ensure correct saving format
            base_img_name = image_name.replace(".jpg", "")
            save_name = os.path.join(class_name, f"{class_name}_{base_img_name}.png")

            plt.savefig(save_name, bbox_inches='tight', dpi=300)
            plt.close()  # Critical: clear RAM between plots to prevent out-of-memory crashes

            print(f"Saved: {save_name}")


if __name__ == "__main__":
    main()