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
    # Hypersensitive wrapper for SHAP logic
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
    lists = get_image_lists()

    for target_class_id, image_list in enumerate(lists):
        class_name = CLASS_NAMES[target_class_id]
        os.makedirs(class_name, exist_ok=True)
        print(f"\n--- Starting Batch Analysis for: {class_name.upper()} ---")

        for image_name in image_list:
            full_image_path = os.path.join(BASE_IMAGE_DIR, image_name)
            print(f"Processing: {image_name}")

            # 1. Load the raw image
            img_bgr = cv2.imread(full_image_path)
            if img_bgr is None:
                continue
            img_bgr = cv2.resize(img_bgr, (640, 640))

            # ---------------------------------------------------------
            # NEW: Generate the image with Bounding Boxes
            # We use conf=0.25 so it only draws real, confident detections
            visual_results = model.predict(img_bgr, imgsz=640, verbose=False, conf=0.25)
            img_boxed_bgr = visual_results[0].plot()  # YOLO's built-in drawing tool

            # Convert both images to RGB and normalize for SHAP
            img_boxed_rgb = cv2.cvtColor(img_boxed_bgr, cv2.COLOR_BGR2RGB)
            img_boxed_normalized = img_boxed_rgb.astype(np.float32) / 255.0

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_normalized = img_rgb.astype(np.float32) / 255.0
            # ---------------------------------------------------------

            # 2. Calculate SHAP using the RAW image so gradients aren't ruined
            masker = shap.maskers.Image("blur(20,20)", img_normalized.shape)
            explainer = shap.Explainer(yolo_predict_wrapper, masker, output_names=CLASS_NAMES)
            shap_values = explainer(np.expand_dims(img_normalized, axis=0), max_evals=500, batch_size=10)

            # 3. Plot the results!
            target_shap_array = shap_values.values[..., target_class_id]

            # We pass the BOXED image as the pixel_values so it shows on the left
            shap.image_plot(
                shap_values=[target_shap_array],
                pixel_values=np.expand_dims(img_boxed_normalized, axis=0),
                show=False
            )

            # Save the figure
            base_img_name = image_name.replace(".jpg", "")
            save_name = os.path.join(class_name, f"{class_name}_{base_img_name}.png")
            plt.savefig(save_name, bbox_inches='tight', dpi=300)
            plt.close()

            print(f"Saved: {save_name}")


if __name__ == "__main__":
    main()