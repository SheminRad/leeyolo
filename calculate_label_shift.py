import os
import torch
import numpy as np
from collections import Counter
from ultralytics import YOLO


def calculate_bbse_weights():
    print("🧮 Running Black Box Shift Estimation (BBSE)")

    # ==========================================
    # 1. CONFIGURATION
    # ==========================================
    MODEL_PATH = "/home/shemin/PycharmProjects/leeyolo/runs/detect/lee_yolo/weights/best.pt"  # Your Phase 1 model

    SOURCE_VAL_YAML = "japan.yaml"  # The Drone dataset
    TARGET_TRAIN_DIR = "clustered_JapanUS/United_States/train/images"  # The Car dataset (IMAGES ONLY)
    SOURCE_TRAIN_LABELS = "clustered_JapanUS/Japan/train/labels"  # To get P_train(y)

    num_classes = 4
    model = YOLO(MODEL_PATH)

    # ==========================================
    # 2. GET CONFUSION MATRIX (C) FROM SOURCE
    # ==========================================
    print("\n1/4: Calculating Source Confusion Matrix (C)...")
    # Run validation on the SOURCE set to see how the model behaves
    metrics = model.val(data=SOURCE_VAL_YAML, split='val', verbose=False)

    # Ultralytics CM includes a 'background' class at the end, we slice it [0:4, 0:4]
    # YOLO CM format: rows are True classes, columns are Predicted classes
    raw_cm = metrics.confusion_matrix.matrix[0:num_classes, 0:num_classes]

    # We need C_{ij} = P(pred=i | true=j).
    # Transpose YOLO's CM so rows=pred, cols=true, then column-normalize.
    C = raw_cm.T
    col_sums = C.sum(axis=0)

    # Avoid division by zero
    col_sums[col_sums == 0] = 1
    C_norm = C / col_sums

    # ==========================================
    # 3. GET PREDICTIONS (q) ON TARGET
    # ==========================================
    print("\n2/4: Running Unsupervised Inference on Target Domain to get (q)...")
    # We run the model on Drone images and just count what it *thinks* it sees
    results = model.predict(source=TARGET_TRAIN_DIR, stream=True, verbose=False, conf=0.25)

    pred_counts = np.zeros(num_classes)
    total_preds = 0

    for r in results:
        for c in r.boxes.cls:
            pred_counts[int(c)] += 1
            total_preds += 1

    q = pred_counts / total_preds if total_preds > 0 else pred_counts

    # ==========================================
    # 4. SOLVE BBSE: P_test(y) = C^-1 * q
    # ==========================================
    print("\n3/4: Solving matrix inversion: P_test = C^-1 * q")
    # We use pseudo-inverse (pinv) because C might be ill-conditioned/singular
    C_inv = np.linalg.pinv(C_norm)
    p_test_estimated = np.dot(C_inv, q)

    # Mathematical safeguard: clip negative probabilities to 0
    p_test_estimated = np.clip(p_test_estimated, 0, None)

    # Re-normalize to ensure they sum to 1.0
    if p_test_estimated.sum() > 0:
        p_test_estimated /= p_test_estimated.sum()

    # ==========================================
    # 5. GET SOURCE DISTRIBUTION & FINAL WEIGHTS
    # ==========================================
    print("\n4/4: Calculating Final Importance Weights (w)...")
    # Count true labels from Source (Car) to get P_train(y)
    true_source_counts = np.zeros(num_classes)
    total_source = 0
    for file in os.listdir(SOURCE_TRAIN_LABELS):
        if file.endswith(".txt"):
            with open(os.path.join(SOURCE_TRAIN_LABELS, file), 'r') as f:
                for line in f:
                    cls_id = int(line.strip().split()[0])
                    true_source_counts[cls_id] += 1
                    total_source += 1

    p_train = true_source_counts / total_source

    weights = []
    print("\n📊 BBSE Final Results:")
    for i in range(num_classes):
        train_prob = p_train[i] if p_train[i] > 0 else 1e-6
        w_y = p_test_estimated[i] / train_prob
        weights.append(w_y)

        print(f"Class {i}:")
        print(f"  q (Predicted on Drone): {q[i]:.4f}")
        print(f"  P_test (Reconstructed): {p_test_estimated[i]:.4f}")
        print(f"  P_train (Source True):  {train_prob:.4f}")
        print(f"  --> Weight: {w_y:.4f}\n")

    # Save for LEE-YOLO
    weight_tensor = torch.tensor(weights, dtype=torch.float32)
    torch.save(weight_tensor, "domain_weights.pt")
    print("✅ BBSE weights mathematically derived and saved to domain_weights.pt")


if __name__ == "__main__":
    calculate_bbse_weights()