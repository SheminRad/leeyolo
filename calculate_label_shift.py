import os
from collections import Counter
import numpy as np
import torch


def count_classes(labels_dir, num_classes=4):
    """Counts the occurrences of each class in a YOLO labels directory."""
    class_counts = Counter({i: 0 for i in range(num_classes)})

    if not os.path.exists(labels_dir):
        print(f"⚠️ Directory not found: {labels_dir}")
        return class_counts

    for filename in os.listdir(labels_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(labels_dir, filename), 'r') as f:
                for line in f:
                    class_id = int(line.split()[0])
                    if 0 <= class_id < num_classes:
                        class_counts[class_id] += 1

    return class_counts


def main():
    # 1. Define your label directories (Update these paths if needed!)
    # Source Domain (p)
    china_labels = "RDD2022/China_Drone/train/labels"
    # Target Domain (q)
    czech_labels = "RDD2022/Czech/train/labels"

    print("🔍 Scanning Source Domain (China)...")
    p_counts = count_classes(china_labels)

    print("🔍 Scanning Target Domain (Czech)...")
    q_counts = count_classes(czech_labels)

    # 2. Convert raw counts to probabilities: P(y)
    total_p = sum(p_counts.values()) or 1
    total_q = sum(q_counts.values()) or 1

    p_y = np.array([p_counts[i] / total_p for i in range(4)])
    q_y = np.array([q_counts[i] / total_q for i in range(4)])

    # 3. Calculate Importance Weights: w(y) = q(y) / p(y)
    # We add a tiny epsilon (1e-6) to prevent dividing by zero if a class is missing
    epsilon = 1e-6
    w_y = (q_y + epsilon) / (p_y + epsilon)

    # Normalize weights so they average out to 1.0 (maintains overall learning rate stability)
    w_y_normalized = w_y / (np.sum(w_y) / len(w_y))

    # =================================================================
    # NEW: AUTOMATIC TENSOR EXPORT
    # =================================================================
    weights_tensor = torch.tensor(w_y_normalized, dtype=torch.float32)

    # Save the tensor to the root of your PyCharm project
    save_path = "domain_weights.pt"
    torch.save(weights_tensor, save_path)

    print("\n" + "=" * 50)
    print("✅ AUTOMATION SUCCESSFUL")
    print(f"Label-shift weights saved locally to: {save_path}")
    print(f"Weights: {w_y_normalized}")
    print("=" * 50)

if __name__ == "__main__":
    main()