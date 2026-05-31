import os
import torch
from collections import Counter

# =====================================================================
# 1. CONFIGURATION
# =====================================================================
SOURCE_LABELS_DIR = "clustered_JapanUS/Japan/train/labels"
TARGET_LABELS_DIR = "clustered_JapanUS/United_States/train/labels"
NUM_CLASSES = 4

# ---> THE MATHEMATICAL DAMPENER <---
# This prevents Catastrophic Forgetting in edge-constrained (Nano) models.
# It forces the multipliers to stay within a safe, stable range.
MIN_WEIGHT = 0.5  # Do not drop any class importance below 50%
MAX_WEIGHT = 2.0  # Do not boost any class importance above 200%


# =====================================================================

def get_class_distribution(labels_dir):
    """Reads all YOLO txt files and calculates the true P(y) distribution."""
    counts = Counter({i: 0 for i in range(NUM_CLASSES)})
    total_objects = 0

    if not os.path.exists(labels_dir):
        raise FileNotFoundError(f"Directory not found: {labels_dir}")

    for filename in os.listdir(labels_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(labels_dir, filename), 'r') as f:
                for line in f:
                    try:
                        class_id = int(line.split()[0])
                        if class_id < NUM_CLASSES:
                            counts[class_id] += 1
                            total_objects += 1
                    except (ValueError, IndexError):
                        continue

    # Convert absolute counts to probabilities (percentages)
    probabilities = {k: v / total_objects if total_objects > 0 else 0 for k, v in counts.items()}
    return probabilities, counts


def main():
    print("📊 Calculating Direct 'Oracle' Label-Shift Weights...")

    # 1. Read true distributions from both countries
    p_source, count_source = get_class_distribution(SOURCE_LABELS_DIR)
    p_target, count_target = get_class_distribution(TARGET_LABELS_DIR)

    raw_weights = []
    dampened_weights = []

    print("\n" + "=" * 50)
    print(f"{'Class ID':<10} | {'Raw Math':<15} | {'Dampened Weight':<15}")
    print("=" * 50)

    # 2. Calculate and Dampen the weights
    for i in range(NUM_CLASSES):
        # Prevent division by zero if a class is entirely missing
        if p_source[i] == 0:
            raw_w = 1.0
        else:
            raw_w = p_target[i] / p_source[i]

        # Apply the clipping dampener
        safe_w = max(MIN_WEIGHT, min(MAX_WEIGHT, raw_w))

        raw_weights.append(raw_w)
        dampened_weights.append(safe_w)

        print(f"Class {i:<6} | {raw_w:<15.4f} | {safe_w:<15.4f}")

    print("=" * 50)

    # 3. Save the protected weights to PyTorch tensor
    weights_tensor = torch.tensor(dampened_weights, dtype=torch.float32)
    save_path = os.path.join(os.getcwd(), "domain_weights.pt")
    torch.save(weights_tensor, save_path)

    print(f"\n✅ Safe Domain Weights successfully saved to: {save_path}")
    print("🚀 You may now run `train_robust.py`")


if __name__ == '__main__':
    main()