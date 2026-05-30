import matplotlib.pyplot as plt
from ultralytics import YOLO

# =====================================================================
# 1. CONFIGURATION VARIABLES
# =====================================================================
MODEL_DISPLAY_NAME = "Lee Yolo Baseline"
MODEL_PATH = "runs/detect/lee_yolo_rdd2022/weights/best.pt"
DATASET_YAML = "czech.yaml"
KNOWN_GFLOPS = 4.8  # From your previous YOLOv10n console log!


# =====================================================================

def save_overall_table(model_name, dataset, params, gflops, latency, fps, map50, map50_95, precision, recall):
    """Draws the overall hardware and accuracy table."""
    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
    ax.axis('off')
    ax.axis('tight')

    table_data = [
        ["Parameters", f"{params / 1e6:.2f} Million"],
        ["Computational Demand", f"{gflops:.2f} GFLOPs"],
        ["Inference Latency", f"{latency:.2f} ms/image"],
        ["Raw Throughput", f"{fps:.1f} FPS"],
        ["Overall mAP@50", f"{map50 * 100:.2f}%"],
        ["Overall mAP@50-95", f"{map50_95 * 100:.2f}%"],
        ["Overall Precision", f"{precision * 100:.2f}%"],
        ["Overall Recall", f"{recall * 100:.2f}%"]
    ]

    columns = ["Metric", "Value"]
    table = ax.table(cellText=table_data, colLabels=columns, cellLoc='center', loc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.8)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#4A4A4A')
        elif row % 2 == 0:
            cell.set_facecolor('#F3F3F3')

    plt.title(f"Overall Benchmarking: {model_name}\nDataset: {dataset}", pad=20, weight='bold')

    safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "").lower()
    filename = f"table_overall_{safe_name}.png"
    plt.savefig(filename, bbox_inches="tight")
    plt.close()
    print(f"✅ Overall table saved: {filename}")


def save_class_table(model_name, dataset, model_names_dict, class_indices, ap50_array, ap_array):
    """Draws the per-class accuracy breakdown table."""
    fig, ax = plt.subplots(figsize=(8, 3), dpi=300)
    ax.axis('off')
    ax.axis('tight')

    table_data = []

    # Loop through the detected classes and format their scores
    for i, class_idx in enumerate(class_indices):
        class_name = model_names_dict[class_idx].replace("_", " ").title()
        class_map50 = f"{ap50_array[i] * 100:.2f}%"
        class_map = f"{ap_array[i] * 100:.2f}%"
        table_data.append([class_name, class_map50, class_map])

    columns = ["Damage Type", "mAP@50", "mAP@50-95"]

    # If no classes were detected (like your Czech dataset earlier), add a placeholder
    if not table_data:
        table_data.append(["No classes detected", "0.00%", "0.00%"])

    table = ax.table(cellText=table_data, colLabels=columns, cellLoc='center', loc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.8)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#2B5B84')  # A nice blue header to distinguish it from the overall table
        elif row % 2 == 0:
            cell.set_facecolor('#F0F4F8')

    plt.title(f"Per-Class Breakdown: {model_name}\nDataset: {dataset}", pad=20, weight='bold')

    safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "").lower()
    filename = f"table_classes_{safe_name}.png"
    plt.savefig(filename, bbox_inches="tight")
    plt.close()
    print(f"✅ Class breakdown table saved: {filename}")


def main():
    print(f"\n🚀 Loading Model: {MODEL_DISPLAY_NAME}")

    model = YOLO(MODEL_PATH)

    # Architecture Metrics
    params = sum(p.numel() for p in model.model.parameters())
    gflops = KNOWN_GFLOPS

    print("⏳ Running Validation...")
    metrics = model.val(data=DATASET_YAML, split="val", device="mps", verbose=False)

    # Overall Metrics
    map50 = metrics.box.map50
    map50_95 = metrics.box.map
    precision = metrics.box.mp
    recall = metrics.box.mr
    latency_ms = metrics.speed['inference']
    fps = 1000 / latency_ms if latency_ms > 0 else 0

    # 1. Generate Overall Image Table
    save_overall_table(MODEL_DISPLAY_NAME, DATASET_YAML, params, gflops, latency_ms, fps, map50, map50_95, precision,
                       recall)

    # 2. Generate Per-Class Image Table
    class_indices = metrics.box.ap_class_index
    ap50_array = metrics.box.ap50
    ap_array = metrics.box.ap
    save_class_table(MODEL_DISPLAY_NAME, DATASET_YAML, model.names, class_indices, ap50_array, ap_array)


if __name__ == '__main__':
    main()