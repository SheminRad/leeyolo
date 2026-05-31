import matplotlib.pyplot as plt
from ultralytics import YOLO

# =====================================================================
# 1. CONFIGURATION VARIABLES
# =====================================================================
MODEL_DISPLAY_NAME = "LEE-YOLO"
MODEL_PATH = "/home/shemin/PycharmProjects/leeyolo/runs/detect/robust/detect/lee_yolo_fully_robust_m4/weights/best.pt"
DATASET_YAML = "china_drone.yaml"
dataset_name = DATASET_YAML.split(".")[0]
KNOWN_GFLOPS = 4.91

# NEW: Set the strictness of your model.
# 0.25 is standard for YOLO edge deployments.
CONFIDENCE_THRESHOLD = 0.25
# =====================================================================

def save_table_as_image(model_name, dataset, params, gflops, latency, fps, map50, map50_95, precision, recall, conf):
    """Draws a beautiful, high-res table using Matplotlib and saves it as a PNG."""
    print("🎨 Generating high-resolution image table for thesis...")

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300) # Slightly taller to fit the new row
    ax.axis('off')
    ax.axis('tight')

    # Prepare the data matrix (Added Confidence Threshold here)
    table_data = [
        ["Parameters", f"{params / 1e6:.2f} Million"],
        ["Computational Demand", f"{gflops:.2f} GFLOPs"],
        ["Inference Latency", f"{latency:.2f} ms/image"],
        ["Raw Throughput", f"{fps:.1f} FPS"],
        ["Validation Confidence", f"{conf * 100:.0f}% Threshold"], # <--- NEW
        ["mAP@50", f"{map50 * 100:.2f}%"],
        ["mAP@50-95", f"{map50_95 * 100:.2f}%"],
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
        else:
            if row % 2 == 0:
                cell.set_facecolor('#F3F3F3')

    plt.title(f"Benchmarking Results: {model_name}\nDataset: {dataset}", pad=20, weight='bold')

    safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "").lower()
    filename = f"{dataset_name}_{safe_name}.png"

    plt.savefig(filename, bbox_inches="tight")
    plt.close()

    print(f"✅ Image saved successfully as: {filename}")


def main():
    print(f"\n🚀 Loading Model: {MODEL_DISPLAY_NAME}")
    print(f"📁 Weights: {MODEL_PATH}")
    print(f"📊 Dataset: {DATASET_YAML}\n")

    model = YOLO(MODEL_PATH)

    params = sum(p.numel() for p in model.model.parameters())
    gflops = KNOWN_GFLOPS

    print(f"\n⏳ Running Validation at {CONFIDENCE_THRESHOLD*100}% Confidence...")
    metrics = model.val(
        data=DATASET_YAML,
        split="val",
        device=0,
        conf=CONFIDENCE_THRESHOLD,  # <--- NEW: Forces the model to only count confident predictions
        verbose=False
    )

    map50 = metrics.box.map50
    map50_95 = metrics.box.map
    precision = metrics.box.mp
    recall = metrics.box.mr
    latency_ms = metrics.speed['inference']
    fps = 1000 / latency_ms if latency_ms > 0 else 0

    print("\n" + "=" * 70)
    print(f"{'THESIS BENCHMARKING RESULTS':^70}")
    print("=" * 70)
    print(f"Model Name:      {MODEL_DISPLAY_NAME}")
    print(f"Evaluation Data: {DATASET_YAML}")
    print("-" * 70)
    print(f"{'Metric':<30} | {'Value':<35}")
    print("-" * 70)
    print(f"{'Parameters':<30} | {params / 1e6:.2f} Million")
    print(f"{'Computational Demand':<30} | {gflops:.2f} GFLOPs")
    print(f"{'Inference Latency':<30} | {latency_ms:.2f} ms/image")
    print(f"{'Raw Throughput':<30} | {fps:.1f} FPS")
    print(f"{'Validation Confidence':<30} | {CONFIDENCE_THRESHOLD * 100:.0f}% Threshold") # <--- NEW
    print(f"{'mAP@50':<30} | {map50 * 100:.2f}%")
    print(f"{'mAP@50-95':<30} | {map50_95 * 100:.2f}%")
    print(f"{'Overall Precision':<30} | {precision * 100:.2f}%")
    print(f"{'Overall Recall':<30} | {recall * 100:.2f}%")
    print("=" * 70 + "\n")

    # Pass the confidence threshold to the image generator
    save_table_as_image(MODEL_DISPLAY_NAME, DATASET_YAML, params, gflops, latency_ms, fps, map50, map50_95, precision, recall, CONFIDENCE_THRESHOLD)

if __name__ == '__main__':
    main()