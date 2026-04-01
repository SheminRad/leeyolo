import matplotlib.pyplot as plt
from sympy import discriminant

from ultralytics import YOLO

# =====================================================================
# 1. CONFIGURATION VARIABLES
# =====================================================================
MODEL_DISPLAY_NAME = "LEE-YOLO"
MODEL_PATH = "/home/shemin/PycharmProjects/leeyolo/runs/detect/lee_yolo_rdd2022/weights/best.pt"
DATASET_YAML = "china_drone.yaml"
dataset_name = DATASET_YAML.split(".")[0]
KNOWN_GFLOPS = 4.91
# =====================================================================

def save_table_as_image(model_name, dataset, params, gflops, latency, fps, map50, map50_95, precision, recall):
    """Draws a beautiful, high-res table using Matplotlib and saves it as a PNG."""
    print("🎨 Generating high-resolution image table for thesis...")

    # Create a figure and axis with no borders
    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
    ax.axis('off')
    ax.axis('tight')

    # Prepare the data matrix
    table_data = [
        ["Parameters", f"{params / 1e6:.2f} Million"],
        ["Computational Demand", f"{gflops:.2f} GFLOPs"],
        ["Inference Latency", f"{latency:.2f} ms/image"],
        ["Raw Throughput", f"{fps:.1f} FPS"],
        ["mAP@50", f"{map50 * 100:.2f}%"],
        ["mAP@50-95", f"{map50_95 * 100:.2f}%"],
        ["Overall Precision", f"{precision * 100:.2f}%"],
        ["Overall Recall", f"{recall * 100:.2f}%"]
    ]

    columns = ["Metric", "Value"]

    # Draw the table
    table = ax.table(cellText=table_data, colLabels=columns, cellLoc='center', loc='center')

    # Styling to make it look academic and clean
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.8)  # Stretch cells slightly for readability

    # Style the header row
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#4A4A4A')  # Dark grey header
        else:
            if row % 2 == 0:
                cell.set_facecolor('#F3F3F3')  # Subtle alternating row colors

    # Add a title
    plt.title(f"Benchmarking Results: {model_name}\nDataset: {dataset}", pad=20, weight='bold')

    # Clean up the filename so it doesn't break if you use spaces
    safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "").lower()
    filename = f"{dataset_name}_{safe_name}.png"

    # Save the file
    plt.savefig(filename, bbox_inches="tight")
    plt.close()

    print(f"✅ Image saved successfully as: {filename}")


def main():
    print(f"\n🚀 Loading Model: {MODEL_DISPLAY_NAME}")
    print(f"📁 Weights: {MODEL_PATH}")
    print(f"📊 Dataset: {DATASET_YAML}\n")

    model = YOLO(MODEL_PATH)

    # Extract Hardware/Architecture Metrics
    # Extract Architecture Metrics using bulletproof native PyTorch
    params = sum(p.numel() for p in model.model.parameters())
    gflops = KNOWN_GFLOPS

    print("\n⏳ Running Validation...")
    metrics = model.val(
        data=DATASET_YAML,
        split="val",
        device=0,
        verbose=False
    )

    # Extract Accuracy & Latency Metrics
    map50 = metrics.box.map50
    map50_95 = metrics.box.map
    precision = metrics.box.mp
    recall = metrics.box.mr
    latency_ms = metrics.speed['inference']
    fps = 1000 / latency_ms if latency_ms > 0 else 0

    # Print the Terminal Table
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
    print(f"{'mAP@50':<30} | {map50 * 100:.2f}%")
    print(f"{'mAP@50-95':<30} | {map50_95 * 100:.2f}%")
    print(f"{'Overall Precision':<30} | {precision * 100:.2f}%")
    print(f"{'Overall Recall':<30} | {recall * 100:.2f}%")
    print("=" * 70 + "\n")

    # Call the new Image Export function
    save_table_as_image(MODEL_DISPLAY_NAME, DATASET_YAML, params, gflops, latency_ms, fps, map50, map50_95, precision,
                        recall)


if __name__ == '__main__':
    main()