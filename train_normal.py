from ultralytics import YOLO
import os
import shutil

os.environ["LEE_YOLO_ROBUST_MODE"] = "False"
# 1. Define the Custom Callback
def save_epoch_snapshot(trainer):
    """Copies the live results.png and saves it with the epoch number."""
    # trainer.save_dir is the current runs folder (e.g., runs/detect/train)
    source_graph = os.path.join(trainer.save_dir, 'results.png')

    # Check if the graph has been generated yet
    if os.path.exists(source_graph):
        target_graph = os.path.join(trainer.save_dir, f'loss_epoch{trainer.epoch}.png')
        shutil.copy(source_graph, target_graph)
        print(f"📸 Saved epoch {trainer.epoch} graph snapshot!")

def main():
    print("1. Initializing Custom LEE-YOLO Architecture...")
    model = YOLO("lee-yolov8n.yaml")
    # 2. Inject the callback into the model BEFORE training starts
    # "on_fit_epoch_end" triggers exactly when the epoch finishes its validation and drawing
    model.add_callback("on_fit_epoch_end", save_epoch_snapshot)

    print("2. Starting Training on Japan Dataset with Nvidia RTX...")
    results = model.train(
        data="japan.yaml",
        epochs=30,
        batch=-1,  # <-- FIX 1: Drop batch size from 32 to 16
        imgsz=640,
        device=0,
        optimizer="AdamW",
        name="lee_yolo",
        deterministic=False
    )

    print("✅ Training Complete!")


if __name__ == '__main__':
    main()