from ultralytics import YOLO

# Load the custom YAML architecture
model = YOLO("lee-yolov8n.yaml")

# Train the model (using a dummy or real dataset)
results = model.train(data="coco8.yaml", epochs=1, imgsz=640)