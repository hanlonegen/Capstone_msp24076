from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n-pose.pt")

# Train the model
results = model.train(data="./cfg/ours-COCO.yaml", epochs=100, imgsz=640, workers=0)


