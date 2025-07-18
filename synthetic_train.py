from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")

# Train the model
results = model.train(data="./cfg/synthetic_images_0625.yaml", epochs=100, imgsz=640, workers=0)


