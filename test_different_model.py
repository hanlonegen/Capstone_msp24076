from ultralytics import YOLO, FastSAM

# Load a model
# model = YOLO("yolov5n.pt")

# model = YOLO("yolov8n.pt")

# model = YOLO("yolov10n.pt")

model = YOLO("yolo11n.pt")

model.train(
    data="./cfg/NEU.yaml",
    epochs=1,
    imgsz=640,
    workers=0,
)

# Validate the model Fastsam
# model = FastSAM("FastSAM-s.pt")
#
# results = model.val(data="./cfg/coco128-seg.yaml", workers=0)

