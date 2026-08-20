from ultralytics import YOLO

print("Loading YOLO model...")

model = YOLO("../yolo11n.pt")

print("YOLO model loaded successfully!")

results = model.predict(
    source="../bus.jpg",
    device="cpu",
    verbose=False
)

print("YOLO CPU inference: WORKING")
print("Detected objects:")

for box in results[0].boxes:
    class_id = int(box.cls[0])
    confidence = float(box.conf[0])
    class_name = model.names[class_id]

    print(f"- {class_name}: {confidence:.2f}")

print("VisionEdge basic detection test completed!")