from collections import Counter

from ultralytics import YOLO
from config import MODEL_PATH, TEST_IMAGE, CONFIDENCE_THRESHOLD, DEVICE


print("Loading YOLO model...")

model = YOLO(str(MODEL_PATH))

print("YOLO model loaded successfully!")

results = model.predict(
    source=str(TEST_IMAGE),
    device=DEVICE,
    conf=CONFIDENCE_THRESHOLD,
    verbose=False
)

result = results[0]

print("YOLO CPU inference: WORKING")
print("Detected objects:")

detected_objects = []

for box in result.boxes:
    class_id = int(box.cls[0])
    confidence = float(box.conf[0])
    class_name = model.names[class_id]

    detected_objects.append(class_name)

    print(f"- {class_name}: {confidence:.2f}")

print()
print(f"Total objects detected: {len(detected_objects)}")

object_counts = Counter(detected_objects)

for object_name, count in object_counts.items():
    print(f"{object_name}: {count}")

print()
print("VisionEdge basic detection test completed!")