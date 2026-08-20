import cv2
from ultralytics import YOLO
from config import MODEL_PATH, CONFIDENCE_THRESHOLD, DEVICE


print("Loading YOLO model...")
model = YOLO(str(MODEL_PATH))
print("YOLO model loaded successfully!")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    raise SystemExit

print("Webcam opened successfully!")
print("A camera window should appear.")
print("Press Q inside the camera window to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read webcam frame.")
        break

    results = model.predict(
        source=frame,
        device=DEVICE,
        conf=CONFIDENCE_THRESHOLD,
        verbose=False
    )

    annotated_frame = results[0].plot()

    cv2.imshow("VisionEdge - Live Detection", annotated_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        print("Q pressed. Stopping...")
        break

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)

print("VisionEdge webcam detection stopped.")