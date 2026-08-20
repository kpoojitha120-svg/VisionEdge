from flask import Flask, render_template, request
from ultralytics import YOLO
from pathlib import Path
from collections import Counter
import os

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "yolo11n.pt"

print("Loading YOLO model...")
model = YOLO(str(MODEL_PATH))
print("YOLO model loaded successfully!")

UPLOAD_FOLDER = BASE_DIR / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)


@app.route("/", methods=["GET", "POST"])
def index():
    detections = []
    total_objects = 0
    object_counts = {}

    if request.method == "POST":
        file = request.files.get("image")

        if file and file.filename:
            image_path = UPLOAD_FOLDER / file.filename
            file.save(image_path)

            results = model.predict(
                source=str(image_path),
                device="cpu",
                conf=0.25,
                verbose=False
            )

            result = results[0]

            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = model.names[class_id]

                detections.append({
                    "name": class_name,
                    "confidence": round(confidence, 2)
                })

            total_objects = len(detections)

            object_counts = dict(
                Counter(item["name"] for item in detections)
            )

            os.remove(image_path)

    return render_template(
        "index.html",
        detections=detections,
        total_objects=total_objects,
        object_counts=object_counts
    )


if __name__ == "__main__":
    print("Starting VisionEdge web application...")
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
