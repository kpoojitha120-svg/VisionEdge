from pathlib import Path

# VisionEdge project paths
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "yolo11n.pt"
TEST_IMAGE = BASE_DIR / "bus.jpg"

# Detection settings
CONFIDENCE_THRESHOLD = 0.25
DEVICE = "cpu"