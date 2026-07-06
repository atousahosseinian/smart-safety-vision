from PIL import Image
from ultralytics import YOLO


class SafetyDetector:
    def __init__(self, model_name: str = "yolo11n.pt"):
        self.model = YOLO(model_name)

    def detect_image(self, image: Image.Image, confidence: float = 0.25):
        results = self.model(image, conf=confidence)
        return results[0]