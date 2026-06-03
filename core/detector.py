from ultralytics import YOLO
from config.configurations import USE_YOLO, YOLO_MODEL

class PersonDetector:

    def __init__(self):
        if USE_YOLO:
            self.model = YOLO(YOLO_MODEL)

    def detect(self, frame):

        if not USE_YOLO:
            h, w = frame.shape[:2]

            return [(0, 0, w, h)]

        results = self.model(frame, imgsz=416, verbose=False)[0]

        boxes = []

        for box in results.boxes:

            if int(box.cls[0]) == 0:

                x1, y1, x2, y2 = map(int,box.xyxy[0])

                boxes.append((x1, y1, x2, y2))

        return boxes