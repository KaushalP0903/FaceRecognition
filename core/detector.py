# from ultralytics import YOLO
# from config.configurations import USE_YOLO, YOLO_MODEL

# class PersonDetector:

#     def __init__(self):
#         if USE_YOLO:
#             self.model = YOLO(YOLO_MODEL)

#     def detect(self, frame):

#         if not USE_YOLO:
#             h, w = frame.shape[:2]

#             return [(0, 0, w, h)]

#         results = self.model(frame, imgsz=416, verbose=False)[0]

#         boxes = []

#         for box in results.boxes:

#             if int(box.cls[0]) == 0:

#                 x1, y1, x2, y2 = map(int,box.xyxy[0])

#                 boxes.append((x1, y1, x2, y2))

#         return boxes

import torch

from transformers import (
    RTDetrForObjectDetection,
    RTDetrImageProcessor
)

from config.configurations import (
    USE_PERSON_DETECTOR,
    RTDETR_MODEL,
    PERSON_CONF_THRESHOLD
)


class PersonDetector:

    def __init__(self):

        self.model = None
        self.processor = None

        if USE_PERSON_DETECTOR:

            self.processor = (
                RTDetrImageProcessor.from_pretrained(
                    RTDETR_MODEL,
                    local_files_only=True
                )
            )

            self.model = (
                RTDetrForObjectDetection.from_pretrained(
                    RTDETR_MODEL,
                    local_files_only=True
                )
            )

            self.model.eval()

    def detect(self, frame):

        if not USE_PERSON_DETECTOR:

            h, w = frame.shape[:2]

            return [(0, 0, w, h)]

        inputs = self.processor(
            images=frame,
            return_tensors="pt"
        )

        with torch.no_grad():

            outputs = self.model(
                **inputs
            )

        target_sizes = torch.tensor(
            [frame.shape[:2]]
        )

        results = (
            self.processor.post_process_object_detection(
                outputs,
                target_sizes=target_sizes,
                threshold=PERSON_CONF_THRESHOLD
            )[0]
        )

        persons = []

        for score, label, box in zip(
            results["scores"],
            results["labels"],
            results["boxes"]
        ):

            if int(label) != 0:
                continue

            x1, y1, x2, y2 = map(
                int,
                box.tolist()
            )

            persons.append(
                (x1, y1, x2, y2)
            )

        return persons