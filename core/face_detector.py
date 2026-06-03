from insightface.model_zoo import get_model

from config.configurations import (CTX_ID,FACE_CONF_THRESHOLD,FACE_MARGIN,RETINAFACE_MODEL)

class FaceDetector:

    def __init__(self):

        self.detector = get_model(RETINAFACE_MODEL)
        self.detector.prepare(ctx_id=CTX_ID, input_size=(640, 640), nms=0.4)

    def detect(self, frame, person_boxes):

        detections = []
        frame_h, frame_w = frame.shape[:2]

        for (x1, y1, x2, y2) in person_boxes:
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame_w, x2)
            y2 = min(frame_h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            roi = frame[y1:y2, x1:x2]
            faces, landmarks = self.detector.detect(roi)

            if faces is None:
                continue

            for i, face in enumerate(faces):
                fx1, fy1, fx2, fy2, score = face

                if score < FACE_CONF_THRESHOLD:
                    continue

                fx1, fy1, fx2, fy2 = map(int, [fx1, fy1, fx2, fy2])

                mx1 = max(0, x1 + fx1 - FACE_MARGIN)
                my1 = max(0, y1 + fy1 - FACE_MARGIN)
                mx2 = min(frame_w, x1 + fx2 + FACE_MARGIN)
                my2 = min(frame_h, y1 + fy2 + FACE_MARGIN)

                abs_landmarks = [
                    (float(px + x1), float(py + y1))
                    for px, py in landmarks[i]
                ]

                detections.append({"bbox": (mx1,my1,mx2,my2), "landmarks": abs_landmarks})

        return detections
