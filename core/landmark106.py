import numpy as np

from insightface.app.common import Face
from insightface.model_zoo import get_model

from config.configurations import (LANDMARK106_MODEL,CTX_ID)


class Landmark106:

    def __init__(self):

        self.model = get_model(LANDMARK106_MODEL)

        if self.model is None:
            raise RuntimeError("Failed to load 106 landmark model")

        self.model.prepare(ctx_id=CTX_ID)

    def get_landmarks(self, frame, bbox):

        x1, y1, x2, y2 = bbox
        face_crop = frame[y1:y2, x1:x2]

        if face_crop.size == 0:
            return None

        fake_face = Face(bbox=np.array([0, 0, face_crop.shape[1], face_crop.shape[0]]))

        landmarks = self.model.get(face_crop, fake_face)

        if landmarks is None:
            return None

        landmarks = landmarks.reshape(-1, 2)

        landmarks[:, 0] += x1
        landmarks[:, 1] += y1

        return landmarks