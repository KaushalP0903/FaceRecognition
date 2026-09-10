import cv2
import numpy as np

class FaceAligner:

    def __init__(self):

        self.dst = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041]
        ], dtype=np.float32)

    def align(self, image, landmarks106):

        if image is None or image.size == 0:
            return None

        left_eye = landmarks106[38]
        right_eye = landmarks106[88]
        nose = landmarks106[86]
        left_mouth = landmarks106[52]
        right_mouth = landmarks106[61]

        src = np.array([left_eye, right_eye, nose, left_mouth, right_mouth], dtype=np.float32)

        M = cv2.estimateAffinePartial2D(src, self.dst)[0]

        if M is None:
            return None

        aligned = cv2.warpAffine(image, M, (112, 112))

        return aligned