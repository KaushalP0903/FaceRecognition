import cv2
from config.configurations import FRAME_SIZE, CAMERA_SOURCE

class FrameManager:

    def __init__(self):
        self.cap = cv2.VideoCapture(CAMERA_SOURCE)
        self.frame_count = 0

    def get_frame(self):

        ret, frame = self.cap.read()

        if not ret:
            return None

        self.frame_count += 1
        frame = cv2.resize(frame, FRAME_SIZE)

        return frame

    def release(self):
        self.cap.release()