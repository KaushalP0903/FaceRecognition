import cv2
import numpy as np

from insightface.app.common import Face
from insightface.model_zoo import get_model

from config.configurations import (POSE_MODEL,CTX_ID,POSE_ESTIMATION)

class PoseEstimator:

    def __init__(self):

        self.model = get_model(POSE_MODEL)

        if self.model is None:
            raise RuntimeError("Failed to load pose model")

        self.model.prepare(ctx_id=CTX_ID)

        self.model_points = np.array([
            (0.0, 0.0, 0.0),         # Nose
            (-30.0, -30.0, -30.0),   # Left eye
            (30.0, -30.0, -30.0),    # Right eye
            (-25.0, 30.0, -30.0),    # Left mouth
            (25.0, 30.0, -30.0),     # Right mouth
            (0.0, 55.0, -20.0)       # Chin
        ], dtype=np.float32)

    def estimate_pose(self, aligned_face):

        h, w = aligned_face.shape[:2]

        fake_face = Face(bbox=np.array([0, 0, w, h]))

        result = self.model.get(aligned_face, fake_face)

        if result is None:
            return None

        landmarks3d = result.reshape(-1, 3)

        if len(landmarks3d) < 68:
            return None

        image_points = np.array([
            landmarks3d[30][:2],   # Nose
            landmarks3d[36][:2],   # Left eye
            landmarks3d[45][:2],   # Right eye
            landmarks3d[48][:2],   # Left mouth
            landmarks3d[54][:2],   # Right mouth
            landmarks3d[8][:2]     # Chin
        ], dtype=np.float32)

        focal_length = w

        center = (w / 2, h / 2)

        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float32)

        dist_coeffs = np.zeros((4, 1))

        success, rotation_vector, translation_vector = (
            cv2.solvePnP(
                self.model_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
        )

        if not success:
            return None

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

        pose_matrix = cv2.hconcat([rotation_matrix,translation_vector])

        _, _, _, _, _, _, euler_angles = (cv2.decomposeProjectionMatrix(pose_matrix))

        pitch = float(euler_angles[0][0])
        yaw = float(euler_angles[1][0])
        roll = float(euler_angles[2][0])
        
        return {"yaw": yaw, "pitch": pitch, "roll": roll}