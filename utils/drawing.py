import cv2
import numpy as np

def draw_tracks(frame, tracks):

    for t in tracks:

        x1, y1, x2, y2 = t["bbox"]
        identity = t.get("identity","Unknown")
        score = t.get("score", 0)

        label = f"{identity} ({score:.2f})"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        landmarks106 = t.get("landmarks106",None)

        if landmarks106 is not None:

            landmarks106 = np.array(landmarks106,dtype=np.int32)

            for point in landmarks106:

                px = int(float(point[0]))
                py = int(float(point[1]))

                cv2.circle(frame, (px, py), 1, (0, 0, 0), -1)

            key_indices = {
                "left_eye": 38,
                "right_eye": 88,
                "nose": 86,
                "left_mouth": 52,
                "right_mouth": 61,
                "chin": 8
            }

            for name, idx in key_indices.items():

                point = landmarks106[idx]

                px = int(float(point[0]))
                py = int(float(point[1]))

                cv2.circle(frame, (px, py), 2, (0, 0, 255), -1)
                #cv2.putText(frame, name, (px + 6, py - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)